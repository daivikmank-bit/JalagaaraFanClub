"""Policy loader supporting PyYAML when available with a deterministic stdlib fallback."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any

DEFAULT_POLICY: dict[str, Any] = {
    "runtime": {
        "max_json_depth": 4000,
        "residual_passes": 3,
        "psn_chars": 8,
        "reference_doc_byte_copy": True,
    },
    "dates": {
        "shift_min_days": 30,
        "shift_max_days": 365,
        "default_order": "dmy",
        "partial_year": "nearest_or_ref",
        "dataset_ref_year": 2026,
    },
    "age": {
        "band_years": 10,
        "top_band": "85+",
        "pool_lattice": [0, 5, 10, 18, 30, 40, 50, 60, 70, 85],
        "truthful": True,
    },
    "dob": {"mode": "suppress"},
    "location": {
        "keep": "district",
        "pin": "none",
        "remove": ["village", "tehsil", "house", "ward", "mohalla", "landmark"],
    },
    "controller": {
        "enabled": True,
        "assumed_amplification": 4.0,
        "k_target": 5,
        "k_dx": 3,
        "max_qi_kinds": 6,
        "uniqueness_target": 0.25,
        "ladder": ["age20", "state", "dx_rollup", "suppress_qi"],
    },
    "linkage": {
        "tau_high_bits": 12.0,
        "tau_low_bits": 6.0,
        "max_component_size": 200,
        "match_weight_units": "log2",
        "vetoes": ["sex", "age", "relative"],
        "generic_name_tokens": ["devi", "kumar", "singh", "kaur", "md", "mohd", "ji", "sahab", "bhai", "begum"],
    },
    "detect": {
        "stack": "recall_first",
        "arbiter_threshold": 0.35,
        "accept_name": 0.35,
        "crf_weight": 0.25,
        "vehicle_plates": True,
        "doctor_reg_numbers": "remove_digits",
        "label_induction": {"min_values": 3, "min_share": 0.6},
        "default_deny": {
            "enabled": True,
            "score": 0.45,
            "min_letters": 2,
            "lowercase_gazetteer_hits": True,
        },
    },
    "privacy": {
        "age_band_years": 10,
        "date_shift_min_days": 30,
        "date_shift_max_days": 365,
        "min_group_size": 5,
        "rename_date_filenames": False,
    },
    "residual": {
        "layer_b": True,
        "readback": True,
        "b2_min_digits": 6,
        "b2_long_digits": 10,
        "doctor_span_exempt": "registry_full_name_only",
    },
    "security": {
        "min_secret_bytes": 1,
        "min_distinct_bytes": 1,
        "pbkdf2_iterations": 600000,
        "pbkdf2_salt": "kavachmed/v5/stretch",
        "hkdf_version": "v5",
        "log": "counts_only",
        "quarantine_vault": False,
        "kdf_version": "kavachmed/v5",
        "psn_chars": 8,
        "dumpable": False,
        "reexec_scrub": False,
    },
    "compliance": {"terminology": "de-identified"},
}


def _strip_comment(line: str) -> str:
    in_sq = False
    in_dq = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif ch == "#" and not in_sq and not in_dq:
            if i == 0 or line[i - 1].isspace():
                return line[:i]
    return line


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    in_sq = False
    in_dq = False
    for ch in s:
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif not in_sq and not in_dq:
            if ch in "[{":
                depth += 1
            elif ch in "]}":
                depth = max(0, depth - 1)
            elif ch == sep and depth == 0:
                parts.append("".join(cur).strip())
                cur = []
                continue
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_scalar(val: str) -> Any:
    v = val.strip()
    if not v or v in ("null", "~", "None"):
        return None
    if v in ("true", "True", "yes"):
        return True
    if v in ("false", "False", "no"):
        return False
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(x) for x in _split_top_level(inner)]
    if v.startswith("{") and v.endswith("}"):
        inner = v[1:-1].strip()
        out: dict[str, Any] = {}
        if not inner:
            return out
        for item in _split_top_level(inner):
            if ":" in item:
                k, raw_v = item.split(":", 1)
                out[k.strip().strip("'\"")] = _parse_scalar(raw_v)
        return out
    try:
        if v.lstrip("-").isdigit():
            return int(v)
        return float(v)
    except ValueError:
        pass
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, rest = stripped.split(":", 1)
        key = key.strip().strip("'\"")
        rest = rest.strip()
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if rest == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(rest)
    return root


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_policy(path: str | Path | None = None) -> dict[str, Any]:
    """Load policy YAML with defaults merged; uses PyYAML if installed else stdlib parser."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config" / "policy.yaml"
    else:
        path = Path(path)
    if not path.is_file():
        return copy.deepcopy(DEFAULT_POLICY)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
    except ImportError:
        loaded = _parse_simple_yaml(text)
    if not isinstance(loaded, dict):
        return copy.deepcopy(DEFAULT_POLICY)
    return _deep_merge(DEFAULT_POLICY, loaded)
