"""KavachMed CLI entry logic: secret gate before I/O, policy load, pipeline run."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from kavachmed import pipeline, policy as policy_mod, security


def _load_policy(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return policy_mod.load_policy(path)


def main(argv: list[str] | None = None) -> int:
    security.install_excepthook()
    security.harden_process()

    ap = argparse.ArgumentParser(description="KavachMed privacy firewall")
    ap.add_argument("--input", required=True, help="Input directory")
    ap.add_argument("--output", required=True, help="Output directory")
    args = ap.parse_args(argv)

    policy_path = Path(os.environ.get("KAVACHMED_POLICY", "config/policy.yaml"))
    policy = _load_policy(policy_path)
    sec_cfg = policy.get("security", {})

    # Fail-closed secret gate BEFORE reading input or creating output/
    try:
        keys = security.load_keys(
            os.environ,
            pbkdf2_iterations=int(sec_cfg.get("pbkdf2_iterations", 600_000)),
            pbkdf2_salt=str(sec_cfg.get("pbkdf2_salt", "kavachmed/v5/stretch")).encode("utf-8"),
            hkdf_version=str(sec_cfg.get("hkdf_version", "v5")).encode("utf-8"),
        )
    except security.StageError as exc:
        sys.stderr.write(f"ERROR {exc}\n")
        return 2
    finally:
        os.environ.pop("RAKSHA_SECRET", None)

    return pipeline.run(Path(args.input), Path(args.output), keys, policy)

