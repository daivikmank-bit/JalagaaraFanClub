"""ABDM-aware FHIR R4 Bundle adapter, projector, attachment handler, and validator (D7, D8, D12)."""
from __future__ import annotations

import base64
import binascii
import copy
import datetime as dt
import html
import json
import re
from typing import Any

from kavachmed import dates as dates_mod
from kavachmed.contracts import (
    Edit,
    Evidence,
    Field,
    FileEntry,
    LocEntry,
    LocMap,
    ParsedFile,
    PrivacyPlan,
    RecKey,
    Record,
    Resolution,
)
from kavachmed.privacy import bands as bands_mod
from kavachmed.privacy import location as loc_mod
from kavachmed.security import CROCKFORD_ALPHABET, Keys, StageError

REGISTER_FHIR = True

_FHIR_DATE_RE = re.compile(
    r"^\d{4}(?:-\d{2}(?:-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?)?)?$"
)
_ISO_PREFIX_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})(T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?$"
)
_TAG_SPLIT_RE = re.compile(r"(<[^>]+>)")

_PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
_AADHAAR_RE = re.compile(r"(?<!\d)[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_PIN_RE = re.compile(r"\b[1-9]\d{5}\b")
_MRN_RE = re.compile(r"\b(?:SNJ|JJH|ASS|DHH|UHID|MRN|IPD|OPD|ADM|[A-Z]{2,5})[-/](?:\d{2,4}[-/])?\d{4,8}\b")


def _encode_crockford40(digest: bytes) -> str:
    val = int.from_bytes(digest[:5], "big")
    return "".join(CROCKFORD_ALPHABET[(val >> (5 * (7 - i))) & 31] for i in range(8))


def _dumps_iterative(obj: Any) -> str:
    """Iterative indented JSON serializer matching json.dumps(..., indent=2, ensure_ascii=False) without recursion."""
    parts: list[str] = []
    # Stack items: ("VAL", val, depth) | ("STR", s)
    stack: list[tuple[Any, ...]] = [("VAL", obj, 0)]
    while stack:
        op = stack.pop()
        tag = op[0]
        if tag == "LIT":
            parts.append(op[1])
            continue
        val, depth = op[1], op[2]
        if val is None or isinstance(val, (bool, int, float, str)):
            parts.append(json.dumps(val, ensure_ascii=False))
        elif isinstance(val, dict):
            if not val:
                parts.append("{}")
                continue
            indent = "  " * (depth + 1)
            close_indent = "  " * depth
            parts.append("{\n")
            stack.append(("LIT", f"\n{close_indent}" + "}"))
            items = list(val.items())
            for idx in range(len(items) - 1, -1, -1):
                k, v = items[idx]
                if idx < len(items) - 1:
                    stack.append(("LIT", ",\n"))
                stack.append(("VAL", v, depth + 1))
                stack.append(("LIT", f"{indent}{json.dumps(str(k), ensure_ascii=False)}: "))
        elif isinstance(val, list):
            if not val:
                parts.append("[]")
                continue
            indent = "  " * (depth + 1)
            close_indent = "  " * depth
            parts.append("[\n")
            stack.append(("LIT", f"\n{close_indent}]"))
            for idx in range(len(val) - 1, -1, -1):
                if idx < len(val) - 1:
                    stack.append(("LIT", ",\n"))
                stack.append(("VAL", val[idx], depth + 1))
                stack.append(("LIT", indent))
        else:
            parts.append(json.dumps(val, ensure_ascii=False))
    return "".join(parts)


def _shift_iso_preserve_precision(val: str, days: int) -> str:
    m = _ISO_PREFIX_RE.match(val.strip())
    if not m:
        return val
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        shifted = dt.date(y, mo, d) + dt.timedelta(days=days)
    except ValueError:
        return val
    suffix = m.group(4) or ""
    return f"{shifted.year:04d}-{shifted.month:02d}-{shifted.day:02d}{suffix}"


def _surrogate_id(keys: Keys | None, system: str, orig: str, psn: str | None) -> str:
    if keys is not None:
        digest = keys.keyed("surr", system, orig)
        return f"ID-{_encode_crockford40(digest)}"
    return psn or "[ID]"


def _surrogate_telecom(keys: Keys | None, system: str, orig: str) -> str:
    if keys is not None:
        digest = keys.keyed("surr", "telecom", system, orig)
        if system.lower() == "email":
            return f"patient.{_encode_crockford40(digest).lower()}@redacted.invalid"
        return f"TEL-{_encode_crockford40(digest)}"
    return "[EMAIL]" if system.lower() == "email" else "[PHONE]"


def _sanitize_plain_pii(
    text: str,
    known_names: list[str],
    psn: str | None,
    shift_days: int,
    age_band: str | None,
) -> str:
    out = text
    for nm in sorted({n.strip() for n in known_names if len(n.strip()) >= 2}, key=len, reverse=True):
        out = re.sub(rf"(?i)(?<!\w){re.escape(nm)}(?!\w)", psn or "[NAME]", out)
    out = _PHONE_RE.sub("[PHONE]", out)
    out = _AADHAAR_RE.sub("[AADHAAR]", out)
    out = _EMAIL_RE.sub("[EMAIL]", out)
    out = _PAN_RE.sub("[PAN]", out)
    out = _MRN_RE.sub(psn or "[MRN]", out)

    tmp_rec = Record(key=("fhir_text", "doc"), fields=[Field(idx=0, path="text", text=out)])
    refs = dates_mod.find(tmp_rec, {})
    for ref in sorted(refs, key=lambda r: r.start, reverse=True):
        rep = "[DOB]" if ref.is_dob else dates_mod.shift(ref, shift_days)
        out = out[: ref.start] + rep + out[ref.end :]

    return bands_mod.replace_ages_in_text(out, age_band or "[AGE]", is_structured_age_field=False)


def _sanitize_xhtml_div(
    div_html: str,
    known_names: list[str],
    psn: str | None,
    shift_days: int,
    age_band: str | None,
) -> str:
    """Sanitize XHTML text nodes while preserving tags and valid XML escaping."""
    parts = _TAG_SPLIT_RE.split(div_html)
    out_parts: list[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith("<") and part.endswith(">"):
            out_parts.append(part)
        else:
            unescaped = html.unescape(part)
            cleaned = _sanitize_plain_pii(unescaped, known_names, psn, shift_days, age_band)
            cleaned = _PIN_RE.sub("[PIN]", cleaned)
            out_parts.append(html.escape(cleaned, quote=False))
    return "".join(out_parts)


def _collect_patient_names(bundle: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for entry in bundle.get("entry") or []:
        res = entry.get("resource") or {}
        rt = res.get("resourceType")
        if rt == "Patient":
            for nm in res.get("name") or []:
                if isinstance(nm, dict):
                    if nm.get("text"):
                        names.append(str(nm["text"]))
                    for g in nm.get("given") or []:
                        names.append(str(g))
                    if nm.get("family"):
                        names.append(str(nm["family"]))
            for ct in res.get("contact") or []:
                if isinstance(ct, dict) and isinstance(ct.get("name"), dict) and ct["name"].get("text"):
                    names.append(str(ct["name"]["text"]))
            for ex in res.get("extension") or []:
                if isinstance(ex, dict) and "name" in str(ex.get("url", "")).lower() and ex.get("valueString"):
                    names.append(str(ex["valueString"]))
        elif rt == "Composition":
            subj = res.get("subject") or {}
            if subj.get("display"):
                names.append(str(subj["display"]))
    expanded: list[str] = []
    for n in names:
        expanded.append(n)
        for tok in n.split():
            if len(tok) >= 3:
                expanded.append(tok)
    return expanded


def validate_fhir_bundle(bundle: Any, expected_profiles: list[str] | None = None) -> list[str]:
    """Structural R4 + ABDM-minimum pre-write validator (D8, FR-F2)."""
    errors: list[str] = []
    if not isinstance(bundle, dict) or bundle.get("resourceType") != "Bundle":
        return ["not_a_bundle"]
    if not bundle.get("id"):
        errors.append("bundle_missing_id")
    if expected_profiles is not None:
        actual_profiles = list((bundle.get("meta") or {}).get("profile") or [])
        if actual_profiles != expected_profiles:
            errors.append("bundle_meta_profile_modified")

    entries = bundle.get("entry")
    if not isinstance(entries, list) or not entries:
        return errors + ["bundle_missing_entries"]

    known_targets: set[str] = set()
    for ent in entries:
        if not isinstance(ent, dict):
            errors.append("entry_not_dict")
            continue
        fu = ent.get("fullUrl")
        if fu:
            known_targets.add(str(fu))
        res = ent.get("resource")
        if isinstance(res, dict) and res.get("resourceType") and res.get("id"):
            known_targets.add(f"{res['resourceType']}/{res['id']}")

    def _check_ref(ref_obj: Any, ctx: str) -> None:
        if isinstance(ref_obj, dict) and "reference" in ref_obj:
            ref = str(ref_obj["reference"])
            if ref not in known_targets:
                errors.append(f"unresolved_ref:{ctx}:{ref}")

    def _check_date(val: Any, ctx: str) -> None:
        if val is None:
            return
        if not isinstance(val, str) or not _FHIR_DATE_RE.match(val):
            errors.append(f"invalid_date:{ctx}")

    if "timestamp" in bundle:
        _check_date(bundle.get("timestamp"), "Bundle.timestamp")
    if isinstance(bundle.get("meta"), dict) and "lastUpdated" in bundle["meta"]:
        _check_date(bundle["meta"].get("lastUpdated"), "Bundle.meta.lastUpdated")

    for i, ent in enumerate(entries):
        res = ent.get("resource") if isinstance(ent, dict) else None
        if not isinstance(res, dict):
            errors.append(f"entry_{i}_missing_resource")
            continue
        rt = res.get("resourceType")
        if not rt or not res.get("id"):
            errors.append(f"entry_{i}_missing_rt_or_id")
            continue

        if rt == "Composition":
            for req in ("status", "type", "subject", "date", "author", "title", "section"):
                if req not in res:
                    errors.append(f"Composition_missing_{req}")
            _check_date(res.get("date"), "Composition.date")
            _check_ref(res.get("subject"), "Composition.subject")
            _check_ref(res.get("encounter"), "Composition.encounter")
            for a in res.get("author") or []:
                _check_ref(a, "Composition.author")
            for sec in res.get("section") or []:
                if isinstance(sec, dict):
                    for se in sec.get("entry") or []:
                        _check_ref(se, "Composition.section.entry")
        elif rt == "Patient":
            if "gender" in res and res["gender"] not in ("male", "female", "other", "unknown"):
                errors.append("Patient_invalid_gender")
        elif rt == "Practitioner":
            if not res.get("name"):
                errors.append("Practitioner_missing_name")
        elif rt == "Organization":
            if not res.get("name"):
                errors.append("Organization_missing_name")
        elif rt == "Encounter":
            for req in ("status", "class", "subject"):
                if req not in res:
                    errors.append(f"Encounter_missing_{req}")
            _check_ref(res.get("subject"), "Encounter.subject")
            _check_ref(res.get("serviceProvider"), "Encounter.serviceProvider")
            per = res.get("period") or {}
            if isinstance(per, dict):
                _check_date(per.get("start"), "Encounter.period.start")
                _check_date(per.get("end"), "Encounter.period.end")
        elif rt == "Condition":
            if "code" not in res or "subject" not in res:
                errors.append("Condition_missing_required")
            _check_ref(res.get("subject"), "Condition.subject")
            _check_ref(res.get("encounter"), "Condition.encounter")
            _check_date(res.get("recordedDate"), "Condition.recordedDate")
        elif rt == "MedicationRequest":
            for req in ("status", "intent", "subject"):
                if req not in res:
                    errors.append(f"MedicationRequest_missing_{req}")
            if "medicationCodeableConcept" not in res and "medicationReference" not in res:
                errors.append("MedicationRequest_missing_medication")
            _check_ref(res.get("subject"), "MedicationRequest.subject")
            _check_date(res.get("authoredOn"), "MedicationRequest.authoredOn")
        elif rt == "Observation":
            for req in ("status", "code", "subject"):
                if req not in res:
                    errors.append(f"Observation_missing_{req}")
            _check_ref(res.get("subject"), "Observation.subject")
            _check_date(res.get("effectiveDateTime"), "Observation.effectiveDateTime")
        elif rt == "Procedure":
            for req in ("status", "code", "subject"):
                if req not in res:
                    errors.append(f"Procedure_missing_{req}")
            _check_ref(res.get("subject"), "Procedure.subject")
            _check_date(res.get("performedDateTime"), "Procedure.performedDateTime")
        elif rt == "DocumentReference":
            if "status" not in res or "content" not in res:
                errors.append("DocumentReference_missing_required")
            _check_ref(res.get("subject"), "DocumentReference.subject")
            for c in res.get("content") or []:
                att = (c or {}).get("attachment") or {}
                data_b64 = att.get("data")
                if data_b64 is not None:
                    try:
                        base64.b64decode(str(data_b64).encode("ascii"), validate=True)
                    except (binascii.Error, UnicodeEncodeError):
                        errors.append("DocumentReference_invalid_base64")
        elif rt == "Binary":
            data_b64 = res.get("data")
            if data_b64 is not None:
                try:
                    base64.b64decode(str(data_b64).encode("ascii"), validate=True)
                except (binascii.Error, UnicodeEncodeError):
                    errors.append("Binary_invalid_base64")

    return errors


def _extract_fhir_fields(bundle: dict[str, Any]) -> tuple[list[Field], dict[str, Any]]:
    """Extract key URM fields and compact anonymized QI summary from a FHIR bundle."""
    fields: list[Field] = []
    idx = 0
    qi: dict[str, Any] = {"dx_terms": []}
    for ent in bundle.get("entry") or []:
        res = ent.get("resource") or {}
        rt = res.get("resourceType")
        if rt == "Patient":
            for nm in res.get("name") or []:
                if isinstance(nm, dict) and nm.get("text"):
                    fields.append(Field(idx=idx, path="Patient.name", text=str(nm["text"])))
                    idx += 1
            for id_obj in res.get("identifier") or []:
                if isinstance(id_obj, dict) and id_obj.get("value"):
                    fields.append(Field(idx=idx, path="Patient.identifier", text=str(id_obj["value"])))
                    idx += 1
            for tel in res.get("telecom") or []:
                if isinstance(tel, dict) and tel.get("value"):
                    fields.append(Field(idx=idx, path=f"Patient.telecom.{tel.get('system', 'phone')}", text=str(tel["value"])))
                    idx += 1
            if res.get("gender"):
                fields.append(Field(idx=idx, path="Patient.gender", text=str(res["gender"])))
                qi["sex"] = str(res["gender"])
                idx += 1
            if res.get("birthDate"):
                bd = str(res["birthDate"])
                fields.append(Field(idx=idx, path="Patient.birthDate", text=bd))
                m_yr = re.match(r"^(\d{4})", bd)
                if m_yr:
                    qi["birth_year"] = int(m_yr.group(1))
                idx += 1
            for addr in res.get("address") or []:
                if isinstance(addr, dict):
                    if addr.get("district"):
                        fields.append(Field(idx=idx, path="Patient.address.district", text=str(addr["district"])))
                        qi.setdefault("district", str(addr["district"]))
                        idx += 1
                    if addr.get("state"):
                        fields.append(Field(idx=idx, path="Patient.address.state", text=str(addr["state"])))
                        idx += 1
        elif rt == "Condition":
            code = res.get("code") or {}
            if code.get("text"):
                fields.append(Field(idx=idx, path="Condition.code.text", text=str(code["text"])))
                qi["dx_terms"].append(str(code["text"]))
                idx += 1
        elif rt == "Organization":
            if res.get("name"):
                fields.append(Field(idx=idx, path="Organization.name", text=str(res["name"])))
                qi.setdefault("hospital", str(res["name"]))
                idx += 1
    return fields, qi


def _merge_clinical_entries(
    clin_spans: list[tuple[int, int, str]],
    rec_key: RecKey,
    total_len: int,
) -> list[LocEntry]:
    """Build non-overlapping LocEntries so clinical_ranges() covers all FHIR clinical/code regions."""
    valid = sorted(
        ((max(0, s), min(total_len, e), kind) for s, e, kind in clin_spans if e > s and s < total_len),
        key=lambda x: (x[0], -x[1]),
    )
    merged: list[tuple[int, int, str]] = []
    for s, e, kind in valid:
        if not merged or s >= merged[-1][1]:
            merged.append((s, e, kind))
        else:
            prev_s, prev_e, prev_k = merged[-1]
            if e > prev_e:
                merged[-1] = (prev_s, e, prev_k)
    return [LocEntry(s, e, rec_key, 0, 0, 0, exempt_kind=kind) for s, e, kind in merged]


def _derive_shift_days(keys: Keys | None, fam: str, pol: dict[str, Any]) -> int:
    priv_cfg = pol.get("privacy") or {}
    dates_cfg = pol.get("dates") or {}
    min_shift = int(dates_cfg.get("shift_min_days", priv_cfg.get("date_shift_min_days", 30)))
    max_shift = int(dates_cfg.get("shift_max_days", priv_cfg.get("date_shift_max_days", 365)))
    span = max(1, max_shift - min_shift)
    if keys is None:
        return min_shift
    digest = keys.keyed("shift", fam)
    mag = min_shift + (int.from_bytes(digest[:4], "big") % span)
    sign = 1 if (digest[4] & 1) == 0 else -1
    return sign * mag


def parse(entry: FileEntry, policy: dict[str, Any], raw_obj: dict[str, Any] | None = None) -> ParsedFile:
    """Parse an ABDM FHIR R4 Bundle and attach its structural projector."""
    if raw_obj is None:
        if entry.text is None:
            raise StageError("parse", "E_INVALID_JSON")
        try:
            raw_obj = json.loads(entry.text)
        except json.JSONDecodeError as exc:
            raise StageError("parse", "E_INVALID_JSON") from exc

    if not isinstance(raw_obj, dict) or raw_obj.get("resourceType") != "Bundle":
        raise StageError("parse", "E_INVALID_JSON")
    if not isinstance(raw_obj.get("entry"), list):
        raise StageError("parse", "E_IO")

    rec_key: RecKey = (entry.rel_path, "doc")
    orig_profiles = list((raw_obj.get("meta") or {}).get("profile") or [])
    fields, qi_summary = _extract_fhir_fields(raw_obj)
    rec = Record(key=rec_key, fields=fields, meta={"is_fhir": True})
    fhir_ctx: dict[str, Any] = {"resolution": None, "plan": None, "keys": None, "policy": policy}

    def _project(
        edits_or_ctx: list[Edit] | dict[str, Any] | None = None,
        psn_map: dict[RecKey, str | None] | None = None,
        *,
        protected_spans: list[Any] | None = None,
    ) -> tuple[bytes, LocMap]:
        _ = protected_spans
        if isinstance(edits_or_ctx, dict):
            ctx_d = edits_or_ctx
            repair_edits: list[Edit] = []
        else:
            ctx_d = fhir_ctx
            repair_edits = list(edits_or_ctx or [])

        res: Resolution | None = ctx_d.get("resolution")
        plan: PrivacyPlan = ctx_d.get("plan") or PrivacyPlan()
        keys: Keys | None = ctx_d.get("keys")
        pol: dict[str, Any] = ctx_d.get("policy") or policy

        patient_of = dict(res.patient_of) if res is not None else dict(psn_map or {})
        cluster_of = dict(res.cluster_of) if res is not None else {}
        family_of = dict(res.family_of) if res is not None else {}
        shift_days_map = dict(res.shift_days) if res is not None else {}

        psn = patient_of.get(rec_key)
        cid = cluster_of.get(rec_key) or f"orphan:{rec_key[0]}|{rec_key[1]}"
        fam = family_of.get(rec_key, rec_key[0])
        shift_days = shift_days_map.get(fam) or _derive_shift_days(keys, fam, pol)

        age_band = plan.age_band.get(cid)
        if age_band is None and qi_summary.get("birth_year") is not None:
            ref_yr = int((pol.get("dates") or {}).get("dataset_ref_year", 2026))
            est_age = max(0, min(120, ref_yr - int(qi_summary["birth_year"])))
            age_cfg = pol.get("age") or {}
            age_band = bands_mod.band_for_ages(
                [est_age],
                lattice=list(age_cfg.get("pool_lattice") or bands_mod.DEFAULT_LATTICE_10),
                top_band=str(age_cfg.get("top_band", "85+")),
            )

        loc_level = plan.location_level.get(rec_key, "district")
        pin_level = plan.pin_level.get(rec_key, "none")
        dob_mode = str((pol.get("dob") or {}).get("mode", "suppress"))
        doc_reg_mode = str((pol.get("detect") or {}).get("doctor_reg_numbers", "remove_digits"))
        dx_rollup_label = plan.dx_rollup.get(rec_key)
        is_suppressed_qi = rec_key in plan.suppress_qi

        bundle = copy.deepcopy(raw_obj)
        known_names = _collect_patient_names(bundle)
        lgd = loc_mod.get_lgd_index()

        if isinstance(bundle.get("meta"), dict):
            if "source" in bundle["meta"]:
                bundle["meta"]["source"] = "[SOURCE]"
            if "lastUpdated" in bundle["meta"]:
                bundle["meta"]["lastUpdated"] = _shift_iso_preserve_precision(str(bundle["meta"]["lastUpdated"]), shift_days)
        if "timestamp" in bundle:
            bundle["timestamp"] = _shift_iso_preserve_precision(str(bundle["timestamp"]), shift_days)

        clinical_texts: list[str] = []

        for ent in bundle.get("entry") or []:
            r = ent.get("resource")
            if not isinstance(r, dict):
                continue
            rt = r.get("resourceType")

            if isinstance(r.get("meta"), dict) and "source" in r["meta"]:
                r["meta"]["source"] = "[SOURCE]"

            for ref_key in ("subject", "patient"):
                if isinstance(r.get(ref_key), dict) and "display" in r[ref_key]:
                    r[ref_key]["display"] = psn or "[NAME]"

            if rt == "Patient":
                r["name"] = [{"text": psn or "[NAME]"}]
                for id_obj in r.get("identifier") or []:
                    if isinstance(id_obj, dict) and "value" in id_obj:
                        id_obj["value"] = _surrogate_id(keys, str(id_obj.get("system", "")), str(id_obj["value"]), psn)
                for tel in r.get("telecom") or []:
                    if isinstance(tel, dict) and "value" in tel:
                        tel["value"] = _surrogate_telecom(keys, str(tel.get("system", "phone")), str(tel["value"]))
                if dob_mode == "year" and isinstance(r.get("birthDate"), str) and len(r["birthDate"]) >= 4:
                    r["birthDate"] = r["birthDate"][:4]
                else:
                    r.pop("birthDate", None)
                r.pop("contact", None)
                new_exts = []
                for ex in r.get("extension") or []:
                    if isinstance(ex, dict) and any(k in str(ex.get("url", "")).lower() for k in ("name", "mother", "father", "relative", "phone", "aadhaar")):
                        ex2 = dict(ex)
                        if "valueString" in ex2:
                            ex2["valueString"] = "[RELATIVE]"
                        new_exts.append(ex2)
                    else:
                        new_exts.append(ex)
                if "extension" in r:
                    r["extension"] = new_exts

                new_addrs = []
                for addr in r.get("address") or []:
                    if not isinstance(addr, dict):
                        continue
                    if loc_level == "none":
                        continue
                    na: dict[str, Any] = {}
                    raw_dist = str(addr.get("district") or addr.get("city") or "")
                    info = lgd.lookup_district(raw_dist) if raw_dist else None
                    if loc_level == "district" and raw_dist:
                        na["district"] = info[1] if info else raw_dist
                    state_val = addr.get("state") or (info[2] if info else None)
                    if state_val:
                        na["state"] = state_val
                    if addr.get("country"):
                        na["country"] = addr["country"]
                    if pin_level == "pin3" and addr.get("postalCode"):
                        na["postalCode"] = loc_mod.generalize_pin(str(addr["postalCode"]), "pin3")
                    if na:
                        new_addrs.append(na)
                if "address" in r:
                    r["address"] = new_addrs
                if is_suppressed_qi and "gender" in r:
                    r["gender"] = "unknown"
                if isinstance(r.get("text"), dict) and isinstance(r["text"].get("div"), str):
                    r["text"]["div"] = _sanitize_xhtml_div(r["text"]["div"], known_names, psn, shift_days, age_band)

            elif rt == "Practitioner":
                for nm in r.get("name") or []:
                    if isinstance(nm, dict) and nm.get("text"):
                        clinical_texts.append(str(nm["text"]))
                if doc_reg_mode == "remove_digits":
                    for id_obj in r.get("identifier") or []:
                        if isinstance(id_obj, dict) and "value" in id_obj:
                            id_obj["value"] = "[REG_NO]"
                    for q in r.get("qualification") or []:
                        if isinstance(q, dict):
                            for id_obj in q.get("identifier") or []:
                                if isinstance(id_obj, dict) and "value" in id_obj:
                                    id_obj["value"] = "[REG_NO]"

            elif rt == "Organization":
                if r.get("name"):
                    clinical_texts.append(str(r["name"]))

            elif rt == "Composition":
                if "date" in r:
                    r["date"] = _shift_iso_preserve_precision(str(r["date"]), shift_days)
                if isinstance(r.get("text"), dict) and isinstance(r["text"].get("div"), str):
                    r["text"]["div"] = _sanitize_xhtml_div(r["text"]["div"], known_names, psn, shift_days, age_band)

            elif rt == "Encounter":
                for id_obj in r.get("identifier") or []:
                    if isinstance(id_obj, dict) and "value" in id_obj:
                        id_obj["value"] = _surrogate_id(keys, str(id_obj.get("system", "")), str(id_obj["value"]), psn)
                if isinstance(r.get("period"), dict):
                    if "start" in r["period"]:
                        r["period"]["start"] = _shift_iso_preserve_precision(str(r["period"]["start"]), shift_days)
                    if "end" in r["period"]:
                        r["period"]["end"] = _shift_iso_preserve_precision(str(r["period"]["end"]), shift_days)
                for loc_ent in r.get("location") or []:
                    if isinstance(loc_ent, dict) and isinstance(loc_ent.get("location"), dict) and loc_ent["location"].get("display"):
                        clinical_texts.append(str(loc_ent["location"]["display"]))

            elif rt == "Condition":
                if "recordedDate" in r:
                    r["recordedDate"] = _shift_iso_preserve_precision(str(r["recordedDate"]), shift_days)
                code = r.get("code") or {}
                if isinstance(code, dict):
                    if dx_rollup_label:
                        code["text"] = dx_rollup_label
                        for cd in code.get("coding") or []:
                            if isinstance(cd, dict):
                                cd["display"] = dx_rollup_label
                                if "code" in cd and isinstance(cd["code"], str) and "." in cd["code"]:
                                    cd["code"] = cd["code"].split(".", 1)[0]
                    if code.get("text"):
                        clinical_texts.append(str(code["text"]))

            elif rt == "MedicationRequest":
                if "authoredOn" in r:
                    r["authoredOn"] = _shift_iso_preserve_precision(str(r["authoredOn"]), shift_days)
                med = r.get("medicationCodeableConcept") or {}
                if isinstance(med, dict) and med.get("text"):
                    clinical_texts.append(str(med["text"]))
                for d_inst in r.get("dosageInstruction") or []:
                    if isinstance(d_inst, dict) and d_inst.get("text"):
                        clinical_texts.append(str(d_inst["text"]))

            elif rt == "Observation":
                if "effectiveDateTime" in r:
                    r["effectiveDateTime"] = _shift_iso_preserve_precision(str(r["effectiveDateTime"]), shift_days)
                for c in (r.get("code") or {}).get("coding") or []:
                    if isinstance(c, dict) and c.get("display"):
                        clinical_texts.append(str(c["display"]))

            elif rt == "Procedure":
                if "performedDateTime" in r:
                    r["performedDateTime"] = _shift_iso_preserve_precision(str(r["performedDateTime"]), shift_days)
                code = r.get("code") or {}
                if isinstance(code, dict) and code.get("text"):
                    clinical_texts.append(str(code["text"]))

            elif rt == "DocumentReference":
                if "date" in r:
                    r["date"] = _shift_iso_preserve_precision(str(r["date"]), shift_days)
                for c in r.get("content") or []:
                    att = c.get("attachment") if isinstance(c, dict) else None
                    if not isinstance(att, dict):
                        continue
                    ct = str(att.get("contentType", "")).lower()
                    raw_b64 = att.get("data")
                    if raw_b64 is not None:
                        try:
                            decoded = base64.b64decode(str(raw_b64).encode("ascii"), validate=True)
                            if ct.startswith("text/") or ct == "application/json" or not ct:
                                try:
                                    txt = decoded.decode("utf-8", errors="strict")
                                    san_txt = _sanitize_plain_pii(txt, known_names, psn, shift_days, age_band)
                                    att["data"] = base64.b64encode(san_txt.encode("utf-8")).decode("ascii")
                                except UnicodeDecodeError:
                                    att.pop("data", None)
                                    att["title"] = "[BINARY_ATTACHMENT]"
                            else:
                                att.pop("data", None)
                                att["title"] = "[BINARY_ATTACHMENT]"
                        except (binascii.Error, UnicodeEncodeError):
                            att.pop("data", None)
                            att["title"] = "[BINARY_ATTACHMENT]"

            elif rt == "Binary":
                ct = str(r.get("contentType", "")).lower()
                raw_b64 = r.get("data")
                if raw_b64 is not None:
                    try:
                        decoded = base64.b64decode(str(raw_b64).encode("ascii"), validate=True)
                        if ct.startswith("text/") or ct == "application/json":
                            txt = decoded.decode("utf-8", errors="strict")
                            san_txt = _sanitize_plain_pii(txt, known_names, psn, shift_days, age_band)
                            r["data"] = base64.b64encode(san_txt.encode("utf-8")).decode("ascii")
                        else:
                            r["data"] = base64.b64encode(b"[BINARY_NEUTRALIZED]").decode("ascii")
                    except (binascii.Error, UnicodeDecodeError, UnicodeEncodeError):
                        r["data"] = base64.b64encode(b"[BINARY_NEUTRALIZED]").decode("ascii")

            elif rt == "Media":
                r["status"] = "entered-in-error"
                r.pop("content", None)
                r["note"] = [{"text": "[MEDIA_DROPPED]"}]

        val_errors = validate_fhir_bundle(bundle, expected_profiles=orig_profiles)
        if val_errors:
            raise StageError("parse", "E_IO")

        code_strings: list[str] = list(orig_profiles)
        for ent in bundle.get("entry") or []:
            res_i = ent.get("resource") or {}
            rt_i = res_i.get("resourceType")
            if isinstance(res_i.get("title"), str) and rt_i == "Composition":
                clinical_texts.append(res_i["title"])
            if isinstance(res_i.get("class"), dict):
                for ck in ("system", "code", "display"):
                    if res_i["class"].get(ck):
                        code_strings.append(str(res_i["class"][ck]))
            for k_sys in ("type", "code", "clinicalStatus", "medicationCodeableConcept"):
                sub = res_i.get(k_sys)
                if isinstance(sub, dict):
                    if sub.get("text") and rt_i != "DocumentReference":
                        clinical_texts.append(str(sub["text"]))
                    for cd in sub.get("coding") or []:
                        if isinstance(cd, dict):
                            for ck in ("system", "code", "display"):
                                if cd.get(ck):
                                    code_strings.append(str(cd[ck]))
            for id_obj in res_i.get("identifier") or []:
                if isinstance(id_obj, dict):
                    if id_obj.get("system"):
                        code_strings.append(str(id_obj["system"]))
                    tp = id_obj.get("type")
                    if isinstance(tp, dict):
                        for cd in tp.get("coding") or []:
                            if isinstance(cd, dict):
                                for ck in ("system", "code", "display"):
                                    if cd.get(ck):
                                        code_strings.append(str(cd[ck]))
            vq_top = res_i.get("valueQuantity")
            if isinstance(vq_top, dict):
                clinical_texts.append(json.dumps(vq_top, ensure_ascii=False))
                if vq_top.get("value") is not None:
                    code_strings.append(str(vq_top["value"]))
                for ck in ("system", "code", "unit"):
                    if vq_top.get(ck):
                        code_strings.append(str(vq_top[ck]))
            for comp in res_i.get("component") or []:
                if isinstance(comp, dict):
                    for cd in (comp.get("code") or {}).get("coding") or []:
                        if isinstance(cd, dict):
                            for ck in ("system", "code", "display"):
                                if cd.get(ck):
                                    code_strings.append(str(cd[ck]))
                    vq = comp.get("valueQuantity")
                    if isinstance(vq, dict):
                        if vq.get("value") is not None:
                            code_strings.append(str(vq["value"]))
                        for ck in ("system", "code", "unit"):
                            if vq.get(ck):
                                code_strings.append(str(vq[ck]))
            for sec in res_i.get("section") or []:
                if isinstance(sec, dict) and sec.get("title"):
                    clinical_texts.append(str(sec["title"]))
            for q in res_i.get("qualification") or []:
                if isinstance(q, dict):
                    qt = (q.get("code") or {}).get("text")
                    if qt:
                        clinical_texts.append(str(qt))
            for ex in res_i.get("extension") or []:
                if isinstance(ex, dict) and ex.get("url"):
                    code_strings.append(str(ex["url"]))
        if isinstance(bundle.get("identifier"), dict) and bundle["identifier"].get("system"):
            code_strings.append(str(bundle["identifier"]["system"]))

        out_str = _dumps_iterative(bundle)

        # Apply any gate repair edits targeting specific substrings if triggered
        for red in repair_edits:
            if 0 <= red.field_idx < len(fields):
                target = fields[red.field_idx].text[red.start:red.end]
                if target and len(target) >= 2:
                    out_str = out_str.replace(json.dumps(target, ensure_ascii=False), json.dumps(red.text, ensure_ascii=False))

        clin_spans: list[tuple[int, int, str]] = []
        for ctxt, c_kind in [(t, "clinical") for t in sorted(set(clinical_texts))] + [(t, "code") for t in sorted(set(code_strings))]:
            if len(ctxt) < 2:
                continue
            start = 0
            while True:
                pos = out_str.find(ctxt, start)
                if pos == -1:
                    break
                clin_spans.append((pos, pos + len(ctxt), c_kind))
                start = pos + len(ctxt)
        for m_vq in re.finditer(r'"valueQuantity"\s*:\s*\{[^}]+\}', out_str):
            clin_spans.append((m_vq.start(), m_vq.end(), "lab"))

        loc_entries = _merge_clinical_entries(clin_spans, rec_key, len(out_str))
        return out_str.encode("utf-8"), LocMap(loc_entries)

    return ParsedFile(
        rel_path=entry.rel_path,
        sha256=entry.sha256,
        fmt="fhir",
        encoding=entry.encoding or "utf-8",
        newline="\n",
        records=[rec],
        raw_bytes=entry.raw_bytes,
        projector=_project,
        extra={"fhir_ctx": fhir_ctx, "fhir_qi": qi_summary},
    )
