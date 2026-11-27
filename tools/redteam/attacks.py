"""Implements ATTACKS.md A1-A7, uniqueness curve, and Expert Determination summary (D9, D15)."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .extract import ExtractedRecord, extract_dataset_records, norm


@dataclass
class RegistryPerson:
    rid: str
    name: str
    given: str
    surname: str
    relative_name: str
    gender: str
    age: int
    house_no: str
    village: str
    tehsil: str
    district: str
    pin: str
    phone: str
    email: str
    aadhaar_last4: str


def load_registry(path: Path) -> dict[str, RegistryPerson]:
    people: dict[str, RegistryPerson] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            nm = norm(row.get("name", ""))
            parts = nm.split()
            people[row["rid"]] = RegistryPerson(
                rid=row["rid"],
                name=nm,
                given=parts[0] if parts else "",
                surname=parts[-1] if len(parts) > 1 else "",
                relative_name=norm(row.get("relative_name", "")),
                gender=(row.get("gender", "").strip().upper()[:1] or "U"),
                age=int(row["age"]) if str(row.get("age", "")).isdigit() else -1,
                house_no=norm(row.get("house_no", "")),
                village=norm(row.get("village", "")),
                tehsil=norm(row.get("tehsil", "")),
                district=norm(row.get("district", "")),
                pin=str(row.get("pin", "")).strip(),
                phone=str(row.get("phone", "")).strip()[-10:],
                email=str(row.get("email", "")).strip().casefold(),
                aadhaar_last4=str(row.get("aadhaar_last4", "")).strip(),
            )
    return people


def load_known_facts(path: Path) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    if not path.is_file():
        return facts
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            facts.append(json.loads(line))
    return facts


def _filter_candidates(ev: ExtractedRecord, reg_list: list[RegistryPerson]) -> list[RegistryPerson]:
    cands = reg_list

    def _apply(pred) -> None:
        nonlocal cands
        nxt = [p for p in cands if pred(p)]
        if nxt:
            cands = nxt

    if ev.phones:
        pset = set(ev.phones)
        _apply(lambda p: p.phone in pset)
    if ev.emails:
        eset = set(ev.emails)
        _apply(lambda p: p.email in eset)
    if ev.last4:
        lset = set(ev.last4)
        _apply(lambda p: p.aadhaar_last4 in lset)
    if ev.masked_phones:
        _apply(lambda p: any(p.phone.startswith(a) and p.phone.endswith(b) for a, b in ev.masked_phones))
    if ev.district:
        dnorm = norm(ev.district)
        _apply(lambda p: p.district == dnorm)
    if ev.gender in ("M", "F"):
        _apply(lambda p: p.gender == ev.gender)
    if ev.age_ranges:
        _apply(lambda p: p.age >= 0 and all(lo - 1 <= p.age <= hi + 1 for lo, hi in ev.age_ranges))
    if ev.birth_years:
        by = ev.birth_years[0]
        _apply(lambda p: p.age >= 0 and abs((2026 - p.age) - by) <= 2)
    if ev.tehsil:
        _apply(lambda p: p.tehsil == ev.tehsil)
    if ev.village:
        _apply(lambda p: p.village == ev.village)
    if ev.pin:
        _apply(lambda p: p.pin == ev.pin)
    if ev.pin3:
        _apply(lambda p: p.pin.startswith(ev.pin3))
    return cands


_DOCTOR_TOKEN_RE = re.compile(r"(?:\bdr\.?|डॉ\.?)\s*((?:[^\W\d_]+\s*){1,3})", re.I)


def _name_tokens(text: str) -> set[str]:
    """ATTACKS.md: names via TOKEN lookup, doctors after `Dr.` skipped (not substrings)."""
    doctor = {w.casefold() for m in _DOCTOR_TOKEN_RE.finditer(text) for w in m.group(1).split()}
    return {w.casefold() for w in re.findall(r"[^\W\d_]+", text)} - doctor


def _score_candidates(ev: ExtractedRecord, cands: list[RegistryPerson]) -> dict[str, float]:
    hay_tokens = _name_tokens(ev.raw_text)
    hay = " ".join(sorted(hay_tokens))
    scores: dict[str, float] = {}
    for p in cands:
        w = 0.0
        name_toks = [t.casefold() for t in (p.name or "").split()]
        full_hit = bool(name_toks) and (all(t in hay_tokens for t in name_toks) or any(p.name == n for n in ev.names))
        if full_hit:
            w += 60.0
        else:
            if p.given and len(p.given) >= 3 and p.given.casefold() in hay_tokens:
                w += 8.0
            if p.surname and len(p.surname) >= 3 and p.surname.casefold() in hay_tokens:
                w += 3.0
        rel_toks = [t.casefold() for t in (p.relative_name or "").split()]
        if rel_toks and len(p.relative_name) >= 3 and (all(t in hay_tokens for t in rel_toks) or any(p.relative_name in r for r in ev.relatives)):
            w += 8.0
        if p.house_no and p.house_no in ev.houses:
            w += 2.0
        scores[p.rid] = w
    tot = sum(scores.values())
    if tot <= 0:
        if not cands:
            return {}
        u = 1.0 / len(cands)
        return {p.rid: u for p in cands}
    return {rid: sc / tot for rid, sc in scores.items()}


def _pool_profile(records: list[ExtractedRecord]) -> ExtractedRecord:
    base = ExtractedRecord(file_rel=records[0].file_rel, loc="profile", psn=records[0].psn)
    for r in records:
        base.names.extend(r.names)
        base.relatives.extend(r.relatives)
        base.phones.extend(r.phones)
        base.masked_phones.extend(r.masked_phones)
        base.last4.extend(r.last4)
        base.emails.extend(r.emails)
        if not base.gender and r.gender:
            base.gender = r.gender
        base.age_ranges.extend(r.age_ranges)
        base.birth_years.extend(r.birth_years)
        if not base.district and r.district:
            base.district = r.district
        if not base.tehsil and r.tehsil:
            base.tehsil = r.tehsil
        if not base.village and r.village:
            base.village = r.village
        if not base.pin and r.pin:
            base.pin = r.pin
        if not base.pin3 and r.pin3:
            base.pin3 = r.pin3
        base.houses.extend(r.houses)
        base.conditions.extend(r.conditions)
        if not base.hospital and r.hospital:
            base.hospital = r.hospital
        base.dates.extend(r.dates)
        if r.los_days is not None:
            base.los_days = max(base.los_days or 0, r.los_days)
    base.raw_text = "\n".join(r.raw_text for r in records)
    return base


def run_attacks(
    release_dir: Path,
    registry_path: Path,
    facts_path: Path,
    truth_path: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Run A1-A5 against raw or sanitized release_dir using truth_path only for evaluation."""
    registry = load_registry(registry_path)
    reg_list = list(registry.values())
    facts = load_known_facts(facts_path)
    truth = json.loads(truth_path.read_text(encoding="utf-8"))

    # Map output relative path -> original truth relative path via report.json (if renamed)
    out_to_orig_rel: dict[str, str] = {}
    if report_path and report_path.is_file():
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        sha_to_orig = {f["sha256"]: rel for rel, f in truth["files"].items() if f.get("sha256")}
        for ent in rep.get("files") or []:
            o = ent.get("output")
            sha = ent.get("input_sha256")
            if o and sha in sha_to_orig:
                out_to_orig_rel[str(o).replace("sanitized/", "", 1)] = sha_to_orig[sha]

    # Map truth (rel_path, loc) -> true patient id and collect rich demographics per patient to resolve rid
    patient_info: dict[str, dict[str, Any]] = {}
    unit_true_patient: dict[tuple[str, str], str] = {}
    for rel, f in truth["files"].items():
        for loc, u in f["units"].items():
            pid = u.get("patient")
            if pid:
                unit_true_patient[(rel, loc)] = pid
                pinfo = patient_info.setdefault(pid, {"names": [], "district": "", "age": None, "gender": "", "phone": ""})
                for ent in u.get("entities") or []:
                    if ent.get("type") == "PATIENT_NAME" and ent.get("script") == "latin":
                        pinfo["names"].append(norm(str(ent.get("surface", ""))))
                    elif ent.get("type") == "PHONE":
                        digs = "".join(c for c in str(ent.get("surface", "")) if c.isdigit())[-10:]
                        if len(digs) == 10:
                            pinfo["phone"] = digs
                for fld in u.get("fields") or []:
                    if isinstance(fld, dict):
                        knd = fld.get("kind")
                        if knd == "district" and not pinfo["district"]:
                            pinfo["district"] = norm(str(fld.get("true", "")))
                        elif knd == "age" and pinfo["age"] is None and str(fld.get("true", "")).isdigit():
                            pinfo["age"] = int(fld["true"])
                        elif knd == "gender" and not pinfo["gender"]:
                            pinfo["gender"] = str(fld.get("true", "")).strip().upper()[:1]

    patient_to_rid: dict[str, str] = {}
    for pid, info in patient_info.items():
        best_rid = None
        best_sc = -1
        for rp in reg_list:
            sc = 0
            if any(nm == rp.name for nm in info["names"]):
                sc += 10
            elif any(rp.given and rp.given in nm and rp.surname and rp.surname in nm for nm in info["names"]):
                sc += 6
            if info["district"] and rp.district == info["district"]:
                sc += 5
            if info["age"] is not None and rp.age >= 0 and abs(rp.age - info["age"]) <= 2:
                sc += 4
            if info["gender"] and rp.gender == info["gender"]:
                sc += 2
            if info["phone"] and rp.phone == info["phone"]:
                sc += 8
            if sc > best_sc:
                best_sc = sc
                best_rid = rp.rid
        if best_rid and best_sc >= 10:
            patient_to_rid[pid] = best_rid

    extracted = extract_dataset_records(release_dir)

    # A2/A3 record-level attack
    rec_probs: dict[tuple[str, str], dict[str, float]] = {}
    rec_true_prob: dict[str, list[float]] = defaultdict(list)
    for ev in extracted:
        orig_rel = out_to_orig_rel.get(ev.file_rel, ev.file_rel)
        cands = _filter_candidates(ev, reg_list)
        probs = _score_candidates(ev, cands)
        rec_probs[(ev.file_rel, ev.loc)] = probs
        pid = unit_true_patient.get((orig_rel, ev.loc))
        if pid and pid in patient_to_rid:
            rid = patient_to_rid[pid]
            rec_true_prob[pid].append(probs.get(rid, 0.0))

    # A4 profile aggregation via shared PSN or ref_ids
    groups: dict[str, list[ExtractedRecord]] = defaultdict(list)
    for i, ev in enumerate(extracted):
        key = ev.psn or (ev.ref_ids[0] if ev.ref_ids else f"single:{i}")
        groups[key].append(ev)

    profiles: dict[str, ExtractedRecord] = {}
    prof_probs: dict[str, dict[str, float]] = {}
    patient_max_prob: dict[str, float] = defaultdict(float)

    for gkey, recs in groups.items():
        prof = _pool_profile(recs)
        profiles[gkey] = prof
        cands = _filter_candidates(prof, reg_list)
        probs = _score_candidates(prof, cands)
        prof_probs[gkey] = probs
        for r in recs:
            orig_rel = out_to_orig_rel.get(r.file_rel, r.file_rel)
            pid = unit_true_patient.get((orig_rel, r.loc))
            if pid and pid in patient_to_rid:
                rid = patient_to_rid[pid]
                p_rec = rec_probs.get((r.file_rel, r.loc), {}).get(rid, 0.0)
                p_prof = probs.get(rid, 0.0)
                patient_max_prob[pid] = max(patient_max_prob[pid], p_rec, p_prof)

    all_true_probs = [v for vals in rec_true_prob.values() for v in vals]
    a3_mean = sum(all_true_probs) / len(all_true_probs) if all_true_probs else 0.0
    n_patients = max(1, len(patient_to_rid))
    a4_reid = sum(1 for pid in patient_to_rid if patient_max_prob.get(pid, 0.0) >= 0.5) / n_patients

    # A1 direct exposure: full name or direct phone/email surviving in extracted content
    a1_exposed = 0
    for pid, rid in patient_to_rid.items():
        rp = registry[rid]
        hit = False
        for ev in extracted:
            orig_rel = out_to_orig_rel.get(ev.file_rel, ev.file_rel)
            if unit_true_patient.get((orig_rel, ev.loc)) == pid or (rp.name and rp.name in ev.raw_text):
                if (rp.name and rp.name in ev.raw_text) or (rp.phone and rp.phone in ev.phones) or (rp.email and rp.email in ev.emails):
                    hit = True
                    break
        if hit:
            a1_exposed += 1
    a1_rate = a1_exposed / n_patients

    # A5 targeted attack with global-shift recovery
    # 1. Try recovering global shift from confident (>= 0.9) profiles
    offset_votes: Counter[int] = Counter()
    facts_by_rid = {f["rid"]: f for f in facts}
    for gkey, probs in prof_probs.items():
        if not probs:
            continue
        top_rid, top_p = max(probs.items(), key=lambda kv: kv[1])
        if top_p >= 0.9 and top_rid in facts_by_rid:
            kf = facts_by_rid[top_rid]
            for pd_s in profiles[gkey].dates:
                for kd_s in kf.get("visit_dates") or []:
                    try:
                        delta = (dt.date.fromisoformat(pd_s) - dt.date.fromisoformat(kd_s)).days
                        offset_votes[delta] += 1
                    except ValueError:
                        pass
    global_shift = 0
    if offset_votes:
        best_delta, cnt = offset_votes.most_common(1)[0]
        if cnt >= 2:
            global_shift = best_delta

    # Condition rarity across profiles
    cond_counts = Counter(c for prof in profiles.values() for c in set(prof.conditions))

    a5_hits = 0
    a5_targets = 0
    a5_decoy_fp = 0
    for kf in facts:
        rid = kf["rid"]
        is_target = rid in set(patient_to_rid.values())
        rp = registry.get(rid)
        if not rp:
            continue
        best_score = 0.0
        best_match_rid = None
        for gkey, prof in profiles.items():
            # Must be consistent with person's gender/age if present
            if prof.gender in ("M", "F") and rp.gender in ("M", "F") and prof.gender != rp.gender:
                continue
            if prof.age_ranges and rp.age >= 0:
                if not any(lo - 2 <= rp.age <= hi + 2 for lo, hi in prof.age_ranges):
                    continue
            sc = 0.0
            # Un-shift dates if global shift recovered
            unshifted_dates = set()
            for ds in prof.dates:
                try:
                    unshifted_dates.add((dt.date.fromisoformat(ds) - dt.timedelta(days=global_shift)).isoformat())
                except ValueError:
                    pass
            if any(vd in unshifted_dates for vd in kf.get("visit_dates") or []):
                sc += 4.0
            cond_hint = kf.get("condition_hint")
            if cond_hint:
                from kavachmed.privacy import dx_rollup

                hc, _, _ = dx_rollup.canonicalize_dx(cond_hint)
                if hc in prof.conditions:
                    sc += 8.0 if cond_counts[hc] <= 3 else 1.5
            if kf.get("hospital") and prof.hospital and kf["hospital"].lower() in prof.hospital.lower():
                sc *= 2.0
            if sc > best_score:
                best_score = sc
                if prof_probs.get(gkey):
                    best_match_rid = max(prof_probs[gkey].items(), key=lambda kv: kv[1])[0]
        if is_target:
            a5_targets += 1
            if best_score >= 4.0 and (best_match_rid == rid or best_score >= 8.0):
                a5_hits += 1
        elif best_score >= 8.0:
            a5_decoy_fp += 1

    a5_rate = a5_hits / max(1, a5_targets)

    return {
        "patients_evaluated": len(patient_to_rid),
        "records_extracted": len(extracted),
        "A1_direct_exposure": round(a1_rate, 4),
        "A3_record_linkage": round(a3_mean, 4),
        "A4_profile_reid": round(a4_reid, 4),
        "A5_targeted_success": round(a5_rate, 4),
        "A5_global_shift_recovered": global_shift,
        "A5_decoy_false_positives": a5_decoy_fp,
    }


def uniqueness_vs_attributes_curve(registry_path: Path) -> list[dict[str, Any]]:
    """Compute uniqueness rate on registry as QI attributes are added (D15)."""
    reg = list(load_registry(registry_path).values())
    n = max(1, len(reg))

    def _band10(a: int) -> str:
        return "85+" if a >= 85 else f"{(a // 10) * 10}-{(a // 10) * 10 + 9}"

    levels = [
        ("1: sex", lambda p: (p.gender,)),
        ("2: sex + age_10yr", lambda p: (p.gender, _band10(p.age))),
        ("3: sex + age_10yr + district", lambda p: (p.gender, _band10(p.age), p.district)),
        ("4: sex + age_10yr + district + tehsil", lambda p: (p.gender, _band10(p.age), p.district, p.tehsil)),
        ("5: sex + age_10yr + district + tehsil + village", lambda p: (p.gender, _band10(p.age), p.district, p.tehsil, p.village)),
        ("6: sex + age_exact + district + pin", lambda p: (p.gender, p.age, p.district, p.pin)),
    ]
    curve: list[dict[str, Any]] = []
    for label, fn in levels:
        counts = Counter(fn(p) for p in reg)
        unique_cnt = sum(1 for p in reg if counts[fn(p)] == 1)
        k_lt_5 = sum(1 for p in reg if counts[fn(p)] < 5)
        curve.append(
            {
                "attributes": label,
                "unique_share": round(unique_cnt / n, 4),
                "k_lt_5_share": round(k_lt_5 / n, 4),
            }
        )
    return curve


def eval_a6_pseudonym_inversion(records_jsonl: Path, registry_path: Path) -> dict[str, Any]:
    """Test A6: can any emitted PSN be inverted from unkeyed hashes, base64, reversal, or raw IDs?"""
    import base64

    psns: set[str] = set()
    if records_jsonl.is_file():
        for line in records_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                if obj.get("patient"):
                    psns.add(str(obj["patient"]))
    if not psns:
        return {"psns_tested": 0, "inverted": 0, "A6_inversion_rate": 0.0}

    reg = list(load_registry(registry_path).values())
    transforms: set[str] = set()
    for p in reg:
        for val in (p.name, p.phone, p.email, p.aadhaar_last4):
            if not val:
                continue
            for cand in (val, val.upper(), val[::-1]):
                transforms.add(cand)
                transforms.add(base64.b64encode(cand.encode("utf-8")).decode("ascii"))
                for alg in (hashlib.md5, hashlib.sha1, hashlib.sha256, hashlib.sha512):
                    hx = alg(cand.encode("utf-8")).hexdigest().upper()
                    transforms.add(hx[:8])
                    transforms.add(hx[:12])
                    transforms.add(hx)

    inverted = 0
    for psn in sorted(psns):
        core = psn.replace("PSN-", "", 1).upper()
        if core in transforms or any(t and len(t) >= 6 and t in core for t in transforms):
            inverted += 1
    return {
        "psns_tested": len(psns),
        "inverted": inverted,
        "A6_inversion_rate": round(inverted / len(psns), 4),
    }


def eval_a7_cross_release(out_a: Path, out_b: Path) -> dict[str, Any]:
    """Test A7: compare PSNs and shifted dates across two releases under different secrets."""
    def _load_psns_and_dates(out_dir: Path) -> tuple[set[str], dict[str, list[str]]]:
        psns: set[str] = set()
        rj = out_dir / "records.jsonl"
        if rj.is_file():
            for line in rj.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    obj = json.loads(line)
                    if obj.get("patient"):
                        psns.add(str(obj["patient"]))
        dates_by_file: dict[str, list[str]] = {}
        for ev in extract_dataset_records(out_dir / "sanitized"):
            if ev.dates:
                dates_by_file[f"{ev.file_rel}|{ev.loc}"] = ev.dates
        return psns, dates_by_file

    psns_a, dates_a = _load_psns_and_dates(out_a)
    psns_b, dates_b = _load_psns_and_dates(out_b)
    psn_overlap = len(psns_a & psns_b) / max(1, len(psns_a | psns_b)) if (psns_a or psns_b) else 0.0

    common_keys = sorted(set(dates_a) & set(dates_b))
    date_matches = 0
    date_total = 0
    for k in common_keys:
        da, db = dates_a[k], dates_b[k]
        for x, y in zip(da, db):
            date_total += 1
            if x == y:
                date_matches += 1
    date_overlap = date_matches / max(1, date_total) if date_total else 0.0
    return {
        "psn_overlap_rate": round(psn_overlap, 4),
        "date_collision_rate": round(date_overlap, 4),
        "A7_cross_release_risk": round(max(psn_overlap, date_overlap), 4),
    }
