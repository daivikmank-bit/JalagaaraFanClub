"""Adaptive risk controller, attribute budget, and pool-consistent PrivacyPlan (D14, FR-P5, FR-P9)."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Literal

from kavachmed.contracts import Evidence, ParsedFile, PrivacyPlan, RecKey, Record, Resolution
from kavachmed.privacy.bands import compute_cluster_bands
from kavachmed.privacy.dx_rollup import plan_dx_rollups
from kavachmed.privacy.location import extract_district_from_record, get_lgd_index


def _los_bucket(los_days: int | None) -> str:
    if los_days is None:
        return "none"
    if los_days <= 0:
        return "0"
    if los_days <= 3:
        return "1-3"
    if los_days <= 7:
        return "4-7"
    if los_days <= 14:
        return "8-14"
    return "15+"


def _extract_sex(rec: Record | None, ev: Evidence | None, qi: dict[str, Any] | None = None) -> str:
    if qi and qi.get("sex"):
        val = str(qi["sex"]).strip().casefold()
        if val.startswith(("m", "पु")):
            return "M"
        if val.startswith(("f", "म", "स्")):
            return "F"
    if ev and ev.sex:
        return ev.sex.upper()[:1]
    if rec is not None:
        for fld in rec.fields:
            col = fld.path.split("/")[-1].strip().casefold()
            if col in ("gender", "sex", "ling", "लिंग"):
                val = fld.text.strip().casefold()
                if val.startswith(("m", "पु")):
                    return "M"
                if val.startswith(("f", "म", "स्")):
                    return "F"
    return "U"


def _uniqueness_proxy(est_registry_k: float, qi_count: int) -> float:
    """Monotone uniqueness proxy combining registry-amplified class size and attribute budget."""
    base = 1.0 / max(1.0, est_registry_k)
    attr_factor = 1.0 - math.exp(-max(0, qi_count - 2) / max(1.0, est_registry_k + 6.0))
    return min(1.0, round(0.5 * base + 0.5 * attr_factor, 4))


def build_plan(
    res: Resolution,
    evidences: list[Evidence],
    parsed_all: list[ParsedFile],
    policy: dict[str, Any],
) -> PrivacyPlan:
    """Compute PrivacyPlan with patient-level bands, escalation ladder, and pool consistency."""
    ctrl_cfg = policy.get("controller") or {}
    loc_cfg = policy.get("location") or {}
    enabled = bool(ctrl_cfg.get("enabled", True))
    amplification = float(ctrl_cfg.get("assumed_amplification", 4.0))
    k_target = float(ctrl_cfg.get("k_target", 5))
    k_dx = int(ctrl_cfg.get("k_dx", 3))
    max_qi_kinds = int(ctrl_cfg.get("max_qi_kinds", 6))
    uniqueness_target = float(ctrl_cfg.get("uniqueness_target", 0.25))
    ladder: list[str] = list(ctrl_cfg.get("ladder") or ["age20", "state", "dx_rollup", "suppress_qi"])

    default_loc: Literal["district", "state", "none"] = str(loc_cfg.get("keep", "district"))  # type: ignore[assignment]
    if default_loc not in ("district", "state", "none"):
        default_loc = "district"
    default_pin: Literal["none", "pin3"] = "pin3" if str(loc_cfg.get("pin", "none")) == "pin3" else "none"

    records_by_key: dict[RecKey, Record] = {}
    qi_extra_by_key: dict[RecKey, dict[str, Any]] = {}
    extra_ages_by_key: dict[RecKey, list[int]] = {}
    ref_yr = int((policy.get("dates") or {}).get("dataset_ref_year", 2026))

    for pf in parsed_all:
        for rec in pf.records:
            if not rec.key[1].startswith("@"):
                records_by_key[rec.key] = rec
        # Support Pass-1 lightweight ParsedFile objects whose records were released
        fqi = (pf.extra or {}).get("fhir_qi")
        if isinstance(fqi, dict):
            rk: RecKey = (pf.rel_path, "doc")
            qi_extra_by_key[rk] = fqi
            ages_here: list[int] = []
            if fqi.get("age") is not None:
                ages_here.append(int(fqi["age"]))
            if fqi.get("birth_year") is not None:
                ages_here.append(max(0, min(120, ref_yr - int(fqi["birth_year"]))))
            if ages_here:
                extra_ages_by_key[rk] = ages_here
        # Pass-1 per-record QI summaries (privacy.spans.record_summary) for non-FHIR files
        rqi = (pf.extra or {}).get("record_qi")
        if isinstance(rqi, dict):
            for loc in sorted(rqi):
                summ = rqi[loc] or {}
                rk2: RecKey = (pf.rel_path, loc)
                if summ.get("ages"):
                    extra_ages_by_key.setdefault(rk2, []).extend(int(a) for a in summ["ages"])
                if summ.get("district") and rk2 not in qi_extra_by_key:
                    qi_extra_by_key[rk2] = {"district": summ["district"]}

    ev_by_key = {ev.key: ev for ev in evidences}
    all_keys = sorted(
        set(records_by_key)
        | set(qi_extra_by_key)
        | {k for k in ev_by_key if not k[1].startswith("@")}
        | {k for k in res.patient_of if not k[1].startswith("@")}
        | {k for k in res.cluster_of if not k[1].startswith("@")}
    )

    lgd = get_lgd_index()

    cluster_of: dict[RecKey, str] = {}
    cluster_records: dict[str, list[RecKey]] = {}
    for rk in all_keys:
        cid = res.cluster_of.get(rk) or f"orphan:{rk[0]}|{rk[1]}"
        cluster_of[rk] = cid
        cluster_records.setdefault(cid, []).append(rk)

    bands = compute_cluster_bands(
        cluster_of, evidences, records_by_key, policy, extra_ages_by_key=extra_ages_by_key
    )

    cluster_sex: dict[str, str] = {}
    cluster_dist: dict[str, str] = {}
    cluster_los: dict[str, str] = {}
    cluster_dx: dict[str, list[str]] = {}
    cluster_hospitals: dict[str, set[str]] = {}
    cluster_proc: dict[str, set[str]] = {}

    for cid, rks in sorted(cluster_records.items()):
        sexes: list[str] = []
        dists: list[str] = []
        los_vals: list[int] = []
        for rk in rks:
            rec = records_by_key.get(rk)
            ev = ev_by_key.get(rk)
            qi = qi_extra_by_key.get(rk)
            sx = _extract_sex(rec, ev, qi)
            if sx != "U":
                sexes.append(sx)
            d = (qi or {}).get("district") or extract_district_from_record(rec, lgd, ev=ev)
            if d:
                dists.append(str(d))
            los_v = getattr(ev, "los_days", None) if ev else None
            if los_v is None and qi and qi.get("los_days") is not None:
                los_v = int(qi["los_days"])
            if los_v is not None:
                los_vals.append(los_v)
            dxs = list(getattr(ev, "dx_terms", []) or []) if ev else []
            if qi and qi.get("dx_terms"):
                dxs.extend(qi["dx_terms"])
            if dxs:
                cluster_dx.setdefault(cid, []).extend(dxs)
            hosp = (ev.hospital if ev else None) or (qi.get("hospital") if qi else None)
            if hosp:
                cluster_hospitals.setdefault(cid, set()).add(str(hosp))
            procs = list(getattr(ev, "procedure_terms", []) or []) if ev else []
            if procs:
                cluster_proc.setdefault(cid, set()).update(procs)
        cluster_sex[cid] = sexes[0] if sexes else "U"
        cluster_dist[cid] = dists[0] if dists else "UNK"
        cluster_los[cid] = _los_bucket(max(los_vals) if los_vals else None)

    dx_rollups_by_cid = plan_dx_rollups(cluster_dx, k_dx=k_dx)

    coarse_mode: dict[str, str] = {cid: "lattice10" for cid in cluster_records}
    loc_level_by_cid: dict[str, Literal["district", "state", "none"]] = {cid: default_loc for cid in cluster_records}
    dx_rolled_by_cid: dict[str, bool] = {cid: False for cid in cluster_records}
    suppress_by_cid: dict[str, bool] = {cid: False for cid in cluster_records}

    def _effective_qi_tuple(cid: str) -> tuple[str, str, str, str]:
        sx = "U" if suppress_by_cid[cid] else cluster_sex[cid]
        bd = bands.get(cid, "Adult")
        if suppress_by_cid[cid] or loc_level_by_cid[cid] == "none":
            loc_val = "NONE"
        elif loc_level_by_cid[cid] == "state":
            loc_val = lgd.state_for_district(cluster_dist[cid]) or "STATE"
        else:
            loc_val = cluster_dist[cid]
        los_v = "none" if suppress_by_cid[cid] else cluster_los[cid]
        return (sx, bd, loc_val, los_v)

    def _qi_kind_count(cid: str) -> int:
        count = 0
        if not suppress_by_cid[cid]:
            if cluster_sex[cid] != "U":
                count += 1
            if cid in bands:
                count += 1
            if loc_level_by_cid[cid] != "none" and cluster_dist[cid] != "UNK":
                count += 1
            if cluster_hospitals.get(cid):
                count += 1
            if cluster_los[cid] != "none":
                count += 1
            if cluster_dx.get(cid) and not dx_rolled_by_cid[cid]:
                count += 1
            if cluster_proc.get(cid):
                count += 1
        return count

    if enabled and cluster_records:
        for step in ladder:
            bands = compute_cluster_bands(
                cluster_of, evidences, records_by_key, policy,
                coarse_by_cluster=coarse_mode, extra_ages_by_key=extra_ages_by_key,
            )
            eq_counts = Counter(_effective_qi_tuple(cid) for cid in cluster_records)
            changed = False
            for cid in sorted(cluster_records):
                est_k = eq_counts[_effective_qi_tuple(cid)] * amplification
                qk = _qi_kind_count(cid)
                u_proxy = _uniqueness_proxy(est_k, qk)
                if est_k >= k_target and qk <= max_qi_kinds and u_proxy <= uniqueness_target:
                    continue
                if step == "age20" and coarse_mode[cid] == "lattice10":
                    coarse_mode[cid] = "age20"
                    changed = True
                elif step == "state" and loc_level_by_cid[cid] == "district":
                    loc_level_by_cid[cid] = "state"
                    changed = True
                elif step == "dx_rollup" and not dx_rolled_by_cid[cid] and cid in dx_rollups_by_cid:
                    dx_rolled_by_cid[cid] = True
                    changed = True
                elif step == "suppress_qi" and not suppress_by_cid[cid]:
                    suppress_by_cid[cid] = True
                    coarse_mode[cid] = "suppress_qi"
                    loc_level_by_cid[cid] = "none"
                    changed = True
            if not changed:
                break
        bands = compute_cluster_bands(
            cluster_of, evidences, records_by_key, policy,
            coarse_by_cluster=coarse_mode, extra_ages_by_key=extra_ages_by_key,
        )

    final_eq = Counter(_effective_qi_tuple(cid) for cid in cluster_records)
    qi_kinds: dict[str, int] = {cid: _qi_kind_count(cid) for cid in sorted(cluster_records)}
    loc_level_rec: dict[RecKey, Literal["district", "state", "none"]] = {}
    pin_level_rec: dict[RecKey, Literal["none", "pin3"]] = {}
    dx_rollup_rec: dict[RecKey, str] = {}
    generalized: list[RecKey] = []
    suppress_recs: set[RecKey] = set()

    for rk in all_keys:
        cid = cluster_of[rk]
        loc_level_rec[rk] = loc_level_by_cid[cid]
        pin_level_rec[rk] = "none" if suppress_by_cid[cid] else default_pin
        if dx_rolled_by_cid[cid] and cid in dx_rollups_by_cid:
            dx_rollup_rec[rk] = sorted(set(dx_rollups_by_cid[cid].values()))[0]
        if suppress_by_cid[cid]:
            suppress_recs.add(rk)
        if loc_level_by_cid[cid] != default_loc or coarse_mode[cid] != "lattice10" or rk in suppress_recs:
            generalized.append(rk)

    k_vals = [final_eq[_effective_qi_tuple(cid)] * amplification for cid in cluster_records]
    risk_summary = {
        "clusters_total": len(cluster_records),
        "min_est_k": min(k_vals) if k_vals else 0.0,
        "median_est_k": sorted(k_vals)[len(k_vals) // 2] if k_vals else 0.0,
        "escalated_clusters": sum(
            1 for cid in cluster_records
            if coarse_mode[cid] != "lattice10" or loc_level_by_cid[cid] != default_loc
        ),
        "max_qi_kinds_observed": max(qi_kinds.values()) if qi_kinds else 0,
    }

    return PrivacyPlan(
        age_band=bands,
        location_level=loc_level_rec,
        dx_rollup=dx_rollup_rec,
        generalized=generalized,
        qi_kinds=qi_kinds,
        pin_level=pin_level_rec,
        suppress_qi=suppress_recs,
        risk_summary=risk_summary,
    )
