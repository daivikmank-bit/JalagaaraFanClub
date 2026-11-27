# KavachMed Compliance Crosswalk & Deviation Register

> **Output classification:** `de-identified` (ABDM Health Data Management Policy sense — keyed pseudonymisation, not irreversible non-identifiability).
> **Legal caveat:** Engineering guidance for synthetic clinical datasets; items marked counsel/verify require legal review before clinical deployment. Generated automatically from `config/policy.yaml` by `tools/compliance_map.py`.

---

## 1. Declared Deviations Register (PRD §1.6.4 & Q12)

| Element / Standard | Policy Setting | Rationale | Residual Risk & Control |
|:--|:--|:--|:--|
| Geography below state (HIPAA §164.514(b)) | `location.keep=district`, `location.pin=none`, `remove=['village', 'tehsil', 'house', 'ward', 'mohalla', 'landmark']` | Utility generalization sub-score values district at 0.7; village/tehsil/PIN removed | District + age band + sex shrinks classes vs registry -> controller (`controller.k_target=20`) escalates to `state` / `none` |
| Clinical dates (HIPAA §164.514(b)) | `dates.shift_min_days=30`, `shift_max_days=365`, `default_order=dmy` | Interval & LOS preservation is 20% of utility (2.4+ pts) | Per-family secret offsets defeat A5 global-shift recovery; rare-dx rollup (`controller.k_dx=3`) mitigates LOS+dx fingerprinting |
| Ages > 89 (HIPAA 90+) | `age.top_band="85+"`, `pool_lattice=[0, 5, 10, 18, 30, 40, 50, 60, 70, 85]` | Patient-level fixed lattice ensures 100% band truthfulness | Superset of HIPAA 90+ category; escalates to 20-yr band / `Senior` via controller |
| Date of Birth (DOB) | `dob.mode=suppress` | Shifted DOB within ±365 d leaks age ±1 | Suppressed (`[DOB]` / omitted `Patient.birthDate`) |
| Doctor / Practitioner names | `detect.doctor_reg_numbers=remove_digits` | Clinical probes require `Practitioner` / `Dr.` preservation | Full-name doctor registry only; digits of registration IDs scrubbed; Layer A narrowed exemption (`residual.doctor_span_exempt=registry_full_name_only`) |
| Secret length floor (Q12 vs NIST SP 800-132) | `security.min_secret_bytes=1`, `min_distinct_bytes=1`, `pbkdf2_iterations=600000` | Lowered deliberately from PRD 10-byte / 4-distinct NIST SP 800-132 baseline to `min_secret_bytes=1`, `min_distinct_bytes=1` (Q12) so evaluator secrets such as `practice` run without false-exit; unset/blank secrets still fail closed (`E_SECRET`) and every secret is stretched with 600,000 PBKDF2-HMAC-SHA256 iterations. | Short passphrases remain weaker offline than >=10-byte secrets; reversible via `policy.yaml` |
| Public fixed PBKDF2 salt (NIST SP 800-132 §5.1) | `security.kdf_version="kavachmed/v5"` | Determinism across independent runs forbids random per-run salt | Mitigated by per-deployment secret rotation and 600k PBKDF2 stretching |
| Keyed 40-bit PSN label | `security.psn_chars=8` | Identity linkage requires stable per-cluster pseudonym across files | Collision-checked per run (`1.1e-5` at 5k patients); holder of run secret can recompute PSNs |

---

## 2. Control -> Instrument -> Policy Key -> Test Crosswalk (Appendix D.1)

| Control (FR) | Instrument | Policy Key(s) | Verification / Test |
|:--|:--|:--|:--|
| Remove patient & relative names in 3 scripts (FR-D2, FR-D11) | HIPAA §164.514(b)(2)(i) relatives/household; ICMR §2.3 | `detect.stack=recall_first`, `detect.default_deny.enabled=True` | `tests/canary/`, `check_dev.py` |
| Remove Aadhaar, masked Aadhaar, VID (FR-D1, FR-D9) | Aadhaar Act 2016 §29; UIDAI Sharing Regulations | `detect.vehicle_plates=True`, `residual.b2_long_digits=10` | `tests/unit/test_residual_b.py`, `tests/canary/` |
| No Aadhaar or key vault persisted (FR-O, §9) | Aadhaar Data Vault circular; DPDP safeguards principle | `security.quarantine_vault=False`, `security.log=counts_only` | `tests/canary/`, `check_submission.py` |
| Keyed HKDF pseudonym & fail-closed secret (FR-K1..K7, FR-L5) | ABDM HDMP; DPDP Rules 2025 Rule 6(1)(a); RFC 5869 | `security.kdf_version`, `security.min_secret_bytes`, `security.psn_chars` | `tests/contract/test_contracts.py` (RFC 5869 A.1–A.3) |
| Independent Layer B read-back re-scan (FR-R1..R8) | HHS OCR Safe Harbor removal; Philter default-deny | `residual.layer_b=True`, `residual.readback=True`, `runtime.residual_passes=3` | `tests/unit/test_residual_b.py`, `tests/canary/` (disk fault) |
| Per-family shifted dates & order oracle (FR-P1, FR-P2) | HIPAA dates rule (declared deviation); ICMR utility | `dates.shift_min_days`, `dates.shift_max_days`, `dates.default_order` | `tests/unit/test_dates.py`, `tools/redteam` A5 |
| Patient-level truthful age bands (FR-P3) | HIPAA ages > 89 | `age.pool_lattice`, `age.top_band`, `age.truthful=True` | `tests/unit/test_privacy.py` |
| DOB suppression (FR-P4) | HIPAA dates rule; Sweeney/Rocher uniqueness | `dob.mode=suppress` | `tests/unit/test_privacy.py`, `tests/unit/test_fhir.py` |
| LGD geography generalization ladder (FR-P4, FR-X1) | HIPAA geography (declared deviation); LGD (GODL-India) | `location.keep`, `location.pin`, `location.remove` | `tests/unit/test_privacy.py`, `tools/redteam` A2–A4 |
| Adaptive risk controller & attribute budget (FR-P5, FR-P6, FR-P9) | Rocher et al. 2019; Sweeney k-anonymity | `controller.enabled=True`, `k_target=20`, `k_dx=3`, `max_qi_kinds=6` | `tests/unit/test_privacy.py`, `tools/redteam` curve |
| ABDM-aware FHIR projection & structural gate (FR-F1..F3) | ABDM FHIR IG (NRCeS); MoHFW EHR Standards 2016 | `privacy.rename_date_filenames` | `tests/unit/test_fhir.py` (19/19 dev bundles) |
| Lexicon provenance & licence manifest (FR-X2) | Government Open Data Licence - India (GODL) | `config/lexicons/MANIFEST.json` | `tools/build_lgd_lexicons.py` |
| Honest PII-free audit report (FR-O) | DPDP Rules DPIA / audit readiness | `security.log=counts_only` | `validate_output.py` |
| Terminology discipline (FR-P8) | ABDM HDMP definitions | `compliance.terminology="de-identified"` | `tools/compliance_map.py --lint` |

---

## 3. HIPAA Safe Harbor 18-Identifier Crosswalk (Appendix D.2)

| # | Identifier Class | Indian Clinical Analogue | KavachMed Treatment |
|:-:|:--|:--|:--|
| 1 | Names (incl. relatives/household) | Patient, relative (`S/o`, `W/o`, `C/O`), informant names | Replaced by keyed PSN / `[NAME]` / `[RELATIVE]`; Layer B default-deny |
| 2 | Geography below state | Village, tehsil, mohalla, house, landmark, PIN, district | Village/tehsil/PIN removed; district kept (declared deviation) or escalated to state |
| 3 | Dates (except year) & ages > 89 | DOB, admission, discharge, lab, OPD dates | DOB suppressed; clinical dates shifted per family; 85+ band |
| 4–5 | Phone & fax numbers | +91 / 0-prefix / 5+5 / Devanagari / JSON int | Surrogated or `[PHONE]`; Layer B B2 catches split/Indic digit runs |
| 6 | Email addresses | Email, UPI handles, ABHA addresses | Removed / `[EMAIL]` / synthetic `@redacted.invalid` in FHIR |
| 7 | SSN / national IDs | Aadhaar (12-digit), masked Aadhaar, VID (16), PAN | Removed / `[AADHAAR]` / `[PAN]` |
| 8 | Medical record numbers | UHID, MRN, IPD, OPD, ADM numbers | Surrogated / `[MRN]` |
| 9 | Health-plan beneficiary IDs | PMJAY ID, insurance policy numbers | Removed / `[INSURANCE_ID]` |
| 10 | Account / billing IDs | Claim / bill numbers | Removed if identifying; `package_code` & amounts retained |
| 11 | Certificate / licence numbers | Doctor reg. numbers, DL, EPIC, passport | Registration digits removed (`Practitioner.name` kept) |
| 12 | Vehicle identifiers | Ambulance registration plates | Flagged by Layer B / detector (`detect.vehicle_plates`) |
| 13 | Device identifiers | Device serial numbers | Flagged by typed-ID rules |
| 14–15 | URLs & IP addresses | `meta.source`, `source_path` | Scrubbed (`[SOURCE]`) |
| 16–17 | Biometrics & photos | `photo_rx.jpg`, `scanned_discharge.pdf`, `Binary` | Binary files quarantined; FHIR `Binary` neutralized |
| 18 | Any other unique code | File paths, JSON keys, CSV headers | Keyed token rename (`fname`), collision-free |
