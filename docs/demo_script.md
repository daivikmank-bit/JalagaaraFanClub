# KavachMed — 60-Second Desk-Review Demo Script (Owner D)

1. **0:00–0:15 · Fail-closed secret & deterministic run**
   - Show `unset RAKSHA_SECRET && python3 main.py --input ../Jalagaara/dev/input --output /tmp/bad` failing immediately with `ERROR security E_SECRET` before touching input or creating `/tmp/bad`.
   - Run `RAKSHA_SECRET=practice-secret-01 python3 main.py --input ../Jalagaara/dev/input --output out` and `python3 ../Jalagaara/validate_output.py --input ../Jalagaara/dev/input --output out` → `CONTRACT OK`.

2. **0:15–0:30 · Red-team replay (raw vs sanitized)**
   - Run `python3 -m tools.redteam --input ../Jalagaara/dev/input --registry ../Jalagaara/dev/attacker/registry.csv --facts ../Jalagaara/dev/attacker/known_facts.jsonl --truth ../Jalagaara/dev/truth/truth.json --output out`.
   - Point out `Raw A4 = 100.0%` dropping to `Sanitized A4 = 2.1%` (with `A1 = 0.0%`, `A5 = 7.1%`, `0` decoy false positives), plus the 6-step attribute-budget uniqueness curve.

3. **0:30–0:45 · Unseen-name canary & disk-dirty fault caught live by Layer B**
   - Run `make canary` (`tests/canary/test_canary_suite.py`) showing unseen Latin/Devanagari/lowercase names in cells, keys, headers, `source_path`, base64, FHIR `text.div`, and a memory-clean/disk-dirty writer fault all caught by `residual_b.py` on read-back with 0 survivors.

4. **0:45–1:00 · ABDM-aware FHIR & live policy-driven compliance crosswalk**
   - Show 19/19 dev FHIR bundles passing `validate_fhir_bundle` with `meta.profile` and `Composition.section.entry` UUID references intact and `Practitioner.name` preserved.
   - Flash `docs/compliance.md`, explaining why output is classified as **de-identified** (not irreversibly anonymous) and how changing `config/policy.yaml` updates the deviation register and fails lint if banned terminology appears.
