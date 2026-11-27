# Jalagaara (KavachMed) — Clinical Data Privacy Firewall

**Track 3 · Secret Wards · Offline, CPU-Only, LLM-Free De-Identification Pipeline**

Jalagaara is a deterministic, two-pass streaming clinical de-identification firewall designed to sanitize heterogeneous hospital records (CSV, TSV, JSON, FHIR R4, and clinical text notes) while preserving medical utility and enforcing fail-closed quarantine on unsafe or unsupported inputs.

---

## 1. Setup & Offline Execution

### Prerequisites
- Python 3.12+ (tested on Python 3.12.16 Linux and Python 3.14.3 Windows)
- Pinned dependencies in [requirements.txt](requirements.txt) (no LLMs, no pretrained transformers)

### Install (network allowed during setup only)
```bash
python -m pip install -r requirements.txt
```

### Run Offline (no network)
```bash
export RAKSHA_SECRET="your-secret-key"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python main.py --input /path/to/input --output /path/to/output
```

If `RAKSHA_SECRET` is unset or blank, `main.py` exits immediately with code `2` before reading any input files.

### Verification & Self-Audit
```bash
pytest -q tests/unit tests/contract tests/integration
python tools/self_audit.py --kit .
```

---

## 2. Entry Point & Configuration

- **Manifest:** [raksha.toml](raksha.toml) declares `name = "Jalagaara"`, `setup`, and `run = "python main.py --input {input} --output {output}"`, with `KAVACHMED_POLICY = "config/policy.yaml"`.
- **CLI Entry Point:** [main.py](main.py) parses `--input` and `--output`, derives HKDF-SHA256 subkeys via `kavachmed.security.load_keys()`, loads [config/policy.yaml](config/policy.yaml), and executes `kavachmed.pipeline.run()`.
- **CI Workflow:** [.github/workflows/ci.yml](.github/workflows/ci.yml) runs unit/contract tests, two-run determinism checks on synthetic fixtures, and `tools/self_audit.py --kit .`.

---

## 3. Output Contract

Running `main.py` writes three artifacts into `{output}`:

```text
{output}/
├── sanitized/      # Sanitized files (UTF-8, mirrored or privacy-renamed relative paths)
├── report.json     # Schema "raksha-report/1", keyed by input_sha256 (never filenames)
└── records.jsonl   # One JSON object per public record: {"file", "loc", "patient"}
```

- **Locators (`loc`):** `row:N` (0-based CSV/TSV data row, where `row:0` is the first data row after the header), RFC 6901 JSON pointer (`/entry/0/...` or `/`), or `doc` (text notes).
- **Quarantined files:** Have `status: "quarantined"`, `output: null`, no file in `sanitized/`, and no rows in `records.jsonl`.
- **Output Ownership Preflight:** Before writing any sanitized artifact, the pipeline checks existing files in `sanitized/` against any prior `report.json` manifest and aborts before mutation on unowned collisions.

---

## 4. Supported Input Dispositions

| Format | Disposition | Notes |
|---|---|---|
| CSV / TSV | Sanitized | Strict delimiter/column-width validation; UTF-8/BOM/UTF-16/cp1252 decode; preserves headers and `row:N` locators |
| JSON / FHIR R4 | Sanitized | Depth-guarded parser (`<= 4,000` via `config/policy.yaml`); preserves numeric/boolean JSON types and RFC 6901 pointers |
| Text (`.txt`, `.md`, `.log`) | Sanitized | Clinical notes and unstructured narratives (`loc: "doc"`) |
| XML (`.xml`) | **Quarantined** | Parsed with `xml.parsers.expat` rejecting DTDs, external entities (XXE), and depth `> 500`; never copied raw |
| Unsupported / Empty / Corrupt | **Quarantined** | Fail-closed (`unsupported_format`, `empty_file`, `invalid_json`, `undecodable`, `excessive_depth`) |

---

## 5. Architecture

Jalagaara executes a **two-pass streaming pipeline** (`kavachmed/pipeline.py`) to bound peak memory on large corpora:

1. **Pass 1 (Stream & Evidence):** Streams files via `inventory.iter_scan()`, parses each file once, extracts clinical shield spans and detection spans, collects compact `Evidence` and `_FileMeta`, and releases raw file buffers.
2. **Cross-File Resolution & Planning:** Runs `linkage.resolve()`, `privacy.plan()` (truthful patient-level age banding across `[0, 5, 10, 18, 30, 40, 50, 60, 70, 85]`, LGD geography generalization, ICD-10/English/Devanagari rare-diagnosis rollup, and adaptive k-anonymity/attribute-budget controller), deterministic filename allocation (`rename.allocate_output_paths()`), output ownership preflight, and `residual_a.compile_keys()`.
3. **Pass 2 (Project, Gate & Emit):** Re-reads one file at a time (`inventory.read_entry()`), verifies its SHA-256 digest (fail-closed on mismatch), projects edits (including ABDM-aware FHIR R4 Bundle sanitization in `kavachmed/adapters/fhir.py` and per-family date shifting in `kavachmed/dates.py`), runs the bounded 3-pass residual gate (`residual_a` + independent default-deny `residual_b` with NFKC/homoglyph/zero-width/base64/%xx/entity decoding), and writes `sanitized/` before releasing per-file buffers.
4. **Emit:** Writes `records.jsonl` and `report.json` (`raksha-report/1`) with counts-only stage timings.

### Key Plan D Privacy, FHIR & Assurance Components
- **Date Engine & Order Oracle (`kavachmed/dates.py`):** Per-source `DMY`/`MDY` inference (hospital stickiness + header hints), ISO/numeric/named-month/Devanagari/chat/year-less format preservation, and DOB vs clinical event disambiguation.
- **Privacy Layer & LGD Lexicons (`kavachmed/privacy/`, `tools/build_lgd_lexicons.py`):** Truthful patient-level age bands (excluding clinical durations), LGD states/districts/subdistricts/hashed villages (`config/lexicons/MANIFEST.json`), multilingual concept rollup (`k_dx`), and risk-controller escalation (`k_target`, `max_qi_kinds`).
- **ABDM FHIR R4 Adapter (`kavachmed/adapters/fhir.py`):** Registered (`REGISTER_FHIR = True`), preserves `meta.profile` and UUID references, protects `Practitioner`/`Organization`/`Observation.valueQuantity`, sanitizes XHTML `text.div` and text `DocumentReference`/`Binary`, drops `Media`, and runs `validate_fhir_bundle()` pre-write.
- **Layer B Residual Re-Scanner (`kavachmed/residual_b.py`):** Strictly isolated from `detect`/`shield`/`evidence`/`linkage`/`residual_a`/`typing`/`privacy`; runs `B1`–`B4` checks on bytes read back from disk.
- **Red Team, Compliance & Brief (`tools/redteam`, `tools/compliance_map.py`, `tools/brief_numbers.py`):** Simulated `A1`–`A7` adversary evaluation (`docs/expert_determination.md`), policy-generated `docs/compliance.md` (with de-identification terminology linting), `tools/run_d_experiments.py` (`docs/d_experiments.json`), and measured 2-page A4 [Project Brief](docs/brief.pdf).

For compliance and security details, see [docs/compliance.md](docs/compliance.md) and the 2-page [Project Brief](docs/brief.pdf).

---

## 6. Measured Results & Remaining Team Dependencies

- **Contract & Determinism (200-file practice corpus):** `CONTRACT OK` (194 sanitized, 6 quarantined, 823 public records, 19/19 ABDM FHIR R4 bundles valid); byte-identical `sanitized/` and `records.jsonl` on repeat runs with the same `RAKSHA_SECRET`; unlinkable outputs under a different secret; `SELF AUDIT OK`.
- **De-Identification & Red-Team Exposure (`check_dev.py` & `tools/redteam`):** `0/3,230` (0.0%) personal-data pieces leaked across Latin/Devanagari/Hinglish and all hidden channels, `0/4` planted canaries leaked, `1,016/4,855` (20.9%) medical values kept (with Owner B `detect`/`shield` and Owner C `linkage`/`residual_a` still at Phase 1 baseline/stubs), and simulated adversary exposure reduced from raw `A1=1.000, A3=0.687, A4=1.000, A5=1.000` to sanitized `A1=0.000, A3=0.008, A4=0.000, A5=0.071, A6_inv=0.000` (`very_small_risk`).
- **Remaining Owner B/C Integration:** Replacing Owner B's Phase 1 fail-closed `detect` stub (for higher non-FHIR medical utility retention and custom header/key sensitivity) and Owner C's `linkage` / `residual_a` stubs (for cross-file patient pseudonyms `PSN-XXXXXXXX` and linkage F0.5).
- **Scale Memory:** Full 20× Linux Python 3.12 RSS measurement after final B/C/D integration remains to be recorded.
