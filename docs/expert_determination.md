# Expert-Determination-Style Risk Summary (KavachMed Red Team)

> **Scope:** Quantitative evaluation against the published A1–A7 registry & known-facts adversary (`docs/ATTACKS.md`). Not a legal certification.

## 1. Methods & Attacker Assumptions
- Population registry (`registry.csv`, 4–5x patient count) with demographic + leaked-contact overlay (60%).
- Known-facts auxiliary knowledge (`known_facts.jsonl`) with visit dates, hospital, rare condition, and decoys.
- Attacker attempts A1 (direct), A2/A3 (filtered Sweeney linkage), A4 (cross-file profile pooling), A5 (targeted + global-shift recovery), A6 (PSN inversion), and A7 (cross-release comparison).

## 2. Measured Attack Exposure (Raw vs Sanitized)

| Attack | Raw Input | Sanitized Release | Target |
|:--|:--:|:--:|:--:|
| A1 Direct Exposure | 100.0% | 0.0% | 0% |
| A3 Record Linkage (mean prob) | 69.1% | 2.0% | low |
| A4 Profile Re-ID (>=0.5) | 100.0% | 2.1% | <=5% |
| A5 Targeted Attack | 100.0% | 0.0% | <=15% |
| A5 Decoy False Positives | 0 | 0 | 0 |
| A6 Pseudonym Inversion | — | 0.0% | 0% |

## 3. Uniqueness vs Attribute Count Curve

| Released Quasi-Identifiers | Registry Unique | k < 5 Share |
|:--|:--:|:--:|
| 1: sex | 0.0% | 0.0% |
| 2: sex + age_10yr | 0.0% | 3.3% |
| 3: sex + age_10yr + district | 20.8% | 82.1% |
| 4: sex + age_10yr + district + tehsil | 49.6% | 100.0% |
| 5: sex + age_10yr + district + tehsil + village | 69.2% | 100.0% |
| 6: sex + age_exact + district + pin | 95.8% | 100.0% |

**Determination Verdict:** `very_small_risk`
