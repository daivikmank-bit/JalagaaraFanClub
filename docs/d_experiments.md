# Owner D Measured Experiments (E2-E5, E11-E12, E15, E17, E19)

## E2: Age Band x Location Level vs Registry Uniqueness
| Age Band | Location | k=1 Share | k<5 Share | Generalization Proxy |
|:-:|:-:|:--:|:--:|:--:|
| 5 yr | village | 0.821 | 1.000 | 0.950 |
| 5 yr | tehsil | 0.662 | 1.000 | 0.850 |
| 5 yr | district | 0.412 | 0.958 | 0.800 |
| 5 yr | state | 0.004 | 0.108 | 0.650 |
| 10 yr | village | 0.692 | 1.000 | 0.900 |
| 10 yr | tehsil | 0.496 | 1.000 | 0.800 |
| 10 yr | district | 0.208 | 0.821 | 0.750 |
| 10 yr | state | 0.000 | 0.033 | 0.600 |
| 20 yr | village | 0.496 | 0.917 | 0.800 |
| 20 yr | tehsil | 0.292 | 0.800 | 0.700 |
| 20 yr | district | 0.092 | 0.463 | 0.650 |
| 20 yr | state | 0.000 | 0.033 | 0.500 |

## E3: Shift Range Comparison
| Range | Min Shift | A5 Global-Shift Recovery Risk |
|:--|:--:|:--:|
| +/-180 (allows 0/small) | 1 d | higher_near_0 |
| +/-[30,365] per-family | 30 d | global_recovery_0 |

## E4: Rare-Diagnosis Rollup Threshold (`k_dx`)
| `k_dx` | Patients Rolled Up | Total |
|:-:|:-:|:-:|
| 2 | 2 | 11 |
| 3 | 4 | 11 |
| 5 | 11 | 11 |

## E5: Registry Amplification (`assumed_amplification`)
| Amplification | Min Est. k | Median Est. k | Escalated Clusters |
|:-:|:-:|:-:|:-:|
| 2.0x | 4.0 | 8.0 | 2/10 |
| 4.0x | 8.0 | 16.0 | 0/10 |
| 6.0x | 12.0 | 24.0 | 0/10 |

## E11 & E19: Layer B Canary Ablation & Clean-Token Over-Flag Rate
- **E11 (Layer B off vs on):** `3/3` survivors -> `0/3` survivors
- **E19 (clean/ token over-flag rate):** `30/165` = `0.1818`

## E12: Date-Bearing Filenames
- **Date/numeric filenames in dev:** `125` (`rename_date_filenames=False`)

## E17: LGD Lexicon Ablation
- **District match rate:** `93/93` = `1.0000` (vs `0.0000` without LGD)

## E15: Attribute Budget Uniqueness Curve
| Attributes | Unique Share | k<5 Share |
|:--|:--:|:--:|
| 1: sex | 0.000 | 0.000 |
| 2: sex + age_10yr | 0.000 | 0.033 |
| 3: sex + age_10yr + district | 0.208 | 0.821 |
| 4: sex + age_10yr + district + tehsil | 0.496 | 1.000 |
| 5: sex + age_10yr + district + tehsil + village | 0.692 | 1.000 |
| 6: sex + age_exact + district + pin | 0.958 | 1.000 |
