# NaN QC report — neurodags feature table

- source: `outputs/aggregate_full_raw.csv`
- shape: 362 recordings x 14 features x 273 sensors (union) = 1383564 cells
- **total NaN: 8204 (0.59%)** = structural 7392 (90%) + scattered 812 (10%)

## Per-dataset

| dataset | recordings | sensors present | NaN frac | NaN cells |
|---|---:|---:|---:|---:|
| lsd | 226 | 271 | 0.0073 | 6328 |
| perampanel | 40 | 272 | 0.0073 | 1120 |
| ketamine | 36 | 272 | 0.0039 | 532 |
| psilocybin | 30 | 273 | 0.0020 | 224 |
| tiagabine | 30 | 273 | 0.0000 | 0 |

## Dead channels (present sensor, ALL features NaN)

58 (recording x sensor) instances.

### Bad-sensor ranking
| sensor | dead in N recordings |
|---|---:|
| MRT36 | 54 |
| MLO11 | 2 |
| MLT37 | 2 |

## Partial-channel NaN

None — every scattered NaN is a whole dead channel.
