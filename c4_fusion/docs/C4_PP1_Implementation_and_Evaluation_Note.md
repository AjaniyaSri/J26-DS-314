# FloodRisk Sri Lanka — C4 PP1 Implementation and Evaluation Note

**Component:** C4 — Independent multimodal breadth  
**Research stage:** Progress Presentation 1; retrospective proof of concept  
**Data window:** 17 November–26 December 2025 (forecast origins through 25 December)  
**Evaluation:** Seven ablations, shared chronological holdout, station-specific flood-warning thresholds

## 1. Objective, design revision and research contribution

C4 tests whether combining **satellite, weather and environmental information** improves the prediction of river conditions compared with one or two modalities. The seven independently trained configurations (M1–M7) preserve the proposal's central controlled-ablation objective. C4 does **not** reuse trained C1–C3 encoders, model outputs, weights or their engineered features; sharing source references, gauge identities, catchment polygons and observed targets is acceptable.

The proposal originally targeted future flood condition or probability from 8–10 historical events and used event-level cross-validation. For PP1, the response variable changed to the **observed next-day river stage** because reliably time-matched flood-extent reference masks were difficult to establish for every event and satellite acquisition. This is a **material revision**, not the original flood-probability task completed without change. The current model predicts the continuous stage change (metres), with a linear output, then calculates:

`predicted_stage(T+1, 09:00) = observed_stage(T, 09:00) + predicted_stage_change`.

**Research value at this stage:** a reproducible comparison of modality combinations on one genuinely aligned station-level regression dataset, including a persistence baseline and evaluation of misses relative to each gauge's warning levels. **Not yet demonstrated:** statistically supported multimodal superiority, reliable warning of severe floods, performance on unseen independent flood events, or operational readiness. The novelty is the *tested combination and comparative design for the revised task* rather than a claim of a new neural architecture or a proven best predictor.

## 2. Data and quality-screened sample

We assembled **1,600 possible station-day slots** from 40 geographic gauges and 40 daily DMC reports; **1,429 valid observed T/T+1 numerical stage pairs** were available across **39 labelled gauges**. Thambuththegama is retained in the 40-gauge geographic registry but has no corresponding labelled pair in this period. We preserved genuine negative stages and converted Nagalagam Street's feet-based source measurements to metres. Numerical records were not fabricated to fill missing labels.

All **1,429 labelled rows** were successfully matched to source-verified inputs from the three modalities and passed Notebook 06's joint completeness and temporal/identity tests. This is **feature completeness after quality screening**, not proof that all data have equal measurement accuracy or that all catchments are officially verified.

| Modality | Actual source and input shape | Verified preparation and quality controls | Remaining uncertainty |
|---|---|---|---|
| **Satellite — Sentinel-1** | GEE `COPERNICUS/S1_GRD` VV/VH, 12.8 × 12.8 km near each gauge, `[256,256,2]` per selected scene | 18-day maximum acquisition age; strictly pre-origin scene times; ≥97% actual valid pixels; scene/patch SHA-256, shape checks; 264 unique physical patch files verified. Originally 1,408/1,429 station-days available; three same-pass Glencourse mosaics recovered the other **21** (9+6+6), each >99.99% valid union. Reuse of images is explicitly recorded. | Sparse acquisitions and image age; SAR is not necessarily contemporaneous with peak flood stage; ImageNet pretrained features are not SAR-specific. |
| **Satellite — ASCAT SWI** | Copernicus ASCAT SWI V4; SWI001, SWI005, SWI010 and quality/age/coverage metadata; catchment summaries | Cached 45 source-product dates; date and quality/SSF screening; native-grid, polygon area-weighted extraction; retained all 1,429 records, none needing nearest-cell fallback in the final aligned rows. The newest accepted source is **T−1 or earlier**, allowed up to **T−7**; median observed age is **1 day**. | Coarse native spatial resolution; a T−1 product is not proven to have been disseminated by 09:00 on T; source-derived quality flags are not independent soil measurements. |
| **Weather** | GEE `ECMWF/ERA5_LAND/HOURLY`, `[1429,14,4]` with precipitation, temperature, dewpoint and surface pressure | 2,026 unique station/catchment-day summaries cached; 24 hourly records checked per day; precipitation summed from the **hourly increment band**, other variables averaged, units converted; area-weighted native-grid polygon aggregation; no missing daily summaries, negative rainfall, or low source-hour coverage in recorded QA. | Reanalysis, not live forecast/field observations; ~9–11 km native spatial information is coarse for some small catchments; real product publication delay not simulated. |
| **Environment** | Copernicus DEM GLO-30, ESA WorldCover v200, JRC Global Surface Water and HydroRIVERS; `[1429,9]` from 40 static gauge/catchment vectors | One vector computed per candidate catchment, with cached source versions and polygon identifiers; nine numerical descriptors checked for finite values (area, mean elevation/slope, four land-cover fractions, historical persistent water, river density). Same static vector reused across that station's dates. | Candidate watershed boundaries are provisional; JRC is historical water context, not the 2025 flood extent; static descriptors can encode station identity. |

**Spatial QA:** Candidate upstream polygons were generated from MERIT Hydro routing and kept separate from the rectangular extraction bounds. All 39 labelled polygons passed existing geometry/identity checks. Five priority catchments retained documented warnings, and the original gauge points for Ellagawa, Pitabeddara and Putupaula were **33.6, 0.4 and 28.7 m** outside their candidate polygons. Their small offsets were visually screened but **not hydrologically certified**. The canonical `shared/spatial/catchments/QA_README.md` retains the full evidence and unresolved flags.

### Timing and availability assumptions

The model's **simulated prediction origin** is **09:00 Sri Lanka time on day T**, predicting the gauge stage at **09:00 on T+1 (24-hour observation horizon)**. DMC readings refer to 09:00 but the nominal report is **09:30**. **We assume the T 09:00 gauge value is known at the simulated 09:00 origin** for retrospective study; actual public availability at 09:00 has **not** been demonstrated. This is an explicit limitation, not an operational forecast issued at that time.

ERA5-Land provides 14 chronological rows of 24-hour catchment-weather summaries, with the latest period based on hourly data preceding T 09:00. Because GEE source timestamps are hourly and 09:00 SLST is 03:30 UTC, the last hourly timestamp is **03:00 UTC (08:30 SLST)**; hourly intervals and product timestamp semantics are documented. Thus the period is conventionally described as **T−1 09:00 to T 09:00**, implemented with the eligible discrete hourly records—not a full future calendar-day T aggregate. **This remains a retrospective reanalysis-availability assumption** rather than a real-time ingestion guarantee.

ASCAT is **daily rather than hourly**: use the latest quality-accepted previous-day product (usually **T−1**, otherwise up to T−7), never the completed T daily product. SAR scenes are accepted only when both acquisition time and age precede/satisfy the T origin. The three Glencourse mosaics have both original slice acquisitions preceding their assigned forecast origins; the 21 recovered records pass the recorded temporal QA.

## 3. Architecture implemented versus originally proposed

The implemented neural networks **learn their embeddings during supervised training**; the data loaders provide screened source arrays and train-only-scaled numerical inputs, not pretrained C4 embeddings.

| Branch | Implemented encoder | Learned output |
|---|---|---:|
| Satellite | Trainable SAR 2→3 channel adapter → **ImageNet-initialized EfficientNet-B0 frozen backbone** → 64-D SAR projection; ASCAT values/quality/auxiliary MLP →16-D; SAR age/availability/quality metadata MLP →8-D; concatenate 64+16+8→Dense(128) | **128-D** |
| Weather | One-layer bidirectional LSTM on 14×4 weather history → Dense projection | **16-D** |
| Environment | 9 descriptors → Dense(32) → Dense(16) | **16-D** |

For M7, these representations are concatenated (**160-D**) and combined with the same normalized **current-stage reference scalar** given to *every* configuration, then processed by a dense 64→32→1 regression head. Models have independent learned weights and are trained separately; persistence (`predicted_change = 0`) is an external, no-training comparator.

**Precise fusion disclosure:** **Yes, intermediate fusion of modality-specific learned representations was implemented. No, the proposed learned modality-attention/gating layer that dynamically assigns weights to satellite, weather and environment was *not* implemented.** Current fusion is concatenation plus a dense regression head. Satellite **availability gating** is implemented inside the satellite branch, but it is **not cross-modality attention**. Therefore describe the current model as **modality-specific intermediate concatenation fusion**, not as a completed “attention-weighted modality-aware fusion” architecture. Attention remains a documented future extension requiring a new controlled experiment. The proposal's 12-band SAR/optical/DEM image, engineered CHIRPS weather, HAND/TWI-rich environmental vector and sigmoid classification head were not preserved; these were deliberately reduced or replaced for feasibility and the new stage-regression target.

## 4. Training and evaluation design

The **chronological split** was 809 train (17 Nov–9 Dec), 269 validation (10–16 Dec), and 351 test (17–25 Dec) observations, with next-day targets extending to 26 December. All seven models used exactly the same labelled sample identities, with numerical scalers and SAR channel statistics fitted on **training rows only**, validation-based early stopping, checkpoint selection, and held-out test evaluation. Training was conducted from the complete aligned dataset; the seven modality combinations were not assigned different sample subsets.

**Important departure from proposal:** This first 40-day, multi-gauge temporal holdout is **not** the originally proposed event-level grouped cross-validation on 8–10 independent floods. It does not establish geographic, event-level or nationwide generalization. The promised event-level confidence intervals, paired statistical tests and effect sizes have **not yet been carried out**; tiny differences must not be represented as statistically meaningful.

**Seven held-out ablations** — errors in metres, **lower is better**:

| Config. | Modalities | Validation MAE | Test MAE | Test RMSE |
|---|---|---:|---:|---:|
| Persistence | Current stage unchanged | — | **0.2172** | 0.4066 |
| M1 | Satellite | 0.2265 | 0.2140 | 0.4036 |
| M2 | Weather | 0.2352 | 0.2201 | **0.3993** |
| M3 | Environment | 0.2293 | **0.2137** | 0.4009 |
| M4 | Satellite + weather | 0.2268 | 0.2157 | 0.4046 |
| M5 | Satellite + environment | **0.2259** | 0.2161 | 0.4060 |
| M6 | Weather + environment | 0.2331 | 0.2159 | 0.4010 |
| **M7** | **Satellite + weather + environment** | 0.2271 | **0.2152** | 0.4040 |

All seven experiments completed. M3 has the lowest *observed test* MAE, whereas M5 has the lowest *validation* MAE. We must not choose the “winning model” by repeatedly consulting test results. The main planned C4 contrast is **M7 versus M6**: M7 reduces test MAE by **0.000717 m (~0.72 mm)** but has **higher** RMSE (0.4040 versus 0.4010 m). M7 also improves test MAE relative to persistence by only **0.00204 m (~2 mm)**. These are **small, mixed differences**, not convincing evidence that all three modalities improve prediction.

### Why does training error exceed test error?

| Split | n | Persistence MAE | Median absolute daily stage change | Rises ≥0.5 m |
|---|---:|---:|---:|---:|
| Train | 809 | **0.6654 m** | **0.29 m** | 162 |
| Validation | 269 | 0.2349 m | 0.13 m | 17 |
| Test | 351 | 0.2172 m | 0.08 m | 24 |

The training interval contains substantially more large stage changes; **32 training changes exceed 3 m**, versus none in validation or test. Therefore a training MAE around 0.64–0.68 m alongside validation MAE around 0.23 m is at least partly explained by **different time-period difficulty**, not evidence of either successful generalization or overfitting on its own. The empirical cumulative distribution figure illustrates this: the blue training curve lies below the other curves at most thresholds.

### Operationally relevant warning-level diagnostics

Flood status was evaluated *after* numerical prediction using **each gauge's own source-reported alert, minor-flood and major-flood levels**. No universal height threshold was imposed and these thresholds were not used to train the regression models.

| Observed next-day class | Train | Validation | Test |
|---|---:|---:|---:|
| Below alert | 696 | 268 | **343** |
| Alert | 56 | 1 | **7** |
| Minor flood | 37 | 0 | **1** |
| Major flood | 20 | 0 | **0** |

On the test period there are only **eight observations at or above alert**. **M7 identifies 4/8 (50% recall)** and misses four, with three false alerts. Persistence identifies **5/8 (62.5% recall)** and misses three, also with three false alerts. M7's MAE on the **seven observed alert-category records** is **~0.711 m**, versus ~0.205 m for the 343 normal records; its sole minor-flood record has **~1.250 m absolute error**. The alert-category sample is tiny, and there are **no major-flood cases** in the held-out set, so a claim of robust flood-warning ability is **not justified**.

Per-station QA finds that **M7 beats persistence on test MAE at 26 of 39 stations** (each station has only nine held-out days); M3 beats it at 29, M6 at 28. Station-level signs of improvement do not necessarily imply a practically significant gain. Some spatial QA warnings and the short evaluation period also restrict interpretation.

## 5. Current conclusion, limitations and next steps

**What PP1 demonstrates:** a real-data, quality-screened and cached three-modality dataset; independent learned encoders; the seven model ablations; reproducible chronological evaluation; and transparent performance comparison against persistence and gauge-specific river-warning levels. The full C4 architecture and ablation training are working in the revised **24-hour stage-regression** experiment.

**What PP1 does not demonstrate:** superior full fusion, true implemented cross-modality attention, prospective 09:00 data availability, robust prediction of rising or major-flood cases, or transportability to different historic events and sites. The training/test distribution shift, coarse reanalysis inputs, stale SAR scenes and candidate-catchment uncertainty are research constraints to address—not reasons to discard the complete initial benchmark.

**PP2 research plan:** obtain approved, longer high-frequency water-level time series and verified catchment metadata; extend to independent events/basins; repeat chronological and event-grouped evaluation with confidence intervals and paired effect sizes; examine rapid-rise errors and alert misses; and test a *separately named* learned modality-attention variant against the existing concatenation baseline. Once hourly or finer stage labels are truly available, define **6 h, 12 h, 18 h, 24 h, 48 h and 72 h** forecasts with correctly time-aligned input availability and changing lead-time evaluation. Current once-daily 09:00 reports **cannot supervise genuine 6 h/12 h/18 h next-stage outcomes**; 48 h and 72 h could be constructed from daily gauge labels if sufficient continuous observations exist, but they have not been trained in this PP1 study.

**Research positioning:** The contribution is a controlled test of whether multimodal **breadth** adds value to independent **depth** branches in Sri Lankan gauge-based prediction. The current test **does not confirm that it does**. That honest negative or inconclusive outcome is informative and supports the next scientific experiment. The original goal of testing modality contribution is preserved; the target, some source representations, validation scope and attention mechanism are not yet the same as those in the initial proposal.

**Data governance:** Numerical DMC-derived labels, exact per-station warning thresholds, private point-coordinate registries and station-level alert predictions remain in ignored private folders. The research report can show aggregate performance and documented methodology while repository-publication/redistribution permissions are checked.
