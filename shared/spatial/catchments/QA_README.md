# FloodRisk Sri Lanka — Catchment Delineation & Quality Assurance (40 Gauges)

**Research group:** J26-DS-314  
**Document type:** Shared spatial-data QA README  
**Scope:** Seven Kelani Ganga gauges and 33 additional gauges across 18 other basin groups  
**QA stage:** **Research candidate delineations — not officially certified catchment boundaries**  
**Record date:** 17 October 2026  
**Coordinate reference:** WGS 84 / geographic coordinates (EPSG:4326; GeoJSON `[longitude, latitude]`)

> **Important usage restriction:** The 40 watersheds may be used as **candidate upstream areas of interest (AOIs)** in research prototyping, subject to the station-specific flags below. Their agreement with a drainage-area number is **not** proof of exact boundary accuracy. Do not relabel these products as *verified* without additional QA evidence and approval.

## 1. Purpose and repository location

This README documents how the study generated upstream catchment polygons for the 40 research gauges, what the numeric and technical checks establish, and which boundaries still need examination. This is a shared geographic definition for C1–C4, **not** a claim that any gauge-specific watershed has been confirmed by an official basin-polygon source.

**Canonical location of this document:** `shared/spatial/catchments/QA_README.md`.

Expected repository layout:

```text
shared/spatial/catchments/
├── QA_README.md                    # this cross-basin assessment
├── kelani/
│   ├── polygons/                   # 7 station-level candidate polygons
│   ├── bounding_boxes/             # 7 corresponding bbox geometries
│   ├── masks/                      # 7 native-grid masks (ignored by Git)
│   └── kelani-specific metadata (as available)
├── kalu_ganga/, gin_ganga/, ...   # one folder per other basin
│   ├── polygons/
│   ├── bounding_boxes/
│   ├── masks/                      # GeoTIFF files shared out of band
│   └── metadata/
├── additional_33_catchment_QA_registry.csv
└── additional_33_catchment_file_manifest.csv
```

The seven Kelani results were originally generated under `results/catchments/kelani_merit_hybas12_pilot/` and published into the shared Kelani folder by the Kelani export cell. The other 33 are written directly to their basin folders by the Additional-33 notebook. The ignored `results/catchments/` folder contains intermediate downloads, HTML maps and team ZIPs. **Do not stage original MERIT rasters, temporary tile downloads, GeoTIFF masks or private third-party river-reference exports in Git.**

## 2. Inputs and delineation method — short explanation

1. **Locate the gauge.** Retain the original public Irrigation Department dashboard point separately; never modify the recorded station coordinate.
2. **Find a MERIT outlet cell.** Query nearby cells of MERIT Hydro v1.0.1 (`dir`, `upa`), normally within 350 m for the additional-33 workflow and 900 m only as fallback. The selected **computational outlet** is a grid cell used to trace the river network; it is *not* a surveyed replacement gauge location. The outlet ranking favors short distance *and* similarity to a published drainage area. Candidates are not independently validated by that same area match.
3. **Trace upstream cells.** Reverse-traverse MERIT's eight-direction (D8) flow pointers from the selected outlet. Each cell that drains to the outlet joins the catchment mask. MERIT Hydro's native sampling is 3 arc-seconds (about 90 m at the equator), not a traced high-resolution riverbank.
4. **Create reusable spatial outputs.** Polygonize the upstream raster mask to one irregular GeoJSON catchment, keep the raster mask, and generate a bounding rectangle. All three describe a *modelled upstream contributing area*, not an observed flood-inundation footprint.
5. **Screen the results.** Test flow-pointer logic, ensure the mask does not touch the raster export edge, compare traced area with MERIT `upa`, compare with the published gauge drainage area, and inspect local channel/confluence relationships on the QA maps. HydroBASINS L12 is a structural reference; `UP_AREA` belongs to its own network position and is not the exact gauge-area ground truth.

**Numerical measure:** `100 × abs(derived_catchment_km² − published_gauge_km²) / published_gauge_km²`.

### What the available checks establish

| Check | Reported result | What it establishes / does not establish |
|---|---|---|
| Station coverage | 7 Kelani + 33 additional = **40** candidate polygon/mask/bbox sets | Workflow completed for 40 selected gauges; not 40 certified basins |
| Raster-edge coverage | **0/40** traced watersheds reached their export-window boundary | No detected rectangular clipping; does not guarantee correct downstream routing |
| Internal D8 versus MERIT `upa` | Approximately **0.01%** difference for every station | Reverse-tracing correctly reproduces MERIT's own drainage areas; **not an independent accuracy estimate** |
| Area comparison with published reference | **35/40** stations within **10%** | For 35 stations the **total modelled area** is within ±10% of a reference number; not a claim that 90% of each polygon boundary is correct |
| Five area outliers | **5/40** exceed 10% | Higher priority for investigating alternate stream/cell positions or uncertainty in source reference |
| Official reference catchment polygons | **Not provided** | No direct polygon-overlap / intersection-over-union validation can be claimed |
| Visual QA | Candidate-map inspection, not a signed survey | Useful plausibility screen, not certification |

*These are the values reported from the two completed notebook runs. They should be reproduced from each notebook's registry after any candidate outlet or geometry is changed.*

## 3. Complete 40-gauge QA register

Columns: derived and published drainage areas in km²; **difference** is the absolute percentage mismatch; **outlet distance** is from the original ArcGIS point to the selected MERIT raster-cell center. A status of *Candidate / visual QA pending* **does not mean passed/approved**. The numeric and source flags reported by the notebook are retained verbatim in the machine-readable registry files.

| Station | Basin | MERIT-derived km² | Published km² | Area difference | Outlet distance (m) | Recorded QA status |
|---|---|---:|---:|---:|---:|---|
| Nagalagam Street | Kelani Ganga | 2,272.76 | 2,085 | 9.01% | 58.47 | Candidate / visual QA pending |
| Hanwella | Kelani Ganga | 1,824.35 | 1,782 | 2.38% | 40.32 | Candidate / visual QA pending |
| Glencourse | Kelani Ganga | 1,522.21 | 1,463 | 4.05% | 58.08 | Candidate / visual QA pending |
| Kithulgala | Kelani Ganga | 426.59 | 383 | 11.38% | 45.37 | Candidate / visual QA pending |
| Holombuwa | Kelani Ganga | 153.66 | 155 | 0.87% | 39.86 | Candidate / visual QA pending |
| Deraniyagala | Kelani Ganga | 180.03 | 183 | 1.62% | 12.21 | Candidate / visual QA pending |
| Norwood | Kelani Ganga | 95.24 | 97 | 1.81% | 43.48 | Candidate / visual QA pending |
| Putupaula | Kalu Ganga | 2,616.83 | 2,598 | 0.72% | 76.71 | Candidate / visual QA pending |
| Ellagawa | Kalu Ganga | 1,399.05 | 1,393 | 0.43% | 80.18 | Candidate / visual QA pending |
| Rathnapura | Kalu Ganga | 633.67 | 603 | 5.09% | 31.31 | Candidate / visual QA pending |
| Kalawellawa (Millakanda) | Kalu Ganga | 772.85 | 780 | 0.92% | 57.59 | Candidate / visual QA pending |
| Magura | Kalu Ganga | 160.91 | 152 | 5.86% | 50.52 | Candidate / visual QA pending |
| Baddegama | Gin Ganga | 721.55 | 749 | 3.66% | 58.04 | Candidate / visual QA pending |
| Thawalama | Gin Ganga | 358.51 | 377 | 4.90% | 69.22 | Candidate / visual QA pending |
| Panadugama | Nilwala Ganga | 434.17 | 445 | 2.43% | 17.94 | Candidate / visual QA pending |
| Pitabeddara | Nilwala Ganga | 288.13 | 310 | 7.05% | 47.03 | Candidate / visual QA pending |
| Urawa | Nilwala Ganga | 75.99 | 59 | 28.79% | 54.57 | Review: official drainage difference gt20 |
| Thalgahagoda | Nilwala Ganga | 834.14 | 852 | 2.10% | 334.23 | Candidate / visual QA pending |
| Moraketiya | Walawe Ganga | 1,581.63 | 1,542 | 2.57% | 59.94 | Candidate / visual QA pending |
| Thanamalwila | Kirindi Oya | 697.11 | 749 | 6.93% | 47.75 | Candidate / visual QA pending |
| Wellawaya | Kirindi Oya | 188.38 | 172 | 9.53% | 93.00 | Candidate / visual QA pending |
| Kuda Oya | Kirindi Oya | 300.10 | 291 | 3.13% | 28.09 | Candidate / visual QA pending |
| Katharagama | Menik Ganga | 796.37 | 787 | 1.19% | 41.11 | Candidate / visual QA pending |
| Nakkala | Kumbukkan Oya | 219.90 | 216 | 1.81% | 59.50 | Candidate / visual QA pending |
| Siyambalanduwa | Heda Oya | 289.41 | 295 | 1.89% | 46.12 | Candidate / visual QA pending |
| Padiyathalawa | Maduru Oya | 167.83 | 159 | 5.56% | 48.63 | Candidate / visual QA pending |
| Manampitiya | Mahaweli Ganga | 7,668.56 | 7,418 | 3.38% | 84.80 | Candidate / visual QA pending |
| Weraganthota | Mahaweli Ganga | 4,035.41 | 4,092 | 1.38% | 100.65 | Candidate / visual QA pending |
| Peradeniya | Mahaweli Ganga | 1,140.86 | 1,168 | 2.32% | 38.16 | Candidate / visual QA pending |
| Nawalapitiya | Mahaweli Ganga | 143.36 | 176 | 18.55% | 60.88 | Candidate / visual QA pending |
| Thaldena | Mahaweli Ganga | 281.19 | 276 | 1.88% | 64.23 | Candidate / visual QA pending |
| Horowpothana | Yan Oya | 976.92 | 720 | 35.68% | 36.70 | Review: official drainage difference gt20 |
| Yaka Wewa | Ma Oya | 120.52 | 121 | 0.40% | 352.08 | Review: distant outlet |
| Thanthirimale | Malwathu Oya | 2,114.57 | 2,116 | 0.07% | 30.34 | Candidate / visual QA pending |
| Galgamuwa | Mee Oya | 313.15 | 299 | 4.73% | 72.60 | Candidate / visual QA pending |
| Moragaswewa | Deduru Oya | 2,018.04 | 1,708 | 18.15% | 9.30 | Review: source coordinate discrepancy |
| Badalgama | Maha Oya | 1,329.05 | 1,360 | 2.28% | 106.05 | Candidate / visual QA pending |
| Giriulla | Maha Oya | 1,161.27 | 1,191 | 2.50% | 82.07 | Candidate / visual QA pending |
| Dunamale | Aththanagalu Oya | 159.47 | 153 | 4.23% | 29.15 | Candidate / visual QA pending |
| Thambuththegama | Kala Oya | 1,189.74 | 1,186 | 0.32% | 53.43 | Candidate / visual QA pending |

## 4. Stations requiring focused QA

### Five drainage-area mismatches greater than 10%

| Priority | Station | Difference | Practical question to resolve |
|---|---|---:|---|
| High | **Horowpothana (Yan Oya)** | **35.68%** (977 km² vs 720 km²) | Is the outlet below a tributary junction adding extra drainage, or do MERIT and the published reference define different upstream areas? |
| High | **Urawa (Urubokka Ganga / Nilwala basin)** | **28.79%** (76 km² vs 59 km²) | Is the candidate on Urubokka Ganga at the correct branch, rather than after a confluence? |
| Moderate | **Nawalapitiya (Mahaweli Ganga)** | **18.55%** (143 km² vs 176 km²) | Is a downstream channel cell more consistent with station placement, or is the published area defined differently? |
| Moderate | **Moragaswewa (Deduru Oya)** | **18.15%** (2018 km² vs 1708 km²) | Compare the **precise GeoJSON geometry** with its rounded/disagreeing coordinate properties and look for confluences near the site. Keep both source coordinates recorded until resolved. |
| Moderate | **Kithulgala (Kelani Ganga)** | **11.38%** (427 km² vs 383 km²) | Does the selected cell include an adjacent tributary that the station's published contributing area does not? |

### Additional spatial/hydrological cautions even when drainage areas are close

- **Yaka Wewa (Ma Oya / Mukunu Oya)** — MERIT outlet is **352.08 m** from the original gauge although areas differ only **0.40%**. The original gauge was **not moved**. Verify the snapped cell follows the intended channel and lies on the appropriate side of junctions; the matching area alone is insufficient.
- **Thalgahagoda (Nilwala Ganga)** — outlet offset is **334.23 m**, despite a **2.10%** area mismatch and ordinary automatic candidate status. Large offsets warrant checking even if within the nominal 350 m candidate radius.
- **Nagalagam Street (Kelani Ganga)** — **9.01%** area mismatch near the lower river. Review lower-reach connectivity; tides and downstream/backwater effects may influence stage predictions even when the upstream catchment polygon is plausible.
- **Moragaswewa coordinate discrepancy** — this is **independent of** its area disagreement: the GeoJSON feature's point geometry was used rather than the rounded properties. Do not silently replace the geometry with a rounded value.

## 5. A manageable manual QA protocol (no official watershed polygons required)

Manual review **cannot establish exact boundary correctness** without surveyed/official catchments. It can, however, detect obvious errors and document *provisional acceptance for feature extraction*. For each of the five area outliers, and separately Yaka Wewa and Thalgahagoda:

1. **Open the station's basin QA map.** Toggle Esri Topographic (context), Esri Satellite (visible channels), the cyan MERIT drainage network, and its coloured catchment. The red original gauge marker is the public-source location; the yellow marker is the computational MERIT outlet. Zoom to the yellow point and inspect nearby tributary junctions.
2. **Check same-stream identity.** Follow the labelled/visible river in both directions. The yellow cell should be on the river/tributary named in the source metadata. A modelled centreline offset by one or several 90 m pixels is not itself disqualifying; an outlet across a confluence or on a different tributary is concerning.
3. **Compare the top-ranked alternate candidate cells, not arbitrary spots.** Open the notebook-generated `outlet_candidate_rankings.csv` (for Additional-33: `results/catchments/additional_33_merit_hybas12/`; for Kelani: corresponding pilot results folder). Prefer candidates on the **correct channel** over artificially forcing a perfect drainage-area match. A nearby alternative with a *different* upstream network may warrant rerunning **only that station's watershed**, with a documented manual pixel override; do not silently alter its original gauge coordinate.
4. **Inspect the resulting polygon at basin scale.** Look for obvious cross-divide leakage, missing upstream branches, a polygon disconnected from its yellow outlet, or an implausible abrupt gain/loss of a tributary. Nested polygons along the same downstream river are **normal**, and uncovered areas between the 40 selected gauges are **not errors**.
5. **Record a decision with evidence.** If the outlet/channel and shape are plausible but the published area remains different, retain the original candidate and record `PROVISIONAL_USE_WITH_LIMITATION`. If river identity, confluence position or coordinates remain ambiguous, record `HOLD_FOR_REVIEW` or `SENSITIVITY_ONLY`. Upgrade to `QA_VISUALLY_REVIEWED` only after actually completing documented checks; **never** rename it `OFFICIALLY_VERIFIED` on this evidence.

**Minimal decision log (copy one line per station into a project QA note/CSV):**

```text
station | reviewer | date | map/file checked | correct tributary? |
confluence issue? | polygon plausible? | alternative candidate checked? |
decision (PROVISIONAL_USE_WITH_LIMITATION / HOLD_FOR_REVIEW / SENSITIVITY_ONLY) |
evidence/limitation | changed outlet pixel? | new artifact/version/hash
```

**Fast PP1 review order:** (1) Horowpothana, (2) Urawa, (3) Yaka Wewa, (4) Moragaswewa, (5) Nawalapitiya, (6) Thalgahagoda, (7) Kithulgala, (8) Nagalagam Street. A short documented review of other gauges can follow. This is a *prioritization*, not a statement that the others are certified.

### Example decisions (guidance, not decisions already made)

- **Better candidate:** selected outlet clearly falls on the named channel, near the mapped station, and its upstream polygon follows the expected drainage/tributary network. The area discrepancy is documented as a limitation if still present.
- **Not acceptable without revision:** outlet lies on a neighbouring branch, a major tributary is included/excluded because the outlet is on the wrong side of a confluence, or the polygons are evidently in another basin. Hold the station, inspect alternate ranked cells, and regenerate only after recording the justification.
- **Uncertain but testable:** different plausible outlet cells yield modestly different polygons. Keep the original version and test sensitivity of C2/C3/C4 catchment summaries; do not pick an outlet solely because it produces a nicer percentage.

## 6. Known limitations and implications for predictive modelling

MERIT Hydro v1.0.1 is a global **3-arc-second (~90 m)** D8 flow-direction model. Its authors document limitations in river/channel bifurcations, waterbody/coastline representation, river–lake distinctions, below-sea-level terrain, and water width in braided channels. A D8 cell has only **one downstream destination**, which simplifies split channels and floodplain routes. The dataset's published GRDC benchmark is general external evidence for the product, **not** a station-specific accuracy guarantee for these Sri Lankan gauges.

| Limitation | Potential effect on these catchments and C1–C4 |
|---|---|
| ~90 m cells and misregistered river centreline | Physical gauge and raster-cell centre can differ; small tributaries and narrow floodplains can be simplified. May change satellite spatial aggregation and environmental summaries. |
| Junction/branch or bifurcation errors | A candidate outlet on the wrong side of a confluence can include/exclude substantial upstream area; rainfall, soil moisture and land-cover features could be biased. |
| Coastlines, tidal lower reaches, river/lake separation | Outlet topology and lowland catchments can be less reliable; tides/backwater may affect stage even when an upstream rainfall catchment is correctly traced. |
| Lakes, reservoirs, and regulation not fully represented in topology | Catchment geometry alone cannot model reservoir releases or human flow controls; models require appropriate temporal signals and acknowledge omissions. |
| Published area mismatch or source-coordinate disagreement | Numerical agreement does not demonstrate a boundary match; the research must preserve source versions, QA flags and possible alternative AOIs. |

**Modality guidance:**

- **C1 satellite:** Extract radar features within a clearly defined AOI or a **separately defined gauge-centred image patch**. A watershed boundary is a contextual spatial area, **not a pixel-level flood label**.
- **C2 weather:** Aggregate causally available CHIRPS/ERA5 over the modelled upstream catchment (optionally compare with small buffer/alternate outlet for flagged cases). Prevent forecasts using weather products released *after* the prediction cutoff.
- **C3 environmental:** Terrain, cover and static basin descriptors can use the polygon, but report area/location uncertainties and avoid station-ID leakage in splits.
- **C4 fusion:** Independently preprocess the three modalities using these shared candidate geographical definitions. Do not import C1–C3 learned weights, predictions or engineered modality features where component independence is required. Study robustness to AOI uncertainty separately from model training.
- **Targets:** Ground-truth future gauge water levels remain linked to the *original physical station*, never to an artificially relocated MERIT outlet. Confirm timestamp alignment, units and the forecasting horizon.

## 7. Reproducibility, changes and publication boundaries

- **Source datasets:** `MERIT/Hydro/v1_0_1` and `WWF/HydroSHEDS/v1/Basins/hybas_12` in Earth Engine; original station records from the Sri Lanka Irrigation Department public ArcGIS dashboard; published gauge drainage areas from the Hydrological Annual reference used in the notebooks. The annual publication edition/page should be linked separately when available; the notebook inputs alone do not provide a complete bibliographic record.
- **Scripts:** The seven-gauge Kelani notebook and Additional-33 notebook provide source registry, raster extraction, outlet ranking, D8 tracing, polygons, masks, bbox generation and QA. Their selected outlet coordinates are **versioned computational choices**.
- **Checksums:** Additional-33 Section H writes `additional_33_catchment_file_manifest.csv` containing SHA256 values. Maintain the Kelani manifest separately. Any manual outlet override that changes a polygon must regenerate the corresponding mask, bbox, registry row and hashes and receive a versioned review note.
- **Git:** Commit this README, small candidate polygons, bounding boxes, metadata and notebooks on a reviewed PR. Native-grid TIFF masks and data caches remain ignored and are exchanged separately by authorized group members. Never commit local credentials or raw/private third-party hydrography exports.
- **Licensing caution:** Publicly viewable coordinates are not automatically unrestricted for all redistribution. Check the ArcGIS item terms before republishing raw data. MERIT Hydro is offered with CC BY-NC 4.0 / ODbL 1.0 options and obligations for derived data; keep source attribution, do not assume a blanket repository license applies to third-party-derived geometry, and do not redistribute the complete original MERIT raster. Confirm the chosen license and reuse terms before broad commercial/public downstream distribution.

### References

1. Yamazaki, D., Ikeshima, D., Sosa, J., Bates, P. D., Allen, G. H., & Pavelsky, T. M. (2019). *MERIT Hydro: A High-Resolution Global Hydrography Map Based on Latest Topography Datasets*. Water Resources Research, 55, 5053–5073. DOI: https://doi.org/10.1029/2019WR024873. Earth Engine: https://developers.google.com/earth-engine/datasets/catalog/MERIT_Hydro_v1_0_1
2. Lehner, B., & Grill, G. (2013). *Global river hydrography and network routing: baseline data and new approaches to study the world's large river systems*. Hydrological Processes, 27, 2171–2186. DOI: https://doi.org/10.1002/hyp.9740. HydroBASINS: https://www.hydrosheds.org/products/hydrobasins
3. Sri Lanka Irrigation Department public real-time stations dashboard: https://slirrigation.maps.arcgis.com/apps/dashboards/2cffe83c9ff5497d97375498bdf3ff38#
4. Esri World Topographic Map item: https://www.arcgis.com/home/item.html?id=6e850093c837475e8c23d905ac43b7d0. Basemap waterway portrayal is for contextual QA, not site-specific survey-grade validation.

---

**Current disposition:** **40 candidate AOI sets technically generated. 35/40 areas match their published drainage-area figures within 10%. None is described here as independently/officially verified.** Five area-mismatch stations and additional outlet/location concerns are retained transparently for risk-managed modelling. This is sufficient for a **conditional research proof of concept**, not a claim of operational flood forecast readiness.
