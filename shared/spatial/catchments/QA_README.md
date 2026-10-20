# FloodRisk Sri Lanka — Catchment Delineation and Consolidated QA (40 Gauges)

**Research group:** J26-DS-314  
**Project:** FloodRisk Sri Lanka  
**Document:** Single shared catchment QA reference for C1–C4  
**QA record updated:** 20 October 2026 (incorporating the 16–17 October reviews and the C4 Notebook 01 follow-up)  
**Source spatial reference:** WGS 84, EPSG:4326; GeoJSON positions stored as `[longitude, latitude]`  
**Current disposition:** **40 candidate upstream catchments available for provisional research feature extraction; none independently or officially certified.**

> **Important:** The terms *geometry valid*, *outlet inside polygon*, *area close to reference*, and *visually plausible* refer to different checks. None alone proves an authoritative hydrological boundary. Existing machine-readable QA flags remain unchanged. This report consolidates the evidence; it does not modify gauge coordinates, computational outlets, masks, polygons, or QA registries.

## 1. Purpose, scope and canonical location

This is the **one maintained QA document** for all 40 research catchments. It incorporates the previous cross-basin `QA_README.md`, the Kelani seven-gauge handover, the focused review of **Urawa, Horowpothana, Yaka Wewa, Moragaswewa and Kithulgala**, and the newer three-gauge original-point-containment review of **Ellagawa, Pitabeddara and Putupaula**. Historical source notes can be archived outside the canonical shared documentation once this consolidation has been reviewed and committed.

**Keep this file at:** `shared/spatial/catchments/QA_README.md`

```text
shared/spatial/catchments/
├── QA_README.md                            # ONE canonical narrative QA document
├── kelani/
│   ├── polygons/                           # seven candidate upstream GeoJSONs
│   ├── bounding_boxes/                     # seven bbox GeoJSONs
│   └── masks/                              # raster masks, normally git-ignored
├── kalu_ganga/, nilwala_ganga/, ...        # remaining 18 basin folders
│   ├── polygons/
│   ├── bounding_boxes/
│   ├── masks/
│   └── metadata/                           # where generated
├── additional_33_catchment_QA_registry.csv
└── additional_33_catchment_file_manifest.csv
```

The candidate polygons are **land contributing upstream of a modelled outlet**. The masks represent contributing MERIT raster cells, and each rectangular bounding box is a convenient retrieval extent—not the watershed itself. **None is a flood-inundation label.** The same agreed candidate geographies support the four components while their feature extraction and models remain independent.

## 2. How we delineated these areas

1. **Retain original gauge locations.** We obtained public dashboard coordinates in EPSG:4326. The physical gauge position was **not moved** during watershed tracing.
2. **Select a computational outlet.** We evaluated nearby MERIT Hydro v1.0.1 stream cells using proximity and agreement with the published gauge drainage area. Additional-33 generally used a 350 m initial search with an extended fallback when needed. The resulting outlet is a routing-grid location, **not** a corrected/surveyed gauge coordinate.
3. **Reverse-trace D8 flow.** Starting at that outlet, the code traversed all upstream MERIT `dir` flow connections, collecting land cells that drain to the outlet. Its native grid is approximately 3 arc-seconds (~90 m near the equator). `upa` provides MERIT's modelled upstream drainage area.
4. **Create the three spatial products.** The collected-cell mask became a GeoTIFF; polygonization produced an irregular upstream catchment GeoJSON; its enclosing rectangle became the GeoJSON bounding box.
5. **Run technical and geographical QA.** We checked geometry validity, outlets, internal area consistency, extraction boundaries, published drainage-area comparison, and displayed river/channel relationships. We used **WWF HydroBASINS L12 only for supplementary comparison**, not to create, clip or replace the MERIT delineations.

**Area difference (%):** `100 × abs(MERIT-derived area − published drainage area) / published drainage area`. The published area also influenced outlet ranking; consequently even a small discrepancy is **not a fully independent validation of the chosen outlet**.

### What the recorded checks established

| Check | Reported result | Appropriate interpretation |
|---|---|---|
| Generated catchments | **7 Kelani + 33 additional = 40** polygon/mask/bbox sets | Files were generated; this is not independent certification |
| Raster-edge flag | **0/40** with `touches_edge=True` | No detected export-window truncation; channel routing may still be wrong |
| Derived area versus MERIT `upa` | Approximately **0.01%** difference in recorded runs | Reverse-D8 tracing is internally consistent with MERIT; not real-world accuracy |
| Derived area versus published drainage area | **35/40 within 10%**, **5/40 greater than 10%** | Similar *total area*, not proof of the same boundary or tributaries |
| Focused post-export geometry checks | **5/5** flagged polygons valid and covering computational outlets | Technical saved-geometry checks, not external hydrological verification |
| C4 Notebook 01 spatial checks | **39/39 labelled stations** linked to distinct valid existing polygons; their computational outlets checked when supplied | Suitable for building a provenance-aware model index; three original gauges fall outside |
| Original gauge inside its polygon | **36/39** labelled gauges: yes; **3/39**: no | Diagnostic only; real gauge and selected MERIT cell can differ |
| Official upstream polygon ground truth | **Not supplied** | No defensible exact-boundary overlap, IoU or site-level certification result |

The **40th geographic gauge, Thambuththegama**, has a generated polygon but is absent from the current DMC numerical labels; it was **not part of the 39-gauge C4 Notebook 01 containment check**. The five-gauge-focused report also recorded a subsequent all-40 saved-file geometry/containment check, which should be understood as an **internal file consistency test**, not ground truth.

### Kelani-specific processing notes

The seven Kelani gauges were originally processed using a native-grid export of **1,873 columns × 1,321 rows**; the chunks were checked and reassembled before tracing. The seven catchments all avoided the export boundary. Six of seven have drainage-area difference under 10%; Kithulgala is the exception (11.38%). The Kelani computational outlet locations each intersect one HydroBASINS L12 sub-basin, but an L12 `UP_AREA` does **not** refer necessarily to the precise gauge outlet.

## 3. Full 40-gauge numerical QA register

These are the **recorded notebook results**, transcribed at their reported precision; no geometries were recalculated to create this document. The `Recorded QA status` text is the human-readable presentation from the previous cross-basin register; **the original machine-readable flag values stay authoritative** in each registry and C4 sample manifest.

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

**Automatic QA meaning:** Ordinary candidate status does not mean `VERIFIED`; the two high drainage-area flags remain at Urawa and Horowpothana, the source-coordinate flag at Moragaswewa, and the distant-outlet flag at Yaka Wewa. Kithulgala's 11.38% discrepancy is a **manual priority finding** even though its original automatic status remained an ordinary pending candidate.

## 4. Focused manual QA — five existing priority gauges (17 October review)

We examined saved catchment overlays at local and regional scales with Esri Topographic/Satellite and the MERIT modelled drainage layer (`upa ≥ 5 km²` for the displayed network). We compared metadata and, for these five, tested the original **unsimplified GeoJSON** with Shapely for validity and outlet containment. These tests passed:

```text
Urawa            valid_polygon=True outlet_in_polygon=True
Horowpothana     valid_polygon=True outlet_in_polygon=True
Yaka Wewa        valid_polygon=True outlet_in_polygon=True
Moragaswewa      valid_polygon=True outlet_in_polygon=True
Kithulgala       valid_polygon=True outlet_in_polygon=True
```

| Station | Reason for focused review | Numerical evidence | Visual/file QA observation | Decision / remaining limit |
|---|---|---|---|---|
| **Urawa** (Nilwala Ganga / Urubokka Ganga) | Area discrepancy **28.79%** | **75.987 km²** derived versus **59 km²** published; outlet offset **54.57 m** | Original gauge near the selected modelled drainage corridor; outlet within the saved polygon; no obvious far-away-branch jump in inspected view | **High uncertainty; provisional exploratory use**. Tributary contribution and reference-area discrepancy unresolved; retain `REVIEW_OFFICIAL_DRAINAGE_DIFFERENCE_GT20` |
| **Horowpothana** (Yan Oya) | Area discrepancy **35.68%** | **976.921 km²** versus **720 km²**; outlet offset **36.70 m** | Both points near modelled river close to road crossing; regional polygon appears connected | **High uncertainty; provisional exploratory use**. Possible connectivity/confluence/reference-area issue remains; retain `REVIEW_OFFICIAL_DRAINAGE_DIFFERENCE_GT20` |
| **Yaka Wewa** (Ma Oya / Mukunu Oya) | Distant computational outlet | **120.521 km²** versus **121 km²** (**0.40%**); outlet offset **352.08 m**, 900 m fallback radius used | Selected point lies on the modelled stream; computational outlet covered by saved polygon; displayed river position differs from the original marker | **Conditional exploratory use**. Area agreement does not verify absence of a confluence; retain `REVIEW_DISTANT_OUTLET` |
| **Moragaswewa** (Deduru Oya) | Conflicting point geometry versus coordinate attributes; area mismatch | **2,018.038 km²** versus **1,708 km²** (**18.15%**); outlet offset **9.30 m** | The stored ArcGIS **geometry** point and modelled outlet almost overlap on the depicted stream; polygon valid | **High uncertainty; exploratory use**. Do not guess a new gauge coordinate; source properties conflict remains; retain `REVIEW_SOURCE_COORDINATE_DISCREPANCY` |
| **Kithulgala** (Kelani Ganga) | Moderate area mismatch | **426.59 km²** versus **383 km²** (**11.38%**); outlet offset **45.37 m** | Points close to same MERIT drainage corridor; eight nearby ranked candidates each had ~426.26–426.67 km² upstream area | **Moderate uncertainty; retain current candidate**. Moving among those similar cells would not fix the mismatch; retain `CANDIDATE_PENDING_VISUAL_QA` |

**Interpretation:** Each polygon being valid and covering its computational outlet is necessary for technical use, **not proof the modelled drainage system matches the real gauge's contributing land**. Published reference area describes a total number, not an authoritative polygon. The five reviews documented *plausibility and risks*, not independent clearance.

### Additional cautions not cleared by the five-gauge review

- **Nawalapitiya (Mahaweli Ganga):** **18.55%** lower modelled area than published reference. It is one of the five 10%-area outliers, but was not one of the five focused reviews above.
- **Thalgahagoda (Nilwala Ganga):** **334.23 m** original-to-outlet shift, despite only **2.10%** area difference. Check connectivity before making strong local hydrology claims.
- **Nagalagam Street (Kelani Ganga):** **9.01%** difference, with lower-river/coastal/tidal and human-control limitations. Good area agreement would not establish tidal stage dynamics.

## 5. Follow-up QA — three original gauges just outside their polygons (20 October review)

C4 Notebook 01 Cell 6 found **3/39 labelled original gauge coordinates outside** their corresponding *unsimplified* modelled catchment polygons, even though all 39 polygons were distinct, geometrically valid candidates and the computational outlets were contained where coordinates were available. We computed the **geodesic distance from the original gauge to the nearest point on the actual polygon**, then visually screened the locations using close-up Esri Satellite imagery. The selected computational outlets and original gauge locations were **not changed**.

| Gauge | Basin | Distance **outside** polygon | Gauge → MERIT outlet displacement | Observed in close-up view | Disposition |
|---|---|---:|---:|---|---|
| **Ellagawa** | Kalu Ganga | **33.6 m** | **80.18 m** | Original marker and outlet are on/near **opposite banks of the same visible river reach**; the polygon's gridded edge runs nearby | Retain provisional AOI; record cross-bank/geometric offset |
| **Pitabeddara** | Nilwala Ganga | **0.4 m** | **47.03 m** | Original marker is effectively at the polygon boundary; outlet lies at the nearby mapped watercourse | Retain provisional AOI; tiny boundary discrepancy |
| **Putupaula** | Kalu Ganga | **28.7 m** | **76.71 m** | Both markers lie near the same broad visible river corridor; the candidate boundary does not precisely track a riverbank | Retain provisional AOI; record local bank/grid offset |

All measured distances **outside** the polygons are smaller than MERIT's approximately **90 m nominal grid-cell size**. That makes raster alignment and riverbank-to-grid offsets a **plausible explanation**, but **not a proven error tolerance, accurate-channel test or certification**. A computational outlet can be geographically nearby yet still be on the wrong side of an important confluence. The satellite tiles and simplified map outlines are visual context, not surveyed geospatial ground truth.

**Decision:** Retain `CANDIDATE_PENDING_VISUAL_QA` for all three. Add `original_gauge_inside_polygon=False` (and the recorded outside distance) as a *separate diagnostic* in C4's QA manifest. Do **not** change the source points, computed outlet cells, masks, polygons or earlier status flags merely to put the red markers inside the outlines.

**Status of C4 Notebook 01:** Cell 6 **passed**. The recorded results justify proceeding with **Cells 7–9** without rerunning MERIT tracing or regenerating spatial outputs. It remains appropriate to flag these three for future channel/confluence review if independent evidence appears.

## 6. How all QA evidence affects C1–C4

| Use | Rule |
|---|---|
| **Shared geography** | Preserve one exact version/hash of each selected candidate upstream polygon, its original gauge coordinate, computational outlet and machine-readable QA status across the four components |
| **C1 Satellite** | Build its own SAR/soil wetness processing; gauge-centred SAR squares and polygon-based upstream variables serve different roles |
| **C2 Weather** | Aggregate weather over the **candidate upstream polygon**, respecting each product's native grid and time-window definitions |
| **C3 Environment** | Compute terrain, land-cover, river density and water-history descriptors from polygons while tracking source and spatial uncertainty |
| **C4 Multimodal fusion** | Independently extract raw-source satellite, weather and environmental inputs. Common station IDs/catchments/target/splits are allowed; C1–C3 trained embeddings or engineered features are not reused |
| **Targets and station location** | Water-level labels stay tied to the **original physical gauge record**, never to the computational outlet or an invented alternative coordinate |
| **Model claims** | These candidates support PP1 **feature-extraction feasibility**, not scientifically validated watershed boundaries or operational forecast skill |

Use the provisional AOIs for the **retrospective proof of concept** while keeping the uncertainty metadata. If time and data permit, conduct a sensitivity analysis excluding high-uncertainty gauges (notably Urawa, Horowpothana, Moragaswewa, and optionally other warned outlets), reusing the fixed temporal split. Do not interpret model accuracy as a measurement of polygon accuracy.

## 7. Remaining limitations and future QA gates

1. **Outlet versus physical gauge:** MERIT's roughly 90 m grid can shift a point across a bank or tributary. An outlet on the wrong side of a confluence may change the contributing area substantially even at a small displacement.
2. **MERIT modelling:** D8 assumes one downstream direction per cell; bifurcations, coastal areas, reservoirs and modified drainage may not be faithfully represented. The blue/cyan `upa` threshold layer shows a selected network, not every natural stream nor the whole watershed.
3. **HydroBASINS context:** Level-12 polygon intersection and `UP_AREA` are supplementary data with different reference outlets. Do not treat their boundaries as the true polygon for these 40 gauges.
4. **No official station watershed polygons:** Exact boundary-overlap metrics cannot be estimated. The published area agreement was partly encouraged by candidate ranking. Neither a neat-looking satellite overlay nor a close area match proves correct channel identity.
5. **Model transfer:** A small catchment can occupy a fraction of a coarse ERA5-Land/ASCAT grid cell. Spatial extraction must retain source resolution/coverage metadata. Watershed QA cannot cure temporal leakage or limited flood-event training data.
6. **Outstanding site-specific checks:** Nawalapitiya's area difference, Thalgahagoda's distance, Nagalagam Street's lower-river routing, and the noted high-uncertainty outlets should remain open until evidence supports a documented change.

**If stronger evidence identifies an actual outlet error:** inspect the ranked MERIT candidate cells on the correct named river, document why a different cell is chosen, regenerate **only the affected station's** mask/polygon/bbox and registry/checksum as a newly versioned product, and rerun any downstream modality features dependent on that changed AOI. Do **not** silently edit coordinates or retroactively change this version's claims.

## 8. Reproducibility, source attribution and sharing

- **Upstream hydrology:** MERIT Hydro v1.0.1 (Earth Engine `MERIT/Hydro/v1_0_1`, bands `dir`, `upa`). Kelani native processing window: 1,873 × 1,321 cells.
- **Hydrography reference:** WWF HydroBASINS L12 (Earth Engine `WWF/HydroSHEDS/v1/Basins/hybas_12`) used for contextual QA, not creating polygons.
- **Gauge points:** Sri Lanka Irrigation Department public real-time dashboard. Original geometry and derived MERIT outlet coordinates are documented separately.
- **Reference drainage areas:** Published Irrigation Department hydrological reference values represented in the notebook inputs; preserve edition/page metadata where available.
- **Local audit evidence:** Kelani seven-gauge notebook and its QA registry; Additional-33 notebook, `additional_33_catchment_QA_registry.csv`, the Additional-33 file manifest/checksums, five-station close-up screenshots and local Shapely results; C4 Notebook 01's 39-polygon QA table and three gauge-to-polygon distances.
- **Git repository:** Keep this narrative QA document, the small candidate GeoJSON outputs, source/configuration notebooks and small registries/manifests under appropriate project access rules. Keep temporary native MERIT tiles, large raster masks, local secrets and restricted/private source exports out of ordinary Git.
- **Redistribution:** Preserve MERIT/HydroBASINS and dashboard attribution; independently confirm derivative-geometry and gauge-data reuse permissions before broader public redistribution or commercial downstream use.

References:

1. Yamazaki, D., et al. (2019), *MERIT Hydro: A High-Resolution Global Hydrography Map Based on Latest Topography Datasets*, *Water Resources Research*. https://doi.org/10.1029/2019WR024873
2. Lehner, B., & Grill, G. (2013), *Global river hydrography and network routing*, *Hydrological Processes*. https://doi.org/10.1002/hyp.9740
3. [MERIT Hydro Earth Engine catalogue](https://developers.google.com/earth-engine/datasets/catalog/MERIT_Hydro_v1_0_1)
4. [HydroBASINS Level 12 Earth Engine catalogue](https://developers.google.com/earth-engine/datasets/catalog/WWF_HydroSHEDS_v1_Basins_hybas_12)
5. [Irrigation Department public dashboard](https://slirrigation.maps.arcgis.com/apps/dashboards/2cffe83c9ff5497d97375498bdf3ff38)

---

## 9. Final PP1 disposition

**We have 40 complete candidate AOI sets**, and the available computational checks found technically consistent products. **35/40** published drainage-area comparisons lie within **10%**, while five exceed it. Five previously flagged gauges have completed targeted local geometry and visual checks, and the later C4 Notebook 01 investigation found only **three** original gauge points slightly outside their polygons (**0.4–33.6 m**), all near depicted river corridors. No official station-specific upstream-polygon reference was available for certification.

**Decision: Proceed with C1–C4 preliminary feature extraction clearly marked candidate AOIs.** Retain the source and candidate outlet separately, preserve all QA flags, track these eight documented manual QA cases, and avoid describing any watershed as independently verified.