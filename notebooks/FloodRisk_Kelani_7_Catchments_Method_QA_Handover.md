# FloodRisk Sri Lanka — Kelani Ganga upstream catchments

**Project:** J26-DS-314  
**Stage:** PP1 catchment delineation and shared spatial-input preparation  
**Status:** **Seven generated CANDIDATE catchments; automatic checks completed; manual hydrological verification pending**  
**Evidence:** Outputs transcribed from the executed Kelani MERIT–HydroBASINS notebook and visual screenshots, 16 October 2026.

## 1. Purpose and scope

We delineate the land draining toward each of seven Kelani Ganga hydrometric gauges. The **irregular upstream catchment polygon** defines a hydrological area of interest (AOI); the **native-grid binary mask** selects contributing raster cells; and the **rectangular bounding box** defines the raster extraction extent, **not** an alternative watershed. These are inputs for the station-level forecasting research, **not flood-extent labels**.

The seven source gauge locations are taken from the Irrigation Department public ArcGIS dashboard GeoJSON. We preserve the original point coordinates and separately record the nearby computational outlet selected on the MERIT grid.

## 2. Reproducible procedure

1. **Source locations:** Read dashboard GeoJSON station identifiers, original WGS84 geometry coordinates, and tributary information. Keep original coordinates unchanged.
2. **Hydrology data:** Retrieve MERIT Hydro v1.0.1 `dir` (D8 routing) and `upa` (upstream area) on the *native grid*. We used an EPSG:4326 window of **1,873 × 1,321** cells. The raster chunks were downloaded and checked before assembly. `dir` contained the expected pointer codes and NoData.
3. **Outlet selection:** Search plausible nearby stream cells, retaining candidates and ranking by distance and agreement with published official drainage areas. This automatic ranking does **not** independently verify channel identity.
4. **Watershed tracing:** Starting at each selected outlet, follow the reverse MERIT D8 network to collect all contributing cells. Save a candidate native-grid mask, polygonized catchment and bounding rectangle. Check for truncation at the extraction-window edge.
5. **Hydrological QA:** Compare calculated geodesic polygon area with MERIT `upa` at the selected cell and independently published reference drainage area. Inspect river/tributary connectivity and nesting on imagery.
6. **Independent geographical context:** Intersect each outlet with WWF HydroBASINS Level 12 and record its `HYBAS_ID`, `SUB_AREA`, `UP_AREA`, `NEXT_DOWN`. Level-12 sub-basin `UP_AREA` is generally referenced to a different downstream location: **it is not an exact gauge-area target**.

**References:** [MERIT Hydro v1.0.1](https://developers.google.com/earth-engine/datasets/catalog/MERIT_Hydro_v1_0_1); [HydroBASINS Level 12](https://developers.google.com/earth-engine/datasets/catalog/WWF_HydroSHEDS_v1_Basins_hybas_12); [Irrigation Department water-level dashboard](https://slirrigation.maps.arcgis.com/apps/dashboards/2cffe83c9ff5497d97375498bdf3ff38#).

## 3. Results of the seven-gauge run

| Gauge | MERIT-derived upstream area (km²) | Published area (km²) | Absolute area difference | Outlet displacement | Touches export edge? | Assessment |
|---|---:|---:|---:|---:|---|---|
| Nagalagam Street | 2,272.76 | 2,085 | 9.01% | 58.47 m | No | Coastal/lowland routing review |
| Hanwella | 1,824.35 | 1,782 | 2.38% | 40.32 m | No | Plausible candidate |
| Glencourse | 1,522.21 | 1,463 | 4.05% | 58.08 m | No | Plausible candidate |
| Kithulgala | 426.59 | 383 | **11.38%** | 45.37 m | No | **Priority outlet/tributary review** |
| Holombuwa | 153.66 | 155 | 0.87% | 39.86 m | No | Plausible candidate |
| Deraniyagala | 180.03 | 183 | 1.62% | 12.21 m | No | Plausible candidate |
| Norwood | 95.24 | 97 | 1.81% | 43.48 m | No | Plausible candidate |

- **7/7** raster-to-polygon traces completed, **7/7** did not touch the extraction-grid edge, and **6/7** are within 10% of the published drainage areas.
- The derived polygon areas differed from the corresponding MERIT `upa` values by approximately **0.01%** each. This shows internal consistency with the *same hydrology product*, **not** independent field accuracy.
- The notebook classified **all seven** as `CANDIDATE_PENDING_VISUAL_QA`, not `VERIFIED`.
- Each selected computational outlet is approximately **12–59 m** from its dashboard gauge point. Small displacement does not alone establish that it is on the right branch.
- The screenshots show generally plausible irregular and nested contributing regions. This is visual screening only; specific confluences and lower-basin/coastal routing need detailed review.

## 4. HydroBASINS Level 12 cross-check

| Gauge | Intersected Level-12 ID | `UP_AREA` (km²; **not gauge-equivalent**) |
|---|---|---:|
| Nagalagam Street | 4121601920 | 2,258.8 |
| Hanwella | 4121158680 | 1,818.8 |
| Glencourse | 4121601910 | 1,666.4 |
| Kithulgala | 4121601880 | 544.1 |
| Holombuwa | 4121601790 | 188.9 |
| Deraniyagala | 4121601950 | 201.2 |
| Norwood | 4121158540 | 211.8 |

**Seven of seven** computational outlet locations intersect one Level-12 polygon each. This confirms mapped geographical coverage, not exact catchment-boundary agreement. Apparent disagreement between purple HydroBASINS boundaries and blue MERIT stream lines is expected: the former show drainage divides, the latter modelled flow paths.

## 5. Output inventory and sharing rules

The executed notebook reports files created under:

`results/catchments/kelani_merit_hybas12_pilot/`

Per gauge (seven sets):

- `<station_slug>_catchment_polygon_CANDIDATE.geojson` — irregular upstream area, EPSG:4326.
- `<station_slug>_catchment_mask_CANDIDATE.tif` — cropped **native MERIT grid**; 1 = contributing cell, 0 = outside (do not confuse 0 with flood/non-flood truth).
- `<station_slug>_bounding_box_CANDIDATE.geojson` — rectangle enclosing the polygon, EPSG:4326.

Shared audit and combined outputs:

- `kelani_7_catchment_QA_registry.csv`
- `kelani_7_catchments_CANDIDATE_combined.geojson`
- `selected_computational_outlets_CANDIDATE.csv`
- `outlet_candidate_rankings.csv`
- `merit_kelani_native_dir_upa.tif` — exact-grid routing source.
- `national_18basin_compatibility_screen.csv` — preliminary national coverage screen.

**Handover status:** The result counts and filenames are based on the supplied notebook output; the actual generated GeoJSON/TIFF bytes remain in the researcher's local project and were **not supplied for independent file-level inspection** in this handover. Share the actual files from the above local output directory; sharing this Markdown document alone does not share the geometries.

## 6. How each component uses the AOIs

| Component | Use of shared catchment reference | Independence rule |
|---|---|---|
| **C1 — Satellite Depth** | Extract catchment-scale radar/wetness statistics; use separately defined fixed-scale local Sentinel-1 image patches for CNN spatial inputs if needed | Develop its own single-/multi-temporal satellite representations |
| **C2 — Weather Depth** | Aggregate CHIRPS and relevant ERA5 variables within each upstream catchment | Independently build weather features, preprocessing and models |
| **C3 — Environmental Depth** | Derive DEM/HAND/land-cover/stream features and dynamic environmental summaries for each contributing watershed | Independently implement static/dynamic environmental experiments |
| **C4 — Multimodal Breadth** | Reuse only the **agreed AOI definitions** and common target/sample IDs; independently extract all raw satellite, weather and environmental inputs | **No C1/C2/C3 processed features, encoders, weights or predictions** |

A common catchment registry is shared **experimental geography**, not sharing learned modality outputs. For very small catchments, coarse ERA5/ASCAT pixels may represent a much larger region; record extraction resolution and validity, and never imply downscaled precision.

## 7. Remaining quality gates and known limitations

1. **Kithulgala:** Inspect outlet placement versus the real main channel and nearby confluences; document why MERIT's area exceeds the reference by 11.38%, and compare nearby candidates if necessary.
2. **Nagalagam Street:** Inspect lower-river/coastal/tidal/canal routing. MERIT hydrographic flow is not a model of tidal backwater or artificial control structures.
3. **All seven:** Verify stream identity, expected upstream tributaries, nested catchment topology (e.g., downstream Hanwella should contain appropriate upstream subcatchments), and overlay alignment on satellite/topographic basemaps.
4. **Area checks:** Published-area differences are diagnostic; favorable matches are partially encouraged by the automatic outlet-ranking formula. Do not use them alone as independent validation evidence.
5. **Other datasets:** Sentinel-1 coverage dates, CHIRPS/ERA5/ASCAT product latency, historical gauge labels and temporal train/test leakage still require their **own** checks. Spatial QA does not resolve data-availability or forecasting-accuracy concerns.

**Practical PP1 decision:** **Proceed with feature-extraction prototypes using clearly identified CANDIDATE polygons**, retain the exact geometry/version and QA status per station, and complete the two priority reviews before calling the Kelani AOIs accepted. Keep any failed or unresolved AOI out of final benchmark claims until corrected.

## 8. Reuse for additional basins

Keep the same workflow and validation steps, but change the station registry, drainage-area reference table and a sufficiently large regional MERIT processing window. The regional export must cover **all upstream contributing drainage**, not a small square around the gauge. Never regard a `touches_edge=False` result as a substitute for independent river-connectivity QA.
