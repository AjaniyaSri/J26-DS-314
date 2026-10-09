# PP1 Sentinel-1 Bitemporal Mask Processing Notes — EV2_Nov_2025

**Task:** GC-07.2 — PP1 Sentinel-1 Bitemporal Pilot — Thresholding, Morphological Cleanup, and Permanent-Water QA  
**Status:** Diagnostic prototype; candidate mask is **not accepted as a validated flood mask**.  
**Last updated:** 9 October 2026

## 1. Scope and study area

This record documents the exploratory Sentinel-1 bitemporal VV change experiment for EV2_Nov_2025 over the small PP1 pilot area near Hanwella. The pilot area is a technical test window, not the event boundary and not a claim that the entire area flooded. Thresholded pixels indicate candidate radar backscatter change, not confirmed floodwater; independent event-specific evidence and quality assurance remain necessary.

- Pilot AOI: `[80.055, 6.885, 80.105, 6.935]` (WGS 84 longitude/latitude).
- Notebook-reported pilot area: approximately 30.69 km².
- Larger candidate AOI: approximately 2,540.21 km².
- Sentinel-1 inventory search window: 20 November–24 December 2025. This is an acquisition-search window, not a confirmed event interval.

## 2. Provisional acquisition pair

**Pre-event candidate — 26 November 2025**
- Local acquisition time: approximately 05:55 (Asia/Colombo).
- Sentinel-1A; descending pass; relative orbit 19; IW mode; VV and VH.
- Two intersecting slices were mosaicked before clipping to the pilot AOI.

**Post-event candidate — 2 December 2025**
- Local acquisition time: approximately 05:54 (Asia/Colombo).
- Sentinel-1C; descending pass; relative orbit 19; IW mode; VV and VH.
- The inventory reported 100% scene-footprint overlap for the larger candidate AOI. This is a footprint-intersection measure, not valid-pixel coverage after clipping.

The six-day interval, platform change (S1A to S1C), and pre-event mosaic are limitations. Matching pass direction and relative orbit support an initial comparison but do not remove platform, radiometric, speckle, temporal, or land-surface differences. The complete DMC/gauge chronology has not yet confirmed that these acquisitions capture the most appropriate flood phase. The pair remains provisional for technical prototyping.

## 3. Image preparation

The notebook mosaics the two 26 November slices, clips the pre-event mosaic and 2 December post-event image to the same pilot AOI, and selects VV for the current change experiment. VH is selected but not used in the current mask calculation. Identical visualisation ranges are used for pre/post VV. No additional date-specific speckle filter has been applied; this is a baseline experiment, not a claim that speckle has been removed.

A previous post-clipping geometry check returned 100% because the image had already been clipped to the AOI. That does not measure valid-pixel coverage. Scene-footprint overlap, valid-pixel coverage, and candidate-mask overlap are distinct measures.

## 4. Change-image definition and sign convention

The current notebook uses:

```python
log_ratio_db = pre_vv.subtract(post_vv).rename("VV_log_ratio_dB")
```

For Sentinel-1 GRD dB values, pre-event minus post-event backscatter is the logarithmic expression of the pre/post backscatter ratio. Positive values mean post-event VV is lower than pre-event VV; negative values mean it is higher. This direction describes change, not proof of flooding. Backscatter changes can also result from vegetation, soil moisture, roughness, built surfaces, speckle, acquisition differences, and other processes.

Current reported percentiles:

| Statistic | Value |
|---|---:|
| 2nd percentile | −5.622 dB |
| Median | +0.375 dB |
| 98th percentile | +10.871 dB |

The older `post_vv - pre_vv` image has the opposite sign. For the same images and valid pixels, `post - pre < -3 dB` is equivalent to `pre - post > 3 dB`. The later fixed-threshold map now follows the clearer `pre - post` convention.

## 5. Fixed threshold, Otsu, and morphology

The corrected fixed-threshold mask is based on `log_ratio_db.gt(3)`. The current run reported:

| Candidate-mask method | Valid change pixels selected |
|---|---:|
| Fixed threshold: `pre − post > 3 dB` | 19.49% |
| Raw Otsu mask | 34.07% |
| After morphological opening | 22.59% |
| After opening and closing | 23.67% |

An earlier Otsu run returned approximately **+1.4999 dB**; the threshold should be re-recorded after the current histogram cell is rerun. Otsu partitions the value histogram; it is not flood-aware and does not identify a class as floodwater. The raw Otsu mask appeared fragmented. Opening (focal minimum then focal maximum) removed many selected pixels; closing (focal maximum then focal minimum) restored or joined some. Both use a one-pixel radius. The cleaned mask still appears fragmented, so these operations have not demonstrated a defensible improvement in flood-mask quality. Opening can remove narrow true features and closing can join unrelated patches. The percentages are descriptive selected-pixel proportions, not accuracy metrics.

## 6. JRC historical persistent-water exclusion

### 6.1 Initial threshold issue

The first rule used `occurrence >= 90%`. The diagnostic returned **726 valid occurrence pixels**, a **maximum occurrence of 89**, and **zero pixels ≥90%**. Therefore, that rule selected no pixels and removed no permanent water. The earlier equal `count()` values were misleading: `ee.Reducer.count()` counts valid pixels, not only pixels above a threshold.

### 6.2 Current provisional proxy

The current test combines JRC `transition` classes **1, 2 and 7**:
- Class 1: Permanent water.
- Class 2: New permanent water.
- Class 7: Seasonal to permanent water.

A constant-base 0/1 image was used so masked/unclassified transition pixels remain unexcluded by this proxy instead of automatically propagating the source mask. Unknown pixels are not thereby proven dry.

Current JRC diagnostics:

| Diagnostic | Count |
|---|---:|
| Transition class 1 — Permanent | 148 |
| Transition class 2 — New permanent | 25 |
| Transition class 7 — Seasonal to permanent | 23 |
| Combined transition classes 1, 2 and 7 | 196 |
| Seasonality = 12 months (comparison only) | 196 |
| Occurrence ≥90% | 0 |

The transition proxy is non-empty. Its count happens to match the seasonality-12 count, but these are different definitions; their pixels must not be assumed identical without checking their spatial overlap.

### 6.3 Effect on cleaned Otsu candidates

| Diagnostic | Count |
|---|---:|
| Cleaned Otsu candidate pixels | 72,263 |
| Candidate pixels overlapping historical-water proxy | 570 |
| Candidate pixels excluded | 0.79% |
| Remaining candidate pixels | 71,693 |

The arithmetic is consistent: 72,263 − 570 = 71,693. The JRC proxy was evaluated at its native approximately 30 m scale for its own diagnostic counts; candidate overlap was counted on the Sentinel-1 analysis grid. These 10 m-aligned counts are not independent 10 m JRC observations and do not add spatial detail to the original JRC product.

**Interpretation:** the revised exclusion removes some candidates, but only **0.79%** of the cleaned Otsu candidates overlap the proxy. This is a small effect and does not establish that permanent-water contamination has been adequately removed. The small overlap could relate to candidate-mask distribution, proxy definition, historical data limitations, AOI coverage, or grid alignment; spatial QA is needed rather than selecting a more aggressive threshold simply to remove more pixels.

JRC is historical water-occurrence/transition information from Landsat observations over 1984–2021, not an observation of the November–December 2025 flood. The final mask has not passed independent visual or event-specific QA.

## 7. Planned local Sri Lankan hydrography reference

The next planned improvement is to inspect a credible local hydrography vector source, beginning with the Survey Department's topographic GIS information and the national spatial-data service. The Survey Department describes topographic vector layers with hydrology features in line and polygon geometries, available through GIS data supply; the national spatial-data service exposes separate hydrography line and polygon layers.

Candidate official resources for evaluation:
- [Survey Department — GIS / Geo Information](https://survey.gov.lk/sdweb/pages_service_geo_information.php?id=df658590a4cbb1f955f5d386b242b6be8d5cadc0)
- [SLNSDI Survey 1:50K — hydrography layers](https://www.gisapps.nsdi.gov.lk/server/rest/services/SLNSDI/Survey_50K/MapServer/layers)
- [SLNSDI `HY_Hydro_Pg` polygon layer](https://www.gisapps.nsdi.gov.lk/server/rest/services/SLNSDI/Survey_50K/MapServer/10)

These are candidate resources, not yet an accepted or downloaded authoritative mask. Before use, verify attribution, licence/access conditions, geometry type, relevant water-body classes, creation/revision dates, coverage, attributes, and CRS. Transform the data correctly to the analysis CRS; do not merely relabel its CRS.

A waterways line represents a mapped channel/centre-line, not the actual water surface or event flood extent. Hydrography polygons may map lakes, tanks, reservoirs, canals, or other water features according to their attributes. Initially overlay the vector data with JRC and Sentinel-1 candidate masks for quality assurance. Do not automatically erase a broad buffer around the Kelani River, as that could remove genuine overbank flooding. Any channel-buffer exclusion should remain a separate, documented sensitivity experiment with a stated width and rationale.

JRC and local hydrography have complementary roles: JRC provides historical raster water-occurrence/transition context; local hydrography provides mapped channel and water-body context. Neither alone proves the extent of the 2025 flood.

## 8. Outstanding QA and next work

The following work remains open under GC-07.2:

1. Inspect the transition-class 1/2/7 proxy and seasonality-12 layer side by side; calculate spatial overlap rather than assuming identical footprints.
2. Obtain and inspect the official/local hydrography resource; verify source, licence/access, geometry, relevant classes, revision dates, coverage, attributes, and CRS.
3. Overlay local hydrography, JRC, pre/post VV, change image, raw Otsu, after-opening, after-closing, and post-exclusion outputs; record river-aligned artefacts and disagreements.
4. Calculate valid Sentinel-1 pixel coverage using the VV image mask and document alignment of 30 m JRC/vector features with the analysis grid.
5. Reassess the acquisition pair against DMC reports and relevant gauge timing to confirm its event-phase suitability.
6. Obtain independent C1 review and record acceptance conditions. Do not designate the mask as canonical before this review.
7. Record the chosen water-exclusion rule, source/licence/CRS, parameter choices, QA findings, and reasons for retaining or rejecting candidate areas.

## 9. Review outcome

**Status: DIAGNOSTIC PROTOTYPE — WATER EXCLUSION TESTED; QA NOT COMPLETE**

The acquisition pair, sign convention, fixed-threshold baseline, Otsu mask, morphological stages, and JRC exclusion have been documented. The occurrence ≥90% rule selected no pixels because the maximum occurrence in the pilot AOI was 89. A transition-class 1/2/7 proxy was then tested and removed 570 of 72,263 cleaned Otsu candidates (0.79%). This confirms that the revised implementation excludes some candidates, but the small overlap does not demonstrate adequate persistent-water removal or flood-mask accuracy. Local hydrography inspection, spatial comparison, event chronology review, valid-data coverage assessment, and independent C1 QA remain outstanding. The output must not be treated as the canonical shared flood-reference mask.

## Data references

- [Sentinel-1 GRD — Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD)
- [JRC Global Surface Water v1.4 — Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_GlobalSurfaceWater)
- [Earth Engine `Image.where`](https://developers.google.com/earth-engine/apidocs/ee-image-where)
- [Earth Engine `Image.unmask`](https://developers.google.com/earth-engine/apidocs/ee-image-unmask)
- [Earth Engine morphological operations](https://developers.google.com/earth-engine/guides/image_morph)
- [Survey Department — GIS / Geo Information](https://survey.gov.lk/sdweb/pages_service_geo_information.php?id=df658590a4cbb1f955f5d386b242b6be8d5cadc0)
- [SLNSDI Survey 1:50K hydrography layers](https://www.gisapps.nsdi.gov.lk/server/rest/services/SLNSDI/Survey_50K/MapServer/layers)
- [SLNSDI `HY_Hydro_Pg` polygon layer](https://www.gisapps.nsdi.gov.lk/server/rest/services/SLNSDI/Survey_50K/MapServer/10)
