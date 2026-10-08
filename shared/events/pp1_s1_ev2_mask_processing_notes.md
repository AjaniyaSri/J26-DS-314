# PP1 Sentinel-1 Bi-temporal Mask Processing Notes — EV2_Nov_2025

## Candidate pair

Pre-event:
- Date: 2025-11-26
- Platform: Sentinel-1A
- Pass: Descending
- Relative orbit: 19
- Polarisation: VV, VH
- Two intersecting slices used to form the pre-event mosaic.

Post-event:
- Date: 2025-12-02
- Platform: Sentinel-1C
- Pass: Descending
- Relative orbit: 19
- Polarisation: VV, VH
- Reported candidate-AOI coverage: 100%.

## Pair rationale

The pair was selected as the closest currently identified candidate
pre/post acquisition combination with descending geometry, matching
relative orbit, VV/VH availability, and suitable spatial coverage.

The 26-November acquisition is represented by two intersecting slices,
which are mosaicked before clipping to the pilot AOI.

## Known mismatches

- Six-day temporal separation.
- Sentinel-1A to Sentinel-1C platform change.
- Pre-event acquisition requires a two-slice mosaic.
- Final suitability remains provisional until the event chronology and full PP1 spatial evidence are reviewed.

## Comparable-image processing

- Both acquisitions clipped to the same PP1 pilot AOI.
- VV selected for the initial change experiment.
- Identical visualisation ranges used for pre/post VV images.
- No additional date-specific speckle filtering applied.
- Change image calculated as post-event VV minus pre-event VV in dB.

## Candidate mask

A provisional threshold-based change mask is generated for technical
validation only. It is not the final shared flood-reference mask.

The planned final workflow remains:
log-ratio -> Otsu thresholding -> morphological cleanup ->
permanent-water masking.

## Review outcome

Status: PROVISIONAL CANDIDATE

Next review:
- confirm event chronology against DMC evidence;
- inspect pre/post/change layers over the pilot AOI;
- assess whether the platform mismatch materially affects the comparison;
- replace the provisional threshold with the documented final
  change-detection procedure.