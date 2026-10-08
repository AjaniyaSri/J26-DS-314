# PP1 Bi-temporal SAR Mask Pilot Area - Hanwella

**Pilot ID:** `PP1_MASK_PILOT_HANWELLA_01`
**Status:** PROVISIONAL / TECHNICAL PILOT
**Purpose:** Validate the FloodRisk Sri Lanka bi-temporal Sentinel-1 flood-change processing workflow before scaling it to the full PP1 event study window.

## 1. Pilot geometry

The pilot area is a small rectangular subset within the provisional PP1 Kelani study window.

| Parameter             |                     Value |
| --------------------- | ------------------------: |
| West                  |                 80.055° E |
| South                 |                  6.885° N |
| East                  |                 80.105° E |
| North                 |                  6.935° N |
| Coordinate convention |       Longitude, Latitude |
| Area                  |      Approximately 30 km² |
| CRS for geometry      | WGS 84 longitude/latitude |

The area is intentionally small so that the first Earth Engine processing run can be inspected quickly and debugging can be performed before processing the full event window.

## 2. Why this pilot area is being used

The pilot area is located within the provisional Kelani-focused PP1 study window and is suitable as an initial technical test area because it is small enough for rapid experimentation while remaining geographically relevant to the selected study area.

The pilot geometry does **not** represent the complete flood extent of either historical event and must not be interpreted as an event boundary.

Its purpose is to test:

* Sentinel-1 image retrieval
* pre/post image compatibility
* spatial overlap
* VV/VH inspection
* bi-temporal change calculation
* thresholding
* persistent-water exclusion
* basic morphological cleaning
* raster export
* visual quality control

## 3. Event relationship

The pilot will initially be tested against a candidate event for which a suitable Sentinel-1 pre-event/flood-phase pair can be identified.

The first event to test should be selected using:

1. documented DMC evidence for flooding relevant to the broader study area
2. availability of Sentinel-1 observations before and during/after the documented flood phase
3. adequate spatial overlap over the pilot area
4. compatible acquisition geometry

The pilot event and exact Sentinel-1 scene pair remain **TBD until the acquisition inventory has been reviewed**.

## 4. Important methodological limitation

This pilot output is an exploratory SAR-change product.

It is not yet the official shared flood-reference mask.

The authoritative mask can only be generated after:

* the event chronology has been reconstructed
* the selected flood-phase observation has been justified
* the pre/post Sentinel-1 pair has been selected
* the common grid and CRS have been frozen
* the persistent-water rule has been agreed
* and independent QC has been completed

## 5. Expected output

The pilot should produce:

* the selected pre-event scene ID
* the selected flood-phase scene ID
* acquisition dates and times
* orbit/pass information
* processing parameters
* candidate flood-change raster
* visual comparison of pre-event and flood-phase imagery
* QC notes
* and an experiment record

**Pilot result status:** `PENDING`