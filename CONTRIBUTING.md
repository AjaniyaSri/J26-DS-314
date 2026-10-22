# Contributing

This team project has **four independently developed components** and one shared station-day data contract.

1. Update `main` locally: `git switch main && git pull --ff-only origin main`.
2. Create your branch: `git switch -c feat/c1-model-audit`.
3. Commit only reviewed source and approved docs (for example, `feat(c1): add station inference`).
4. Run syntax checks and relevant tests; inspect `git diff --cached --name-only`.
5. Push, open a pull request, and request a teammate's review before merging.

**Research checks:** unchanged `station_id + forecast_date` join keys; chronological TRAIN/VALIDATION/TEST splits; no future-target leakage; train-only preprocessing; source availability and QA flags; matched baseline comparisons with sample counts.

**Publication checks:** no credentials, raw gauge tables, satellite patches, private licensed assets, generated outputs or model binaries.
