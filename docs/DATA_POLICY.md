# Data access and publication

The shared PP1 agreement requires verifying original source terms before publishing numerical observations or downloaded scientific source data.

**Do not publish without permission:** `secrets.env`, API credentials, DMC source-derived station-day labels, satellite SAR or SWI downloads, image patches, derived feature tables, private hydrography assets, local trained models (`.joblib`), station photographs, or unverified third-party imagery.

**Appropriate for source control after review:** application and experiment code, tests, sanitized examples, data schemas, aggregate summaries with methodology and QA, documentation, and authorized original artwork.

Keep a **private data manifest** with origin, product version, license and access rules, collection date, preprocessing, QA, checksums and intended use. `.gitignore` only prevents *future* accidental additions; it does **not** remove previously committed data from Git history.

If any secret was pushed, revoke/rotate it immediately and contact the repository owner about history cleanup.
