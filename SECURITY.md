# Security and responsible use

This code is an academic **retrospective research prototype**, not an emergency prediction service. It should not be used to make safety-critical evacuation or flood-management decisions.

Never disclose credentials or restricted source datasets in commits, issues or pull requests. If a token was exposed, revoke and rotate it: removing a file from a new commit does not remove it from Git history.

The C1 API is intended for local use on `127.0.0.1`. Review authentication, CORS, dependency security, data licenses, and model validity before considering external deployment.
