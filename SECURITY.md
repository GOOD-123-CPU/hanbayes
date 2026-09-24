# Security Policy

## Supported versions

Security-related maintenance targets the latest revision on the `main` branch.
Historical commits and archived experiment artifacts are not independently maintained.

## Reporting a vulnerability

Please do **not** open a public issue for a suspected vulnerability. Prefer GitHub
private vulnerability reporting when available; otherwise use the maintainer contact
listed on the GitHub profile.

A useful report includes the affected revision, a minimal reproduction, realistic
impact, and any proposed mitigation. Never include credentials or private datasets.

## Data and research integrity

HanBayes is a research-oriented statistical/NLP project. Reports are also welcome for
issues that can compromise reproducibility or silently invalidate published artifacts,
including:

- unsafe loading of untrusted serialized objects;
- accidental inclusion of private or licensed datasets;
- seed/configuration drift that makes frozen results unverifiable;
- metric code paths that can silently produce invalid statistical conclusions;
- generated artifacts whose provenance or hashes no longer match recorded metadata.

Statistical correctness bugs that do not create a security impact should be filed as
normal public issues with a minimal reproducible example.

## Scope

The project is not a hosted service and does not claim a production security posture.
This policy covers vulnerabilities in the repository code, workflows, packaging, and
published artifacts.
