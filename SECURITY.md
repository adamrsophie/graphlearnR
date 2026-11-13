# Security Policy

## Supported Versions

Security fixes are considered for the active development branch and the most
recent release.

| Version | Supported |
|---------|-----------|
| `0.1.x` | Yes |
| `< 0.1.0` | No |

## Reporting a Vulnerability

Please do not report security vulnerabilities in public GitHub issues.

Report suspected vulnerabilities by email to `adamrsophie@gmail.com`, or use
GitHub private vulnerability reporting if it is enabled for the repository.

Include as much detail as you can:

- affected version or commit,
- operating system and R version,
- minimal reproduction steps,
- any proof-of-concept code.

## Response Process

The maintainer will acknowledge reports as soon as practical, investigate the
issue, and coordinate a fix if the report is confirmed. Public disclosure should
wait until a fix or mitigation is available, unless there is an overriding user
safety reason to disclose earlier.

## Scope

Security-sensitive areas include:

- dataset download and extraction: `download_tu_dataset()` and
  `download_planetoid_raw()` fetch archives over HTTPS, check MD5 sums and
  extract them; a path traversal or a way to skip the checksum would be a
  vulnerability,
- deserialization: `load_graph_fit()` and `graph_dataset_disk()` read `.rds`
  files, which can carry arbitrary R objects; load only files you trust,
- arbitrary file writes, for example through `checkpoint_path` or
  `write_graph_dataset()`,
- unbounded memory or time on small inputs, for example in dense batching or
  shortest path encodings,
- exposure of private data in examples, tests, or logs.

A numerically wrong result that does not create a software-security
vulnerability should be opened as a normal GitHub issue instead.
