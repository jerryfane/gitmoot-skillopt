# Changelog

## v0.2.0b2

- Fix Claude Code structured-output compatibility by using the public
  `--json-schema` flag, with legacy `--schema` fallback when advertised by the
  installed Claude CLI.
- Fail early with actionable guidance when a Claude CLI has no schema-backed
  structured-output flag.
