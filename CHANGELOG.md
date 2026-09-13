# Changelog

## 1.0.0

Fork of [`better-custom`](https://github.com/ratatulieoi/better-custom) v0.4.2 with
OpenRouter metadata enrichment, difficulty-based auto thinking, and self-healing
reasoning effort.

### Added

- **OpenRouter metadata enrichment** — the probe keeps `architecture.input_modalities`,
  `context_length`, `top_provider.max_completion_tokens`, `pricing.*` (per-token →
  per-million, including `overrides` → `cost.tiers`) and `name`, so a new model is
  written with real values instead of guesswork.
- **Catalog lookup for proxy endpoints** — bare ids from a non-OpenRouter endpoint
  (e.g. `claude-fable-5`) are matched against the public OpenRouter catalog (exact id,
  then a unique basename) to recover metadata.
- **Auto thinking level** — `/better-custom → Thinking mode (auto)` scores each prompt
  (length, code/logs, hard-task keywords, question count, images, context fill, and how
  rough the previous turn was) and sets the cheapest level that should handle it, with a
  configurable fallback floor.
- **Effort guard** — for custom providers, an outgoing `reasoning_effort` above the
  declared/learned limit is clamped before the request. Non-standard `xhigh`/`max` are
  pruned by default on plain OpenAI-style endpoints.
- **Effort learning + retry** — an `invalid_enum_value` rejection on an effort field is
  parsed for allowed options, persisted to `better-custom.state.json`, mirrored into
  `thinkingLevelMap` in `models.json`, and the request is retried with a supported value.
- **New wizard actions** — `Sync metadata from OpenRouter`, `Thinking mode (auto)`,
  `Reasoning effort limits`.
- **Thinking level map support** — reasoning levels are derived from OpenRouter's
  `reasoning.supported_efforts` / `reasoning.mandatory`, including `max`.

### Changed

- `yaml` is resolved lazily via `createRequire`, so the extension also runs from
  `~/.pi/agent/extensions/` (OMP `models.yml` support preserved).
- `pickMany` returns full probe items (with metadata) instead of bare ids.
- `ModelOptions` / `buildModelEntry` / `buildProviderConfig` carry `name`, `maxTokens`
  and `cost` in addition to the previous fields.

### Fixed

- Cost tiers always emit numeric `input`/`output`/`cacheRead`/`cacheWrite`; a missing
  value previously dropped the key and made `models.json` fail pi's schema validation.
- Provider-id and state caches are refreshed on `session_start` and `model_select`.

## Upstream

Based on `better-custom` 0.4.2 by ratatulieoi (MIT).
