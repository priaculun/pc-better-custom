# pc-better-custom

A Pi / Oh My Pi (OMP) extension for managing custom providers through an
interactive wizard — with **OpenRouter metadata enrichment**, **difficulty-based
auto thinking**, and **self-healing reasoning effort**.

A fork of [`better-custom`](https://github.com/ratatulieoi/better-custom) by
ratatulieoi (MIT).

## Why

Adding a custom provider usually means hand-writing `models.json`, guessing the
context window, and discovering the hard way that a provider rejects
`reasoning_effort: "max"`. This extension does the probing, keeps the real
metadata, picks a sensible thinking effort per prompt, and fixes a rejected
effort value before you ever see the error.

## Features

### Provider wizard

- Add, edit, or delete custom providers from an interactive wizard
- OpenAI-compatible, Anthropic-compatible, and Ollama-compatible endpoints
- Auto-probe `/models` with a searchable multi-select picker
- Unique provider names; explicit confirmation before overriding a built-in id
- API key modes: literal, `$ENV`, `!command`, or none (writes a placeholder)
- Uses the host's agent dir (`models.json` for Pi, `models.yml`/`yaml` for OMP)

### OpenRouter metadata enrichment

When the endpoint is OpenRouter (or a proxy whose ids match the public catalog),
each model is written with real values instead of defaults:

| OpenRouter field | pi model field |
|---|---|
| `architecture.input_modalities` (image) | `input: ["text","image"]` |
| `context_length` | `contextWindow` |
| `top_provider.max_completion_tokens` | `maxTokens` |
| `pricing.prompt` / `completion` / `input_cache_read` / `input_cache_write` | `cost` (per-million) |
| `pricing.overrides[].min_prompt_tokens` | `cost.tiers[].inputTokensAbove` |
| `reasoning.supported_efforts` + `reasoning.mandatory` | `thinkingLevelMap` |
| `name` | `name` |

Proxies that return bare ids (`claude-fable-5` instead of
`anthropic/claude-fable-5`) are resolved against the public OpenRouter catalog:
exact id first, then a **unique** basename. Ambiguous matches are skipped rather
than guessed, and an offline/failed lookup silently falls back to defaults.

`Edit provider → Sync metadata from OpenRouter` backfills missing fields on existing
models. It asks before recomputing reasoning levels, since that replaces a ceiling
you may have set on purpose.

### Auto thinking level

`/better-custom → Thinking mode (auto)` scores each prompt and sets the cheapest
thinking level that should still handle it:

| Score | Level | Typical prompt |
|---|---|---|
| 0 | `minimal` | `hi`, thanks |
| 1 | `low` | short factual question |
| 2–3 | `medium` | focused change request |
| 4–5 | `high` | refactor + explanation |
| 6 | `xhigh` | big multi-part job |
| 7+ | `max` | hardest prompts |

Signals: prompt length, code fences / file paths / stack traces, hard-task
keywords, question count, numbered lists, image attachments, context fill, and how
rough the previous turn was (tool-call volume and tool errors). A configurable
floor (default `medium`) keeps non-trivial prompts from dropping too low, and the
chosen level is always snapped to what the model supports.

### Self-healing reasoning effort

Some providers reject effort values their upstream does not accept:

```
400 VALIDATION_ERROR
issues: [{ received: "max", code: "invalid_enum_value",
           options: ["low","medium","high"], path: ["reasoning_effort"] }]
```

Two layers handle this, for **custom providers only** (built-in catalogs carry
accurate metadata and are left alone):

1. **Effort guard** — before each request, a top-level `reasoning_effort` above
   the declared/learned limit is clamped to the highest allowed value.
   Non-standard `xhigh`/`max` are pruned by default on plain OpenAI-style
   endpoints (OpenAI documents only `low`/`medium`/`high`); re-enable them per
   provider if your endpoint really accepts them.
2. **Learning + retry** — an `invalid_enum_value` rejection on an effort field is
   parsed for the allowed `options`, persisted, mirrored into that model's
   `thinkingLevelMap` in `models.json`, and the request is retried with a
   supported value. The original error detail is preserved, so a genuinely
   unfixable failure still surfaces once the retry budget is spent.

`/better-custom → Reasoning effort limits` lists learned limits per
`provider/model`, lets you forget one or all, and toggles auto-retry and the
extended-effort cap.

## Install

From GitHub:

```bash
pi install https://github.com/<you>/pc-better-custom
```

From a local checkout:

```bash
pi install /absolute/path/to/pc-better-custom
```

Or drop it in an auto-discovered location and `/reload`:

```
~/.pi/agent/extensions/pc-better-custom/index.ts   # global
.pi/extensions/pc-better-custom/index.ts           # project-local
```

## Usage

```text
/better-custom
```

The menu offers:

1. **Add provider**
2. **Edit provider** — re-probe, sync metadata, set context window, edit per model,
   add models, rename
3. **Delete provider**
4. **Thinking mode (auto)** — enable auto effort, set the fallback floor
5. **Reasoning effort limits** — learned limits, auto-retry, extended-effort cap

## Runtime state

Written to `~/.pi/agent/better-custom.state.json` (not `models.json`, whose schema
only allows model metadata):

```json
{
  "autoThinking": false,
  "autoThinkingFallback": "medium",
  "autoRetryEffort": true,
  "allowExtendedEffort": false,
  "effortAllow": {
    "ontoken/deepseek-v4.1-flash": ["low", "medium", "high"]
  }
}
```

## Development

```bash
npm install
npm run check      # node --check index.ts
npm run typecheck  # tsc --noEmit
```

`tsconfig.json` maps the `@mariozechner/*` specifiers (which Pi aliases at runtime)
onto the published `@earendil-works/*` type packages used for typechecking.

`index.ts` is the whole extension — a single file so it can be dropped into
`extensions/` without a build step.

## How it maps to pi

pi exposes thinking levels `off, minimal, low, medium, high, xhigh, max`. When a
model has `reasoning: true`, `minimal`…`high` are available by default; `xhigh`
and `max` are opt-in and only appear when explicitly mapped. Any level set to
`null` is hidden. The wizard writes `thinkingLevelMap` to match what the provider
actually accepts.

## License

MIT. Original `better-custom` © ratatulieoi; fork modifications © the
pc-better-custom contributors. See [LICENSE](LICENSE) and
[CHANGELOG.md](CHANGELOG.md).
