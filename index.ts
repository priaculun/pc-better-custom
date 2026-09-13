import * as CodingAgent from "@mariozechner/pi-coding-agent";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Key, matchesKey, truncateToWidth } from "@mariozechner/pi-tui";
import { execSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

type ProviderApi = "openai-completions" | "anthropic-messages";
type ProviderStyle = "openai" | "anthropic" | "ollama";
type ApiKeyMode = "env" | "literal" | "shell" | "none";
// pi's reasoning levels. "off" means no reasoning; the rest are the levels a
// model is allowed to use. See pi-ai getSupportedThinkingLevels / EXTENDED_THINKING_LEVELS.
type ReasoningCeiling = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";
const REASONING_LEVELS: ReasoningCeiling[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

// Per-model knobs the wizard can write. apiKey lives at provider scope, not here.
type ModelOptions = {
	reasoning: ReasoningCeiling;
	vision: boolean;
	contextWindow?: number;
	maxTokens?: number;
	name?: string;
	cost?: ModelCost;
	// Provider-declared thinking level map (OpenRouter supported_efforts). When
	// set, it overrides pi's default level assumptions.
	reasoningMap?: ThinkingLevelMap;
};

type ModelsConfig = {
	providers?: Record<string, any>;
};

type ProbeItem = {
	value: string;
	label: string;
	description?: string;
	// Rich metadata surfaced by OpenRouter's /models endpoint. Undefined for
	// plain OpenAI-compatible probes that only return a list of ids.
	meta?: OpenRouterModelMeta;
};

// Subset of the OpenRouter /models payload we care about when pre-filling a
// custom model entry. All fields are optional so a partial payload is harmless.
type ModelCost = {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	tiers?: Array<{ inputTokensAbove: number; input: number; output: number; cacheRead: number; cacheWrite: number }>;
};

type OpenRouterModelMeta = {
	name?: string;
	contextLength?: number;
	maxCompletionTokens?: number;
	modalities: string[];
	supportedParameters: string[];
	cost?: ModelCost;
	// OpenRouter's `reasoning` object. `supportedEfforts` are the provider-facing
	// effort strings ("low"…"max", "none"); `mandatory` means thinking cannot be
	// turned off.
	supportedEfforts?: string[];
	mandatoryReasoning?: boolean;
};

type SelectItem = {
	value: string;
	label: string;
	suffix?: string;
	description?: string;
	searchText?: string;
};

type CommandContext = Parameters<Parameters<ExtensionAPI["registerCommand"]>[1]["handler"]>[1];

// Persisted tuning that the wizard writes to a sidecar file rather than
// models.json, whose schema only allows model metadata.
type BetterCustomState = {
	// Difficulty-based thinking level (see autoThinking).
	autoThinking?: boolean;
	// Level used when auto thinking cannot classify a prompt.
	autoThinkingFallback?: ReasoningCeiling;
	// Retry once with a clamped level when a provider rejects an effort value.
	autoRetryEffort?: boolean;
	// Opt into sending xhigh/max on a plain OpenAI-compatible endpoint.
	allowExtendedEffort?: boolean;
	// Provider/model -> provider-facing effort strings the endpoint actually
	// accepts. Learned from `invalid_enum_value` responses and then used to clamp
	// outgoing requests so a bad value never reaches the provider again.
	effortAllow?: Record<string, string[]>;
};

const AGENT_DIR = CodingAgent.getAgentDir();
const IS_OMP = "logger" in CodingAgent || /(^|[\\/])\.?omp([\\/]|$)/i.test(AGENT_DIR);
// OMP prefers YAML and still accepts legacy JSON; normal Pi uses JSON only.
const MODELS_JSON_PATH = (IS_OMP ? ["models.yml", "models.yaml", "models.json"] : ["models.json"])
	.map((name) => join(AGENT_DIR, name))
	.find(existsSync) ?? join(AGENT_DIR, IS_OMP ? "models.yml" : "models.json");
const IS_YAML_CONFIG = /\.ya?ml$/i.test(MODELS_JSON_PATH);
const BUILTIN_PROVIDER_IDS = new Set([
	"anthropic",
	"openai",
	"azure-openai",
	"google",
	"vertex",
	"bedrock",
	"mistral",
	"groq",
	"cerebras",
	"xai",
	"openrouter",
	"vercel-ai-gateway",
	"zai",
	"huggingface",
	"kimi-for-coding",
	"minimax",
	"ollama",
]);

function ensureConfigDir() {
	mkdirSync(dirname(MODELS_JSON_PATH), { recursive: true });
}

const STATE_PATH = join(AGENT_DIR, "better-custom.state.json");

// Cached so the provider-request hook stays cheap; refreshed on session start.
let stateCache: BetterCustomState | undefined;

function loadState(): BetterCustomState {
	if (stateCache) return stateCache;
	try {
		if (!existsSync(STATE_PATH)) return (stateCache = {});
		const raw = readFileSync(STATE_PATH, "utf8").trim();
		if (!raw) return (stateCache = {});
		const parsed = JSON.parse(raw);
		stateCache = parsed && typeof parsed === "object" && !Array.isArray(parsed) ? (parsed as BetterCustomState) : {};
	} catch {
		stateCache = {};
	}
	return stateCache;
}

function refreshState() {
	stateCache = undefined;
}

function saveState(state: BetterCustomState) {
	stateCache = state;
	try {
		writeFileSync(STATE_PATH, `${JSON.stringify(state, null, 2)}\n`, "utf8");
	} catch {
		// Best-effort: a read-only agent dir must not break the agent loop.
	}
}

// `yaml` is only needed for OMP-style configs. Resolve it lazily so the
// extension also works when it lives in ~/.pi/agent/extensions (where node
// module resolution can't see pi's managed npm directory).
let yamlModule: { parse: (raw: string) => unknown; stringify: (value: unknown, opts?: any) => string } | null | undefined;
function getYaml() {
	if (yamlModule === undefined) {
		yamlModule = null;
		const candidates: Array<() => any> = [
			() => createRequire(import.meta.url)("yaml"),
			() => createRequire(join(AGENT_DIR, "npm", "package.json"))("yaml"),
		];
		for (const load of candidates) {
			try {
				yamlModule = load();
				break;
			} catch {
				// try the next resolution root
			}
		}
	}
	if (!yamlModule) {
		throw new Error('The "yaml" package is required to read/write YAML models configs. Install it with "npm i yaml".');
	}
	return yamlModule;
}

function loadModelsConfig(): ModelsConfig {
	ensureConfigDir();
	if (!existsSync(MODELS_JSON_PATH)) {
		return { providers: {} };
	}

	const raw = readFileSync(MODELS_JSON_PATH, "utf8").trim();
	if (!raw) return { providers: {} };

	const parsed = (IS_YAML_CONFIG ? getYaml().parse(raw) : JSON.parse(raw)) as ModelsConfig;
	if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
		throw new Error("models config must be an object");
	}
	if (!parsed.providers || typeof parsed.providers !== "object") {
		parsed.providers = {};
	}
	return parsed;
}

function saveModelsConfig(config: ModelsConfig) {
	ensureConfigDir();
	const content = IS_YAML_CONFIG ? getYaml().stringify(config, { lineWidth: 0 }) : JSON.stringify(config, null, 2);
	writeFileSync(MODELS_JSON_PATH, `${content.trimEnd()}\n`, "utf8");
	refreshProviderIds();
}

// Provider ids the user defined in models.json. The runtime effort guard only
// touches these: built-in provider catalogs already carry accurate effort info,
// so clamping them would silently downgrade models that really do support
// xhigh/max.
let providerIdsCache: Set<string> | undefined;
function customProviderIds(): Set<string> {
	if (providerIdsCache) return providerIdsCache;
	try {
		providerIdsCache = new Set(Object.keys(loadModelsConfig().providers ?? {}));
	} catch {
		providerIdsCache = new Set();
	}
	return providerIdsCache;
}

function isCustomProvider(providerId: string | undefined): boolean {
	return !!providerId && customProviderIds().has(providerId);
}

function refreshProviderIds() {
	providerIdsCache = undefined;
}

function hasExplicitScheme(input: string): boolean {
	return /^[a-z]+:\/\//i.test(input.trim());
}

function addDefaultScheme(input: string): string {
	if (hasExplicitScheme(input)) return input;
	const lower = input.toLowerCase();
	const isLocal =
		lower.startsWith("localhost") ||
		lower.startsWith("127.") ||
		lower.startsWith("0.0.0.0") ||
		lower.startsWith("10.") ||
		lower.startsWith("192.168.") ||
		lower.startsWith("172.16.") ||
		lower.startsWith("172.17.") ||
		lower.startsWith("172.18.") ||
		lower.startsWith("172.19.") ||
		lower.startsWith("172.20.") ||
		lower.startsWith("172.21.") ||
		lower.startsWith("172.22.") ||
		lower.startsWith("172.23.") ||
		lower.startsWith("172.24.") ||
		lower.startsWith("172.25.") ||
		lower.startsWith("172.26.") ||
		lower.startsWith("172.27.") ||
		lower.startsWith("172.28.") ||
		lower.startsWith("172.29.") ||
		lower.startsWith("172.30.") ||
		lower.startsWith("172.31.") ||
		lower.startsWith("[");
	return `${isLocal ? "http" : "https"}://${input}`;
}

function stripSuffix(pathname: string, suffix: string): string {
	return pathname.endsWith(suffix) ? pathname.slice(0, -suffix.length) || "/" : pathname;
}

function normalizeEndpoint(input: string, api: ProviderApi): string {
	const url = new URL(addDefaultScheme(input.trim()));
	let pathname = url.pathname.replace(/\/+$/, "") || "/";

	if (api === "openai-completions") {
		pathname = stripSuffix(pathname, "/chat/completions");
		pathname = stripSuffix(pathname, "/responses");
		pathname = stripSuffix(pathname, "/completions");
		pathname = stripSuffix(pathname, "/models");
	} else {
		pathname = stripSuffix(pathname, "/messages");
	}

	pathname = pathname === "/" ? "" : pathname;
	const port = url.port ? `:${url.port}` : "";
	return `${url.protocol}//${url.hostname}${port}${pathname}`;
}

function slugify(value: string): string {
	return value
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "-")
		.replace(/^-+|-+$/g, "")
		.replace(/--+/g, "-");
}

function suggestProviderId(endpoint: string): string {
	const url = new URL(addDefaultScheme(endpoint));
	const host = url.hostname.replace(/^www\./, "").replace(/^api\./, "");
	const hostSlug = slugify(`${host}${url.port ? `-${url.port}` : ""}`) || "provider";
	return `custom-${hostSlug}`;
}

function buildProbeUrl(baseUrl: string): string {
	const withSlash = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
	return new URL("models", withSlash).toString();
}

function resolveApiKeyForProbe(mode: ApiKeyMode, storedValue?: string): string | undefined {
	if (!storedValue || mode === "none") return undefined;
	if (mode === "literal") return storedValue;
	if (mode === "env") return process.env[storedValue]?.trim() || undefined;
	if (mode === "shell") {
		try {
			return execSync(storedValue, {
				encoding: "utf8",
				stdio: ["ignore", "pipe", "pipe"],
			}).trim();
		} catch {
			return undefined;
		}
	}
	return undefined;
}

function serializeApiKey(mode: ApiKeyMode, value?: string, style?: ProviderStyle): string | undefined {
	if (mode === "none") return style === "ollama" ? "ollama" : "dummy";
	if (!value) return undefined;
	// pi resolves an apiKey by prefix: "!cmd" runs a shell command, "$VAR" reads an
	// env var, anything else is a literal. See pi-ai resolve-config-value.
	if (mode === "shell") return value.startsWith("!") ? value : `!${value}`;
	if (mode === "env") return value.startsWith("$") ? value : `$${value}`;
	return value;
}

// OpenRouter returns per-model metadata (context length, modality, pricing,
// supported params). Plain OpenAI-compatible endpoints only return ids, so we
// detect OpenRouter by host and parse the richer payload when present.
function isOpenRouterEndpoint(baseUrl: string): boolean {
	try {
		return new URL(addDefaultScheme(baseUrl)).hostname.toLowerCase().endsWith("openrouter.ai");
	} catch {
		return false;
	}
}

// OpenRouter prices are per-token strings; pi's model `cost` is per-million tokens.
function perTokenToPerMillion(value: unknown): number | undefined {
	if (typeof value !== "string" && typeof value !== "number") return undefined;
	const parsed = typeof value === "number" ? value : Number.parseFloat(value);
	if (!Number.isFinite(parsed)) return undefined;
	// Round away float noise from the multiply (e.g. 0.00000003 * 1e6).
	return Math.round(parsed * 1_000_000 * 1e6) / 1e6;
}

const REASONING_PARAM_KEYS = ["reasoning", "include_reasoning", "reasoning_effort"];

function parseOpenRouterModelMeta(raw: any): OpenRouterModelMeta | undefined {
	if (!raw || typeof raw !== "object") return undefined;
	// OpenRouter entries always carry an `architecture` object; require it so we
	// don't misclassify arbitrary OpenAI-style payloads that happen to reuse ids.
	const architecture = raw.architecture;
	if (!architecture || typeof architecture !== "object") return undefined;

	const inputModalities: string[] = Array.isArray(architecture.input_modalities)
		? architecture.input_modalities.filter((m: unknown): m is string => typeof m === "string")
		: [];
	// pi only distinguishes text vs image, so collapse anything richer (file,
	// audio, video) down to modimage support.
	const modalities = ["text", ...(inputModalities.includes("image") ? ["image"] : [])];

	const topProvider = raw.top_provider && typeof raw.top_provider === "object" ? raw.top_provider : undefined;
	const contextLength =
		typeof raw.context_length === "number"
			? raw.context_length
			: typeof topProvider?.context_length === "number"
				? topProvider.context_length
				: undefined;
	const maxCompletionTokens = typeof topProvider?.max_completion_tokens === "number" ? topProvider.max_completion_tokens : undefined;

	const pricing = raw.pricing && typeof raw.pricing === "object" ? raw.pricing : undefined;
	const input = perTokenToPerMillion(pricing?.prompt);
	const output = perTokenToPerMillion(pricing?.completion);
	const cacheRead = perTokenToPerMillion(pricing?.input_cache_read);
	const cacheWrite = perTokenToPerMillion(pricing?.input_cache_write);

	// OpenRouter expresses long-context pricing as `overrides` keyed by a prompt
	// token threshold; pi models that as cost `tiers`.
	const tiers: ModelCost["tiers"] = [];
	if (Array.isArray(pricing?.overrides)) {
		for (const override of pricing.overrides) {
			const threshold = typeof override?.min_prompt_tokens === "number" ? override.min_prompt_tokens : undefined;
			const tierInput = perTokenToPerMillion(override?.prompt);
			const tierOutput = perTokenToPerMillion(override?.completion);
			if (threshold === undefined || (tierInput === undefined && tierOutput === undefined)) continue;
			tiers.push({
				inputTokensAbove: threshold,
				input: tierInput ?? input ?? 0,
				output: tierOutput ?? output ?? 0,
				// pi's schema requires all four rates as numbers on every tier, so
				// never let these resolve to undefined (JSON.stringify would drop
				// the key and the whole models.json would fail validation).
				cacheRead: perTokenToPerMillion(override?.input_cache_read) ?? cacheRead ?? 0,
				cacheWrite: perTokenToPerMillion(override?.input_cache_write) ?? cacheWrite ?? 0,
			});
		}
	}

	const cost: ModelCost | undefined =
		input !== undefined || output !== undefined
			? {
					input: input ?? 0,
					output: output ?? 0,
					cacheRead: cacheRead ?? 0,
					cacheWrite: cacheWrite ?? 0,
					...(tiers.length > 0 ? { tiers } : {}),
				}
			: undefined;

	const rawReasoning = raw.reasoning && typeof raw.reasoning === "object" ? raw.reasoning : undefined;

	return {
		name: typeof raw.name === "string" ? raw.name : undefined,
		contextLength,
		maxCompletionTokens,
		modalities,
		supportedParameters: Array.isArray(raw.supported_parameters)
			? raw.supported_parameters.filter((p: unknown): p is string => typeof p === "string")
			: [],
		cost,
		supportedEfforts: Array.isArray(rawReasoning?.supported_efforts)
			? rawReasoning.supported_efforts.filter((e: unknown): e is string => typeof e === "string")
			: undefined,
		mandatoryReasoning: typeof rawReasoning?.mandatory === "boolean" ? rawReasoning.mandatory : undefined,
	};
}

// ---- Reasoning / thinking-level mapping -----------------------------------
// OpenRouter advertises a per-model `reasoning.supported_efforts` array and a
// `mandatory` flag. Those effort names line up with pi's thinking levels, so we
// build a `thinkingLevelMap` that reflects what the provider actually accepts:
//   - a named level is supported when it appears in supported_efforts
//   - unsupported levels are nulled so pi hides them
//   - `off` is unsupported when reasoning is mandatory, otherwise it maps to the
//     provider value "none"
// Only levels we have real information about are written; anything else stays
// unset so pi's defaults apply.
type ThinkingLevelMap = Partial<Record<ReasoningCeiling, string | null>>;

function mapFromMeta(meta?: OpenRouterModelMeta): ThinkingLevelMap | undefined {
	const efforts = meta?.supportedEfforts;
	const mandatory = meta?.mandatoryReasoning;
	if ((!efforts || efforts.length === 0) && mandatory === undefined) return undefined;

	const supported = new Set(efforts ?? []);
	const map: ThinkingLevelMap = {};

	// `mandatory` means the model always thinks; the doc-level "none" effort also
	// disables it. OpenRouter is inconsistent about listing "none", so treat
	// mandatory:false as "can be turned off".
	map.off = mandatory === true ? null : "none";

	if (efforts && efforts.length > 0) {
		// "minimal" is a pi-only alias some providers expose directly.
		map.minimal = supported.has("minimal") ? "minimal" : null;
		for (const level of REASONING_LEVELS) {
			if (level === "off" || level === "minimal") continue;
			map[level] = supported.has(level) ? level : null;
		}
	}
	return map;
}

// Whether a thinking level is usable given a (possibly absent) level map.
// A null entry means unsupported; a string means supported; an omitted entry
// falls back to pi's default, which enables the standard levels only.
function mapLevelSupported(map: ThinkingLevelMap | undefined, level: ReasoningCeiling): boolean {
	if (level === "off") return true;
	const declared = map?.[level];
	if (declared === null) return false;
	if (typeof declared === "string") return true;
	return level !== "xhigh" && level !== "max";
}

// A model supports reasoning when OpenRouter advertises any reasoning parameter.
function metaSupportsReasoning(meta?: OpenRouterModelMeta): boolean {
	return !!meta && meta.supportedParameters.some((param) => REASONING_PARAM_KEYS.includes(param));
}

function buildMetaDescription(meta: OpenRouterModelMeta): string {
	const parts: string[] = [];
	if (meta.name) parts.push(meta.name);
	if (meta.contextLength) parts.push(`ctx ${Math.round(meta.contextLength / 1000)}k`);
	if (meta.modalities.includes("image")) parts.push("vision");
	if (metaSupportsReasoning(meta)) {
		const levels = REASONING_LEVELS.filter((l) => l !== "off" && (meta.supportedEfforts?.includes(l) ?? (l !== "xhigh" && l !== "max")));
		const off = meta.mandatoryReasoning === true ? "always" : "+off";
		parts.push(`reasoning ${levels.length > 0 ? levels.join("/") : "?"} ${off}`);
	}
	if (meta.maxCompletionTokens) parts.push(`max ${Math.round(meta.maxCompletionTokens / 1000)}k`);
	return parts.join(" • ");
}

async function probeOpenAIModels(baseUrl: string, apiKeyMode: ApiKeyMode, apiKeyValue?: string): Promise<ProbeItem[]> {
	const headers: Record<string, string> = {
		accept: "application/json",
		"accept-encoding": "identity",
	};
	const resolvedKey = resolveApiKeyForProbe(apiKeyMode, apiKeyValue);
	if (resolvedKey) {
		headers.authorization = `Bearer ${resolvedKey}`;
	}

	const response = await fetch(buildProbeUrl(baseUrl), { headers });
	if (!response.ok) {
		const body = await response.text().catch(() => "");
		throw new Error(`Probe failed (${response.status} ${response.statusText})${body ? `: ${body.slice(0, 200)}` : ""}`);
	}

	const json = (await response.json()) as any;
	const rawModels = Array.isArray(json) ? json : Array.isArray(json?.data) ? json.data : [];
	const isOpenRouter = isOpenRouterEndpoint(baseUrl);

	const byId = new Map<string, ProbeItem>();
	for (const item of rawModels) {
		const id = typeof item?.id === "string" ? item.id.trim() : "";
		if (!id) continue;
		const meta = isOpenRouter ? parseOpenRouterModelMeta(item) : undefined;
		const description = meta ? buildMetaDescription(meta) : undefined;
		byId.set(id, {
			value: id,
			label: id,
			description: description || undefined,
			meta,
		});
	}

	return Array.from(byId.values()).sort((a, b) => a.value.localeCompare(b.value));
}

// ---- OpenRouter catalog lookup --------------------------------------------
// Even when the configured endpoint is a proxy (not openrouter.ai), its model
// ids often match OpenRouter's catalog (e.g. "claude-fable-5" vs
// "anthropic/claude-fable-5"). We fetch the public catalog once and use it to
// fill in the real context window, modality, output cap and pricing.
const OPENROUTER_CATALOG_URL = "https://openrouter.ai/api/v1/models";

type OpenRouterCatalog = {
	byId: Map<string, OpenRouterModelMeta>;
	byBasename: Map<string, OpenRouterModelMeta[]>;
};

let openRouterCatalog: OpenRouterCatalog | null = null;
let openRouterCatalogPromise: Promise<OpenRouterCatalog | null> | null = null;

async function loadOpenRouterCatalog(): Promise<OpenRouterCatalog | null> {
	if (openRouterCatalog) return openRouterCatalog;
	if (openRouterCatalogPromise) return openRouterCatalogPromise;

	openRouterCatalogPromise = (async (): Promise<OpenRouterCatalog | null> => {
		try {
			const response = await fetch(OPENROUTER_CATALOG_URL, {
				headers: { accept: "application/json", "accept-encoding": "identity" },
			});
			if (!response.ok) return null;
			const json = (await response.json()) as any;
			const rawModels = Array.isArray(json?.data) ? json.data : [];

			const byId = new Map<string, OpenRouterModelMeta>();
			const byBasename = new Map<string, OpenRouterModelMeta[]>();
			for (const item of rawModels) {
				const id = typeof item?.id === "string" ? item.id.trim() : "";
				const meta = parseOpenRouterModelMeta(item);
				if (!id || !meta) continue;
				byId.set(id, meta);
				const basename = id.split("/").pop() ?? id;
				const bucket = byBasename.get(basename);
				if (bucket) bucket.push(meta);
				else byBasename.set(basename, [meta]);
			}
			openRouterCatalog = { byId, byBasename };
			return openRouterCatalog;
		} catch {
			// Offline / rate limited: metadata enrichment is best-effort only.
			return null;
		} finally {
			openRouterCatalogPromise = null;
		}
	})();

	return openRouterCatalogPromise;
}

// Exact id wins; otherwise a unique basename match (so bare proxy ids resolve).
// Ambiguous basenames are skipped rather than guessing the wrong model.
function matchCatalogMeta(catalog: OpenRouterCatalog, id: string): OpenRouterModelMeta | undefined {
	const exact = catalog.byId.get(id);
	if (exact) return exact;
	const basename = id.split("/").pop() ?? id;
	const bucket = catalog.byBasename.get(basename);
	return bucket && bucket.length === 1 ? bucket[0] : undefined;
}

// Attach catalog metadata to probe items that don't already have it, when the
// endpoint did not return rich data itself. Returns how many were enriched.
async function enrichProbeItemsWithCatalog(items: ProbeItem[], isOpenRouterEndpointUrl: boolean): Promise<number> {
	if (isOpenRouterEndpointUrl) return 0;
	const catalog = await loadOpenRouterCatalog();
	if (!catalog) return 0;
	let enriched = 0;
	for (const item of items) {
		if (item.meta) continue;
		const meta = matchCatalogMeta(catalog, item.value);
		if (!meta) continue;
		item.meta = meta;
		item.description = buildMetaDescription(meta) || item.description;
		enriched++;
	}
	return enriched;
}

function normalizeSelectItems(items: Array<string | SelectItem>): SelectItem[] {
	return items.map((item) => (typeof item === "string" ? { value: item, label: item } : item));
}

async function selectOne(
	ctx: CommandContext,
	title: string,
	items: Array<string | SelectItem>,
	options?: { initialIndex?: number },
): Promise<string | null> {
	const normalizedItems = normalizeSelectItems(items);
	if (normalizedItems.length === 0) return null;

	return await ctx.ui.custom<string | null>((tui, theme, _kb, done) => {
		let cursor = Math.max(0, Math.min(options?.initialIndex ?? 0, normalizedItems.length - 1));
		let query = "";
		let cachedLines: string[] | undefined;
		const maxVisible = 12;

		function getVisibleItems() {
			const lowerQuery = query.trim().toLowerCase();
			if (!lowerQuery) return normalizedItems;
			return normalizedItems.filter((item) => {
				const haystack = `${item.label} ${item.suffix ?? ""} ${item.description ?? ""} ${item.searchText ?? ""}`.toLowerCase();
				return haystack.includes(lowerQuery);
			});
		}

		function refresh() {
			const visibleItems = getVisibleItems();
			if (visibleItems.length === 0) cursor = 0;
			else if (cursor >= visibleItems.length) cursor = visibleItems.length - 1;
			cachedLines = undefined;
			tui.requestRender();
		}

		return {
			render(width: number) {
				if (cachedLines) return cachedLines;

				const visibleItems = getVisibleItems();
				const safeWidth = Math.max(10, width);
				const lines: string[] = [];
				const add = (line = "") => lines.push(truncateToWidth(line, safeWidth));
				const border = theme.fg("accent", "─".repeat(safeWidth));

				add(border);
				add(` ${theme.fg("accent", theme.bold(title))}`);
				add(` ${theme.fg("text", `Search: ${query || "-"}`)}`);
				add();

				if (visibleItems.length === 0) {
					add(theme.fg("warning", " No matches."));
				} else {
					const start = Math.max(0, Math.min(cursor - Math.floor(maxVisible / 2), Math.max(0, visibleItems.length - maxVisible)));
					const end = Math.min(visibleItems.length, start + maxVisible);

					for (let i = start; i < end; i++) {
						const item = visibleItems[i];
						const active = i === cursor;
						const prefix = active ? theme.fg("accent", "> ") : "  ";
						const label = active ? theme.fg("accent", item.label) : theme.fg("text", item.label);
						const suffix = item.suffix ? theme.fg("dim", item.suffix) : "";
						add(`${prefix}${label}${suffix}`);
						if (item.description) {
							for (const line of item.description.split("\n")) {
								add(`   ${theme.fg("muted", line)}`);
							}
						}
					}

					if (visibleItems.length > maxVisible) {
						add();
						add(theme.fg("dim", ` ${start + 1}-${end} of ${visibleItems.length}`));
					}
				}

				add();
				add(theme.fg("dim", " Type to search • ↑↓ move (wraps) • enter confirm • backspace delete • esc cancel"));
				add(border);

				cachedLines = lines;
				return lines;
			},
			invalidate() {
				cachedLines = undefined;
			},
			handleInput(data: string) {
				const visibleItems = getVisibleItems();
				if (matchesKey(data, Key.up)) {
					if (visibleItems.length === 0) return;
					cursor = cursor === 0 ? visibleItems.length - 1 : cursor - 1;
					refresh();
					return;
				}
				if (matchesKey(data, Key.down)) {
					if (visibleItems.length === 0) return;
					cursor = cursor === visibleItems.length - 1 ? 0 : cursor + 1;
					refresh();
					return;
				}
				if (matchesKey(data, Key.enter)) {
					const item = visibleItems[cursor];
					done(item?.value ?? null);
					return;
				}
				if (matchesKey(data, Key.escape)) {
					done(null);
					return;
				}
				if (data === "\u007f" || data === "\b") {
					if (query.length > 0) {
						query = query.slice(0, -1);
						refresh();
					}
					return;
				}
				if (data >= " " && data !== "\u001b" && data !== "\r" && data !== "\n") {
					query += data;
					cursor = 0;
					refresh();
				}
			},
		};
	});
}

async function pickMany(
	ctx: CommandContext,
	title: string,
	items: ProbeItem[],
): Promise<ProbeItem[] | null> {
	return await ctx.ui.custom<ProbeItem[] | null>((tui, theme, _kb, done) => {
		let cursor = 0;
		let query = "";
		const selected = new Set<string>();
		let cachedLines: string[] | undefined;
		const maxVisible = 12;

		function getVisibleItems() {
			const lowerQuery = query.trim().toLowerCase();
			if (!lowerQuery) return items;
			return items.filter((item) => {
				const haystack = `${item.label} ${item.value} ${item.description ?? ""}`.toLowerCase();
				return haystack.includes(lowerQuery);
			});
		}

		function refresh() {
			const visibleItems = getVisibleItems();
			if (visibleItems.length === 0) cursor = 0;
			else if (cursor >= visibleItems.length) cursor = visibleItems.length - 1;
			cachedLines = undefined;
			tui.requestRender();
		}

		return {
			render(width: number) {
				if (cachedLines) return cachedLines;

				const visibleItems = getVisibleItems();
				const safeWidth = Math.max(10, width);
				const lines: string[] = [];
				const add = (line = "") => lines.push(truncateToWidth(line, safeWidth));
				const border = theme.fg("accent", "─".repeat(safeWidth));

				add(border);
				add(` ${theme.fg("accent", theme.bold(title))}`);
				add(` ${theme.fg("text", `Search: ${query || "-"}`)}`);
				add(` ${theme.fg("muted", `${selected.size} selected • ${visibleItems.length}/${items.length} shown`)}`);
				add();

				if (visibleItems.length === 0) {
					add(theme.fg("warning", " No matching models."));
				} else {
					const start = Math.max(0, Math.min(cursor - Math.floor(maxVisible / 2), Math.max(0, visibleItems.length - maxVisible)));
					const end = Math.min(visibleItems.length, start + maxVisible);

					for (let i = start; i < end; i++) {
						const item = visibleItems[i];
						const active = i === cursor;
						const checked = selected.has(item.value);
						const prefix = active ? theme.fg("accent", "> ") : "  ";
						const box = checked ? theme.fg("success", "[x]") : theme.fg("muted", "[ ]");
						const label = active ? theme.fg("accent", item.label) : theme.fg("text", item.label);
						add(`${prefix}${box} ${label}`);
						if (item.description) {
							add(`     ${theme.fg("muted", item.description)}`);
						}
					}

					if (visibleItems.length > maxVisible) {
						add();
						add(theme.fg("dim", ` ${start + 1}-${end} of ${visibleItems.length}`));
					}
				}

				add();
				add(theme.fg("dim", " Type to search • ↑↓ move (wraps) • space toggle • ctrl+a toggle all • enter confirm • backspace delete • esc cancel"));
				if (selected.size === 0) {
					add(theme.fg("warning", " Select at least one model before confirming."));
				}
				add(border);

				cachedLines = lines;
				return lines;
			},
			invalidate() {
				cachedLines = undefined;
			},
			handleInput(data: string) {
				const visibleItems = getVisibleItems();
				if (matchesKey(data, Key.up)) {
					if (visibleItems.length === 0) return;
					cursor = cursor === 0 ? visibleItems.length - 1 : cursor - 1;
					refresh();
					return;
				}
				if (matchesKey(data, Key.down)) {
					if (visibleItems.length === 0) return;
					cursor = cursor === visibleItems.length - 1 ? 0 : cursor + 1;
					refresh();
					return;
				}
				if (matchesKey(data, Key.ctrl("a"))) {
					// Toggle every model in the current visible set (search-filtered):
					// any unselected visible model -> select all visible; else clear all visible.
					const anyUnselected = visibleItems.some((item) => !selected.has(item.value));
					for (const item of visibleItems) {
						if (anyUnselected) selected.add(item.value);
						else selected.delete(item.value);
					}
					refresh();
					return;
				}
				if (data === " ") {
					const value = visibleItems[cursor]?.value;
					if (!value) return;
					if (selected.has(value)) selected.delete(value);
					else selected.add(value);
					refresh();
					return;
				}
				if (matchesKey(data, Key.enter)) {
					if (selected.size > 0) {
						// Preserve the probe order (and its metadata) for the selection.
						done(items.filter((item) => selected.has(item.value)));
					}
					return;
				}
				if (matchesKey(data, Key.escape)) {
					done(null);
					return;
				}
				if (data === "\u007f" || data === "\b") {
					if (query.length > 0) {
						query = query.slice(0, -1);
						refresh();
					}
					return;
				}
				if (data >= " " && data !== "\u001b" && data !== "\r" && data !== "\n") {
					query += data;
					cursor = 0;
					refresh();
				}
			},
		};
	});
}

async function promptApiKey(
	ctx: CommandContext,
): Promise<{ mode: ApiKeyMode; value?: string } | null> {
	const choice = await selectOne(ctx, "API key", [
		{ value: "literal", label: "API key", description: "Stored verbatim in the active models config" },
		{ value: "none", label: "None", description: "No key; a placeholder is written so the provider still loads" },
	]);
	if (!choice) return null;
	if (choice === "none") return { mode: "none" };

	const value = await ctx.ui.input("API key", "saved directly in the active models config");
	if (value === undefined) return null;
	const trimmed = value.trim();
	if (!trimmed) return { mode: "none" };
	return { mode: "literal", value: trimmed };
}

function reasoningLabel(level: ReasoningCeiling): string {
	if (level === "off") return "Off - no reasoning";
	if (level === "xhigh") return "xhigh - maximum (only if the model supports it)";
	return `${level} - cap reasoning at ${level}`;
}

// Prompts for a reasoning ceiling. Returns null if cancelled.
async function promptReasoning(ctx: CommandContext, current?: ReasoningCeiling): Promise<ReasoningCeiling | null> {
	const items: SelectItem[] = REASONING_LEVELS.map((level) => ({
		value: level,
		label: reasoningLabel(level),
	}));
	const initialIndex = current ? REASONING_LEVELS.indexOf(current) : 0;
	const choice = await selectOne(ctx, "Reasoning", items, { initialIndex: Math.max(0, initialIndex) });
	return (choice as ReasoningCeiling | null) ?? null;
}

// When a model is capped at xhigh, some providers name that level differently
// (e.g. "max"). Offer an optional override for the provider-facing string.
async function promptXhighProviderString(ctx: CommandContext, current?: string): Promise<string | undefined> {
	const value = await ctx.ui.input(
		"xhigh provider value (blank = xhigh)",
		current && current !== "xhigh" ? `current: ${current}` : 'e.g. max (leave blank to send "xhigh")',
	);
	if (value === undefined) return undefined;
	const trimmed = value.trim();
	return trimmed || undefined;
}

async function promptVision(ctx: CommandContext, current?: boolean): Promise<boolean | null> {
	const choice = await selectOne(ctx, "Image input (vision)", [
		{ value: "yes", label: "Yes - send text + images", description: "Sets input: [text, image]" },
		{ value: "no", label: "No - text only", description: "Sets input: [text]" },
	], { initialIndex: current === false ? 1 : 0 });
	if (!choice) return null;
	return choice === "yes";
}

// Prompts for a context window size in tokens. Returns:
//   number  -> set/replace contextWindow
//   0       -> clear contextWindow (user typed 0)
//   null    -> cancelled, leave unchanged
async function promptContextWindow(ctx: CommandContext, current?: number): Promise<number | null> {
	const value = await ctx.ui.input(
		"Context window (tokens)",
		current ? `current: ${current} (blank = keep, 0 = clear)` : "e.g. 128000 (blank = unset)",
	);
	if (value === undefined) return null;
	const trimmed = value.trim();
	if (!trimmed) return null;
	const parsed = Number.parseInt(trimmed.replace(/[_,]/g, ""), 10);
	if (!Number.isFinite(parsed) || parsed < 0) {
		ctx.ui.notify("Enter a whole number of tokens (0 to clear).", "warning");
		return null;
	}
	return parsed;
}

// Prompts for max output tokens. Same return contract as promptContextWindow:
// number to set, 0 to clear, null to leave unchanged.
async function promptMaxTokens(ctx: CommandContext, current?: number): Promise<number | null> {
	const value = await ctx.ui.input(
		"Max output tokens",
		current ? `current: ${current} (blank = keep, 0 = clear)` : "e.g. 8192 (blank = unset)",
	);
	if (value === undefined) return null;
	const trimmed = value.trim();
	if (!trimmed) return null;
	const parsed = Number.parseInt(trimmed.replace(/[_,]/g, ""), 10);
	if (!Number.isFinite(parsed) || parsed < 0) {
		ctx.ui.notify("Enter a whole number of tokens (0 to clear).", "warning");
		return null;
	}
	return parsed;
}

// Read the reasoning ceiling + vision flags already stored on a model entry,
// mirroring pi's getSupportedThinkingLevels so edit defaults match reality.
// Extended levels (xhigh/max) count only when explicitly mapped; a mapped level
// set to null is treated as unavailable.
function readModelOptions(model: any): ModelOptions {
	const vision = Array.isArray(model?.input) ? model.input.includes("image") : true;
	const contextWindow = typeof model?.contextWindow === "number" ? model.contextWindow : undefined;
	const maxTokens = typeof model?.maxTokens === "number" ? model.maxTokens : undefined;
	const name = typeof model?.name === "string" ? model.name : undefined;
	const cost = model?.cost && typeof model.cost === "object" ? (model.cost as ModelCost) : undefined;
	if (!model || model.reasoning !== true) {
		return { reasoning: "off", vision, contextWindow, maxTokens, name, cost };
	}

	const map = model.thinkingLevelMap;
	let ceiling: ReasoningCeiling = "high";
	if (map && typeof map === "object") {
		for (let i = REASONING_LEVELS.length - 1; i >= 1; i--) {
			const level = REASONING_LEVELS[i];
			if (!mapLevelSupported(map, level)) continue;
			ceiling = level;
			break;
		}
	}
	return { reasoning: ceiling, vision, contextWindow, maxTokens, name, cost };
}

function readXhighProviderString(model: any): string | undefined {
	const v = model?.thinkingLevelMap?.xhigh;
	return typeof v === "string" ? v : undefined;
}

// Extract provider-declared levels already stored on a model so editing the
// reasoning ceiling preserves per-level provider values (and explicit nulls).
function declaredMapFromModel(model: any): ThinkingLevelMap | undefined {
	const m = model?.thinkingLevelMap;
	if (!m || typeof m !== "object") return undefined;
	const out: ThinkingLevelMap = {};
	let any = false;
	for (const level of REASONING_LEVELS) {
		const v = m[level];
		if (v === null) {
			out[level] = null;
			any = true;
		} else if (typeof v === "string") {
			out[level] = v;
			any = true;
		}
	}
	return any ? out : undefined;
}

async function promptModelIdsOneByOne(
	ctx: CommandContext,
	style: ProviderStyle,
): Promise<string[] | null> {
	const modelIds: string[] = [];
	const firstPlaceholder =
		style === "anthropic"
			? "e.g. claude-sonnet-4-5 (blank to finish)"
			: style === "ollama"
				? "e.g. llama3.1:8b or qwen2.5-coder:7b (blank to finish)"
				: "e.g. gpt-4o-mini or qwen/qwen3-coder (blank to finish)";
	const nextPlaceholder =
		style === "anthropic"
			? "another Anthropic-style model id (blank to finish)"
			: style === "ollama"
				? "another Ollama model id (blank to finish)"
				: "another OpenAI-style model id (blank to finish)";

	while (true) {
		const value = await ctx.ui.input(modelIds.length === 0 ? "Model id" : "Add another model id", modelIds.length === 0 ? firstPlaceholder : nextPlaceholder);
		if (value === undefined) return null;
		const trimmed = value.trim();
		if (!trimmed) {
			if (modelIds.length === 0) {
				ctx.ui.notify("Add at least one model.", "warning");
				continue;
			}
			return modelIds;
		}
		if (modelIds.includes(trimmed)) {
			ctx.ui.notify(`Model already added: ${trimmed}`, "warning");
			continue;
		}
		modelIds.push(trimmed);
	}
}

// Apply reasoning settings to an entry in place, preserving other fields.
//
// `ceiling` is the wizard's cap (off → max). `levelMap` is an optional
// provider-declared map (from OpenRouter's `supported_efforts`) describing which
// levels the model actually supports; when present it wins over pi's defaults,
// so unsupported levels are hidden instead of being offered and rejected.
//
// pi semantics (see getSupportedThinkingLevels): off/minimal/low/medium/high are
// available by default when reasoning is true; xhigh/max are available ONLY when
// explicitly mapped; any level set to null is removed.
function applyReasoning(
	entry: any,
	ceiling: ReasoningCeiling,
	providerStringOverride?: string,
	levelMap?: ThinkingLevelMap,
) {
	if (ceiling === "off") {
		delete entry.reasoning;
		delete entry.thinkingLevelMap;
		return;
	}
	entry.reasoning = true;
	const ceilingIndex = REASONING_LEVELS.indexOf(ceiling);
	const map: Record<string, string | null> = {};

	// Reasoning that cannot be disabled must null out `off`, otherwise pi would
	// offer an "off" toggle the provider ignores.
	if (levelMap && levelMap.off === null) map.off = null;

	for (const level of REASONING_LEVELS) {
		if (level === "off") continue;
		const index = REASONING_LEVELS.indexOf(level);
		const declared = levelMap?.[level];

		const defaultSupported = level !== "xhigh" && level !== "max";
		// declared null = provider says unsupported; string = supported with that
		// provider value; undefined = not mentioned. For an unmentioned level we use
		// pi's default (standard levels only) but still unlock an extended level when
		// the user explicitly picked it as the ceiling.
		const supported =
			declared === null ? false : typeof declared === "string" ? true : defaultSupported || index <= ceilingIndex;
		if (!supported || index > ceilingIndex) {
			map[level] = null;
			continue;
		}

		// Only extended levels need an explicit provider value; pi already maps the
		// standard levels to their own names. A declared provider string is kept as
		// given so non-default names (e.g. a custom "low") survive.
		if (typeof declared === "string" && level !== "xhigh" && level !== "max") {
			map[level] = declared;
		} else if (level === "xhigh") {
			map.xhigh = typeof declared === "string" ? declared : providerStringOverride?.trim() || "xhigh";
		} else if (level === "max") {
			map.max = typeof declared === "string" ? declared : "max";
		}
	}
	if (Object.keys(map).length > 0) entry.thinkingLevelMap = map;
	else delete entry.thinkingLevelMap;
}

function buildModelEntry(id: string, opts: ModelOptions, providerStringOverride?: string): any {
	const entry: any = {
		id,
		// Default to text+image so pi forwards images upstream. Without this,
		// custom models default to text-only and images are silently dropped.
		input: opts.vision ? ["text", "image"] : ["text"],
	};

	if (opts.name) entry.name = opts.name;

	if (typeof opts.contextWindow === "number" && opts.contextWindow > 0) {
		entry.contextWindow = opts.contextWindow;
	}
	if (typeof opts.maxTokens === "number" && opts.maxTokens > 0) {
		entry.maxTokens = opts.maxTokens;
	}
	if (opts.cost) entry.cost = opts.cost;

	applyReasoning(entry, opts.reasoning, providerStringOverride, opts.reasoningMap);
	return entry;
}

// Derive wizard defaults from OpenRouter metadata so a freshly added model
// carries the right context window, modality, output cap, pricing and reasoning
// state instead of blanket text+image/xhigh guesses.
function optionsFromMeta(meta: OpenRouterModelMeta | undefined): ModelOptions {
	// Without metadata (plain OpenAI-compatible endpoint) keep the historical
	// permissive default: assume vision + reasoning.
	if (!meta) return { reasoning: "xhigh", vision: true };

	const reasoningMap = mapFromMeta(meta);
	// Cap new models at the highest level the provider advertises, so the default
	// thinking level is actually accepted instead of silently clamped.
	let ceiling: ReasoningCeiling = "off";
	if (metaSupportsReasoning(meta)) {
		const highest = [...REASONING_LEVELS].reverse().find((level) => mapLevelSupported(reasoningMap, level));
		ceiling = highest ?? "high";
	}

	return {
		reasoning: ceiling,
		vision: meta.modalities.includes("image"),
		contextWindow: meta.contextLength,
		maxTokens: meta.maxCompletionTokens,
		name: meta.name,
		cost: meta.cost,
		reasoningMap,
	};
}

function buildProviderConfig(
	style: ProviderStyle,
	api: ProviderApi,
	baseUrl: string,
	apiKey: { mode: ApiKeyMode; value?: string },
	models: Array<{ id: string; opts: ModelOptions }>,
	providerStringOverride?: string,
) {
	const serializedApiKey = serializeApiKey(apiKey.mode, apiKey.value, style);
	const config: any = {
		baseUrl,
		api,
		...(serializedApiKey ? { apiKey: serializedApiKey } : {}),
		models: models.map(({ id, opts }) => buildModelEntry(id, opts, providerStringOverride)),
	};

	if (style === "ollama") {
		if (!config.apiKey) config.apiKey = "ollama";
		config.compat = {
			supportsDeveloperRole: false,
			supportsReasoningEffort: false,
		};
	}

	return config;
}

function describeProvider(providerId: string, provider: any): string {
	const modelCount = Array.isArray(provider?.models) ? provider.models.length : 0;
	const endpoint = typeof provider?.baseUrl === "string" ? provider.baseUrl : "(no baseUrl)";
	const api = typeof provider?.api === "string" ? provider.api : "(no api)";
	return `${providerId}\n${api} • ${modelCount} model${modelCount === 1 ? "" : "s"}\n${endpoint}`;
}

function describeProviderInline(providerId: string, provider: any): { label: string; suffix: string; searchText: string } {
	const modelCount = Array.isArray(provider?.models) ? provider.models.length : 0;
	const endpoint = typeof provider?.baseUrl === "string" ? provider.baseUrl : "(no baseUrl)";
	const api = typeof provider?.api === "string" ? provider.api : "(no api)";
	const suffix = ` • ${api} • ${endpoint} • ${modelCount} model${modelCount === 1 ? "" : "s"}`;
	return {
		label: providerId,
		suffix,
		searchText: `${providerId} ${api} ${endpoint} ${modelCount}`,
	};
}

function providerModelItems(provider: any): SelectItem[] {
	const models = Array.isArray(provider?.models) ? provider.models : [];
	return models
		.map((model: any) => {
			const id = typeof model === "string" ? model.trim() : typeof model?.id === "string" ? model.id.trim() : "";
			if (!id) return null;

			const details: string[] = [];
			if (model && typeof model === "object") {
				if (model.reasoning === true) {
					const opts = readModelOptions(model);
					details.push(`reasoning:${opts.reasoning}`);
				}
				if (Array.isArray(model.input) && model.input.includes("image")) details.push("vision");
				if (typeof model.contextWindow === "number") details.push(`context ${model.contextWindow}`);
				if (typeof model.maxTokens === "number") details.push(`max ${model.maxTokens}`);
			}

			return {
				value: id,
				label: id,
				suffix: details.length > 0 ? ` • ${details.join(" • ")}` : "",
				searchText: `${id} ${details.join(" ")}`,
			};
		})
		.filter((item: SelectItem | null): item is SelectItem => item !== null);
}

function normalizeStoredEndpoint(provider: any): string {
	const endpoint = typeof provider?.baseUrl === "string" ? provider.baseUrl.trim() : "";
	if (!endpoint) return "";
	const api: ProviderApi = provider?.api === "anthropic-messages" ? "anthropic-messages" : "openai-completions";
	try {
		return normalizeEndpoint(endpoint, api);
	} catch {
		return endpoint.replace(/\/+$/, "");
	}
}

function findProvidersByEndpoint(config: ModelsConfig, endpoint: string): string[] {
	return Object.entries(config.providers ?? {})
		.filter(([, provider]) => normalizeStoredEndpoint(provider) === endpoint)
		.map(([providerId]) => providerId)
		.sort((a, b) => a.localeCompare(b));
}

async function editProviderFlow(ctx: CommandContext) {
	let cursor = 0;

	while (true) {
		let config: ModelsConfig;
		try {
			config = loadModelsConfig();
		} catch (error) {
			ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
			return;
		}

		config.providers ||= {};
		const providerIds = Object.keys(config.providers).sort((a, b) => a.localeCompare(b));
		if (providerIds.length === 0) {
			ctx.ui.notify(`No providers found in ${MODELS_JSON_PATH}`, "warning");
			return;
		}

		const choice = await selectOne(
			ctx,
			"Edit provider",
			providerIds.map((providerId) => {
				const inline = describeProviderInline(providerId, config.providers?.[providerId]);
				return {
					value: providerId,
					label: inline.label,
					suffix: inline.suffix,
					searchText: inline.searchText,
				};
			}),
			{ initialIndex: Math.min(cursor, providerIds.length - 1) },
		);
		if (!choice) return;

		cursor = providerIds.indexOf(choice);
		await editSingleProvider(ctx, choice);
	}
}

// Per-provider action menu. Returns when the user backs out to the provider list.
async function editSingleProvider(ctx: CommandContext, providerId: string) {
	while (true) {
		let config: ModelsConfig;
		try {
			config = loadModelsConfig();
		} catch (error) {
			ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
			return;
		}
		const provider = config.providers?.[providerId];
		if (!provider) {
			ctx.ui.notify(`Provider "${providerId}" no longer exists.`, "warning");
			return;
		}

		const modelCount = Array.isArray(provider.models) ? provider.models.length : 0;
		const action = await selectOne(ctx, `Edit ${providerId}`, [
			{ value: "probe", label: "Re-probe for new models", description: "Query /models again and add ones not already configured" },
			{ value: "sync", label: "Sync metadata from OpenRouter", description: "Fill context window, max tokens, vision, cost (blank fields only) and refresh reasoning levels from OpenRouter" },
			{ value: "context", label: "Set context window (all models)", description: `Apply one contextWindow to all ${modelCount} model${modelCount === 1 ? "" : "s"}` },
			{ value: "models", label: "Edit per model", description: `${modelCount} model${modelCount === 1 ? "" : "s"} — reasoning, vision, context, max tokens, headers, delete` },
			{ value: "add", label: "Add models manually", description: "Type model ids to add" },
			{ value: "rename", label: "Rename provider", description: "Change the provider name in the models config" },
			{ value: "back", label: "Back", description: "Return to the provider list" },
		]);
		if (!action || action === "back") return;

		if (action === "models") {
			await editProviderModels(ctx, providerId);
		} else if (action === "probe") {
			await reprobeProvider(ctx, providerId);
		} else if (action === "sync") {
			await syncProviderMetadata(ctx, providerId);
		} else if (action === "context") {
			await setProviderContextWindow(ctx, providerId);
		} else if (action === "add") {
			await addModelsToProvider(ctx, providerId);
		} else if (action === "rename") {
			// Reassign so the menu keeps editing the same provider under its new name.
			const renamed = await renameProvider(ctx, providerId);
			if (renamed) providerId = renamed;
		}
	}
}

// Rename a provider's key in the models config, preserving its config and original
// position in the file. Returns the new id on success, or null if cancelled,
// unchanged, or rejected. Only touches the models config — a currently-selected model
// pinned to the old provider id must be reselected via /model afterwards.
async function renameProvider(ctx: CommandContext, providerId: string): Promise<string | null> {
	let config: ModelsConfig;
	try {
		config = loadModelsConfig();
	} catch (error) {
		ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return null;
	}
	config.providers ||= {};
	if (!config.providers[providerId]) {
		ctx.ui.notify(`Provider "${providerId}" no longer exists.`, "warning");
		return null;
	}

	const input = await ctx.ui.input("Rename provider", `current: ${providerId}`);
	if (input === undefined) return null;
	// Slugify so names stay consistent with the Add flow.
	const newId = slugify(input.trim());
	if (!newId || newId === providerId) return null;

	if (config.providers[newId]) {
		ctx.ui.notify(`Provider "${newId}" already exists. Choose a different name.`, "warning");
		return null;
	}

	if (BUILTIN_PROVIDER_IDS.has(newId)) {
		const ok = await ctx.ui.confirm(
			"Override built-in provider?",
			`"${newId}" matches a built-in provider id. Saving this will override that provider in the active models config. Continue?`,
		);
		if (!ok) return null;
	}

	// Rebuild key-by-key so the renamed entry keeps its position rather than
	// jumping to the bottom (a naive delete + reassign would reorder it).
	const rebuilt: Record<string, any> = {};
	for (const [key, value] of Object.entries(config.providers)) {
		rebuilt[key === providerId ? newId : key] = value;
	}
	config.providers = rebuilt;

	try {
		saveModelsConfig(config);
	} catch (error) {
		ctx.ui.notify(`Could not write ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return null;
	}

	ctx.ui.notify(`Renamed "${providerId}" → "${newId}".`, "info");
	return newId;
}

// Apply a single contextWindow value to every model in the provider, preserving
// each model's reasoning/vision config. A value of 0 clears it from all models.
async function setProviderContextWindow(ctx: CommandContext, providerId: string) {
	let provider: any;
	try {
		provider = loadModelsConfig().providers?.[providerId];
	} catch (error) {
		ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return;
	}
	const models = Array.isArray(provider?.models) ? provider.models : [];
	if (models.length === 0) {
		ctx.ui.notify(`Provider "${providerId}" has no models.`, "warning");
		return;
	}

	// Prefill with the shared value if every model already agrees, else blank.
	const windows = models.map((m: any) => (typeof m?.contextWindow === "number" ? m.contextWindow : undefined));
	const shared = windows.every((w: number | undefined) => w === windows[0]) ? windows[0] : undefined;

	const result = await promptContextWindow(ctx, shared);
	if (result === null) return;

	const saved = await mutateProvider(ctx, providerId, (p) => {
		const list = Array.isArray(p.models) ? p.models : [];
		for (const m of list) {
			const opts = readModelOptions(m);
			opts.contextWindow = result === 0 ? undefined : result;
			const rebuilt = buildModelEntry(modelIdOf(m), opts, readXhighProviderString(m));
			Object.assign(m, rebuilt);
			if (result === 0) delete m.contextWindow;
		}
		return true;
	});
	if (saved) {
		ctx.ui.notify(
			result === 0
				? `Cleared context window on all ${models.length} model${models.length === 1 ? "" : "s"}.`
				: `Set context window ${result} on all ${models.length} model${models.length === 1 ? "" : "s"}.`,
			"info",
		);
	}
}

// Load config, hand the provider to a mutator, and save if it returns true.
async function mutateProvider(
	ctx: CommandContext,
	providerId: string,
	mutate: (provider: any) => boolean | Promise<boolean>,
): Promise<boolean> {
	let config: ModelsConfig;
	try {
		config = loadModelsConfig();
	} catch (error) {
		ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return false;
	}
	const provider = config.providers?.[providerId];
	if (!provider) {
		ctx.ui.notify(`Provider "${providerId}" no longer exists.`, "warning");
		return false;
	}

	const changed = await mutate(provider);
	if (!changed) return false;

	try {
		saveModelsConfig(config);
	} catch (error) {
		ctx.ui.notify(`Could not write ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return false;
	}
	return true;
}

// Pick a model, then a field to edit. Each edit mutates one field in place so
// other fields (headers, overrides, cost) are preserved.
async function editProviderModels(ctx: CommandContext, providerId: string) {
	let cursor = 0;
	while (true) {
		let provider: any;
		try {
			provider = loadModelsConfig().providers?.[providerId];
		} catch (error) {
			ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
			return;
		}
		const modelItems = providerModelItems(provider);
		if (modelItems.length === 0) {
			ctx.ui.notify(`Provider "${providerId}" has no models.`, "warning");
			return;
		}

		const choice = await selectOne(ctx, `Edit model in ${providerId}`, modelItems, {
			initialIndex: Math.min(cursor, modelItems.length - 1),
		});
		if (!choice) return;
		cursor = modelItems.findIndex((item) => item.value === choice);

		const deleted = await editSingleModel(ctx, providerId, choice);
		if (deleted) cursor = Math.max(0, cursor - 1);
	}
}

// Field-picker for one model. Returns true if the model was deleted (so the
// caller can adjust its cursor).
async function editSingleModel(ctx: CommandContext, providerId: string, modelId: string): Promise<boolean> {
	while (true) {
		let model: any;
		try {
			model = findModel(loadModelsConfig().providers?.[providerId], modelId);
		} catch (error) {
			ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
			return false;
		}
		if (!model) {
			ctx.ui.notify(`Model "${modelId}" no longer exists.`, "warning");
			return false;
		}

		const opts = readModelOptions(model);
		const ctxWin = typeof model.contextWindow === "number" ? model.contextWindow : "unset";
		const maxTok = typeof model.maxTokens === "number" ? model.maxTokens : "unset";
		const hasHeaders = model.headers && Object.keys(model.headers).length > 0;
		const override = model.baseUrl || model.api ? "set" : "unset";

		const field = await selectOne(ctx, `Edit ${modelId}`, [
			{ value: "reasoning", label: "Reasoning", suffix: ` • ${opts.reasoning}`, description: "Set the reasoning ceiling (off → max)" },
			{ value: "vision", label: "Vision", suffix: ` • ${opts.vision ? "on" : "off"}`, description: "Toggle image input (text+image vs text-only)" },
			{ value: "context", label: "Context window", suffix: ` • ${ctxWin}`, description: "Max context tokens for this model" },
			{ value: "maxtokens", label: "Max output tokens", suffix: ` • ${maxTok}`, description: "Max tokens this model may generate" },
			{ value: "override", label: "Headers / endpoint override", suffix: ` • ${hasHeaders ? "headers" : override}`, description: "Per-model HTTP headers and api/baseUrl override" },
			{ value: "delete", label: "Delete this model", description: "Remove this model from the provider" },
			{ value: "back", label: "Back", description: "Return to the model list" },
		]);
		if (!field || field === "back") return false;

		if (field === "reasoning") {
			const reasoning = await promptReasoning(ctx, opts.reasoning);
			if (reasoning === null) continue;
			let xhigh: string | undefined;
			if (reasoning === "xhigh") xhigh = await promptXhighProviderString(ctx, readXhighProviderString(model));
			const declared = declaredMapFromModel(model);
			await mutateModel(ctx, providerId, modelId, (m) => applyReasoning(m, reasoning, xhigh, declared));
		} else if (field === "vision") {
			const vision = await promptVision(ctx, opts.vision);
			if (vision === null) continue;
			await mutateModel(ctx, providerId, modelId, (m) => { m.input = vision ? ["text", "image"] : ["text"]; });
		} else if (field === "context") {
			const result = await promptContextWindow(ctx, typeof model.contextWindow === "number" ? model.contextWindow : undefined);
			if (result === null) continue;
			await mutateModel(ctx, providerId, modelId, (m) => { if (result === 0) delete m.contextWindow; else m.contextWindow = result; });
		} else if (field === "maxtokens") {
			const result = await promptMaxTokens(ctx, typeof model.maxTokens === "number" ? model.maxTokens : undefined);
			if (result === null) continue;
			await mutateModel(ctx, providerId, modelId, (m) => { if (result === 0) delete m.maxTokens; else m.maxTokens = result; });
		} else if (field === "override") {
			await editModelOverride(ctx, providerId, modelId);
		} else if (field === "delete") {
			const ok = await ctx.ui.confirm("Delete model?", `Remove "${modelId}" from "${providerId}"?`);
			if (!ok) continue;
			const saved = await mutateProvider(ctx, providerId, (p) => {
				const models = Array.isArray(p.models) ? p.models : [];
				const index = models.findIndex((m: any) => modelIdOf(m) === modelId);
				if (index === -1) return false;
				models.splice(index, 1);
				return true;
			});
			if (saved) ctx.ui.notify(`Deleted "${modelId}".`, "info");
			return true;
		}
	}
}

// Mutate a single model entry in place and save.
async function mutateModel(ctx: CommandContext, providerId: string, modelId: string, mutate: (model: any) => void): Promise<boolean> {
	return mutateProvider(ctx, providerId, (p) => {
		const models = Array.isArray(p.models) ? p.models : [];
		const index = models.findIndex((m: any) => modelIdOf(m) === modelId);
		if (index === -1) return false;
		// Strings become objects so per-field knobs have somewhere to live.
		if (typeof models[index] === "string") models[index] = { id: modelId, input: ["text", "image"] };
		mutate(models[index]);
		return true;
	}).then((saved) => {
		if (saved) ctx.ui.notify(`Updated "${modelId}".`, "info");
		return saved;
	});
}

// Edit per-model HTTP headers and api/baseUrl endpoint override.
async function editModelOverride(ctx: CommandContext, providerId: string, modelId: string) {
	let model: any;
	try {
		model = findModel(loadModelsConfig().providers?.[providerId], modelId);
	} catch {
		model = undefined;
	}
	const currentBase = typeof model?.baseUrl === "string" ? model.baseUrl : "";
	const currentHeaders = model?.headers && typeof model.headers === "object" ? JSON.stringify(model.headers) : "";

	const base = await ctx.ui.input("baseUrl override (blank = use provider, \"-\" to clear)", currentBase || "e.g. https://api.example.com/v1");
	if (base === undefined) return;
	const headers = await ctx.ui.input("Headers as JSON (blank = keep, \"-\" to clear)", currentHeaders || 'e.g. {"x-api-version":"2024-01"}');
	if (headers === undefined) return;

	let parsedHeaders: Record<string, string> | null | undefined;
	const trimmedHeaders = headers.trim();
	if (trimmedHeaders === "-") parsedHeaders = null;
	else if (trimmedHeaders) {
		try {
			const obj = JSON.parse(trimmedHeaders);
			if (!obj || typeof obj !== "object" || Array.isArray(obj)) throw new Error("not an object");
			parsedHeaders = obj;
		} catch (error) {
			ctx.ui.notify(`Invalid headers JSON: ${error instanceof Error ? error.message : String(error)}`, "error");
			return;
		}
	}

	await mutateModel(ctx, providerId, modelId, (m) => {
		const trimmedBase = base.trim();
		if (trimmedBase === "-") delete m.baseUrl;
		else if (trimmedBase) m.baseUrl = trimmedBase;
		if (parsedHeaders === null) delete m.headers;
		else if (parsedHeaders) m.headers = parsedHeaders;
	});
}

function modelIdOf(model: any): string {
	return typeof model === "string" ? model.trim() : typeof model?.id === "string" ? model.id.trim() : "";
}

function findModel(provider: any, id: string): any {
	const models = Array.isArray(provider?.models) ? provider.models : [];
	return models.find((m: any) => modelIdOf(m) === id);
}

// Resolve a stored provider's apiKey reference back into mode+value so we can
// reuse it for probing. Anything other than $VAR or !cmd is treated as literal.
function apiKeyFromProvider(provider: any): { mode: ApiKeyMode; value?: string } {
	const raw = typeof provider?.apiKey === "string" ? provider.apiKey : "";
	if (!raw || raw === "dummy" || raw === "ollama") return { mode: "none" };
	if (raw.startsWith("!")) return { mode: "shell", value: raw.slice(1) };
	if (raw.startsWith("$")) return { mode: "env", value: raw.slice(1) };
	return { mode: "literal", value: raw };
}

async function addModelEntriesToProvider(ctx: CommandContext, providerId: string, items: Array<string | ProbeItem>) {	const existing = new Set<string>();
	try {
		const provider = loadModelsConfig().providers?.[providerId];
		for (const m of Array.isArray(provider?.models) ? provider.models : []) existing.add(modelIdOf(m));
	} catch {
		// fall through; mutateProvider re-reads and reports errors
	}

	// Normalize string ids and probe items (which carry OpenRouter metadata).
	const seen = new Set<string>();
	const fresh = items
		.map((item) => (typeof item === "string" ? { id: item.trim() } : { id: item.value.trim(), meta: item.meta }))
		.filter(({ id }) => {
			if (!id || existing.has(id) || seen.has(id)) return false;
			seen.add(id);
			return true;
		});
	if (fresh.length === 0) {
		ctx.ui.notify("Nothing to add — all selected models already exist.", "info");
		return;
	}

	// Prefer OpenRouter metadata (context window, modality, output cap, pricing,
	// reasoning support). Fall back to the permissive default when it is absent.
	const saved = await mutateProvider(ctx, providerId, (p) => {
		const models = Array.isArray(p.models) ? p.models : [];
		for (const { id, meta } of fresh) models.push(buildModelEntry(id, optionsFromMeta(meta)));
		p.models = models;
		return true;
	});
	if (saved) ctx.ui.notify(`Added ${fresh.length} model${fresh.length === 1 ? "" : "s"} to "${providerId}".`, "info");
}

async function reprobeProvider(ctx: CommandContext, providerId: string) {
	let provider: any;
	try {
		provider = loadModelsConfig().providers?.[providerId];
	} catch (error) {
		ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return;
	}
	if (provider?.api === "anthropic-messages") {
		ctx.ui.notify("Anthropic-style endpoints don't expose /models. Use 'Add models manually'.", "warning");
		return;
	}
	const baseUrl = typeof provider?.baseUrl === "string" ? provider.baseUrl : "";
	if (!baseUrl) {
		ctx.ui.notify(`Provider "${providerId}" has no baseUrl to probe.`, "error");
		return;
	}

	const apiKey = apiKeyFromProvider(provider);
	let probed: ProbeItem[];
	try {
		ctx.ui.notify(`Probing ${buildProbeUrl(baseUrl)} ...`, "info");
		probed = await probeOpenAIModels(baseUrl, apiKey.mode, apiKey.value);
	} catch (error) {
		ctx.ui.notify(`Probe failed: ${error instanceof Error ? error.message : String(error)}`, "error");
		return;
	}

	await enrichProbeItemsWithCatalog(probed, isOpenRouterEndpoint(baseUrl));

	const existing = new Set((Array.isArray(provider.models) ? provider.models : []).map(modelIdOf));
	const novel = probed.filter((item) => !existing.has(item.value));
	if (novel.length === 0) {
		ctx.ui.notify("No new models — everything the endpoint returned is already configured.", "info");
		return;
	}

	const picked = await pickMany(ctx, `New models for ${providerId}`, novel);
	if (!picked || picked.length === 0) return;
	await addModelEntriesToProvider(ctx, providerId, picked);
}

// Backfill metadata (context window, max tokens, vision, reasoning, cost) on
// already-configured models from the OpenRouter catalog. Only blank fields are
// filled so hand-tuned values are never clobbered.
async function syncProviderMetadata(ctx: CommandContext, providerId: string) {
	let provider: any;
	try {
		provider = loadModelsConfig().providers?.[providerId];
	} catch (error) {
		ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return;
	}
	const models = Array.isArray(provider?.models) ? provider.models : [];
	if (models.length === 0) {
		ctx.ui.notify(`Provider "${providerId}" has no models.`, "warning");
		return;
	}

	const catalog = await loadOpenRouterCatalog();
	if (!catalog) {
		ctx.ui.notify("Could not reach the OpenRouter catalog (offline or blocked).", "error");
		return;
	}

	// Reasoning maps are the one field worth overwriting: an older sync wrote a
	// bare `{ xhigh: "xhigh" }` that unlocks a level the provider may reject. Ask
	// first because it replaces any per-level values you tuned by hand.
	const anyMatched = models.some((m: any) => matchCatalogMeta(catalog, modelIdOf(m)) !== undefined);
	let refreshReasoningMaps = false;
	if (anyMatched) {
		refreshReasoningMaps = await ctx.ui.confirm(
			"Refresh reasoning levels?",
			"Rebuild reasoning and thinkingLevelMap from the levels OpenRouter declares for each model.\n\nThis replaces reasoning ceilings you may have set manually (e.g. a deliberate cap), so pick No to only fill in missing fields.",
		);
	}

	let matched = 0;
	let changed = 0;
	const saved = await mutateProvider(ctx, providerId, (p) => {
		const list = Array.isArray(p.models) ? p.models : [];
		for (const model of list) {
			// Strings have nowhere to store metadata; promote them to objects.
			const id = modelIdOf(model);
			if (!id) continue;
			const meta = matchCatalogMeta(catalog, id);
			if (!meta) continue;
			matched++;
			let localChanged = false;
			if (typeof model === "string") {
				const index = list.indexOf(model);
				list[index] = buildModelEntry(id, optionsFromMeta(meta));
				changed++;
				continue;
			}
			if (model.name === undefined && meta.name) {
				model.name = meta.name;
				localChanged = true;
			}
			if (model.contextWindow === undefined && meta.contextLength) {
				model.contextWindow = meta.contextLength;
				localChanged = true;
			}
			if (model.maxTokens === undefined && meta.maxCompletionTokens) {
				model.maxTokens = meta.maxCompletionTokens;
				localChanged = true;
			}
			if (!Array.isArray(model.input) && meta.modalities.length > 0) {
				model.input = [...meta.modalities];
				localChanged = true;
			}

			const declaredMap = mapFromMeta(meta);
			if (refreshReasoningMaps && metaSupportsReasoning(meta)) {
				// Full refresh: recompute both the ceiling and the per-level map from what
				// the provider actually declares. This also drops a stale xhigh override
				// written by older versions of this wizard.
				const before = JSON.stringify([model.reasoning, model.thinkingLevelMap]);
				applyReasoning(model, optionsFromMeta(meta).reasoning, undefined, declaredMap);
				if (JSON.stringify([model.reasoning, model.thinkingLevelMap]) !== before) localChanged = true;
			} else if (model.reasoning === undefined && metaSupportsReasoning(meta)) {
				applyReasoning(model, optionsFromMeta(meta).reasoning, undefined, declaredMap);
				localChanged = true;
			} else if (model.reasoning === true && model.thinkingLevelMap === undefined) {
				const opts = readModelOptions(model);
				applyReasoning(model, opts.reasoning, undefined, declaredMap);
				localChanged = true;
			} else if (refreshReasoningMaps && model.reasoning === true && !metaSupportsReasoning(meta)) {
				// OpenRouter says this model has no reasoning parameters, so drop a stale
				// reasoning flag/map instead of advertising an unusable toggle.
				applyReasoning(model, "off");
				localChanged = true;
			}

			if (model.cost === undefined && meta.cost) {
				model.cost = meta.cost;
				localChanged = true;
			}
			if (localChanged) changed++;
		}
		return changed > 0;
	});

	if (!saved) {
		ctx.ui.notify(
			matched === 0
				? "No models matched the OpenRouter catalog — nothing to sync."
				: "All matching models already have complete metadata.",
			"info",
		);
		return;
	}
	ctx.ui.notify(
		`Synced ${changed} of ${models.length} model${models.length === 1 ? "" : "s"} in "${providerId}"${refreshReasoningMaps ? " (incl. reasoning levels)" : ""}.`,
		"info",
	);
}

async function addModelsToProvider(ctx: CommandContext, providerId: string) {
	let provider: any;
	try {
		provider = loadModelsConfig().providers?.[providerId];
	} catch (error) {
		ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return;
	}
	const style: ProviderStyle =
		provider?.api === "anthropic-messages" ? "anthropic" : provider?.compat ? "ollama" : "openai";
	const ids = await promptModelIdsOneByOne(ctx, style);
	if (!ids || ids.length === 0) return;
	// Manual ids skip /models, but we can still match them against the OpenRouter
	// catalog to recover context window, modality and pricing.
	const items: ProbeItem[] = ids.map((id) => ({ value: id, label: id }));
	await enrichProbeItemsWithCatalog(items, isOpenRouterEndpoint(typeof provider?.baseUrl === "string" ? provider.baseUrl : ""));
	await addModelEntriesToProvider(ctx, providerId, items);
}

async function deleteProviderFlow(ctx: CommandContext) {
	let cursor = 0;
	let deletedAny = false;

	while (true) {
		let config: ModelsConfig;
		try {
			config = loadModelsConfig();
		} catch (error) {
			ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
			return;
		}

		config.providers ||= {};
		const providerIds = Object.keys(config.providers).sort((a, b) => a.localeCompare(b));
		if (providerIds.length === 0) {
			ctx.ui.notify(
				deletedAny ? `No providers left in ${MODELS_JSON_PATH}` : `No providers found in ${MODELS_JSON_PATH}`,
				deletedAny ? "info" : "warning",
			);
			return;
		}

		const choice = await selectOne(
			ctx,
			"Delete provider",
			providerIds.map((providerId) => {
				const inline = describeProviderInline(providerId, config.providers?.[providerId]);
				return {
					value: providerId,
					label: inline.label,
					suffix: inline.suffix,
					searchText: inline.searchText,
				};
			}),
			{ initialIndex: Math.min(cursor, providerIds.length - 1) },
		);
		if (!choice) return;

		const provider = config.providers[choice];
		const confirmed = await ctx.ui.confirm("Delete provider?", describeProvider(choice, provider));
		const selectedIndex = providerIds.indexOf(choice);
		cursor = selectedIndex;
		if (!confirmed) continue;

		cursor = selectedIndex + 1;
		delete config.providers[choice];

		try {
			saveModelsConfig(config);
		} catch (error) {
			ctx.ui.notify(`Could not write ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
			return;
		}

		deletedAny = true;
		ctx.ui.notify(`Deleted provider \"${choice}\" from ${MODELS_JSON_PATH}`, "info");
	}
}

async function promptProviderStyle(
	ctx: CommandContext,
): Promise<{ style: ProviderStyle; api: ProviderApi } | null> {
	const providerStyleLabel = await selectOne(ctx, "Provider style", [
		"OpenAI-compatible",
		"Anthropic-compatible",
		"Ollama-compatible",
	]);
	if (!providerStyleLabel) return null;

	const style: ProviderStyle =
		providerStyleLabel === "Anthropic-compatible"
			? "anthropic"
			: providerStyleLabel === "Ollama-compatible"
				? "ollama"
				: "openai";
	const api: ProviderApi = style === "anthropic" ? "anthropic-messages" : "openai-completions";
	return { style, api };
}

async function promptEndpoint(
	ctx: CommandContext,
	style: ProviderStyle,
	api: ProviderApi,
): Promise<{ normalized: string; raw: string } | null> {
	const endpointInput = await ctx.ui.input(
		"Endpoint",
		style === "anthropic"
			? "e.g. https://api.anthropic-proxy.com/v1"
			: style === "ollama"
				? "e.g. http://localhost:11434/v1"
				: "e.g. https://api.example.com/v1 or http://localhost:11434/v1",
	);
	if (endpointInput === undefined) return null;
	const raw = endpointInput.trim();
	if (!raw) {
		ctx.ui.notify("Endpoint is required.", "error");
		return null;
	}

	try {
		return { normalized: normalizeEndpoint(raw, api), raw };
	} catch (error) {
		ctx.ui.notify(`Invalid endpoint: ${error instanceof Error ? error.message : String(error)}`, "error");
		return null;
	}
}

async function confirmEndpointReuse(ctx: CommandContext, normalizedEndpoint: string): Promise<boolean> {
	let config: ModelsConfig;
	try {
		config = loadModelsConfig();
	} catch (error) {
		ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return false;
	}

	const providersWithSameEndpoint = findProvidersByEndpoint(config, normalizedEndpoint);
	if (providersWithSameEndpoint.length === 0) return true;

	return ctx.ui.confirm(
		"Endpoint already exists",
		`This endpoint is already used by:\n${providersWithSameEndpoint.map((id) => `- ${id}`).join("\n")}\n\nAdd another provider with the same endpoint?`,
	);
}

async function promptProviderId(ctx: CommandContext, normalizedEndpoint: string): Promise<string | null> {
	let existingIds = new Set<string>();
	try {
		existingIds = new Set(Object.keys(loadModelsConfig().providers ?? {}));
	} catch {
		// If config can't be read, persistProvider surfaces the error later.
	}

	const providerIdSuggestion = suggestProviderId(normalizedEndpoint);
	const suggestionTaken = existingIds.has(providerIdSuggestion);

	while (true) {
		const providerNameInput = await ctx.ui.input(
			suggestionTaken ? "Provider name (must be unique)" : `Provider name (blank = ${providerIdSuggestion})`,
			"e.g. custom-example-com",
		);
		if (providerNameInput === undefined) return null;
		const providerId = slugify(providerNameInput.trim() || providerIdSuggestion);
		if (!providerId) {
			ctx.ui.notify("Provider name is required.", "error");
			continue;
		}

		// Provider names must be unique — never silently overwrite an existing one.
		if (existingIds.has(providerId)) {
			ctx.ui.notify(`Provider "${providerId}" already exists. Choose a different name.`, "warning");
			continue;
		}

		if (BUILTIN_PROVIDER_IDS.has(providerId)) {
			const ok = await ctx.ui.confirm(
				"Override built-in provider?",
				`"${providerId}" matches a built-in provider id. Saving this will override that provider in the active models config. Continue?`,
			);
			if (!ok) continue;
		}
		return providerId;
	}
}

async function persistProvider(ctx: CommandContext, providerId: string, providerConfig: any): Promise<boolean> {
	let config: ModelsConfig;
	try {
		config = loadModelsConfig();
	} catch (error) {
		ctx.ui.notify(`Could not read ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return false;
	}

	config.providers ||= {};
	if (config.providers[providerId]) {
		// Names are validated unique at prompt time; this only triggers if the
		// config changed underneath us. Refuse rather than overwrite.
		ctx.ui.notify(`Provider "${providerId}" already exists. Not overwriting.`, "error");
		return false;
	}

	config.providers[providerId] = providerConfig;
	try {
		saveModelsConfig(config);
	} catch (error) {
		ctx.ui.notify(`Could not write ${MODELS_JSON_PATH}: ${error instanceof Error ? error.message : String(error)}`, "error");
		return false;
	}
	return true;
}

async function addProviderFlow(ctx: CommandContext) {
	const styleChoice = await promptProviderStyle(ctx);
	if (!styleChoice) return;
	const { style, api } = styleChoice;

	const endpoint = await promptEndpoint(ctx, style, api);
	if (!endpoint) return;
	if (!(await confirmEndpointReuse(ctx, endpoint.normalized))) return;

	const providerId = await promptProviderId(ctx, endpoint.normalized);
	if (!providerId) return;

	const apiKey = await promptApiKey(ctx);
	if (!apiKey) return;
	if (apiKey.mode === "none") {
		ctx.ui.notify(
			style === "ollama"
				? 'No API key selected. Using "ollama" automatically in the models config.'
				: 'No API key selected. Using "dummy" automatically in the models config.',
			"info",
		);
	}

	const models = await collectModelIds(ctx, style, api, apiKey, endpoint.normalized, endpoint.raw, isOpenRouterEndpoint(endpoint.normalized));
	if (!models || models.length === 0) return;

	const providerConfig = buildProviderConfig(
		style,
		api,
		endpoint.normalized,
		apiKey,
		// Each selected model carries its OpenRouter metadata (when available) so
		// defaults match the real context window, modality and pricing.
		models.map((item) => ({
			id: item.value,
			opts: item.meta ? optionsFromMeta(item.meta) : { reasoning: "xhigh" as ReasoningCeiling, vision: true },
		})),
	);
	if (!(await persistProvider(ctx, providerId, providerConfig))) return;

	ctx.ui.notify(`Saved provider \"${providerId}\" to ${MODELS_JSON_PATH}`, "info");
	ctx.ui.notify("Open /model to use your new provider.", "info");
}

async function collectModelIds(
	ctx: CommandContext,
	style: ProviderStyle,
	api: ProviderApi,
	apiKey: { mode: ApiKeyMode; value?: string },
	normalizedEndpoint: string,
	trimmedEndpointInput: string,
	isOpenRouterEndpointUrl: boolean,
): Promise<ProbeItem[] | null> {
	// Manual entry has no metadata to attach, so wrap ids with an undefined meta.
	if (api !== "openai-completions") {
		const ids = await promptModelIdsOneByOne(ctx, style);
		return ids?.map((id) => ({ value: id, label: id })) ?? null;
	}

	const modelMode = await selectOne(ctx, "Models", ["Auto probe from /models", "Add manually"]);
	if (!modelMode) return null;
	if (modelMode !== "Auto probe from /models") {
		const ids = await promptModelIdsOneByOne(ctx, style);
		return ids?.map((id) => ({ value: id, label: id })) ?? null;
	}

	try {
		ctx.ui.notify(`Probing ${buildProbeUrl(normalizedEndpoint)} ...`, "info");
		const probedModels = await probeOpenAIModels(normalizedEndpoint, apiKey.mode, apiKey.value);
		if (probedModels.length === 0) {
			ctx.ui.notify("Probe succeeded but returned no models. Switching to manual entry.", "warning");
			const ids = await promptModelIdsOneByOne(ctx, style);
			return ids?.map((id) => ({ value: id, label: id })) ?? null;
		}
		await enrichProbeItemsWithCatalog(probedModels, isOpenRouterEndpointUrl);
		return pickMany(ctx, "Select models", probedModels);
	} catch (error) {
		const schemeHint = hasExplicitScheme(trimmedEndpointInput) ? "" : "\n\nNo http:// or https:// was provided.";
		ctx.ui.notify(
			`Auto probe failed: ${error instanceof Error ? error.message : String(error)}.${schemeHint}\n\nSwitching to manual entry.`,
			"warning",
		);
		const ids = await promptModelIdsOneByOne(ctx, style);
		return ids?.map((id) => ({ value: id, label: id })) ?? null;
	}
}

// ===========================================================================
// Auto thinking level
// ===========================================================================
// pi exposes thinking levels in increasing effort order. Auto mode scores the
// incoming prompt and picks the cheapest level that should still handle it.
const AUTO_LEVEL_ORDER: ReasoningCeiling[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

// Prompt signals that usually mean real reasoning work rather than a lookup.
const HARD_TASK_PATTERNS: RegExp[] = [
	/\b(refactor|architect|redesign|migrat|optimi[sz]e|debug|diagnos|root cause|investigat)\w*/i,
	/\b(implement|build|design|write|create|add|fix)\b[\s\S]{0,40}\b(feature|system|module|component|api|service|pipeline|algorithm)\b/i,
	/\b(why|explain|analy[sz]e|compare|trade-?offs?|prove|derive|reason)\b/i,
	/\b(concurren|race condition|deadlock|memory leak|performance|latency|throughput|scal)\w*/i,
];

const SIMPLE_TASK_PATTERNS: RegExp[] = [
	/^\s*(hi|hey|hello|yo|thanks|thank you|ok|okay|sure|yes|no|cool|nice|great)\b[\s!.?]*$/i,
	/^\s*\/(help|status|model|thinking|clear)\s*$/i,
];

function scorePromptDifficulty(prompt: string, images: unknown[] | undefined, contextTokens: number): number {
	const text = (prompt ?? "").trim();
	if (!text) return 1;

	let score = 1;
	const length = text.length;

	// Length is the strongest cheap signal: long asks tend to be multi-step.
	if (length > 1500) score += 3;
	else if (length > 600) score += 2;
	else if (length > 200) score += 1;

	// Code, logs, or stack traces imply understanding a concrete artifact.
	if (/```/.test(text)) score += 2;
	if (/\b[A-Za-z0-9_./-]+\.(ts|tsx|js|jsx|py|go|rs|java|rb|php|c|cpp|h|json|ya?ml|toml|sql|sh)\b/.test(text)) score += 1;
	if (/\b(error|exception|traceback|stack trace|failed|panic|segfault)\b/i.test(text)) score += 1;
	if (/(^|\n)\s*at\s+\S+\(.*:\d+:\d+\)/.test(text)) score += 1;

	// Multiple questions or a numbered plan point at a bigger job.
	const questionMarks = (text.match(/\?/g) ?? []).length;
	if (questionMarks >= 3) score += 1;
	if (/^\s*(\d+[.)]|[-*])\s+/m.test(text) && /\n/.test(text)) score += 1;

	if (HARD_TASK_PATTERNS.some((pattern) => pattern.test(text))) score += 2;
	if (SIMPLE_TASK_PATTERNS.some((pattern) => pattern.test(text))) score -= 1;
	if (images && images.length > 0) score += 1;

	// Working in a nearly full context usually means a long session; favor room
	// for thinking over token thrift once past ~60% of a typical window.
	if (contextTokens > 120_000) score += 1;

	return Math.max(0, score);
}

// score -> pi level. Extended levels (xhigh/max) are reserved for the hardest
// prompts so a routine ask never pays for them.
function levelForScore(score: number): ReasoningCeiling {
	if (score <= 0) return "minimal";
	if (score <= 1) return "low";
	if (score <= 3) return "medium";
	if (score <= 5) return "high";
	if (score <= 6) return "xhigh";
	return "max";
}

// Snap a desired level to the cheapest supported level that is still at least as
// capable, so auto mode never silently drops below the intended effort.
function clampToSupported(desired: ReasoningCeiling, supported: ReasoningCeiling[]): ReasoningCeiling {
	const desiredIndex = AUTO_LEVEL_ORDER.indexOf(desired);
	const atLeast = supported
		.slice()
		.sort((a, b) => AUTO_LEVEL_ORDER.indexOf(a) - AUTO_LEVEL_ORDER.indexOf(b))
		.find((level) => AUTO_LEVEL_ORDER.indexOf(level) >= desiredIndex);
	return atLeast ?? supported[supported.length - 1] ?? "off";
}

function supportedLevelsForModel(model: any): ReasoningCeiling[] {
	if (!model?.reasoning) return ["off"];
	const map = model.thinkingLevelMap as ThinkingLevelMap | undefined;
	return AUTO_LEVEL_ORDER.filter((level) => mapLevelSupported(map, level));
}

function pickAutoThinkingLevel(
	prompt: string,
	images: unknown[] | undefined,
	ctx: { getContextUsage?: () => { tokens?: number | null } | undefined },
	model: any,
	state: BetterCustomState,
	previousToolCalls: number,
	previousToolErrors: number,
): ReasoningCeiling {
	const supported = supportedLevelsForModel(model);
	if (supported.length <= 1) return supported[0] ?? "off";

	const usage = ctx.getContextUsage?.();
	let score = scorePromptDifficulty(prompt, images, usage?.tokens ?? 0);

	// A previous turn that needed many tools or hit failures is a strong hint the
	// follow-up is still hard, so bias upward without going straight to max.
	if (previousToolErrors > 0) score += 2;
	if (previousToolCalls >= 8) score += 2;
	else if (previousToolCalls >= 4) score += 1;

	const desired = levelForScore(score);
	const fallback = state.autoThinkingFallback ?? "medium";
	const target = supported.includes(desired) ? desired : clampToSupported(desired, supported);
	// Never let auto mode drop the fallback floor for non-trivial prompts.
	if (score >= 2) {
		const floor = clampToSupported(fallback, supported);
		return AUTO_LEVEL_ORDER.indexOf(target) >= AUTO_LEVEL_ORDER.indexOf(floor) ? target : floor;
	}
	return target;
}

async function configureAutoThinking(ctx: CommandContext) {
	const state = loadState();
	const enabled = state.autoThinking === true;
	const choice = await selectOne(ctx, "Thinking mode", [
		{ value: "on", label: "Auto", suffix: ` • ${enabled ? "current" : "off"}`, description: "Pick thinking effort from task difficulty, level by level" },
		{ value: "off", label: "Manual", suffix: ` • ${enabled ? "" : "current"}`, description: "Keep the thinking level you set with /thinking" },
		{ value: "floor", label: "Auto fallback level", suffix: ` • ${state.autoThinkingFallback ?? "medium"}`, description: "Minimum level auto mode uses for non-trivial prompts" },
	]);
	if (!choice) return;

	if (choice === "on" || choice === "off") {
		state.autoThinking = choice === "on";
		saveState(state);
		try {
			ctx.ui.setStatus("better-custom-auto", state.autoThinking ? "auto-thinking: on" : undefined);
		} catch {
			// status UI is optional
		}
		ctx.ui.notify(
			state.autoThinking
				? "Auto thinking enabled. Effort follows task difficulty (may override /thinking each turn)."
				: "Auto thinking disabled. /thinking is now the source of truth.",
			"info",
		);
		return;
	}

	const level = await selectOne(ctx, "Auto fallback level", AUTO_LEVEL_ORDER.map((l) => ({
		value: l,
		label: l,
		description: l === "off" ? "Always allow auto mode to turn thinking off" : `Never go below ${l} for non-trivial prompts`,
	})));
	if (!level) return;
	state.autoThinkingFallback = level as ReasoningCeiling;
	saveState(state);
	ctx.ui.notify(`Auto thinking fallback set to ${level}.`, "info");
}

// ===========================================================================
// Effort limits (self-healing)
// ===========================================================================
function effortKey(providerId: string, modelId: string): string {
	return `${providerId}/${modelId}`;
}

// The payload field carrying reasoning effort differs by API/thinking format.
function payloadEffortField(payload: any): string | undefined {
	if (typeof payload?.reasoning_effort === "string") return "reasoning_effort";
	if (typeof payload?.reasoning?.effort === "string") return "reasoning.effort";
	if (typeof payload?.thinking?.effort === "string") return "thinking.effort";
	return undefined;
}

function readPayloadEffort(payload: any, field: string): string | undefined {
	if (field === "reasoning_effort") return typeof payload.reasoning_effort === "string" ? payload.reasoning_effort : undefined;
	if (field === "reasoning.effort") return typeof payload.reasoning?.effort === "string" ? payload.reasoning.effort : undefined;
	if (field === "thinking.effort") return typeof payload.thinking?.effort === "string" ? payload.thinking.effort : undefined;
	return undefined;
}

// Reduce an effort to the most capable value the endpoint still accepts. Values
// are provider strings ("low", "high", "none", …), so ordering is by pi's own
// level order when the name matches a level. Unknown provider names sort last.
function clampEffortValue(value: string, allowed: string[]): string {
	if (allowed.includes(value)) return value;
	const index = AUTO_LEVEL_ORDER.indexOf(value as ReasoningCeiling);
	if (index === -1) return allowed[allowed.length - 1] ?? value;

	const levelIndex = (name: string): number => {
		const i = AUTO_LEVEL_ORDER.indexOf(name as ReasoningCeiling);
		return i === -1 ? Number.MAX_SAFE_INTEGER : i;
	};
	const ordered = allowed.slice().sort((a, b) => levelIndex(a) - levelIndex(b));
	// Highest allowed level that is still no more capable than what was asked.
	const candidates = ordered.filter((name) => levelIndex(name) <= index);
	return candidates[candidates.length - 1] ?? ordered[0] ?? value;
}

// Effort strings this model may send: each supported level's provider value,
// falling back to the level name itself when the map has no explicit mapping.
function declaredEfforts(model: any): string[] | undefined {
	const map = model?.thinkingLevelMap as ThinkingLevelMap | undefined;
	const values: string[] = [];
	for (const level of supportedLevelsForModel(model)) {
		if (level === "off") continue;
		const mapped = map?.[level];
		values.push(typeof mapped === "string" ? mapped : level);
	}
	const unique = Array.from(new Set(values));
	return unique.length > 0 ? unique : undefined;
}

// Non-standard extended effort names. OpenAI's own `reasoning_effort` enum only
// documents low/medium/high, so a bare OpenAI-compatible endpoint that rejects
// them returns a 400 the user should never have to see. They are stripped from
// the allowed set by default ("high" becomes the cap) unless the model's
// declared map was explicitly trusted or a limit was learned from the endpoint.
const EXTENDED_EFFORTS = new Set(["xhigh", "max"]);

// Rewrite an outgoing payload in place when its effort exceeds the learned or
// declared limit. Returns the applied change for reporting, or undefined.
function clampPayloadEffort(payload: any, model: any, state: BetterCustomState): { from: string; to: string } | undefined {
	if (!payload || typeof payload !== "object") return undefined;
	const field = payloadEffortField(payload);
	if (!field) return undefined;
	const value = readPayloadEffort(payload, field);
	if (!value) return undefined;

	const learned = state.effortAllow?.[effortKey(model.provider, model.id)];
	let allowed: string[] | undefined;
	if (learned && learned.length > 0) {
		// Learned values are authoritative: they came from the endpoint itself.
		allowed = learned.slice();
	} else {
		const declared = declaredEfforts(model);
		if (!declared) return undefined;
		allowed = declared;
		// Only the plain OpenAI `reasoning_effort` path needs the conservative
		// cap; OpenRouter-style `reasoning: { effort }` is expected to understand
		// extended names (that is what the catalog advertises).
		if (field === "reasoning_effort" && state.allowExtendedEffort !== true) {
			const pruned = allowed.filter((name) => !EXTENDED_EFFORTS.has(name));
			if (pruned.length > 0) allowed = pruned;
		}
	}
	if (allowed.length === 0) return undefined;

	const clamped = clampEffortValue(value, allowed);
	if (clamped === value) return undefined;

	if (field === "reasoning_effort") payload.reasoning_effort = clamped;
	else if (field === "reasoning.effort") payload.reasoning.effort = clamped;
	else payload.thinking.effort = clamped;
	return { from: value, to: clamped };
}

// Fall back to the model's declared map when nothing has been learned yet. Only
// string mappings count: null means the level is unsupported.
// Parse a provider 400 body for `invalid_enum_value` on an effort field. The
// error message embeds the JSON body, so the allowed options can be recovered.
function parseEffortRejection(errorMessage: string): string[] | undefined {
	if (!/invalid_enum_value|invalid enum|expected .*received/i.test(errorMessage)) return undefined;
	if (!/reasoning_effort|reasoning\.effort|thinking\.effort|\beffort\b/i.test(errorMessage)) return undefined;

	// Prefer the structured issue payload when present.
	const optionsMatch = errorMessage.match(/"options"\s*:\s*\[([^\]]*)\]/i);
	if (optionsMatch) {
		const options = optionsMatch[1]
			.split(",")
			.map((part) => part.trim().replace(/^"|"$/g, ""))
			.filter(Boolean);
		if (options.length > 0) return options;
	}

	// Fall back to the human-readable form: Expected 'low' | 'medium' | 'high'
	const expected = errorMessage.match(/Expected\s+((?:'[^']+'\s*\|?\s*)+)/i);
	if (expected) {
		const options = Array.from(expected[1].matchAll(/'([^']+)'/g)).map((m) => m[1]);
		if (options.length > 0) return options;
	}
	return undefined;
}

// Mirror a learned limit into models.json so the UI stops offering levels the
// endpoint rejects. Values are provider strings; unmatched level strings are
// left untouched (they may name a provider value we cannot map to a pi level).
function applyEffortAllowToList(providerId: string, modelId: string, allowed: string[]): boolean {
	let config: ModelsConfig;
	try {
		config = loadModelsConfig();
	} catch {
		return false;
	}
	const provider = config.providers?.[providerId];
	const models = Array.isArray(provider?.models) ? provider.models : [];
	const model = models.find((m: any) => modelIdOf(m) === modelId);
	if (!model || typeof model !== "object") return false;

	const map: ThinkingLevelMap = { ...(model.thinkingLevelMap ?? {}) };
	let changed = false;
	for (const level of AUTO_LEVEL_ORDER) {
		if (level === "off") continue;
		const mapped = map[level];
		if (typeof mapped !== "string" || allowed.includes(mapped)) continue;
		map[level] = null;
		changed = true;
	}
	if (!changed) return false;

	model.thinkingLevelMap = map;
	try {
		saveModelsConfig(config);
		return true;
	} catch {
		return false;
	}
}

async function manageEffortLimits(ctx: CommandContext) {
	while (true) {
		const state = loadState();
		const entries = Object.entries(state.effortAllow ?? {}).sort(([a], [b]) => a.localeCompare(b));
		const autoRetry = state.autoRetryEffort !== false;
		const extended = state.allowExtendedEffort === true;

		const items: SelectItem[] = [
			{ value: "__retry", label: "Auto-retry on rejected effort", suffix: ` • ${autoRetry ? "on" : "off"}`, description: autoRetry ? "Retry a request that failed with an invalid effort, using an allowed value" : "Surface the error instead of retrying" },
			{ value: "__extended", label: "Send xhigh/max to OpenAI-style endpoints", suffix: ` • ${extended ? "on" : "off"}`, description: "Off caps plain reasoning_effort at high; OpenRouter-style is unaffected" },
		];
		for (const [key, allowed] of entries) {
			items.push({ value: key, label: key, suffix: ` • ${allowed.join(", ")}`, description: "Forget this limit so the endpoint is probed again" });
		}
		items.push({ value: "__clear", label: "Clear all limits", description: entries.length > 0 ? `Forget ${entries.length} learned limit${entries.length === 1 ? "" : "s"}` : "Nothing learned yet" });

		const choice = await selectOne(ctx, "Reasoning effort limits", items);
		if (!choice) return;

		if (choice === "__retry") {
			state.autoRetryEffort = !autoRetry;
			saveState(state);
			ctx.ui.notify(`Auto-retry on rejected effort ${state.autoRetryEffort ? "enabled" : "disabled"}.`, "info");
			continue;
		}
		if (choice === "__extended") {
			state.allowExtendedEffort = !extended;
			saveState(state);
			ctx.ui.notify(`Extended efforts to OpenAI-style endpoints ${state.allowExtendedEffort ? "enabled" : "disabled"}.`, "info");
			continue;
		}

		if (choice === "__clear") {
			if (entries.length === 0) continue;
			const ok = await ctx.ui.confirm("Clear all limits?", `Forget ${entries.length} learned limit${entries.length === 1 ? "" : "s"}?`);
			if (!ok) continue;
			state.effortAllow = {};
			saveState(state);
			ctx.ui.notify("Cleared learned effort limits.", "info");
			continue;
		}

		const ok = await ctx.ui.confirm("Forget limit?", `Forget the learned effort limit for ${choice}?`);
		if (!ok) continue;
		const next = { ...(state.effortAllow ?? {}) };
		delete next[choice];
		state.effortAllow = next;
		saveState(state);
		ctx.ui.notify(`Forgot effort limit for ${choice}.`, "info");
	}
}

export default function betterCustomWizard(pi: ExtensionAPI) {
	// ---- Auto thinking level -------------------------------------------------
	// Tracks the previous run so difficulty scoring can escalate a prompt that
	// follows a rough turn (tool failures / lots of tool use).
	let lastRunToolCalls = 0;
	let lastRunToolErrors = 0;
	let currentRunToolCalls = 0;
	let currentRunToolErrors = 0;

	pi.on("before_agent_start", async (event, ctx) => {
		const state = loadState();
		lastRunToolCalls = currentRunToolCalls;
		lastRunToolErrors = currentRunToolErrors;
		currentRunToolCalls = 0;
		currentRunToolErrors = 0;

		if (!state.autoThinking) return;
		const model = ctx.model as any;
		if (!model?.reasoning || !ctx.hasUI) return;

		const level = pickAutoThinkingLevel(event.prompt, event.images, ctx, model, state, lastRunToolCalls, lastRunToolErrors);
		try {
			pi.setThinkingLevel(level);
			ctx.ui.setStatus("better-custom-auto", `auto-thinking: ${level}`);
		} catch {
			// Non-fatal: a stale context just skips auto tuning for this turn.
		}
	});

	pi.on("tool_execution_end", async (event) => {
		currentRunToolCalls++;
		if (event.isError) currentRunToolErrors++;
	});

	// ---- Effort guard --------------------------------------------------------
	pi.on("before_provider_request", async (event, ctx) => {
		const model = ctx.model as any;
		// Built-in catalogs already describe supported efforts precisely; only
		// hand-configured providers need the guard. Skip when the payload targets a
		// different model (compaction uses its own), since the level map would be
		// the wrong one.
		if (!model || !isCustomProvider(model.provider)) return;
		const payloadModel = (event.payload as any)?.model;
		if (typeof payloadModel === "string" && payloadModel !== model.id) return;
		const result = clampPayloadEffort(event.payload, model, loadState());
		if (!result) return;
		if (ctx.hasUI) ctx.ui.setStatus("better-custom-effort", `effort: ${result.to} (asked ${result.from})`);
		return event.payload;
	});

	// Learn from a provider rejection so the bad value never goes out again, and
	// ask pi to retry once the limit is recorded. The provider did return an
	// error, and pi's retry classifier treats that as transient, so the retry is
	// just the normal transient-error path with a corrected payload.
	pi.on("message_end", async (event, ctx) => {
		const message = event.message as any;
		if (message?.role !== "assistant" || message.stopReason !== "error") return;
		const allowed = parseEffortRejection(String(message.errorMessage ?? ""));
		if (!allowed || allowed.length === 0) return;
		const model = ctx.model as any;
		if (!model || !isCustomProvider(model.provider)) return;

		const key = effortKey(model.provider, model.id);
		const state = loadState();
		const previous = state.effortAllow?.[key];
		const same = previous && previous.length === allowed.length && previous.every((v, i) => v === allowed[i]);
		if (!same) {
			state.effortAllow = { ...(state.effortAllow ?? {}), [key]: allowed };
			saveState(state);
			const updated = applyEffortAllowToList(model.provider, model.id, allowed);
			if (ctx.hasUI) {
				ctx.ui.notify(
					`${key} only accepts reasoning effort: ${allowed.join(", ")}.${updated ? " models.json updated." : ""}`,
					"info",
				);
			}
		}

		// Retry with the corrected payload so the failure stays invisible. pi only
		// retries errors it classifies as transient, so the message pi reads for that
		// classification is annotated; the original detail is preserved so a real,
		// unfixable error still surfaces after the retry budget runs out.
		if (state.autoRetryEffort !== false && !same) {
			const current = String(message.errorMessage ?? "");
			const note = "provider returned error (invalid reasoning effort); retrying with a supported value";
			if (!current.includes(note)) {
				return { message: { ...message, errorMessage: `${current}\n${note}` } };
			}
		}
	});

	pi.on("session_start", async () => {
		refreshState();
		refreshProviderIds();
		lastRunToolCalls = 0;
		lastRunToolErrors = 0;
		currentRunToolCalls = 0;
		currentRunToolErrors = 0;
	});

	// /model reloads models.json, so drop the cached provider list with it.
	pi.on("model_select", async () => {
		refreshProviderIds();
	});

	pi.registerCommand("better-custom", {
		description: "Wizard for adding, editing, or deleting custom providers in ~/.pi/agent/models.json",
		handler: async (_args, ctx) => {
			const action = await selectOne(ctx, "Better custom", [
				{ value: "add", label: "Add provider" },
				{ value: "edit", label: "Edit provider" },
				{ value: "delete", label: "Delete provider" },
				{ value: "thinking", label: "Thinking mode (auto)", description: "Adjust thinking effort to task difficulty" },
				{ value: "effort", label: "Reasoning effort limits", description: "View/clear learned per-model effort limits" },
			]);
			if (!action) return;
			if (action === "edit") {
				await editProviderFlow(ctx);
				return;
			}
			if (action === "delete") {
				await deleteProviderFlow(ctx);
				return;
			}
			if (action === "thinking") {
				await configureAutoThinking(ctx);
				return;
			}
			if (action === "effort") {
				await manageEffortLimits(ctx);
				return;
			}
			await addProviderFlow(ctx);
		},
	});
}
