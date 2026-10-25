# refusal

[![CI](https://github.com/0ldev/refusal/actions/workflows/ci.yml/badge.svg)](https://github.com/0ldev/refusal/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A [Hermes](https://hermes-agent.nousresearch.com) plugin that detects when a model refuses a request, or deflects to an educational alternative, and retries the turn on your configured fallback model.

Some primary models decline legitimate work, such as security research, penetration-testing help and malware analysis. Instead of leaving you with a dead-end reply, `refusal` reports the refusal to Hermes so the turn continues on the next model in your fallback chain.

## Features

- **Refusal detection.** Direct refusals ("I can't help with that"), policy invocations ("this violates my guidelines"), softer phrasings ("I'm not comfortable providing that") and deflections ("I can't do X, but I can explain Y defensively").
- **Provider-signalled refusals.** Responses the provider itself flags (Anthropic `stop_reason: refusal`, `finish_reason: content_filter`) are recognized even though they carry no text.
- **Few false positives.** Tool-calling responses, code blocks, partial answers ("I can't do X, but here's Y") and idioms such as "can't help but" are treated as genuine answers. See [Limitations](#limitations) for what still gets through.
- **Bounded cost.** The lazy spans and sentence-start whitespace in the patterns are length-bounded, and a reply with tens of thousands of blank lines is classified in a fraction of a second instead of stalling the agent.
- **Retry preprompt (opt-in, bring your own text).** You can configure text that is appended to the system prompt of requests after a refusal is detected in the session, so the fallback retry is not made on a bare prompt. The plugin ships no text of its own and does nothing here until you set one.
- **Public APIs only.** It uses the `llm_execution` and `llm_request` middlewares and two observer hooks, and never patches Hermes core.
- **No dependencies, no network calls.** Detection is local and rule-based.

## Requirements

- Hermes `>= 0.22.1` (the `requires_hermes` floor in `plugin.yaml`)
- Python `>= 3.10`
- At least one entry in `fallback_providers`. Without one, a detected refusal ends the turn (see [Behavior and disclosures](#behavior-and-disclosures)).

## Installation

```bash
hermes plugins install refusal
hermes plugins enable refusal
```

To install from a local checkout:

```bash
ln -s "$(pwd)" ~/.hermes/plugins/refusal
hermes plugins enable refusal
```

Check that it loaded:

```bash
hermes plugins list
hermes plugins capabilities refusal
```

## Configuration

Out of the box the plugin only needs a fallback chain, configured in `~/.hermes/config.yaml` (or with `hermes fallback add`):

```yaml
model:
  default: gemini-3.8-flash
  provider: openrouter

fallback_providers:
  - provider: anthropic
    model: claude-haiku-4-5
```

The retry preprompt and its controls are plugin settings under `plugins.entries.refusal.settings` (the manifest declares a `config_schema`, so Hermes's settings form can render them). All of them are optional:

| Setting | Type | Default | What it does |
|---|---|---|---|
| `preprompt` | string | empty | Text appended to the system prompt after a refusal. Empty turns the whole preprompt feature off. |
| `preprompt_escalated` | string | empty | Text used instead of `preprompt` once the preprompt itself was refused (only with `on_preprompt_refusal: escalate`). Empty reuses `preprompt`. |
| `on_preprompt_refusal` | `disarm` \| `escalate` \| `keep` | `disarm` | What a refusal that happened *with* the preprompt does to the session. |
| `preprompt_scope` | `turn` \| `session` | `turn` | How long a session stays armed. |
| `preprompt_models` | list of globs | empty (any) | Only send the preprompt to matching `model` or `provider/model` names. |
| `arm_on_native_refusal` | bool | `true` | Also arm on provider-signalled refusals. |
| `persist_state` | bool | `false` | Keep armed levels in Hermes's plugin state so they survive a restart and are shared between processes. See [Persistence](#persistence-optional). |
| `max_armed_sessions` | int ≥ 1 | `1024` | How many armed sessions are tracked at once. |

Environment: `REFUSAL_PREPROMPT=0` (or `off`, `false`, `no`) disables arming and injection regardless of the settings. It is read at call time, so it works without a restart.

## How it works

1. The plugin registers an `llm_execution` middleware and calls the model as usual.
2. Responses that contain tool calls (OpenAI `tool_calls` or Anthropic `tool_use`) are returned untouched.
3. For text responses, `RefusalDetector` checks the prose, ignoring fenced code blocks, against refusal, deflection and partial-answer patterns.
4. When a refusal is detected, the plugin sets `finish_reason` to `content_filter` (Anthropic: `stop_reason` to `refusal`).
5. Hermes handles `content_filter` through its content-policy path and activates the next fallback model.

Hermes does not let plugins register custom failover reasons, so a detected refusal appears under Hermes's own `content_policy_blocked` reason.

## Retry preprompt (optional)

By default a refusal only triggers the fallback. If you set a preprompt, the retry is also made with your text in the system prompt:

```yaml
plugins:
  entries:
    refusal:
      settings:
        preprompt: "<text appended to the system prompt after a refusal>"
        preprompt_escalated: "<optional: used instead after the preprompt itself was refused>"
        on_preprompt_refusal: disarm      # disarm | escalate | keep
        preprompt_scope: turn             # turn | session
        preprompt_models: []              # e.g. ["google/gemini-*"]; empty = any model
        arm_on_native_refusal: true
        persist_state: false              # opt in to survive restarts / share between processes
        max_armed_sessions: 1024
```

You are responsible for the text you configure: it is sent to your model provider, and you must comply with that provider's usage policy.

### Lifecycle

Each session has a *level*: 0 (unarmed), 1, or 2.

1. A refusal on a request that did **not** carry the preprompt sets the session to level 1.
2. An `llm_request` middleware appends the text for the current level to the system prompt of later requests in the session, including the fallback retry. The first call of a turn in an unarmed session is untouched, so a refusal on it is genuine.
3. If a request that **did** carry the preprompt is refused too, `on_preprompt_refusal` decides:
   - `disarm`: back to level 0, the preprompt is no longer sent.
   - `escalate`: level 2, which sends `preprompt_escalated` (or `preprompt` if empty).
   - `keep`: the level stays as it is.

   Hermes moves on to the next fallback entry in every case. A refusal from a model the preprompt was not sent to (see `preprompt_models`) does not count as "despite the preprompt".
4. With `preprompt_scope: turn` the level is forgotten when the next user turn starts. With `session` it lasts until the session is finalized or reset. If Hermes gives no `turn_id`, `turn` behaves like `session`.
5. The ending session's armed state is dropped on `on_session_finalize` and `on_session_reset` (for a reset, the *old* session id). At most `max_armed_sessions` sessions are tracked; the least recently armed is evicted first.

### What happens on each outcome

| Situation | Plugin does | Hermes does |
|---|---|---|
| Text refusal, no preprompt sent | Marks `content_filter`, level → 1 | Falls back to the next chain entry |
| Provider-signalled refusal, no preprompt sent | Level → 1 (response untouched) | Falls back |
| Refusal despite the preprompt | Marks it, applies `on_preprompt_refusal` | Falls back; with no entry left, shows the safety notice |
| Normal answer on an armed request | Logs `answered with preprompt`, level unchanged | Continues |
| Tool-call response | Ignored | Continues |
| Refusal with no `session_id` | Marks it, does not arm | Falls back |
| No fallback configured | Arms anyway (it cannot see the chain) | Shows the safety notice |

### Where the text goes

The text is added where each provider reads its system prompt: the `system` message for chat completions, the top-level `system` field for Anthropic and Bedrock (string or text blocks; existing blocks and their `cache_control` markers are left intact), and `instructions` for Responses/Codex. The text is never stored, and the original request object is never mutated.

### Models

`preprompt_models` takes case-insensitive globs matched against the model name or `provider/model`, for example `["google/gemini-*", "claude-*"]`. When set, the preprompt is only sent to matching models, so a last-resort backup on another vendor never receives it.

### Logs and traces

| Level | Message | Meaning |
|---|---|---|
| WARNING | `Refusal detected (kind=…, reason=…, text=…)` | A text refusal was matched and the response marked |
| INFO | `provider-signalled refusal (…)` | The provider flagged the response itself |
| INFO | `preprompt armed (session=… level=1)` | First refusal in the session |
| INFO | `preprompt injected (session=… level=… provider=… model=…)` | A request was modified |
| INFO | `answered with preprompt (…)` | A request that carried the preprompt got a normal answer |
| WARNING | `refused despite preprompt (…); mode=… -> level …` | The preprompt did not help |
| INFO | `refused without preprompt (…); level unchanged` | Armed session, but this request was not sent the text (model filter or scope) |
| INFO | `refusal detected without a session_id; preprompt not armed` | Hermes gave no session id |

Injected requests also carry `source: refusal` and a `reason` in Hermes's middleware trace.

### Persistence (optional)

By default armed levels live in process memory. With `persist_state: true` the plugin keeps them in Hermes's per-plugin state store (`ctx.state`, a JSON file `state.json` in this plugin's directory under `<HERMES_HOME>/plugin-data/`, written atomically under a cross-process file lock), so:

- an armed session survives a gateway restart, and
- several processes serving the same profile see the same levels.

What is stored: a single list of `[session_id, level, turn_id]` rows, at most `max_armed_sessions` of them. **No preprompt text, no message content and no credentials are stored.** Rows are removed when a session is finalized or reset and when the cap evicts them.

Costs and caveats:

- While the preprompt is configured, each request reads the file once (and each arming or disarm writes it). The read is a small JSON file under an advisory lock.
- Updates are a read-merge-write of the whole list, so two processes arming at the same instant can lose one update (last writer wins).
- If the store cannot be read or written, the plugin logs a WARNING and falls back to memory for the rest of the process. A Hermes that has no `ctx.state` also falls back, with a warning.

### Reaching the same model on the retry

Hermes skips fallback entries that resolve to the same backend as the current one (same provider + model), so to retry the *same model* the fallback entry must be a **different route to the same model**:

```yaml
model:
  default: gemini-3.8-flash
  provider: openrouter

fallback_providers:
  - provider: google              # same model, different route -> preprompt retry
    model: gemini-3.8-flash
  - provider: anthropic           # final backup if the preprompt retry also fails
    model: claude-haiku-4-5
```

Call 1: `openrouter/gemini-3.8-flash` without the preprompt. Refusal → call 2: `google/gemini-3.8-flash` with `preprompt`. Refusal again → level 2 → call 3: the next chain entry (here `anthropic/claude-haiku-4-5`). To use `preprompt_escalated` on the same model, list two different routes to it in the chain (e.g. `openrouter` → `google` → a custom endpoint).

Injection happens at request time, and auxiliary calls (compression, titles) are unaffected — they do not pass through the agent's request middleware.

## Behavior and disclosures

- **Every turn is inspected.** The middleware reads each model response before the agent loop uses it. Nothing is stored or sent anywhere.
- **Responses are modified.** A detected text refusal changes the response metadata (`finish_reason`, `stop_reason`) so that Hermes triggers failover. Provider-signalled refusals are not altered.
- **Requests are modified (only when you set a preprompt).** For armed sessions your preprompt text is appended to the system prompt of outgoing requests.
- **The matched snippet is logged.** A detected refusal logs the matching part of the response at WARNING. Your preprompt text is never logged.
- **Fallback costs tokens.** Hermes resends the full conversation to the fallback model, and the fallback provider bills for it.
- **No fallback configured.** Hermes replaces the response with its standard notice, `⚠️ The model declined to respond to this request (safety refusal).` A false positive in this setup discards a valid answer, so configure at least one fallback provider.
- **Persistence is opt-in.** With `persist_state: true` the plugin reads and writes its own file in Hermes's plugin state directory (session ids, levels and turn ids only; see [Persistence](#persistence-optional)). With it off, the plugin writes nothing to disk.
- **One environment variable is read.** `REFUSAL_PREPROMPT` (the kill switch), at call time. It is not a credential and is never written.
- The plugin makes no network calls, runs no subprocesses, starts no background processes, reads no files outside its own package (and, if enabled, its own plugin state) and stores no credentials. Armed state lives in process memory only, unless you opt in to `persist_state` (below).

## Limitations

- **Pattern-based.** An unusual refusal phrasing can be missed, and a reply that quotes refusal language can be flagged. Inability that is explained by a technical reason ("I can't run that because the container has no network", "The repository is read-only") is recognized as a genuine answer, but only for the phrasings the filter knows, and never when the same stretch mentions policy or harm. Configure a fallback so a false positive does not discard the answer.
- **Cannot see the fallback chain.** Hermes does not tell middleware whether a fallback exists, so with none configured the plugin still marks and arms, and Hermes shows the safety notice.
- **Same-model retries need a different route.** Hermes skips fallback entries that resolve to the backend that just failed (see below).
- **State is per process by default.** Armed levels are not shared between gateway workers or processes and are lost on restart, unless you enable [`persist_state`](#persistence-optional).
- **Request copying.** Registering a `llm_request` middleware makes Hermes deep-copy each outgoing payload once or twice (about 1 ms for a 700 KB request in my measurement), whether or not you configured a preprompt. The plugin returns `None` for requests it does not change, so Hermes records no trace entry and makes no further copy for them.
- **Context compression** starts a new session id with no link to the old one that plugins can see, so an armed session is unarmed after a compression. The orphaned entry is removed by the `max_armed_sessions` eviction.
- **Streaming.** The middleware sees the finished response, so a refusal that streamed to the user is still shown before the fallback runs.
- **Sub-agents** have their own session ids and are armed independently of their parent.
- **Auxiliary calls** (compression, titles) never pass through the request middleware and are unaffected.
- **Prompt caching (Anthropic).** The preprompt is added as a new block after the cached system prefix, so the first request after arming cannot reuse the cached message history. Measured live on `claude-haiku-4-5` with an ~11k-token prompt (system plus 10 turns, 4 cache markers): the first armed request re-wrote 6,291 tokens and still read the 5,055-token system prefix from cache; later armed requests read all ~11.3k tokens from cache, history growth cost only the new turn (~635 tokens written), and disarming read the original cache again while it was still within its TTL. So arming costs one re-write of the history region, once per level change. With a short history there is almost nothing to re-write (21 tokens in an ~8.6k-token system-prompt-only test).
- **Tested against.** Live: Anthropic (`claude-haiku-4-5`, OAuth) and OpenRouter chat completions (`poolside/laguna-s-2.1:free`: a real refusal-shaped reply was marked `content_filter`, armed the session, and the next request carried the text in a single system message). The Bedrock and Responses/Codex request shapes are covered by unit tests only.

## Upgrading from 0.1.x

0.2.0 adds the `llm_request` middleware, two session hooks (`on_session_finalize`, `on_session_reset`) and the settings above. With no `preprompt` configured, behavior is identical to 0.1.x except that provider-signalled refusals are now recorded (they were already handled by Hermes). Existing configuration keeps working. See the [changelog](CHANGELOG.md).

## Library usage

The detector works without Hermes:

```python
from refusal import detect, is_refusal

result = detect("I cannot assist with executing cyberattacks.")
result.is_refusal  # True
result.kind        # "refusal"
result.reason      # "direct_refusal"

is_refusal("chmod 600 ~/.ssh/id_rsa")  # False
```

`reason` is one of `direct_refusal`, `policy_refusal` or `off_topic_deflection`. `kind` is `refusal` or `deflection`.

## Development

```bash
python -m pip install -e . pytest
pytest
```

The tests cover the detector (including timing bounds), the fallback marking, and the preprompt state machine, injection shapes and settings parsing.

Validate the packaging against the Hermes catalog rules:

```bash
hermes plugins validate . --install-deps
hermes plugins doctor . --ci
```

### Layout

```text
plugin.yaml            Hermes manifest
CHANGELOG.md           Release notes
__init__.py            Plugin entry point (exports register)
refusal/
  detector.py          RefusalDetector and DetectionResult
  patterns.py          Compiled refusal, deflection and partial-answer patterns
  preprompt.py         Per-api_mode system-prompt injection of the configured text
  plugin.py            register() and the llm_execution + llm_request middlewares
tests/                 Detector, fallback and preprompt tests
```

## License

[MIT](LICENSE)
