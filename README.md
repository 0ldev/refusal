# refusal

[![CI](https://github.com/0ldev/refusal/actions/workflows/ci.yml/badge.svg)](https://github.com/0ldev/refusal/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A [Hermes](https://hermes-agent.nousresearch.com) plugin that detects when a model refuses a request, or deflects to an educational alternative, and retries the turn on your configured fallback model.

Some primary models decline legitimate work, such as security research, penetration-testing help and malware analysis. Instead of leaving you with a dead-end reply, `refusal` reports the refusal to Hermes so the turn continues on the next model in your fallback chain.

## Features

- **Refusal detection.** Direct refusals ("I can't help with that"), policy invocations ("this violates my guidelines") and deflections ("I can't do X, but I can explain Y defensively").
- **Few false positives.** Tool-calling responses, code blocks, partial answers ("I can't do X, but here's Y") and idioms such as "can't help but" are all treated as genuine answers.
- **Retry preprompt (opt-in, bring your own text).** You can configure text that is appended to the system prompt of requests after a refusal is detected in the session, so the fallback retry is not made on a bare prompt. The plugin ships no text of its own and does nothing here until you set one.
- **Public APIs only.** It uses the `llm_execution` and `llm_request` middlewares and never patches Hermes core.
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

The plugin has no settings of its own. Configure the fallback chain in `~/.hermes/config.yaml` (or with `hermes fallback add`):

```yaml
model:
  default: gemini-3.8-flash
  provider: openrouter

fallback_providers:
  - provider: anthropic
    model: claude-haiku-4-5
```

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
        preprompt_escalated: "<optional: used instead after a second refusal in the session>"
```

- **Nothing is bundled.** Both settings default to empty, and with `preprompt` empty requests are never modified. `preprompt_escalated` falls back to `preprompt` when empty.
- **Detection arms the session.** A detected refusal raises the session's level (1, then 2) before the response is marked `content_filter`. An `llm_request` middleware appends the text for the current level to the system prompt of every later request in that session. The first call of a turn in an unarmed session is untouched.
- **Per-provider injection.** The text goes where the provider reads its system prompt: the `system` message for chat completions, the top-level `system` field for Anthropic and Bedrock, and `instructions` for Responses/Codex.
- **Session-scoped.** The armed state persists for the session: once a model refuses, later turns keep the preprompt. It is forgotten on `on_session_finalize` and `on_session_reset`, and at most `max_armed_sessions` sessions are tracked (default 1024; least recently armed evicted first). A false-positive detection arms the session too.
- **Kill switch.** `REFUSAL_PREPROMPT=0` disables arming and injection regardless of the settings (read at call time). Detection and fallback routing still work.
- **Your responsibility.** What you put in the preprompt is sent to your model provider, and you are responsible for complying with that provider's usage policy.

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
- **Responses are modified.** A detected refusal changes the response metadata (`finish_reason`, `stop_reason`) so that Hermes triggers failover.
- **Requests are modified (only when you set a preprompt).** For armed sessions your preprompt text is appended to the system prompt of outgoing requests. It is not stored and is sent only to the model being called.
- **Fallback costs tokens.** Hermes resends the full conversation to the fallback model, and the fallback provider bills for it.
- **No fallback configured.** Hermes replaces the response with its standard notice, `⚠️ The model declined to respond to this request (safety refusal).` A false positive in this setup discards a valid answer, so configure at least one fallback provider.
- The plugin makes no network calls, runs no subprocesses, starts no background processes, reads nothing outside its own package and stores no credentials.

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

Validate the packaging against the Hermes catalog rules:

```bash
hermes plugins validate . --install-deps
hermes plugins doctor . --ci
```

### Layout

```text
plugin.yaml            Hermes manifest
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
