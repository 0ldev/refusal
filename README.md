# refusal

[![CI](https://github.com/0ldev/refusal/actions/workflows/ci.yml/badge.svg)](https://github.com/0ldev/refusal/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A [Hermes](https://hermes-agent.nousresearch.com) plugin that detects when a model refuses a request, or deflects to an educational alternative, and retries the turn on your configured fallback model.

Some primary models decline legitimate work, such as security research, penetration-testing help and malware analysis. Instead of leaving you with a dead-end reply, `refusal` reports the refusal to Hermes so the turn continues on the next model in your fallback chain.

## Features

- **Refusal detection.** Direct refusals ("I can't help with that"), policy invocations ("this violates my guidelines") and deflections ("I can't do X, but I can explain Y defensively").
- **Few false positives.** Tool-calling responses, code blocks, partial answers ("I can't do X, but here's Y") and idioms such as "can't help but" are all treated as genuine answers.
- **Public APIs only.** It uses the `llm_execution` middleware and never patches Hermes core.
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

## Behavior and disclosures

- **Every turn is inspected.** The middleware reads each model response before the agent loop uses it. Nothing is stored or sent anywhere.
- **Responses are modified.** A detected refusal changes the response metadata (`finish_reason`, `stop_reason`) so that Hermes triggers failover.
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
  plugin.py            register() and the llm_execution middleware
tests/                 Detector and fallback tests
```

## License

[MIT](LICENSE)
