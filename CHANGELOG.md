# Changelog

## 0.2.0 (develop, unreleased)

### Added
- **Retry preprompt (opt-in).** `preprompt` and `preprompt_escalated` settings: user-supplied text appended to the system prompt of requests after a refusal is detected in the session. The plugin ships no text; with `preprompt` empty, requests are never modified.
- `llm_request` middleware that performs the injection, per `api_mode`: the `system` message (chat completions), the top-level `system` field (Anthropic, Bedrock; existing blocks and `cache_control` preserved) or `instructions` (Responses/Codex).
- `on_preprompt_refusal` (`disarm` default, `escalate`, `keep`): what happens when the preprompt itself is refused.
- `preprompt_scope` (`turn` default, `session`): how long a session stays armed.
- `preprompt_models`: glob allowlist of `model` / `provider/model` names that may receive the preprompt.
- `arm_on_native_refusal` (default `true`): provider-signalled refusals (`stop_reason=refusal`, `finish_reason=content_filter`, Responses `incomplete`/`content_filter`) arm the session too. Those responses are left unaltered.
- `max_armed_sessions` (default 1024) and cleanup on `on_session_finalize` / `on_session_reset`.
- `REFUSAL_PREPROMPT=0` kill switch.
- Logs for arming, injection, `answered with preprompt` and `refused despite preprompt`; `source` / `reason` in Hermes's middleware trace.
- Softer refusal phrasings ("I'm not comfortable providing that", "I'd rather not walk you through that", "that falls outside what I can help with", …).

### Fixed
- **Quadratic backtracking on whitespace runs.** The shared sentence-start prefix `(?:^|[.!?\n]\s*)` was still quadratic after 0.1.1: a reply with 5,000 blank lines took ~3 s and 20,000 took ~50 s inside the synchronous middleware. It is now `\s{0,20}` (20,000 blank lines: ~0.1 s). Regression tests added.
- Two of the new soft patterns flagged ordinary engineering replies ("I'm not comfortable writing to the production database…", "I have to say no to that flag…"); they now require a request-shaped object.
- Post-call processing in `on_llm_execution` is fail-soft: an internal error is logged and the response is returned unchanged.

### Changed
- Docs: full settings reference, lifecycle, outcome table, log reference, limitations.

## 0.1.1

- Bounded the lazy spans in the refusal patterns (`{0,300}`) after the Hermes catalog review found catastrophic backtracking; added timing tests.

## 0.1.0

- Initial release: `llm_execution` middleware that detects refusals and deflections and marks the response `content_filter` so Hermes activates its fallback model.
