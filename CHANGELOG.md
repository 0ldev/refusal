# Changelog

## 0.2.0

### Added
- **Retry preprompt (opt-in).** `preprompt` and `preprompt_escalated` settings: user-supplied text appended to the system prompt of requests after a refusal is detected in the session. The plugin ships no text; with `preprompt` empty, requests are never modified.
- `llm_request` middleware that performs the injection, per `api_mode`: the `system` message (chat completions), the top-level `system` field (Anthropic, Bedrock; existing blocks and `cache_control` preserved) or `instructions` (Responses/Codex).
- `on_preprompt_refusal` (`disarm` default, `escalate`, `keep`): what happens when the preprompt itself is refused.
- `preprompt_scope` (`turn` default, `session`): how long a session stays armed.
- `preprompt_models`: glob allowlist of `model` / `provider/model` names that may receive the preprompt.
- `arm_on_native_refusal` (default `true`): provider-signalled refusals (`stop_reason=refusal`, `finish_reason=content_filter`, Responses `incomplete`/`content_filter`) arm the session too. Those responses are left unaltered.
- `persist_state` (default `false`): opt in to keep armed levels in Hermes's plugin state so they survive restarts and are shared between processes. Stores session ids, levels and turn ids only; falls back to memory with a warning if the store fails.
- `max_armed_sessions` (default 1024) and cleanup on `on_session_finalize` / `on_session_reset`.
- `REFUSAL_PREPROMPT=0` kill switch.
- Logs for arming, injection, `answered with preprompt` and `refused despite preprompt`; `source` / `reason` in Hermes's middleware trace.
- Softer refusal phrasings ("I'm not comfortable providing that", "I'd rather not walk you through that", "that falls outside what I can help with", …).

### Changed
- The refusal-detection log line now carries `session`, `model`, `provider` and `api_request_id`. Hermes logs and counts every `content_filter` response identically (`reason=content_policy_blocked`, `error_type=ContentPolicyBlocked`) whether the provider signalled it or this plugin synthesized it from matched text — there's no field downstream that tells the two apart. These keys are the only way to join this line back to Hermes's own native log line and know which `content_policy_blocked` events were ours. See [hermes-agent#138900](https://github.com/NousResearch/hermes-agent/issues/138900) for the public seam this plugin asked for (and that would make this correlation unnecessary) that didn't ship.

### Fixed
- Armed state no longer resets when context compression rotates the session id: `post_compaction` carries the level, and a same-turn disarm, to the new id. `on_session_delete` drops deleted or archived sessions.
- Message content returned as a list of text parts (`[{"type": "text", "text": …}]`) was handed to the detector as a list, so a refusal in that shape was silently missed. Content is now flattened to text first.
- A request that already carries the configured preprompt is never injected a second time.
- Explained technical inability ("I can't run that because the container has no network", "The repository is read-only") is no longer treated as a refusal unless the same stretch mentions policy or harm.
- **Quadratic backtracking on whitespace runs.** The shared sentence-start prefix `(?:^|[.!?\n]\s*)` was still quadratic after 0.1.1: a reply with 5,000 blank lines took ~3 s and 20,000 took ~50 s inside the synchronous middleware. It is now `\s{0,20}` (20,000 blank lines: ~0.1 s). Regression tests added.
- Two of the new soft patterns flagged ordinary engineering replies ("I'm not comfortable writing to the production database…", "I have to say no to that flag…"); they now require a request-shaped object.
- `llm_request` returned `{"request": request}` even when nothing changed, which made Hermes copy the payload again and record a spurious `source: plugin` trace entry on every request for every user. It now returns `None` for unchanged requests.
- Session cleanup on `on_session_reset` used the new session id; it now drops the old one (`old_session_id`) as well as the finalizing one.
- Post-call processing in `on_llm_execution` is fail-soft: an internal error is logged and the response is returned unchanged.

### Verified
- Live on Anthropic (`claude-haiku-4-5`) and OpenRouter chat completions (free model): arming from a real response, injection into the right field, model compliance, and prompt-cache behavior with a long history. Persistence through Hermes's real `ctx.state` across two processes.

### Changed
- Docs: full settings reference, lifecycle, outcome table, log reference, limitations.
- `requires_hermes` raised from `>=0.22.1` to `>=0.22.3`, the first release that knows the `on_session_delete` and `post_compaction` hooks (`hermes plugins doctor` rejected them as unknown on 0.22.1 and 0.22.2).

## 0.1.1

- Bounded the lazy spans in the refusal patterns (`{0,300}`) after the Hermes catalog review found catastrophic backtracking; added timing tests.

## 0.1.0

- Initial release: `llm_execution` middleware that detects refusals and deflections and marks the response `content_filter` so Hermes activates its fallback model.
