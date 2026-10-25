"""Keep plugin.yaml, register(), the README and the version numbers from drifting apart."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import refusal
from refusal import plugin as P

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
CHANGELOG = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def _manifest_list(key: str) -> list[str]:
    block = re.search(rf"^{key}:\s*\[\]|^{key}:\n((?:  - .+\n)+)", MANIFEST, re.M)
    return re.findall(r"  - (\S+)", block.group(1)) if block and block.group(1) else []


def _schema_keys() -> list[str]:
    body = MANIFEST.split("config_schema:\n", 1)[1]
    return re.findall(r"^  (\w+):\n    type:", body.split("\nrequires_env", 1)[0], re.M)


def _register_calls():
    ctx = MagicMock()
    asked: list[str] = []

    def get_config(key, default=None):
        asked.append(key)
        return default

    ctx.get_config.side_effect = get_config
    P.register(ctx)
    return ctx, asked


def test_versions_agree():
    manifest = re.search(r"^version:\s*(\S+)", MANIFEST, re.M).group(1)
    pyproject = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT, re.M).group(1)
    changelog = re.search(r"^## (\d+\.\d+\.\d+)", CHANGELOG, re.M).group(1)
    assert manifest == pyproject == refusal.__version__ == changelog


def test_declared_middleware_and_hooks_match_registrations():
    ctx, _ = _register_calls()
    middleware = sorted(c.args[0] for c in ctx.register_middleware.call_args_list)
    hooks = sorted(c.args[0] for c in ctx.register_hook.call_args_list)
    assert middleware == sorted(_manifest_list("provides_middleware"))
    assert hooks == sorted(_manifest_list("provides_hooks"))
    assert _manifest_list("provides_tools") == [] and not ctx.register_tool.called


def test_every_declared_setting_is_read_and_every_read_setting_is_declared():
    _, asked = _register_calls()
    assert sorted(set(asked)) == sorted(_schema_keys())


def test_readme_settings_table_lists_exactly_the_declared_settings():
    table = README.split("| Setting |", 1)[1].split("\n\n", 1)[0]
    documented = re.findall(r"^\| `(\w+)` \|", table, re.M)
    assert sorted(documented) == sorted(_schema_keys())


def test_documented_log_messages_exist_in_the_code():
    source = (ROOT / "refusal" / "plugin.py").read_text(encoding="utf-8")
    for fragment in (
        "Refusal detected", "provider-signalled refusal", "preprompt armed", "preprompt injected",
        "answered with preprompt", "refused despite preprompt", "refused without preprompt",
        "without a session_id",
    ):
        assert fragment in source and fragment.split()[0] in README, fragment


def test_env_kill_switch_name_is_documented():
    assert "REFUSAL_PREPROMPT" in README and "REFUSAL_PREPROMPT" in (ROOT / "refusal" / "plugin.py").read_text()


def test_requires_hermes_floor_is_consistent_and_semver():
    floors = re.findall(r'requires_hermes:\s*"([^"]+)"', MANIFEST)
    assert floors == [">=0.22.1"] and re.fullmatch(r">=\d+\.\d+(\.\d+)?", floors[0])
    assert "0.22.1" in README
