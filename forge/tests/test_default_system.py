"""set_default_system: the system prompt a template inserts by itself is ours, and travels."""

from __future__ import annotations

from pathlib import Path

import pytest

from eullm_forge.identity import set_default_system

jinja2 = pytest.importorskip("jinja2")

# The shape of Ministral-3's template: a default message, with an escaped
# quote in it, inserted when the conversation brings no system message.
TEMPLATE = (
    "{#- Default system message if no system prompt is passed. #}\n"
    "{%- set default_system_message = 'You are Ministral. You power an AI assistant called "
    "Le Chat.\nWhen you\\'re not sure, ask. The current date is {today}.' %}\n"
    "{{- bos_token }}"
    "{%- if messages[0]['role'] != 'system' and default_system_message != '' %}"
    "{{- '[SYSTEM_PROMPT]' + default_system_message + '[/SYSTEM_PROMPT]' }}{%- endif %}"
    "{%- for m in messages %}{%- if m['role'] == 'system' %}"
    "{{- '[SYSTEM_PROMPT]' + m['content'] + '[/SYSTEM_PROMPT]' }}"
    "{%- elif m['role'] == 'user' %}{{- '[INST]' + m['content'] + '[/INST]' }}{%- endif %}"
    "{%- endfor %}"
)


class Tok:
    chat_template = TEMPLATE

    def render(self, messages):
        return jinja2.Template(self.chat_template).render(messages=messages, bos_token="<s>")


def test_the_default_is_replaced_and_an_explicit_system_message_still_wins():
    tok = Tok()
    ours = "Sei legal-it. Cita i testi con [n]; se l'articolo non c'è, dillo."
    assert set_default_system(tok, ours)
    out = tok.render([{"role": "user", "content": "ciao"}])
    assert out == f"<s>[SYSTEM_PROMPT]{ours}[/SYSTEM_PROMPT][INST]ciao[/INST]"
    assert "Le Chat" not in out and "{today}" not in out
    explicit = tok.render([{"role": "system", "content": "altro"},
                           {"role": "user", "content": "ciao"}])
    assert ours not in explicit and "[SYSTEM_PROMPT]altro[/SYSTEM_PROMPT]" in explicit


def test_a_template_without_a_default_is_left_alone():
    class Plain:
        chat_template = "{%- for m in messages %}{{ m['content'] }}{%- endfor %}"
    tok = Plain()
    assert not set_default_system(tok, "x")
    assert tok.chat_template == Plain.chat_template


def test_stage3_takes_the_prompt_from_a_file(tmp_path, monkeypatch):
    import importlib.util
    script = Path(__file__).resolve().parents[1] / "scripts" / "stage3_sft.py"
    spec = importlib.util.spec_from_file_location("stage3_sft", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seen = {}
    monkeypatch.setattr(mod, "fine_tune_identity", lambda cfg: seen.setdefault("cfg", cfg))
    prompt = tmp_path / "system.txt"
    prompt.write_text("Sei legal-it.\n", encoding="utf-8")
    pairs = tmp_path / "p.jsonl"
    pairs.write_text('{"instruction": "a", "output": "b"}\n', encoding="utf-8")
    try:
        mod.main(["--model", str(tmp_path), "--pairs", str(pairs), "--out", str(tmp_path / "o"),
                  "--default-system-file", str(prompt)])
    except SystemExit:
        pass
    assert seen["cfg"].default_system == "Sei legal-it."
