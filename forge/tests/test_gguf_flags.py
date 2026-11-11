"""Whether the engine's `serve` has `--no-ui`, from its `--help` output.

`EullmServer.command` asks the binary once whether `serve` has `--no-ui`,
because on a node of several points every server after the first died at
start-up with the chat UI's port taken. The help is the tool's UTF-8 output;
a host whose default encoding is not UTF-8 must not turn a readable flag
into a decode failure at eval start-up.
"""

from __future__ import annotations

from eullm_forge.eval.gguf import EullmServer


def test_the_help_is_read_as_utf8(monkeypatch):
    import eullm_forge.eval.gguf as gguf

    class Out:
        stdout = "usage: eullm serve \u2014 citt\u00e0\n  --no-ui\n"
        stderr = ""
        returncode = 0

    seen = []

    def run(cmd, **kwargs):
        seen.append(kwargs)
        return Out()

    monkeypatch.setattr(gguf.subprocess, "run", run)
    server = EullmServer("m.gguf", binary="/bin/eullm")
    assert "--no-ui" in server.command()
    assert seen[0].get("encoding") == "utf-8"
