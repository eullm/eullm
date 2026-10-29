"""The tail of a llama-server log, as the engine wrote it.

`LlamaServer.log_tail` is what a failed or not-ready server shows for why it
stopped, so the lines must not depend on the host's default encoding: the
engine writes the log as UTF-8, and a machine whose default is not UTF-8
read the text as mojibake.
"""

from __future__ import annotations

from eullm_forge.eval.gguf import LlamaServer


def test_the_log_tail_is_read_as_utf8(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    log = tmp_path / "m.server.log"
    log.write_text("errore: modello non trovato a citt\u00e0\n", encoding="utf-8")
    server = LlamaServer(gguf, binary=str(tmp_path / "absent"), log_path=log)
    assert "citt\u00e0" in server.log_tail()
