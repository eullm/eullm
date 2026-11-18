"""A missing --mbox path is an error, not an empty mailbox.

mailbox.mbox creates the file when missing (create=True is the default), so
a typo'd path silently grew a stray 0-byte file and the run ended with "no
email found" for mail that never existed.

Run this with `python -m pytest examples/decision-triage`; the Examples
job in CI runs it too.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "triage", Path(__file__).resolve().parent / "triage.py")
_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_mod)


def test_a_missing_mbox_path_is_an_error_and_creates_nothing(tmp_path):
    missing = tmp_path / "mailboox.mbox"  # the typo
    with pytest.raises(SystemExit, match="no such mbox file"):
        _mod.load(SimpleNamespace(dir=None, mbox=str(missing)))
    assert not missing.exists()


def test_a_real_mbox_still_loads(tmp_path):
    mbox = tmp_path / "mail.mbox"
    mbox.write_bytes(
        b"From alice@example.invalid Mon Oct  6 12:00:00 2026\n"
        b"From: alice@example.invalid\n"
        b"Subject: Appalto urgente\n"
        b"\n"
        b"Corpo della mail.\n")
    mails = _mod.load(SimpleNamespace(dir=None, mbox=str(mbox)))
    assert len(mails) == 1
    assert mails[0]["subject"] == "Appalto urgente"
