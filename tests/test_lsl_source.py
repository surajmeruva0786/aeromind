"""Covers the guarded-import failure path in src.inference.lsl_source
without needing real EEG hardware or the native liblsl library."""

from __future__ import annotations

import pytest

import src.inference.lsl_source as lsl_source


def test_stream_lsl_raises_clear_error_when_unavailable(monkeypatch):
    monkeypatch.setattr(lsl_source, "LSL_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="pylsl is not usable"):
        next(lsl_source.stream_lsl())
