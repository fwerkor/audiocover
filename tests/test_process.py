from __future__ import annotations

from audiocover import process


def test_hidden_window_kwargs_is_noop_off_windows(monkeypatch) -> None:
    monkeypatch.setattr(process.sys, "platform", "linux")
    kwargs = {"text": True, "capture_output": True}

    assert process._hidden_window_kwargs(kwargs) is kwargs
    assert "creationflags" not in kwargs


def test_hidden_window_kwargs_hides_windows_console(monkeypatch) -> None:
    class FakeStartupInfo:
        def __init__(self) -> None:
            self.dwFlags = 0
            self.wShowWindow = None

    monkeypatch.setattr(process.sys, "platform", "win32")
    monkeypatch.setattr(process.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(process.subprocess, "STARTUPINFO", FakeStartupInfo, raising=False)
    monkeypatch.setattr(process.subprocess, "STARTF_USESHOWWINDOW", 1, raising=False)
    monkeypatch.setattr(process.subprocess, "SW_HIDE", 0, raising=False)

    kwargs = {"text": True, "creationflags": 0x20}
    hidden = process._hidden_window_kwargs(kwargs)

    assert hidden is not kwargs
    assert hidden["creationflags"] == 0x08000020
    assert hidden["startupinfo"].dwFlags & 1
    assert hidden["startupinfo"].wShowWindow == 0
    assert kwargs["creationflags"] == 0x20
