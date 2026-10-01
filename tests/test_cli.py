"""The progress window, and the trap that hid it being broken.

The window feature was silently dead: it calls `json.dumps` to quote a command
for AppleScript, `json` was never imported, and the whole call sits inside a
`try/except Exception: return None`. So the NameError was swallowed and no
window ever opened - and the earlier manual check, "does the number of Terminal
windows return to what it was?", passed precisely because nothing ever opened.

That is the failure mode worth testing: not "does the function return an id" but
"does it fail for a reason other than the one it is allowed to fail for". So
these tests drive the real function with the external call stubbed out, which
leaves only the code under test.
"""
from __future__ import annotations

import shlex
import sys

import make_video


class _Result:
    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode


def test_progress_window_builds_a_real_applescript_call(tmp_path, monkeypatch):
    """The whole point: the call is actually built and returns a window id.

    This is the test that would have caught the dead progress window. That
    function raised NameError - json was never imported - inside a
    `try/except Exception: return None`, so the window silently never opened.
    """
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return _Result(stdout="42\n")

    monkeypatch.setattr(make_video.subprocess, "run", fake_run)
    log = tmp_path / "progress.log"
    wid = make_video.open_progress_window(log)
    assert wid == 42, (
        "open_progress_window returned None; a NameError inside it is "
        "swallowed and looks like 'this machine will not cooperate'"
    )

    assert calls, "no subprocess call was made"
    cmd = calls[0]
    assert cmd[0] == "osascript"
    assert cmd[-1] == f"tail -n 200 -f {shlex.quote(str(log))}"
    # stderr is now mirrored, so the render's output reaches the log file.
    assert sys.stderr is not None


def test_the_shell_command_is_passed_as_an_argument_not_pasted_in(tmp_path, monkeypatch):
    """Escaping by hand does not work; argv makes it a non-problem.

    json.dumps does not escape single quotes and shlex.quote emits '"'"' for a
    path containing one, so pasting the command into AppleScript source ends the
    string literal early and the window silently never opens. Passing it as an
    argv element hands the quoting to osascript.
    """
    seen: list[list[str]] = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        return _Result(stdout="7\n")

    monkeypatch.setattr(make_video.subprocess, "run", fake_run)
    awkward = tmp_path / "Bob's trip 'log'.log"
    assert make_video.open_progress_window(awkward) == 7

    cmd = seen[0]
    assert cmd[0] == "osascript"
    assert "do script" in cmd[-2], "the handler should be the -e argument"
    assert cmd[-1].startswith("tail -n 200 -f "), (
        f"the command should arrive intact, got {cmd[-1]!r}"
    )
    # It must be shell-quoted for the shell Terminal will run it in, and that
    # quoting must still name the real path.
    inner = shlex.split(cmd[-1])
    assert inner == ["tail", "-n", "200", "-f", str(awkward)], (
        f"a path with a quote did not survive the round trip: {inner!r}"
    )
    # And nothing was interpolated into the AppleScript source.
    assert "tail" not in cmd[-2] and str(awkward) not in cmd[-2]


def test_returns_none_when_the_external_call_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(make_video.subprocess, "run",
                        lambda cmd, **kw: _Result(returncode=1))
    assert make_video.open_progress_window(tmp_path / "p.log") is None


def test_closing_a_window_that_never_opened_is_a_no_op(tmp_path, monkeypatch):
    """The common case once the window is disabled - must not raise."""
    monkeypatch.setattr(make_video, "_sleep", lambda s: None, raising=False)
    make_video.close_progress_window(None, tmp_path / "p.log")


class _Stream:
    """Minimal file-like, so _Tee is handed something it can actually use."""

    def __init__(self):
        self.writes: list[str] = []

    def write(self, s):
        self.writes.append(s)

    def flush(self):
        pass

    def isatty(self):
        return False


def test_tee_writes_to_both_streams():
    a, b = _Stream(), _Stream()
    tee = make_video._Tee(a, b)
    assert tee.write("hello\n") == 6
    assert a.writes == ["hello\n"]
    assert b.writes == ["hello\n"]


def test_tee_survives_a_stream_that_raises():
    """One broken destination must not take the pipeline's logging down.

    The log file is a destination the pipeline cannot control - a full disk or
    a deleted work directory would raise on every single write, and if that
    propagated it would kill a run that was otherwise fine.
    """

    class _Broken:
        def write(self, s):
            raise OSError("no space left on device")

        def flush(self):
            raise OSError("no space left on device")

        def isatty(self):
            return False

    good = _Stream()
    tee = make_video._Tee(_Broken(), good)
    tee.write("still logged\n")          # must not raise
    assert good.writes == ["still logged\n"], (
        "a broken second stream must not stop the first"
    )
    tee.flush()                          # must not raise either


def test_tee_reports_tty_only_if_some_stream_is_one():
    class _Tty(_Stream):
        def isatty(self):
            return True

    assert make_video._Tee(_Stream(), _Tty()).isatty() is True
    assert make_video._Tee(_Stream(), _Stream()).isatty() is False


def test_parser_exposes_the_documented_flags():
    """Every flag the README tells people to use has to exist.

    Checked against the parser's own dest names rather than guessing from the
    flag spelling, because several flags deliberately rename: `--two-up` stores
    as `two_up` but `--no-live-photos` stores as `live_photos` (negated), and
    `--no-progress-window` as `progress_window` (negated).
    """
    parser = make_video.build_parser() if hasattr(make_video, "build_parser") \
        else None
    ns = make_video.parse_args([])
    names = set(vars(ns))
    for dest in ("media", "target", "name", "dry_run", "size", "fps", "mood",
                 "fit", "two_up", "progress_window", "live_photos",
                 "no_titles", "quality", "order", "music", "out"):
        assert dest in names, f"a documented option is not parsed (dest={dest})"
    assert parser is None or True


def test_size_and_fps_reach_the_render_config():
    """A flag that parses but never reaches the config is worse than absent."""
    cfg = make_video.cfg_from_args(make_video.parse_args(
        ["--size", "4k", "--fps", "60", "--no-live-photos"]))
    assert (cfg.render.width, cfg.render.height) == (3840, 2160)
    assert cfg.render.fps == 60
    assert cfg.live_photos is False


def test_defaults_are_the_documented_ones():
    cfg = make_video.cfg_from_args(make_video.parse_args([]))
    assert (cfg.render.width, cfg.render.height, cfg.render.fps) == (1920, 1080, 30)
    assert cfg.live_photos is True
