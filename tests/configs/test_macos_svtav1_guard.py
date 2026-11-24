#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""macOS SVT-AV1 shutdown guard. Platform and encoder version are mocked."""

import logging

import pytest

from lerobot.configs.video import (
    LEROBOT_ALLOW_SVTAV1_ENV,
    DepthEncoderConfig,
    RGBEncoderConfig,
    VideoEncoderConfig,
    macos_svtav1_close_is_unsafe,
    parse_svtav1_version,
    platform as video_platform,
    svtav1_version_from_library_name,
)

_DEFAULT_ENCODERS = {"libsvtav1", "h264", "hevc", "h264_videotoolbox", "libaom-av1"}


@pytest.fixture(autouse=True)
def _reset_svtav1_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lerobot.configs.video._macos_svtav1_warned", False)
    monkeypatch.delenv(LEROBOT_ALLOW_SVTAV1_ENV, raising=False)


def _stub_encoders(monkeypatch: pytest.MonkeyPatch, available: set[str] | None = None) -> None:
    present = _DEFAULT_ENCODERS if available is None else available

    def detect(self: VideoEncoderConfig, encoders: list[str] | str) -> list[str]:
        names = [encoders] if isinstance(encoders, str) else list(encoders)
        return [name for name in names if name in present]

    monkeypatch.setattr(VideoEncoderConfig, "detect_available_encoders", detect)
    monkeypatch.setattr(VideoEncoderConfig, "validate", lambda self: None)


def _darwin(monkeypatch: pytest.MonkeyPatch, version: tuple[int, ...] | None) -> None:
    monkeypatch.setattr(video_platform, "system", lambda: "Darwin")
    monkeypatch.setattr("lerobot.configs.video._bundled_svtav1_version", lambda: version)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("3.0.0", (3, 0, 0)),
        ("SVT-AV1 Encoder Lib v4.2.0", (4, 2, 0)),
        ("v4.3.0", (4, 3, 0)),
        ("not a version", None),
    ],
)
def test_parse_svtav1_version(text: str, expected: tuple[int, int, int] | None) -> None:
    assert parse_svtav1_version(text) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("libSvtAv1Enc.3.0.0.dylib", (3, 0, 0)),
        ("/opt/pyav/lib/libSvtAv1Enc.3.0.0.dylib", (3, 0, 0)),
        ("libSvtAv1Enc-6870dfd5.so.3.0.0", (3, 0, 0)),
        ("libSvtAv1Enc.so.4.2.0", (4, 2, 0)),
        ("libSvtAv1Enc.so.3", None),
        ("libsvtav1.so", None),
    ],
)
def test_svtav1_version_from_library_name(name: str, expected: tuple[int, int, int] | None) -> None:
    assert svtav1_version_from_library_name(name) == expected


@pytest.mark.parametrize(
    ("version", "unsafe"),
    [
        (None, False),
        ((3, 0, 0), True),
        ((4, 2, 0), True),
        ((4, 2), True),
        ((4, 3), False),
        ((4, 3, 0), False),
        ((4, 3, 1), False),
        ((5, 0, 0), False),
    ],
)
def test_macos_svtav1_close_is_unsafe(version: tuple[int, ...] | None, unsafe: bool) -> None:
    assert macos_svtav1_close_is_unsafe(version) is unsafe


@pytest.mark.parametrize(
    ("system", "version", "vcodec", "allow", "available", "expected", "warns"),
    [
        ("Linux", (3, 0, 0), "libsvtav1", False, None, "libsvtav1", False),
        ("Windows", (3, 0, 0), "libsvtav1", False, None, "libsvtav1", False),
        ("Darwin", (4, 3, 0), "libsvtav1", False, None, "libsvtav1", False),
        ("Darwin", (4, 2, 0), "libsvtav1", False, None, "h264", True),
        ("Darwin", (3, 0, 0), "libsvtav1", False, None, "h264", True),
        ("Darwin", (3, 0, 0), "av1", False, None, "h264", True),
        ("Darwin", (3, 0, 0), "libsvtav1", True, None, "libsvtav1", True),
        ("Darwin", None, "libsvtav1", False, None, "libsvtav1", True),
        ("Darwin", (3, 0, 0), "h264", False, None, "h264", False),
        ("Darwin", (3, 0, 0), "hevc", False, None, "hevc", False),
        ("Darwin", (3, 0, 0), "libsvtav1", False, {"libsvtav1"}, "libsvtav1", True),
    ],
)
def test_resolve_replaces_only_known_affected_macos_svtav1(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    system: str,
    version: tuple[int, ...] | None,
    vcodec: str,
    allow: bool,
    available: set[str] | None,
    expected: str,
    warns: bool,
) -> None:
    _stub_encoders(monkeypatch, available)
    monkeypatch.setattr(video_platform, "system", lambda: system)
    monkeypatch.setattr("lerobot.configs.video._bundled_svtav1_version", lambda: version)
    if allow:
        monkeypatch.setenv(LEROBOT_ALLOW_SVTAV1_ENV, "1")

    with caplog.at_level(logging.WARNING, logger="lerobot.configs.video"):
        cfg = RGBEncoderConfig(vcodec=vcodec)

    assert cfg.vcodec == expected
    assert any("SIGTRAP" in record.message for record in caplog.records) is warns
    if expected == "libsvtav1":
        assert cfg.preset == 12
    if expected == "h264" and vcodec in {"libsvtav1", "av1"}:
        assert cfg.preset is None
        options = cfg.get_codec_options()
        assert options["crf"] == 30
        assert options["g"] == 2
        assert "preset" not in options
        assert "svtav1-params" not in options


def test_linux_does_not_probe_the_svtav1_version(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_encoders(monkeypatch)
    monkeypatch.setattr(video_platform, "system", lambda: "Linux")

    def boom() -> tuple[int, int, int]:
        raise AssertionError("SVT-AV1 version was probed off macOS")

    monkeypatch.setattr("lerobot.configs.video._bundled_svtav1_version", boom)
    cfg = RGBEncoderConfig()
    assert cfg.vcodec == "libsvtav1"
    assert cfg.preset == 12


def test_auto_keeps_videotoolbox_without_probing_svtav1(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_encoders(monkeypatch)
    _darwin(monkeypatch, (3, 0, 0))

    def boom() -> tuple[int, int, int]:
        raise AssertionError("hardware auto-selection should not inspect SVT-AV1")

    monkeypatch.setattr("lerobot.configs.video._bundled_svtav1_version", boom)
    cfg = RGBEncoderConfig(vcodec="auto")
    assert cfg.vcodec == "h264_videotoolbox"


def test_auto_without_hardware_uses_h264_on_affected_macos(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _stub_encoders(monkeypatch, {"libsvtav1", "h264"})
    _darwin(monkeypatch, (3, 0, 0))

    with caplog.at_level(logging.WARNING, logger="lerobot.configs.video"):
        cfg = RGBEncoderConfig(vcodec="auto")

    assert cfg.vcodec == "h264"
    assert "falling back to software encoder 'libsvtav1'" not in caplog.text
    assert "Using h264 instead of libsvtav1" in caplog.text


def test_auto_without_hardware_stays_on_libsvtav1_off_macos(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _stub_encoders(monkeypatch, {"libsvtav1", "h264"})
    monkeypatch.setattr(video_platform, "system", lambda: "Linux")

    with caplog.at_level(logging.WARNING, logger="lerobot.configs.video"):
        cfg = RGBEncoderConfig(vcodec="auto")

    assert cfg.vcodec == "libsvtav1"
    assert "falling back to software encoder 'libsvtav1'" in caplog.text


def test_integer_preset_is_dropped_and_string_preset_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_encoders(monkeypatch)
    _darwin(monkeypatch, (3, 0, 0))

    dropped = RGBEncoderConfig(preset=8)
    assert dropped.vcodec == "h264"
    assert dropped.preset is None

    kept = RGBEncoderConfig(preset="fast")
    assert kept.vcodec == "h264"
    assert kept.get_codec_options()["preset"] == "fast"


def test_depth_default_is_not_svtav1(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_encoders(monkeypatch)
    _darwin(monkeypatch, (3, 0, 0))
    cfg = DepthEncoderConfig()
    assert cfg.vcodec == "hevc"


def test_stored_av1_codec_is_not_rewritten(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _stub_encoders(monkeypatch)
    _darwin(monkeypatch, (3, 0, 0))

    with caplog.at_level(logging.WARNING, logger="lerobot.configs.video"):
        cfg = RGBEncoderConfig.from_video_info({"video.codec": "av1", "video.pix_fmt": "yuv420p"})

    assert cfg.vcodec == "libsvtav1"
    assert "stored codec is kept" in caplog.text


def test_missing_stored_codec_still_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_encoders(monkeypatch)
    _darwin(monkeypatch, (3, 0, 0))
    cfg = RGBEncoderConfig.from_video_info({})
    assert cfg.vcodec == "h264"


def test_warning_is_logged_once(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _stub_encoders(monkeypatch)
    _darwin(monkeypatch, (3, 0, 0))

    with caplog.at_level(logging.WARNING, logger="lerobot.configs.video"):
        RGBEncoderConfig()
        RGBEncoderConfig(vcodec="libsvtav1")

    assert len(caplog.records) == 1
