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
# Note: We subclass str so that serialization is straightforward
# https://stackoverflow.com/questions/24481852/serialising-an-enum-member-to-json

"""Video encoder configurations."""

from __future__ import annotations

import logging
import os
import platform
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

import numpy as np

from lerobot.utils.import_utils import require_package

logger = logging.getLogger(__name__)

# List of hardware encoders to probe for auto-selection. Availability depends on the platform and the chosen video backend.
# Determines the order of preference for auto-selection when vcodec="auto" is used.
HW_VIDEO_CODECS = [
    "h264_videotoolbox",  # macOS
    "hevc_videotoolbox",  # macOS
    "h264_nvenc",  # NVIDIA GPU
    "hevc_nvenc",  # NVIDIA GPU
    "h264_vaapi",  # Linux Intel/AMD
    "h264_qsv",  # Intel Quick Sync
]
VALID_VIDEO_CODECS: frozenset[str] = frozenset(
    {"h264", "hevc", "libsvtav1", "libaom-av1", "auto", *HW_VIDEO_CODECS}
)
# Aliases for legacy video codec names.
VIDEO_CODECS_ALIASES: dict[str, str] = {"av1": "libsvtav1"}

LIBSVTAV1_DEFAULT_PRESET: int = 12

# SVT-AV1 before 4.3.0 can SIGTRAP on macOS when an encoder is destroyed.
# libdispatch aborts if a semaphore is freed below its initial value
# (AOMediaCodec/SVT-AV1#2961, fixed by 049193fc in 4.3.0). Linux and Windows
# semaphores do not check, so the same library is safe there. Current PyAV
# wheels still ship an affected SVT-AV1.
SVTAV1_MACOS_SEMAPHORE_FIX: tuple[int, int, int] = (4, 3, 0)
MACOS_SVTAV1_SAFE_ENCODER: str = "h264"
LEROBOT_ALLOW_SVTAV1_ENV: str = "LEROBOT_ALLOW_SVTAV1"
_SVTAV1_MACOS_ISSUE_URL: str = "https://github.com/huggingface/lerobot/issues/5308"
_SVTAV1_VERSION_RE: re.Pattern[str] = re.compile(r"(\d+)\.(\d+)\.(\d+)")
# Auditwheel hashes the SONAME (`libSvtAv1Enc-<hash>.so.3.0.0`). macOS wheels use
# `libSvtAv1Enc.3.0.0.dylib`. A major-only SONAME (`libSvtAv1Enc.so.3`) does not match.
_SVTAV1_LIBRARY_VERSION_RE: re.Pattern[str] = re.compile(
    r"libSvtAv1Enc(?:-[0-9A-Fa-f]+)?(?:\.so)?\.(\d+)\.(\d+)\.(\d+)",
    re.IGNORECASE,
)
# Reconstruction from a dataset's stored codec must not silently switch encoders.
_APPLY_MACOS_SVTAV1_GUARD: ContextVar[bool] = ContextVar("lerobot_apply_macos_svtav1_guard", default=True)
_macos_svtav1_warned: bool = False


def _format_svtav1_version(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)


def parse_svtav1_version(text: str) -> tuple[int, int, int] | None:
    """Return the first ``major.minor.patch`` in *text*, or ``None``."""
    match = _SVTAV1_VERSION_RE.search(text)
    if match is None:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def svtav1_version_from_library_name(name: str) -> tuple[int, int, int] | None:
    """Read an SVT-AV1 version from a ``libSvtAv1Enc`` filename, if it has one.

    PyAV's macOS wheel names the affected build ``libSvtAv1Enc.3.0.0.dylib``.
    A SONAME with only the major version (``libSvtAv1Enc.so.3``) cannot be
    classified and yields ``None``.
    """
    match = _SVTAV1_LIBRARY_VERSION_RE.search(name.replace("\\", "/"))
    if match is None:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def macos_svtav1_close_is_unsafe(version: tuple[int, ...] | None) -> bool:
    """Whether *version* is a known SVT-AV1 that can SIGTRAP on encoder close.

    ``None`` means the version could not be read. That is not treated as a
    known-affected build.
    """
    if not version:
        return False
    parts = tuple(int(part) for part in version[:3])
    padded = parts + (0,) * (3 - len(parts))
    return padded < SVTAV1_MACOS_SEMAPHORE_FIX


def _svtav1_explicitly_allowed() -> bool:
    return os.environ.get(LEROBOT_ALLOW_SVTAV1_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _warn_macos_svtav1_once(message: str) -> None:
    global _macos_svtav1_warned
    if _macos_svtav1_warned:
        return
    _macos_svtav1_warned = True
    logger.warning(message)


def _bundled_svtav1_version() -> tuple[int, int, int] | None:
    # Lazy: pyav_utils imports PyAV, an optional dataset extra. Importing it at
    # module scope would make ``lerobot.configs`` require that extra.
    from lerobot.datasets.pyav_utils import bundled_svtav1_version

    return bundled_svtav1_version()


# Keys persisted under ``features[*]["info"]`` as ``video.<name>`` (from :class:`VideoEncoderConfig`).
# ``vcodec``` and ``pix_fmt`` are derived from the video stream directly.
VIDEO_ENCODER_INFO_FIELD_NAMES: frozenset[str] = frozenset(
    {"g", "crf", "preset", "fast_decode", "extra_options", "video_backend"}
)
VIDEO_ENCODER_INFO_KEYS: frozenset[str] = frozenset(
    f"video.{name}" for name in VIDEO_ENCODER_INFO_FIELD_NAMES
)

# Default depth quantization and encoding parameters.
DEPTH_QUANT_BITS: int = 12
DEPTH_QMAX: int = (1 << DEPTH_QUANT_BITS) - 1  # 4095

DEFAULT_DEPTH_MIN: float = 0.01
DEFAULT_DEPTH_MAX: float = 10.0
DEFAULT_DEPTH_SHIFT: float = 3.5
DEFAULT_DEPTH_USE_LOG: bool = True
DEFAULT_DEPTH_PIX_FMT: str = "gray12le"

DEPTH_METER_UNIT: str = "m"
DEPTH_MILLIMETER_UNIT: str = "mm"
DEFAULT_DEPTH_UNIT: str = DEPTH_MILLIMETER_UNIT


def infer_depth_unit(dtype: np.dtype | type) -> str:
    """Infer the physical unit of raw depth frames from their dtype.

    Floating-point frames are assumed to be in metres, integer frames in millimetres.
    """
    return DEPTH_METER_UNIT if np.issubdtype(np.dtype(dtype), np.floating) else DEPTH_MILLIMETER_UNIT


# Depth-specific tuning fields persisted under ``features[*]["info"]`` as ``video.<name>``.
DEPTH_ENCODER_INFO_FIELD_NAMES: frozenset[str] = frozenset({"depth_min", "depth_max", "shift", "use_log"})


@dataclass
class VideoEncoderConfig:
    """Video encoder configuration."""

    vcodec: str = "libsvtav1"  # Video codec. "auto" picks a hardware codec if available, else libsvtav1. On macOS with SVT-AV1 < 4.3.0, libsvtav1 is replaced with h264.
    pix_fmt: str = "yuv420p"  # Pixel format (e.g. yuv420p).
    g: int | None = 2  # GOP size (keyframe interval).
    crf: int | float | None = 30  # Quality level. Lower means better quality and larger files.
    preset: int | str | None = None  # Speed/quality preset. Accepted values are codec-specific.
    fast_decode: int = 0  # Fast-decode tuning. Accepted values are codec-specific, 0 disables it.
    # TODO(CarolinePascal): add torchcodec support + find a way to unify the
    # two backends (encoding and decoding).
    video_backend: str = "pyav"  # Encoding backend. Only "pyav" is currently supported.
    # Extra codec options merged last, e.g. {"tune": "film"}.
    extra_options: dict[str, Any] = field(default_factory=dict)

    # Source-data channel count this encoder is expected to handle. ``None``
    # disables the pix_fmt channel-count check; concrete subclasses set it
    # (3 for RGB, 1 for depth, etc.).
    _DEFAULT_CHANNELS: ClassVar[int | None] = None

    def __post_init__(self) -> None:
        self.resolve_vcodec()
        # Empty-constructor ergonomics: ``VideoEncoderConfig()`` must "just work".
        if self.preset is None and self.vcodec == "libsvtav1":
            self.preset = LIBSVTAV1_DEFAULT_PRESET
        self.validate()

    @classmethod
    def _kwargs_from_video_info(cls, video_info: dict | None) -> dict[str, Any]:
        """Parse the ``video.*`` keys of a feature ``info`` block into
        constructor kwargs.
        """
        video_info = video_info or {}
        kwargs: dict[str, Any] = {}

        for src_key, dst_field in (("video.codec", "vcodec"), ("video.pix_fmt", "pix_fmt")):
            value = video_info.get(src_key)
            if value is not None:
                kwargs[dst_field] = value

        for field_name in VIDEO_ENCODER_INFO_FIELD_NAMES:
            value = video_info.get(f"video.{field_name}")
            if value is None:
                continue
            # Persisted as ``{}`` after merges with disagreeing sources — treat as default.
            if field_name == "extra_options" and not value:
                continue
            kwargs[field_name] = value

        return kwargs

    @classmethod
    def from_video_info(cls, video_info: dict | None) -> Self:
        """Reconstruct an encoder config from a video feature's ``info`` block.

        Missing or ``None`` values fall back to the class defaults. A codec
        recorded in the block is kept, including ``libsvtav1`` on macOS, so
        editing an existing dataset does not switch codecs.
        """
        video_info = video_info or {}
        codec_was_stored = video_info.get("video.codec") is not None
        token = _APPLY_MACOS_SVTAV1_GUARD.set(not codec_was_stored)
        try:
            return cls(**cls._kwargs_from_video_info(video_info))
        finally:
            _APPLY_MACOS_SVTAV1_GUARD.reset(token)

    def detect_available_encoders(self, encoders: list[str] | str) -> list[str]:
        """Return the subset of available encoders based on the specified video backend.

        Args:
            encoders: List of encoder names to detect. If a string, it is converted to a list.
        Returns:
            List of available encoder names. If the video backend is not "pyav", returns an empty list.
        """
        if self.video_backend == "pyav":
            require_package("av", extra="dataset")
            from lerobot.datasets import detect_available_encoders_pyav

            return detect_available_encoders_pyav(encoders)
        return []

    def validate(self) -> None:
        """Validate the video encoder configuration."""
        if self.video_backend == "pyav":
            require_package("av", extra="dataset")
            from lerobot.datasets import check_video_encoder_parameters_pyav

            check_video_encoder_parameters_pyav(
                self.vcodec, self.pix_fmt, self.get_codec_options(), channels=self._DEFAULT_CHANNELS
            )

    def resolve_vcodec(self) -> None:
        """Check ``vcodec`` and, when it is ``"auto"``, pick a concrete encoder.

        For ``"auto"``, the first hardware encoder in the preference list that is
        available is chosen. If none are available, ``libsvtav1`` is used. On
        macOS, a ``libsvtav1`` choice whose bundled SVT-AV1 is older than 4.3.0
        is replaced with ``h264``. If the resolved codec is not available,
        raises ``ValueError``.

        Stream-derived canonical codec names listed in :data:`VIDEO_CODECS_ALIASES` are
        rewritten to their corresponding encoder name (e.g. ``"av1"`` → ``"libsvtav1"``).
        Codecs read back from dataset metadata are not replaced.
        """
        self.vcodec = VIDEO_CODECS_ALIASES.get(self.vcodec, self.vcodec)
        if self.vcodec not in VALID_VIDEO_CODECS:
            raise ValueError(f"Invalid vcodec '{self.vcodec}'. Must be one of: {sorted(VALID_VIDEO_CODECS)}")
        replaced = False
        if self.vcodec == "auto":
            available = self.detect_available_encoders(HW_VIDEO_CODECS)
            for encoder in HW_VIDEO_CODECS:
                if encoder in available:
                    logger.info(f"Auto-selected video codec: {encoder}")
                    self.vcodec = encoder
                    return
            self.vcodec = "libsvtav1"
            if _APPLY_MACOS_SVTAV1_GUARD.get():
                replaced = self._maybe_replace_macos_svtav1()
            if not replaced:
                logger.warning("No hardware encoder available, falling back to software encoder 'libsvtav1'")
        elif _APPLY_MACOS_SVTAV1_GUARD.get():
            self._maybe_replace_macos_svtav1()
        else:
            self._warn_if_stored_macos_svtav1_is_unsafe()

        if self.detect_available_encoders(self.vcodec):
            logger.info(f"Using video codec: {self.vcodec}")
            return
        raise ValueError(f"Unsupported video codec: {self.vcodec} with video backend {self.video_backend}")

    def _maybe_replace_macos_svtav1(self) -> bool:
        """Replace a known-bad macOS ``libsvtav1`` with software H.264.

        Returns whether the codec was replaced. No-op off macOS, on SVT-AV1
        4.3.0 or newer, when the version cannot be read, and when
        ``LEROBOT_ALLOW_SVTAV1`` is set.
        """
        if self.vcodec != "libsvtav1" or platform.system() != "Darwin":
            return False
        version = _bundled_svtav1_version()
        rendered = "an unknown version" if version is None else _format_svtav1_version(version)
        if _svtav1_explicitly_allowed():
            if version is None or macos_svtav1_close_is_unsafe(version):
                _warn_macos_svtav1_once(
                    f"{LEROBOT_ALLOW_SVTAV1_ENV} is set, so libsvtav1 stays selected on macOS. "
                    f"SVT-AV1 {rendered} can abort the process with SIGTRAP when an encoder closes "
                    f"({_SVTAV1_MACOS_ISSUE_URL})."
                )
            return False
        if version is None:
            _warn_macos_svtav1_once(
                "Could not read the bundled SVT-AV1 version. On macOS, SVT-AV1 older than "
                f"{_format_svtav1_version(SVTAV1_MACOS_SEMAPHORE_FIX)} can abort the process with "
                "SIGTRAP when an encoder closes, so libsvtav1 stays selected. Pass "
                f"vcodec={MACOS_SVTAV1_SAFE_ENCODER}, or install a build that bundles SVT-AV1 "
                f"{_format_svtav1_version(SVTAV1_MACOS_SEMAPHORE_FIX)} or newer "
                f"({_SVTAV1_MACOS_ISSUE_URL})."
            )
            return False
        if not macos_svtav1_close_is_unsafe(version):
            return False
        if not self.detect_available_encoders(MACOS_SVTAV1_SAFE_ENCODER):
            _warn_macos_svtav1_once(
                f"libsvtav1 on macOS is linked to SVT-AV1 {rendered}, which can abort the process "
                "with SIGTRAP when an encoder closes, and "
                f"{MACOS_SVTAV1_SAFE_ENCODER} is not available to replace it "
                f"({_SVTAV1_MACOS_ISSUE_URL})."
            )
            return False
        if isinstance(self.preset, int) and not isinstance(self.preset, bool):
            # SVT-AV1 presets are integers. x264 presets are names, and the integer
            # would fail validation on the replacement encoder.
            self.preset = None
        self.vcodec = MACOS_SVTAV1_SAFE_ENCODER
        _warn_macos_svtav1_once(
            f"Using {MACOS_SVTAV1_SAFE_ENCODER} instead of libsvtav1 on macOS. The bundled SVT-AV1 "
            f"{rendered} can abort the process with SIGTRAP (exit 133) when an encoder closes. "
            f"SVT-AV1 {_format_svtav1_version(SVTAV1_MACOS_SEMAPHORE_FIX)} fixes that race. "
            f"Set {LEROBOT_ALLOW_SVTAV1_ENV}=1 to keep libsvtav1 ({_SVTAV1_MACOS_ISSUE_URL})."
        )
        return True

    def _warn_if_stored_macos_svtav1_is_unsafe(self) -> None:
        """Warn when dataset metadata keeps ``libsvtav1`` on an affected macOS build."""
        if self.vcodec != "libsvtav1" or platform.system() != "Darwin":
            return
        version = _bundled_svtav1_version()
        if version is not None and not macos_svtav1_close_is_unsafe(version):
            return
        rendered = "an unknown version" if version is None else _format_svtav1_version(version)
        _warn_macos_svtav1_once(
            "This dataset video uses libsvtav1. On macOS the bundled SVT-AV1 "
            f"{rendered} can abort the process with SIGTRAP if those videos are re-encoded. "
            f"The stored codec is kept ({_SVTAV1_MACOS_ISSUE_URL})."
        )

    def get_codec_options(
        self, encoder_threads: int | None = None, as_strings: bool = False
    ) -> dict[str, Any]:
        """Translate the tuning fields to codec-specific options.

        ``VideoEncoderConfig.extra_options`` are merged last but never override a structured field.

        Args:
            encoder_threads: Number of encoder threads set globally for all VideoEncoderConfigs.
                For libsvtav1, this is mapped to ``lp`` via ``svtav1-params``.
                For h264/hevc, this is mapped to ``threads``.
                Hardware encoders ignore this parameter.
            as_strings: If ``True``, casts values to strings.
        """
        opts: dict[str, Any] = {}

        def set_if(key: str, value: Any) -> None:
            if value is not None:
                opts[key] = value if not as_strings else str(value)

        # GOP size is not a codec-specific option, so it is always set.
        set_if("g", self.g)

        if self.vcodec == "libsvtav1":
            set_if("crf", self.crf)
            set_if("preset", self.preset)
            svtav1_parts: list[str] = []
            if self.fast_decode is not None:
                svtav1_parts.append(f"fast-decode={max(0, min(2, self.fast_decode))}")
            if encoder_threads is not None:
                svtav1_parts.append(f"lp={encoder_threads}")
            if svtav1_parts:
                set_if("svtav1-params", ":".join(svtav1_parts))
        elif self.vcodec in ("h264", "hevc"):
            set_if("crf", self.crf)
            set_if("preset", self.preset)
            if self.fast_decode:
                set_if("tune", "fastdecode")
            set_if("threads", encoder_threads)
        elif self.vcodec == "libaom-av1":
            set_if("crf", self.crf)
            set_if("preset", self.preset)
            if encoder_threads is not None:
                set_if("threads", encoder_threads)
                set_if("row-mt", 1)
        elif self.vcodec in ("h264_videotoolbox", "hevc_videotoolbox"):
            if self.crf is not None:
                set_if("q:v", max(1, min(100, 100 - self.crf * 2)))
        elif self.vcodec in ("h264_nvenc", "hevc_nvenc"):
            set_if("rc", 0)
            set_if("qp", self.crf)
            set_if("preset", self.preset)
        elif self.vcodec == "h264_vaapi":
            set_if("qp", self.crf)
        elif self.vcodec == "h264_qsv":
            set_if("global_quality", self.crf)
            set_if("preset", self.preset)
        else:
            set_if("crf", self.crf)
            set_if("preset", self.preset)

        # Extra options are merged last but never override structured fields (values are kept as given).
        for k, v in self.extra_options.items():
            if k not in opts:
                set_if(k, v)

        return opts


@dataclass
class RGBEncoderConfig(VideoEncoderConfig):
    """Encoder configuration for RGB camera streams.

    Identical to :class:`VideoEncoderConfig` but declares the 3-channel
    source-data layout so ``pix_fmt`` is validated against RGB inputs.
    """

    _DEFAULT_CHANNELS: ClassVar[int] = 3


def rgb_encoder_defaults() -> RGBEncoderConfig:
    """Return a :class:`RGBEncoderConfig` with RGB-camera defaults."""
    return RGBEncoderConfig()


@dataclass
class DepthEncoderConfig(VideoEncoderConfig):
    """Encoder configuration for depth-map streams.

    Inherits the full :class:`VideoEncoderConfig` surface (codec, GOP, CRF,
    preset, ``extra_options``…) and adds the parameters of the depth quantizer.
    Defaults flip ``vcodec`` to ``"hevc"`` (Main 12 profile) and ``pix_fmt`` to
    ``"gray12le"``.
    """

    vcodec: str = "hevc"  # Video codec name. Defaults to HEVC Main 12 (a 12-bit-capable codec).
    pix_fmt: str = "gray12le"  # Pixel format. Defaults to 12-bit grayscale.
    extra_options: dict[str, Any] = field(default_factory=lambda: {"x265-params": "lossless=1"})

    depth_min: float = DEFAULT_DEPTH_MIN  # Minimum depth in meters, mapped to the lowest quantum.
    depth_max: float = DEFAULT_DEPTH_MAX  # Maximum depth in meters, mapped to the highest quantum.
    shift: float = DEFAULT_DEPTH_SHIFT  # Pre-log offset in meters for numerical stability near zero.
    use_log: bool = DEFAULT_DEPTH_USE_LOG  # Use logarithmic quantization (True) or linear (False).

    _DEFAULT_CHANNELS: ClassVar[int] = 1

    @classmethod
    def _kwargs_from_video_info(cls, video_info: dict | None) -> dict[str, Any]:
        """Layer the depth-specific tuning (``depth_min`` / ``depth_max`` /
        ``shift`` / ``use_log``) on top of the base parser. Missing keys
        fall back to the class defaults.
        """
        kwargs = super()._kwargs_from_video_info(video_info)
        video_info = video_info or {}
        for name in DEPTH_ENCODER_INFO_FIELD_NAMES:
            value = video_info.get(f"video.{name}")
            if value is not None:
                kwargs[name] = value
        return kwargs


def depth_encoder_defaults() -> DepthEncoderConfig:
    """Return a :class:`DepthEncoderConfig` with depth-camera defaults."""
    return DepthEncoderConfig()


def is_depth_map(feature: dict | None) -> bool:
    """Return whether a feature is flagged as a depth map.

    Depth maps are flagged canonically via ``feature['info']['is_depth_map']`` or through the
    legacy ``video.is_depth_map`` key, which may live in ``feature['info']`` or in a separate
    ``feature['video_info']`` dict.
    """
    feature = feature or {}
    info = feature.get("info") or {}
    video_info = feature.get("video_info")
    return bool(
        info.get("is_depth_map")
        or info.get("video.is_depth_map")
        or (isinstance(video_info, dict) and video_info.get("video.is_depth_map"))
    )


def encoder_config_from_video_info(video_info: dict | None) -> VideoEncoderConfig:
    """Build the appropriate encoder config from a feature's ``info`` block.

    Dispatches to :class:`DepthEncoderConfig` when the dict marks the feature
    as a depth map and to :class:`RGBEncoderConfig`
    otherwise.

    Args:
        video_info: A feature's ``info`` dict as persisted in ``info.json``,
            or ``None`` (treated as an empty dict).

    Returns:
        A :class:`DepthEncoderConfig` for depth features, otherwise a
        :class:`RGBEncoderConfig`.
    """
    video_info = video_info or {}
    cls: type[VideoEncoderConfig] = (
        DepthEncoderConfig if is_depth_map({"info": video_info}) else RGBEncoderConfig
    )
    return cls.from_video_info(video_info)
