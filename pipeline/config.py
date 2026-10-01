"""Central settings for the auto-video pipeline.

Everything the user might want to tweak lives here. Defaults are chosen to
produce a watchable travel recap with no arguments at all.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- media types

PHOTO_EXT = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff",
    ".dng", ".arw", ".cr2", ".nef", ".gif",
}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".insv", ".mts", ".m2ts", ".webm", ".3gp"}
AUDIO_EXT = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".aiff", ".aif", ".ogg"}


# Named output sizes, so "4K" means one thing everywhere instead of four
# separate width/height pairs drifting apart.
#
# 4K is worth having for *stills* and not for *video*. A phone photo is
# 4032x3024, so 3840x2160 is a genuine downscale with real detail in it. A
# phone video is 1920x1440 at most, so the same 4K frame is a 2.7x upscale:
# measured near-Nyquist energy rises 10%, which is interpolation rather than
# resolution. It is not worse, but it is not better either, and it costs
# bitrate on pixels carrying no information. So 4K is a choice rather than a
# default, and 1080p60 is the better deal when the library is mostly video.
SIZE_PRESETS = {
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2160p": (3840, 2160),
    "4k": (3840, 2160),
}

# Named here rather than spelled out in a default=, so the CLI help, the
# config default and the test that checks the default all read from one place.
DEFAULT_SIZE = "1080p"


@dataclass
class Render:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    # "blur" = blurred copy fills frame, subject centred on top (never crops).
    # "crop" = centre-crop to fill (may cut content).
    # "pad"  = black bars.
    fit: str = "blur"
    # Apply `fit` per shot rather than to the whole video. A 4:3 landscape photo
    # in a 16:9 frame only has ~25% of its width to give up, so centre-cropping
    # costs a strip of sky and foreground and looks deliberate; blur-filling the
    # same photo puts blurred bars down both sides and looks like a mistake.
    # Portrait shots are the opposite case - cropping one to 16:9 throws away
    # most of the frame - so those keep the blur treatment and nothing is cut.
    # Measured on the real library, 139 of 141 photos are 4:3 landscape, so this
    # is the difference between bars on nearly every shot and bars on none.
    # False restores the old behaviour of using `fit` for everything.
    fit_per_shot: bool = True
    # A shot is filled edge-to-edge when it is at least this fraction as wide as
    # the frame; below that the mismatch is big enough that cropping starts
    # costing real content. 4:3 lands at 0.75 and square at 0.56, so 0.70
    # separates them with margin rather than sitting exactly on the boundary.
    fit_crop_min_aspect: float = 0.70
    # Show two portrait photos side by side rather than one at a time with
    # blurred sides. This library turned out to be 108 portrait photos out of
    # 141, so the single-photo treatment put bars down both edges of three
    # quarters of the film; a spread halves the waste and reads as a photo
    # album rather than as a video that could not decide what shape it was.
    # Only stills pair with stills, so a pair is silent either way and the
    # audio track needs no special case.
    two_up: bool = True
    crf: int = 18
    preset: str = "veryfast"
    # Cap on the long side of the intermediate still frame. The Ken Burns move
    # is supersampled, so the working image is several times the delivery
    # frame; at 1080p that is about 5500px wide, which is already past the
    # 4032px a 12MP iPhone photo provides. 0 means no cap, which is right for
    # a library of phone photos and wrong for one of scanned film.
    still_max_long_side: int = 0
    audio_bitrate: str = "192k"
    music_volume: float = 0.85
    # Ken Burns. 0 disables motion entirely.
    ken_burns: bool = True
    zoom_amount: float = 0.10
    title_card: bool = True
    title: str = ""
    subtitle: str = ""
    end_card: bool = True
    end_text: str = ""


@dataclass
class Analysis:
    """How aggressively we sample and score."""

    # Frames per second pulled from each video for analysis.
    video_sample_fps: float = 3.0
    # Long edge used for EVERY analysis copy, stills and video alike.
    # Sharpness metrics are resolution-dependent, so both media types must be
    # measured at the same scale or video scores unfairly low against photos.
    analysis_long_side: int = 640
    # Shot-boundary sensitivity: lower finds more cuts.
    scene_threshold: float = 0.28
    min_shot_seconds: float = 0.8
    max_shot_seconds: float = 6.0

    # Hard rejects. Thresholds are calibrated for analysis_long_side=640;
    # good phone shots land around 800-2000, unusable ones below ~100.
    min_sharpness: float = 60.0        # variance-of-Laplacian floor
    max_clipped: float = 0.28          # fraction of blown-out pixels
    max_black: float = 0.45            # fraction of crushed pixels

    # Relative weights. Faces/people usually matter most on trips.
    w_sharpness: float = 0.30
    w_exposure: float = 0.20
    w_faces: float = 0.20
    w_color: float = 0.12
    w_contrast: float = 0.10
    w_motion: float = 0.08
    # Multiplier applied to photo score so they compete fairly with video.
    photo_bias: float = 1.0

    # Never use two visually near-identical items.
    dedupe_threshold: int = 10         # Hamming distance on 64-bit dHash
    # Don't stack too much from one moment (bursts / same minute).
    max_per_minute: int = 2


@dataclass
class Music:
    mode: str = "auto"        # "auto" | "generate" | "track"
    track_path: str = ""
    mood: str = "uplifting"   # uplifting | cinematic | calm | energetic | warm
    bpm: int = 0              # 0 = let the pipeline choose
    # Keep music under native audio automatically.
    duck: bool = True


@dataclass
class Pipeline:
    target_seconds: int = 90
    order: str = "chronological"   # chronological | hook (best clip first)
    render: Render = field(default_factory=Render)
    analysis: Analysis = field(default_factory=Analysis)
    music: Music = field(default_factory=Music)

    # Per-item on-screen duration bounds, before beat snapping (seconds).
    photo_min: float = 2.6
    photo_max: float = 4.6
    video_min: float = 1.8
    video_max: float = 5.0

    jobs: int = 0              # 0 = auto (cpu_count - 1)
    keep_temp: bool = False
    verbose: bool = True
    # Play a Live Photo's motion clip and ambient sound instead of panning the
    # still. On a library shot entirely on an iPhone this is most of the
    # footage: it turns a flat pan into real movement, gives the film sound
    # from the moment each picture was taken, and stops the still and its clip
    # competing for the same slot in the timeline.
    live_photos: bool = True
    # Split chapters on the calendar day alone, leaving a day that covered two
    # places as one chapter. False also splits on a 2.5km move, which suits a
    # single-day recap but would cut a travel day into five fragments.
    chapter_on_move: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def default_paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "media": root / "media",
        "music": root / "music",
        "out": root / "out",
        "work": root / ".work",
        "cache": root / ".cache",
    }
