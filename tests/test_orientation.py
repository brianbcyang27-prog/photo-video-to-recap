"""EXIF orientation: the transform that stops a portrait shot appearing sideways.

This exists because the output-level check that used to cover this was rewritten.
That check compared the rendered frame against the two possible orientations of
the source photo, and it could not decide: every still on screen is a window onto
a larger canvas, pushed in and panned, so the photo is neither where it was
composed nor the whole of the picture. Measured on the trip render, a shot at
zoom 0.142 matched its own half at +0.061, and with the frame rotated 90 degrees
- so the sideways reading was plainly true - the check still called two photos of
three upright. It reported two correct photos as sideways, and would not have
caught a genuine one on its own.

That check is kept for the failure it was written for, 106 of 141 photos lying on
their side, but that failure is decided in aggregate rather than per shot. The
individual transform is decided exactly here instead, where it is a pure function
of eight integers and needs no encoder, no frame grab and no correlation.

Two of the tests here were written and then measured to be worthless, which is
worth recording. The first attempt at the renderer test used a JPEG, and dropping
the orientation tag from `_load_source` - the actual historical bug - left it
green. The reason is in `load_image`: PIL applies EXIF itself, so on that branch
the argument is ignored. The real test has to force the sips loader, because that
is the branch the argument exists for.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pipeline.quality import apply_exif_orientation, exif_orientation, load_image


def _marker() -> Image.Image:
    """A wide image with two bright blocks, so every transform is identifiable.

    Asymmetric on both axes and in different colours, because a grey square
    cannot tell FLIP_LEFT_RIGHT from ROTATE_180 - which is exactly the class of
    bug a transformation table is prone to.
    """
    a = np.zeros((40, 80, 3), np.uint8)
    a[3:13, 4:14] = (255, 0, 0)          # top-left
    a[30:38, 66:76] = (0, 0, 255)        # bottom-right
    return Image.fromarray(a)


def _where(img: Image.Image, colour: str) -> tuple[str, str]:
    """Report where a marker ended up, in reading order."""
    a = np.asarray(img.convert("RGB"))
    channel = {"r": 0, "b": 2}[colour]
    ys, xs = np.where(a[:, :, channel] > 200)
    assert len(ys), "the marker went missing, so the test proves nothing"
    return ("top" if ys.mean() < a.shape[0] / 2 else "bottom",
            "left" if xs.mean() < a.shape[1] / 2 else "right")


# The EXIF specification, transcribed. 1 is what the camera meant; 2-4 are
# mirrors that keep the axes; 5-8 add a quarter turn and swap them.
EXPECTED = {
    1: ("landscape", ("top", "left"),   ("bottom", "right")),
    2: ("landscape", ("top", "right"),  ("bottom", "left")),
    3: ("landscape", ("bottom", "right"), ("top", "left")),
    4: ("landscape", ("bottom", "left"), ("top", "right")),
    5: ("portrait", ("top", "left"),   ("bottom", "right")),
    6: ("portrait", ("top", "right"),  ("bottom", "left")),
    7: ("portrait", ("bottom", "right"), ("top", "left")),
    8: ("portrait", ("bottom", "left"), ("top", "right")),
}

# A phone shooting a portrait frame records 6 or 8. These are the two values
# that actually turn a shot on its side, so they are the ones worth naming.
QUARTER_TURNS = (5, 6, 7, 8)


@pytest.mark.parametrize("orientation", sorted(EXPECTED))
def test_each_orientation_lands_the_markers_where_the_spec_says(orientation):
    shape, red, blue = EXPECTED[orientation]
    out = apply_exif_orientation(_marker(), orientation)
    w, h = out.size
    got = "landscape" if w >= h else "portrait"
    assert got == shape, (
        f"orientation {orientation} should display {shape}, got {got} "
        f"({w}x{h}) - getting this wrong is what lays a photo on its side"
    )
    assert _where(out, "r") == red, f"orientation {orientation} misplaced the red block"
    assert _where(out, "b") == blue, f"orientation {orientation} misplaced the blue block"


def test_orientation_one_is_the_identity():
    """It has to return something usable; identity and copy are both fine here."""
    assert np.array_equal(np.asarray(apply_exif_orientation(_marker(), 1)),
                          np.asarray(_marker()))


def test_an_unknown_or_zero_tag_leaves_the_pixels_alone():
    """A missing tag means 1. Guessing a rotation for an unstated one is worse."""
    src = _marker()
    for value in (0, 9, -1):
        assert np.array_equal(np.asarray(apply_exif_orientation(src, value)),
                              np.asarray(src)), (
            f"orientation {value} is not a real EXIF value and must not move "
            f"any pixels"
        )


@pytest.mark.parametrize("orientation", QUARTER_TURNS)
def test_no_orientation_double_rotates(orientation):
    """Applying the transform twice must not be the same as applying it once.

    A pipeline that rotated during analysis and again during composition is the
    common way a correct transform produces a sideways result, and it is
    invisible in a table test.
    """
    src = _marker()
    once = np.asarray(apply_exif_orientation(src, orientation))
    twice = np.asarray(apply_exif_orientation(
        apply_exif_orientation(src, orientation), orientation))
    assert not np.array_equal(once, twice), (
        f"orientation {orientation} is self-inverse on a double apply, so "
        f"this test cannot tell a second rotation from none"
    )


def _quarter_turned_file(tmp_path: Path, orientation: int) -> Path:
    """A portrait shot as a camera stores it: landscape pixels plus a tag.

    Written through PIL with a real Orientation tag rather than mocked, because
    the point of these tests is the whole path - file on disk, tag read back,
    pixels turned. A stubbed tag proves the table and nothing else.
    """
    path = tmp_path / f"shot_{orientation}.jpg"
    exif = _marker().getexif()          # 80x40, so upright means 40x80
    exif[0x0112] = orientation
    _marker().save(path, exif=exif, quality=95)
    return path


def test_a_file_on_disk_comes_back_the_way_up(tmp_path):
    """End to end through the file: write a tag, read it, turn the pixels."""
    path = _quarter_turned_file(tmp_path, 6)
    assert exif_orientation(path) == 6, (
        "the tag was not written or not read back, so this test would pass "
        "without exercising the transform at all"
    )
    out = load_image(path, max_long_side=0, orientation=exif_orientation(path))
    w, h = out.size
    assert h > w, (
        f"a shot tagged 6 must come back portrait, got {w}x{h} - this is the "
        f"sideways-photo bug in its purest form"
    )


def _block_pil_on(monkeypatch, path: Path) -> list[Path]:
    """Make PIL unable to open one file, the way a HEIC defeats it.

    Blocking the source path alone, not `Image.open` wholesale: `_load_via_sips`
    decodes its own converted output with `Image.open` too, so a blanket patch
    makes sips fail as well and the image silently arrives via the ffmpeg
    fallback, which autorotates on its own. That was measured - the test then
    passed for entirely the wrong reason.

    Returns the list of paths PIL refused, so a test can prove it really did
    fall through rather than believing it did.
    """

    import pipeline.quality as quality

    refused: list[Path] = []
    real_open = quality.Image.open

    def _open(p, *args, **kwargs):
        if Path(p) == path:
            refused.append(Path(p))
            raise OSError("stand-in for a format PIL lacks, like HEIC")
        return real_open(p, *args, **kwargs)

    monkeypatch.setattr(quality.Image, "open", _open)
    return refused


def test_the_orientation_argument_is_decided_on_the_sips_branch(tmp_path,
                                                               monkeypatch):
    """Why the renderer passes the tag even though one of three loaders ignores it.

    `load_image` has three loaders and they disagree about EXIF: PIL applies the
    tag itself inside `ImageOps.exif_transpose`, ffmpeg autorotates, and sips
    quietly ignores it - so only the sips branch consults the argument. This
    library is 87% HEIC and therefore almost entirely on that branch: 6,087 of
    6,972 photos cannot be opened by PIL at all.

    Both halves of that are asserted, because the argument is only justified by
    the second: a JPEG really does come back upright either way, which is
    exactly why a test written on one cannot catch the bug.
    """
    path = _quarter_turned_file(tmp_path, 6)

    # PIL's own branch: the tag is honoured regardless of the argument. JPEG is
    # the only format here PIL can open at all, so it is the only way to show it.
    by_pil = load_image(path, max_long_side=0, orientation=1)
    assert by_pil.size[1] > by_pil.size[0], (
        f"PIL applies EXIF itself, so this branch is upright even when the "
        f"caller drops the tag; got {by_pil.size}"
    )

    # sips' branch: the argument is the only thing that can turn it.
    refused = _block_pil_on(monkeypatch, path)
    with_tag = load_image(path, max_long_side=0, orientation=exif_orientation(path))
    without_tag = load_image(path, max_long_side=0, orientation=1)
    assert refused, "PIL was never asked, so this never reached sips"
    assert with_tag.size[1] > with_tag.size[0], (
        "sips ignores the tag, so the argument has to turn it and did not"
    )
    assert without_tag.size[1] < without_tag.size[0], (
        "dropping the tag on the sips branch must lay the photo on its side - if "
        "this stops being true the loader has changed and the orientation "
        "argument may no longer be needed"
    )


@pytest.mark.parametrize("orientation", QUARTER_TURNS)
def test_the_probe_reads_the_tag_and_reports_the_size_it_displays_as(
        tmp_path, orientation):
    """The other end of the handoff, where the tag is read once.

    `_load_source` forwards `info.exif_orientation`, so the value has to reach
    MediaInfo first. This is the probe pass that does it, and it also transposes
    the reported width and height to match - a quarter-turn photo is stored as
    4:3 landscape and displays as 3:4 portrait, and reporting the stored axes
    is what made the renderer centre-crop 100 Orientation-6 photos in the
    finished edit instead of giving them the blur treatment.
    """
    from pipeline.ingest import _true_photo_size
    from pipeline.util import MediaInfo

    row = {"Orientation": orientation, "ImageWidth": 80, "ImageHeight": 40}
    info = MediaInfo(path=tmp_path / "s.jpg", kind="photo")
    _true_photo_size(info, row)
    assert info.exif_orientation == orientation
    assert (info.width, info.height) == (40, 80), (
        f"orientation {orientation} displays as 40x80, reported "
        f"{info.width}x{info.height}"
    )


@pytest.mark.parametrize("value,expected", [(0, 1), (9, 1), (-3, 1), ("", 1),
                                            (None, 1), (6, 6)])
def test_a_tag_the_probe_cannot_use_falls_back_to_one(value, expected):
    """A nonsense tag means no tag. Guessing a rotation would be worse."""
    from pipeline.ingest import _true_photo_size
    from pipeline.util import MediaInfo

    info = MediaInfo(path=Path("s.jpg"), kind="photo")
    _true_photo_size(info, {"Orientation": value, "ImageWidth": 80,
                             "ImageHeight": 40})
    assert info.exif_orientation == expected, (
        f"tag {value!r} should be read as {expected}"
    )


@pytest.mark.parametrize("orientation", QUARTER_TURNS)
def test_the_renderer_hands_the_composer_an_upright_photo(tmp_path, monkeypatch,
                                                          orientation):
    """The one call that decides what gets composited, exercised for real.

    `_load_source` is where a still crosses into the render. If it drops the
    tag, every quarter-turn shot in the film is laid on its side and nothing
    downstream can put it right.
    """
    from pipeline.analysis import Item
    from pipeline.render import _load_source
    from pipeline.select import Entry
    from pipeline.util import MediaInfo

    path = _quarter_turned_file(tmp_path, orientation)
    refused = _block_pil_on(monkeypatch, path)

    # The tag is read once, by the probe, and handed on. That is what the
    # renderer consumes, so this is the value it must forward.
    info = MediaInfo(path=path, kind="photo", width=80, height=40,
                     exif_orientation=orientation)
    entry = Entry(item=Item(info=info, kind="photo"), duration=3.0)
    out = _load_source(entry, None, 0)

    assert refused, "PIL was never asked, so this never reached sips"
    assert out is not None, "the still could not be read at all"
    w, h = out.size
    assert h > w, (
        f"orientation {orientation} reached the composer as {w}x{h}, i.e. on "
        f"its side - every portrait shot in the film would be sideways"
    )
    # And it must be the upright image, not merely the right shape.
    assert _where(out, "r") == EXPECTED[orientation][1], (
        f"orientation {orientation} reached the composer turned the wrong way"
    )


def test_a_video_frame_is_not_given_the_stills_orientation_tag():
    """EXIF orientation describes stills. A video frame is already upright.

    Applying it to a frame grabbed from a clip would turn any MOV carrying a
    stray tag on its side. The Live Photo tail push reads a frame out of a clip,
    so it is the one place this could have gone wrong, and it is checked by
    reading the function's own source because the alternative - building a clip
    whose decoded frame happens to carry a stray tag - is not something a writer
    would produce on demand.
    """
    from pipeline.render import _push_from_last_frame

    src = inspect.getsource(_push_from_last_frame)
    assert "orientation=1" in src, (
        "a frame read out of a video must be used as-is; applying a still's "
        "orientation tag to it would rotate the Live Photo tail"
    )
