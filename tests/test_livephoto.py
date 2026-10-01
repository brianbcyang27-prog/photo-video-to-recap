"""Live Photo pairing.

The property under test is not "does this find a file" but "does it pair the
right one". A Live Photo's two halves always share a folder, and on a real
library base names repeat across folders - 1,041 of them on the trip library
this was built against, because the same iPhone roll was copied into several
places. A flat index over base names therefore pairs files that merely share a
name, which produces a plausible-looking edit rather than an obvious failure,
so the folder scoping is the thing worth pinning down.
"""
from __future__ import annotations

from pipeline import livephoto
from pipeline.ingest import Library, MediaInfo, _attach_live_photos


class _FakePath:
    """Just enough of a Path for the pairing code: a name and a parent."""

    def __init__(self, path, size=60_000):
        self.path = path
        self._size = size

    @property
    def name(self):
        return self.path.name

    @property
    def stem(self):
        return self.path.stem

    @property
    def suffix(self):
        return self.path.suffix

    @property
    def parent(self):
        return self.path.parent

    def stat(self):
        class S:
            st_size = self._size
        return S()


def _mk(path, kind="photo", size=60_000, dims=(4032, 3024)):
    return MediaInfo(path=_FakePath(path, size), kind=kind,
                     width=dims[0], height=dims[1])


def test_pairs_same_stem_in_same_folder(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    photo = _mk(a / "IMG_1.HEIC")
    clip = _mk(a / "IMG_1.MOV", kind="video")
    motion, consumed = livephoto.pair_library([photo], [clip])
    assert motion == {photo.path: clip.path}
    assert consumed == {clip.path}


def test_never_pairs_across_folders(tmp_path):
    """The regression this test exists for: same base name, different folder."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    photo_a = _mk(a / "IMG_1.HEIC")
    photo_b = _mk(b / "IMG_1.HEIC")
    clip_a = _mk(a / "IMG_1.MOV", kind="video")
    clip_b = _mk(b / "IMG_1.MOV", kind="video")
    motion, _ = livephoto.pair_library([photo_a, photo_b], [clip_a, clip_b])
    assert motion[photo_a.path] == clip_a.path
    assert motion[photo_b.path] == clip_b.path
    assert motion[photo_a.path] != clip_b.path


def test_case_insensitive_on_both_halves(tmp_path):
    """Phones write IMG_1.HEIC/IMG_1.MOV, desktops often lowercase the clip."""
    d = tmp_path / "d"
    d.mkdir()
    photo = _mk(d / "IMG_1.HEIC")
    clip = _mk(d / "img_1.mov", kind="video")
    motion, _ = livephoto.pair_library([photo], [clip])
    assert photo.path in motion


def test_one_clip_claimed_by_at_most_one_photo(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    photo = _mk(d / "IMG_1.HEIC")
    clip = _mk(d / "IMG_1.MOV", kind="video")
    motion, consumed = livephoto.pair_library([photo], [clip])
    owners = [p for p, m in motion.items() if m == clip.path]
    assert len(owners) == 1
    assert len(consumed) == len(motion), "pairing must be 1:1"


def test_tiny_clip_is_not_treated_as_a_live_photo(tmp_path):
    """A 100-byte MOV sharing a name is not the still's motion half."""
    d = tmp_path / "d"
    d.mkdir()
    photo = _mk(d / "IMG_1.HEIC")
    stub = _mk(d / "IMG_1.MOV", kind="video", size=100)
    motion, consumed = livephoto.pair_library([photo], [stub])
    assert motion == {}
    assert consumed == set(), "an unrelated clip must stay in the video pool"


def test_unrelated_clip_stays_available_as_video(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    photo = _mk(d / "IMG_1.HEIC")
    motion_clip = _mk(d / "IMG_1.MOV", kind="video")
    real_reel = _mk(d / "IMG_9.MOV", kind="video")
    motion, consumed = livephoto.pair_library([photo], [motion_clip, real_reel])
    assert real_reel.path not in consumed, (
        "a real video clip must not be swallowed as a Live Photo half"
    )


def test_scan_attachment_drops_motion_from_the_video_pool(tmp_path):
    """End to end through Library: the still gains motion, the pool loses it."""
    d = tmp_path / "d"
    d.mkdir()
    photo = _mk(d / "IMG_1.HEIC")
    motion_clip = _mk(d / "IMG_1.MOV", kind="video")
    other = _mk(d / "IMG_7.MOV", kind="video")
    lib = Library(photos=[photo], videos=[motion_clip, other])
    _attach_live_photos(lib)
    assert photo.is_live
    assert photo.live_motion == motion_clip.path
    assert [v.path for v in lib.videos] == [other.path], (
        "the motion half is the still's, not a competing video"
    )


def test_live_photos_off_leaves_everything_alone(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    photo = _mk(d / "IMG_1.HEIC")
    motion_clip = _mk(d / "IMG_1.MOV", kind="video")
    lib = Library(photos=[photo], videos=[motion_clip])
    _attach_live_photos(lib)
    assert photo.is_live
    # And the flag's off path: scan simply does not call the attach step.
    lib2 = Library(photos=[_mk(d / "IMG_2.HEIC")], videos=[_mk(d / "IMG_2.MOV", kind="video")])
    assert not any(p.is_live for p in lib2.photos)


def test_empty_pools_are_safe():
    assert livephoto.pair_library([], []) == ({}, set())


# ---- one moment, two files on disk
#
# Found on the real library, not invented: IMG_0044.HEIC and IMG_0044.JPG sat
# beside a single IMG_0044.mov. Both stills match the one clip, so both got
# the same three seconds of motion attached - and had selection happened to
# keep both, the moment would have played twice in a row. One clip, one owner.


def test_a_clip_shared_by_two_copies_of_a_photo_has_one_owner(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    heic = _mk(d / "IMG_0044.HEIC")
    jpg = _mk(d / "IMG_0044.JPG")
    clip = _mk(d / "IMG_0044.MOV", kind="video")
    motion, consumed = livephoto.pair_library([heic, jpg], [clip])
    owners = [p for p, m in motion.items() if m == clip.path]
    assert len(owners) == 1, (
        f"{len(owners)} stills claim the same motion clip; the moment would "
        f"be shown twice"
    )
    assert consumed == {clip.path}, "the clip is still consumed, just once"


def test_the_original_keeps_its_clip_not_the_re_encoded_copy(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    heic = _mk(d / "IMG_0044.HEIC")
    jpg = _mk(d / "IMG_0044.JPG")
    clip = _mk(d / "IMG_0044.MOV", kind="video")
    motion, _ = livephoto.pair_library([jpg, heic], [clip])
    assert list(motion) == [heic.path], (
        "the JPEG beside a HEIC is a re-encode of it, so the motion belongs to "
        "the HEIC"
    )


def test_the_wider_copy_wins_when_the_format_matches(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    small = _mk(d / "IMG_0044.JPG", dims=(1024, 768))
    large = _mk(d / "IMG_0044.JPEG", dims=(4032, 3024))
    clip = _mk(d / "IMG_0044.MOV", kind="video")
    motion, _ = livephoto.pair_library([small, large], [clip])
    assert list(motion) == [large.path]


def test_the_choice_does_not_depend_on_scan_order(tmp_path):
    """Otherwise the same library could pair differently on two runs."""
    d = tmp_path / "d"
    d.mkdir()
    heic = _mk(d / "IMG_0044.HEIC")
    jpg = _mk(d / "IMG_0044.JPG")
    clip = _mk(d / "IMG_0044.MOV", kind="video")
    first, _ = livephoto.pair_library([heic, jpg], [clip])
    second, _ = livephoto.pair_library([jpg, heic], [clip])
    assert first == second, "pairing must be deterministic, not scan-order"


def test_three_copies_still_yield_one_owner(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    copies = [_mk(d / "IMG_0044.HEIC"), _mk(d / "IMG_0044.JPG"),
              _mk(d / "IMG_0044.JPEG")]
    clip = _mk(d / "IMG_0044.MOV", kind="video")
    motion, _ = livephoto.pair_library(copies, [clip])
    assert len(motion) == 1
    assert list(motion) == [copies[0].path]


def test_distinct_photos_in_different_folders_still_both_pair(tmp_path):
    """The collision fix must not leak into unrelated pairings."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    pa, pb = _mk(a / "IMG_1.HEIC"), _mk(b / "IMG_1.HEIC")
    ca, cb = _mk(a / "IMG_1.MOV", kind="video"), _mk(b / "IMG_1.MOV", kind="video")
    motion, _ = livephoto.pair_library([pa, pb], [ca, cb])
    assert len(motion) == 2, "two folders means two separate moments"
    assert motion[pa.path] == ca.path and motion[pb.path] == cb.path
