# Trip Recap Builder

Drop a folder of unorganised photos and videos in. Get back a finished,
beat-synced travel recap MP4 — plus a report explaining every decision it made.

No timeline to assemble. No export settings to fiddle with. No rendering
dialogue. Nothing leaves your machine.

```
media/     <- you put photos and videos here
music/     <- optional; leave empty and a score is written for you
out/       <- the finished video, report, and edit decision list appear here
```

## Use it

```bash
./setup.sh                 # once, to install dependencies
./make_video.py            # ~90 second recap from everything in media/
```

That is the whole interface. If you want a different length:

```bash
./make_video.py --target 300     # five minutes
./make_video.py --target 600     # ten minutes
```

Two builds can safely run at once — each gets its own scratch directory under
`.work/` — so a long one-minute render need not block a quick five-second
proof. They do compete for CPU, so the second will be slower.

Useful options:

| Flag | What it does |
|---|---|
| `--target N` | length in seconds (default 90) |
| `--mood` | `uplifting`, `cinematic`, `calm`, `energetic`, `warm` |
| `--order hook` | open on the strongest shot instead of chronologically |
| `--fit` | `blur` (default, never crops), `crop`, `pad` |
| `--no-two-up` | one photo at a time, instead of portrait photos in side-by-side pairs |
| `--no-fit-per-shot` | use one `--fit` style for everything, instead of deciding per shot |
| `--music-track FILE` | use your own track instead of a generated score |
| `--title "..."` | your own text for the opening card |
| `--dry-run` | make every decision and write the report, render nothing |
| `--no-titles` | skip the chapter cards |
| `--no-ken-burns` | static photos instead of slow push/pan |
| `--no-geocode` | skip looking up place names from GPS |

## What comes out

Three files in `out/`:

- **`trip_recap.mp4`** — 1920x1080, 30fps, H.264 + AAC, plays anywhere
- **`trip_recap.md`** — a plain-English record of what was chosen and why
- **`trip_recap.edl.json`** — the machine-readable cut list, with the exact
  source file and in-point for every shot

The report is the point as much as the video. It tells you which shots were
rejected and why, which places were visited, how screen time was split across
days, and what the music tempo is. If you disagree with a choice, the EDL tells
you exactly where to change it.

## How it decides

The pipeline follows the order a human editor actually works in.

**1. Read the library.** EXIF first for capture time and GPS, filename and mtime
as fallback. Files that are not real photos or videos get dropped.

**2. Judge every shot.** Each frame is measured on sharpness (variance of the
Laplacian plus Sobel energy), exposure, blown highlights, crushed blacks,
colourfulness, contrast, and — for video — inter-frame motion, which separates
genuine movement from handheld shake. Faces are detected and weighted by
whether the eyes look open, which on a trip is usually the strongest signal
there is. Genuinely unusable frames are rejected; everything else is ranked.

**3. Map the trip.** GPS positions are clustered into stops, gaps of more than
five hours or a move of more than about 2.5 km start a new chapter, and each
chapter gets screen time in proportion to how much usable material it holds
(with a floor, so a quiet day still appears). Real sunrise and sunset times are
computed per location, so golden-hour shots are preferred over harsh midday
ones. Place names are reverse-geocoded for the title cards; offline, it
degrades quietly to coordinates.

**4. Choose.** Deduplicate near-identical frames, cap how much can come from any
one minute, then fill the timeline best-first. Video clips get a reserved share
because motion carries a montage better than a run of stills.

**5. Write the music.** By default an original score is synthesised — chords,
pad, bass, arpeggio and drums in a chosen mood, sidechain-pumped, with a known
beat grid. Because the music is generated, the tempo can be placed exactly on
the video's frame grid, so every cut is mathematically on the beat with nothing
resampled. Supply your own track and the beats are found by spectral-flux onset
detection and a dynamic-programming beat tracker, then the timeline snaps to the
nearest frame; the error stays under 17ms rather than speeding up your music.

**6. Cut.** Every shot length is a whole number of beats, and each cut is placed
on the frame nearest its true beat time. Because the frame index comes from the
*cumulative* beat position, rounding never accumulates — the last shot is as
tight to the beat as the first. Portrait phone shots get a blurred fill rather
than a crop, so nobody's head is cut off. Still photos get a slow push or pan,
varied so the motion does not become one repeated trick.

**7. Mix.** Native audio is pulled per shot, level-matched and faded, then the
music is ducked underneath it with a sidechain compressor so speech stays
clear. Stills get silence, so the music plays through them untouched. Final
loudness is normalised to about -14 LUFS with true peak kept under -1 dBTP.

**8. Report.** Everything above, written down.

## Checking the output

```bash
./.venv/bin/python tools/verify_output.py out/trip_recap.mp4
```

This measures the rendered file rather than trusting the plan. It checks that
the render matches the cut list, that picture and sound end together,
resolution and pixel format, integrated loudness and true peak, that the
soundtrack never goes silent, that title cards actually contain legible text,
that stills really move and move *smoothly*, that portrait shots were filled
rather than cropped, that no black bars crept in, and that every cut lands on
the beat. If any shots carried audio it also measures whether the music is
genuinely being ducked underneath — the easiest thing in the whole pipeline to
get subtly wrong and the hardest to catch by ear.

To run that check across every combination of options:

```bash
SWEEP_MEDIA=test_media ./tools/sweep.sh
```

This renders each case for real and verifies the file, so it takes a while. It
is the regression suite: if a change breaks durations, framing, framing mode,
tempo or the audio mix, this is what will say so.

## Testing with media of your own

Two helpers build throwaway test libraries. They write to `test_media/`, not
`media/`, so a synthetic library can never end up in your recap:

```bash
./.venv/bin/python tools/make_test_media.py              # a synthetic 3-day trip
./.venv/bin/python tools/make_audio_library.py           # -> test_media_audio/
```

The second mints a copy of the library whose video clips carry speech-like
audio at deliberately uneven levels, which is the only way to exercise the
native-audio path: per-shot audio extraction, level matching, and the ducking.
Then:

```bash
./.venv/bin/python make_video.py --media test_media_audio --name audio_test
./.venv/bin/python tools/verify_output.py out/audio_test.mp4
```

A library already generated lives in `test_media/`, so you can try all of this
without regenerating it.

## Requirements

macOS with ffmpeg, ffprobe, exiftool and Python 3.11+. `./setup.sh` checks for
all of them and creates a virtual environment. It does not install the generated
music's dependencies beyond numpy — the score is synthesised from scratch, so
there is no library to download and no licence to worry about.

## Layout

```
make_video.py          command line entry point
pipeline/
  config.py            every tunable in one place
  util.py              process helpers, probing, EXIF
  quality.py           the frame metrics
  ingest.py            library scanning and filtering
  analysis.py          per-shot scoring, video shot detection
  geo.py               GPS clustering, place names, sunrise/sunset
  context.py           the trip map: stops, chapters, screen time
  music.py             score synthesis and beat tracking
  select.py            dedupe, selection, beat-aligned timing
  render.py            ffmpeg and PIL: compositing, Ken Burns, mixing
  report.py            the decision record
tools/
  make_test_media.py      generates a synthetic trip for testing
  make_audio_library.py   adds speech-like audio so the mix path can be tested
  verify_output.py        measures a rendered file against its plan
  sweep.sh                renders and verifies every option combination
test_media/           a generated library, for trying things out
test_media_audio/     the same, with audio on every clip
```

## Tuning

`pipeline/config.py` holds the knobs that matter:

```python
photo_min / photo_max   # how long a still may be on screen, in seconds
video_min / video_max   # same for clips
order                   # "chronological" or "hook"
target_seconds          # default length
w_sharpness             # relative weight of each scoring signal
w_faces                 # (raise this for trips with people in them)
max_per_minute          # diversity cap; relaxed automatically if needed
dedupe_threshold        # how similar two frames must be to count as duplicates
```

`--fit`, `--mood`, `--target` and `--quality` cover most needs without editing
anything. `--quality 16` gives a larger, cleaner file; the default 18 is a good
balance.
