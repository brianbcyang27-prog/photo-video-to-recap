# Trip Recap Builder

[![tests](https://github.com/brianbcyang27-prog/photo-video-to-recap/actions/workflows/tests.yml/badge.svg)](https://github.com/brianbcyang27-prog/photo-video-to-recap/actions/workflows/tests.yml)
[![licence](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![macOS](https://img.shields.io/badge/macOS-ffmpeg%20%2B%20exiftool-informational.svg)](https://ffmpeg.org)

Drop a folder of unorganised photos, videos and iPhone Live Photos in. Get back
a finished, beat-synced travel recap MP4 — plus a report explaining every
decision it made.

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
`.work/` — so a long render need not block a quick proof. They do compete for
CPU, so the second will be slower.

A real library takes a while: on a 12,000-file iPhone export, roughly 10 minutes
of reading and scoring followed by the render. On macOS a second Terminal window
opens and tails the log, so you can leave it and come back;
`--no-progress-window` keeps everything in the one terminal.

Useful options:

| Flag | What it does |
|---|---|
| `--target N` | length in seconds (default 90) |
| `--mood` | `uplifting`, `cinematic`, `calm`, `energetic`, `warm` |
| `--order hook` | open on the strongest shot instead of chronologically |
| `--size` | `1080p` (default), `1440p`, `4k` |
| `--fps` | 30 (default) or 60 |
| `--fit` | `blur` (default, never crops), `crop`, `pad` |
| `--no-two-up` | one photo at a time, instead of portrait photos in side-by-side pairs |
| `--no-fit-per-shot` | use one `--fit` style for everything, instead of deciding per shot |
| `--no-live-photos` | treat Live Photos as plain stills, instead of playing their motion and sound |
| `--no-ken-burns` | static photos instead of slow push/pan |
| `--no-titles` | skip the chapter cards |
| `--no-duck` | do not lower the music under native audio |
| `--music-track FILE` | use your own track instead of a generated score |
| `--bpm N` | force a tempo instead of picking one |
| `--title "..."` | your own text for the opening card |
| `--quality N` | x264 CRF; lower is better quality and a bigger file (default 18) |
| `--jobs N` | parallel workers (default: auto) |
| `--dry-run` | make every decision and write the report, render nothing |
| `--no-geocode` | skip looking up place names from GPS |
| `--no-progress-window` | keep the progress log in this terminal only |

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

## iPhone Live Photos

A Live Photo is a still plus a few seconds of real video of the same moment,
usually taken on an iPhone. Most trip libraries are full of them — on a typical
export more than half the photos have one.

The motion half is normally treated as a duplicate and thrown away, which is a
waste: it is the only part that shows someone actually moving, and it carries
the ambient sound of the room. So by default each Live Photo is played as what
it is — real motion, and the sound that was recorded with it — and shown once,
not as a still with a pan.

A Live Photo's clip is usually far shorter than the beat slot it has to fill.
On a real trip library the clips ran 0.70s to 2.71s against a 2.80s slot, so
most of a Live Photo shot used to be a still frame with nothing happening in
it — measured on one shot, 0 of 54 frames moved after the clip ended, and on
another, three quarters of the shot was frozen. The remainder is now a slow
eased push on the frame the clip ended on, so the shot keeps drifting all the
way to its cut. The clip still plays at its own pace and is never slowed down
to fill the gap, which would turn a gesture into a smear. A shortfall under
0.25s is left alone: it is a few frames, and it is not worth three encodes.

`--no-live-photos` turns it off and puts them back to Ken Burns pans.

Pairing is scoped to the containing folder, because the same photo gets copied
into more than one place on a trip (1,041 base names in the sample library
recurred across folders) and matching across the whole library pairs the wrong
half with the wrong still. On 12,000 files the whole index takes 0.02s.

The motion half is played whole rather than trimmed to match the still, and its
audio is deliberately not trimmed either, so picture and sound stay aligned.
Trimming it to the still's first frame was measured and rejected: aligning a
HEIC key frame against HEVC motion gives a median match of 54% ±26% with no
separation between right and wrong candidates, while a planted control scored
8.9x — the method simply does not work across those two encodings, and a wrong
trim would be worse than none.

## Quality

The default look is not "a slideshow with filters". Two things do most of the
work:

**Supersampled Ken Burns.** `zoompan`, which is what makes a still move, samples
its source nearest-neighbour at whatever size you ask it for. Asking for a 1080p
frame from a 1080p-prepared image loses about 10% of the edge detail — measured
mean edge energy 5.402 against a direct Lanczos reference. Rendering at 2x and
reducing afterwards recovers nearly all of it: 5.688, or −4.6% instead of
−10.0%. A 4x supersample adds nothing (5.700) for twice the cost, so the factor
is 2.

The factor is computed per run rather than fixed, because overshooting it is
waste. A 1080p film gets 2x; a 4K film already needs 3840px from a 4032px photo,
so there is nothing left to sample and it drops to 1x instead of paying four
times the filter cost to interpolate pixels that were never in the file.

**Lanczos video scaling.** Video clips are resampled with Lanczos rather than
ffmpeg's bicubic default, worth 5.6% more edge energy on the same frames.

**Paced camera moves.** Every move used to travel the same distance, so a 2s
stab pushed the frame as far as a 6s drift. Those are different gestures — the
stab reads as a punch because the same distance in a third of the time is three
times the speed — and a film that does it two hundred times stops feeling
edited. Each shot's travel is now scaled against its length on a square-root
law, and modulated by the music section it lands in, so the picture leans into
a crescendo instead of sitting at one size under the whole piece. Measured on a
real 180s render: 2.4x more travel per second on short shots than long ones.

Also, because these are real numbers rather than folklore: the H.264 level is
derived from the frame size and macroblock rate (4.0/4.2/5.1/5.2). A hardcoded
`4.0` still *accepts* a 4K file but writes `level=40` into the tag, and a
hardware decoder that trusts the tag will play a 4K file as though it were
1080p. The GOP is derived from the frame rate, so it does not silently halve the
seek interval when you ask for 60fps.

### About 4K

`--size 4k` is real detail for stills and an upscale for video, and the tool
says so rather than pretending otherwise. A phone photo is 4032x3024, so
3840x2160 is a genuine downscale. A phone video is 1920x1440 at most, so the
same frame is a 2.7x upscale; near-Nyquist energy rises 10%, which is
interpolation rather than resolution. It is not worse — just not better — and it
costs bitrate on pixels carrying no information. If your library is mostly video,
`--fps 60` at 1080p is the better deal.

One output file has one resolution, so mixing them is not an option; that is a
deliberate limit, not an oversight.

There is a second reason 1080p often looks *better* than 4K here, which is
counter-intuitive enough to be worth stating. Stills are animated by `zoompan`,
which resamples its input with a nearest-neighbour kernel, so it is rendered
larger than the delivery size and reduced afterwards — each output pixel then
gets chosen from several source pixels instead of one. Measured on a real 12MP
photo, mean edge energy is 5.252 at 1:1, 5.009 when rendered at 2x and reduced,
and that softening at 1:1 "was the single largest cause of the picture looking
low-resolution". The factor is capped at 2x because source resolution limits
what more can recover, and 2x already covers 1080p (a 4032px photo against a
1920px canvas) but not 4K: `4032 // 3840` is 1, so a 4K render drops the
supersample and animates 1:1. 4K therefore gains pixels and gives up edge
detail on exactly the shots that carry a film. If the picture looks soft rather
than small, `--size 1080p` is the fix, and `--fps 60` is the better deal still.

### Portrait shots

Portrait phone photos are shown upright, two at a time side by side
(`--no-two-up` for one at a time), with a blurred fill behind them rather than a
crop, so nobody's head gets cut off.

## How it decides

The pipeline follows the order a human editor actually works in.

**1. Read the library.** EXIF first for capture time and GPS, filename and mtime
as fallback. Files that are not real photos or videos get dropped. Live Photos
are paired to their motion halves here.

**2. Judge every shot.** Each frame is measured on sharpness (variance of the
Laplacian plus Sobel energy), exposure, blown highlights, crushed blacks,
colourfulness, contrast, and — for video — inter-frame motion, which separates
genuine movement from handheld shake. Faces are detected and weighted by whether
the eyes look open, which on a trip is usually the strongest signal there is.
Genuinely unusable frames are rejected; everything else is ranked.

**3. Map the trip.** GPS positions are clustered into stops, gaps of more than
five hours or a move of more than about 2.5 km start a new chapter, and each
chapter gets screen time in proportion to how much usable material it holds
(with a floor, so a quiet day still appears). Chapters split by calendar date:
the day is the card title, the place is the subtitle. Real sunrise and sunset
times are computed per location, so golden-hour shots are preferred over harsh
midday ones. Place names are reverse-geocoded for the title cards; offline, it
degrades quietly to coordinates.

**4. Choose.** Deduplicate near-identical frames, cap how much can come from any
one minute, then fill the timeline best-first. Video clips get a reserved share
because motion carries a montage better than a run of stills.

**5. Write the music.** By default an original score is synthesised — chords,
pad, bass, arpeggio and drums in a chosen mood, sidechain-pumped, with a known
beat grid. Because the music is generated, the tempo can be placed exactly on the
video's frame grid, so every cut is mathematically on the beat with nothing
resampled.

The score also has a shape, so ten minutes does not sound like ten minutes of
the same four bars. A plan is derived from the target length — exact bar counts,
sections spread evenly, a change every 45–90 seconds, fixed open and close —
and the instrumentation follows it: a section with no drums leaves the bed
alone, a bright one brings the arpeggio forward. Supply your own track and the
beats are found by spectral-flux onset detection and a dynamic-programming beat
tracker, then the timeline snaps to the nearest frame; the error stays under 17ms
rather than speeding up your music.

**6. Cut.** Every shot length is a whole number of beats, and each cut is placed
on the frame nearest its true beat time. Because the frame index comes from the
*cumulative* beat position, rounding never accumulates — the last shot is as
tight to the beat as the first. Portrait phone shots get a blurred fill rather
than a crop, so nobody's head is cut off. Ordinary still photos get a slow push
or pan, varied so the motion does not become one repeated trick: the seven moves
are distributed so no one move takes more than 14% of the timeline and no two
adjacent shots use the same one.

**7. Mix.** Native audio is pulled per shot — including from a Live Photo's
motion half — level-matched and faded, then the music is ducked underneath it
with a sidechain compressor so speech stays clear. Stills with no audio get
silence, so the music plays through them untouched. Final loudness is normalised
to about −14 LUFS with true peak kept under −1 dBTP.

**8. Report.** Everything above, written down.

## Checking the output

```bash
./.venv/bin/python tools/verify_output.py out/trip_recap.mp4
```

This measures the rendered file rather than trusting the plan — 27 checks. It
verifies that the *picture* runs for as long as the cut list says it should (on
the video stream, not the container, which reports whichever stream is longer and
will happily cover for a picture that stopped early), that picture and sound end
together, resolution and pixel format, integrated loudness and true peak, that
the soundtrack never goes silent, that title cards actually contain legible
text, that stills really move and move *smoothly*, that portrait shots were
filled rather than cropped, that no black bars crept in, and that every cut lands
on the beat. If any shots carried audio it also measures whether the music is
genuinely being ducked underneath — the easiest thing in the whole pipeline to
get subtly wrong and the hardest to catch by ear.

To run that check across every combination of options:

```bash
SWEEP_MEDIA=test_media ./tools/sweep.sh
```

This renders each case for real and verifies the file, so it takes a while.

## Tests

```bash
./.venv/bin/python -m pytest tests -q      # 220 tests, about 3.8s
```

No ffmpeg, no media on disk, no network — the suite is pure logic, so a failure
is always a real regression. CI runs it on Python 3.11, 3.12 and 3.13, and
nightly also does a real render on a generated library and verifies the file,
which is the only way to catch an ffmpeg build that no longer accepts a filter
graph.

Every test in this suite was **calibrated against a deliberately broken
control**: a test that has never been shown to fail is not evidence. Breaking
folder scoping in the Live Photo pairing, the motion distribution, the H.264
level call sites, the supersample factor, and the Ken Burns move selection each
produced the failures you would expect, and each fix was kept only because the
control was re-broken and re-observed. Several tests in `tests/test_render.py`
and `tests/test_cli.py` exist specifically to test the *call sites* rather than
the helpers, because the first calibration pass found tests that passed happily
against code that had been reverted to a hardcoded `"4.0"`.

The same rule found a real bug: the macOS progress window called `json.dumps`
without importing `json`, inside a `try/except Exception: return None`. The
window silently never opened, and the earlier manual check that the Terminal
window count returned to its original number passed precisely *because* nothing
ever opened. The quoting was also wrong in a way that would have bitten anyone
whose log path contained a quote: `json.dumps` does not escape single quotes and
`shlex.quote` emits `'"'"'`, which ends the AppleScript string literal early. The
command is now passed to `osascript` as an argument, which makes the escaping a
non-problem.

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

## Requirements

macOS with ffmpeg, ffprobe, exiftool and Python 3.11+. `./setup.sh` checks for
all of them and creates a virtual environment. It does not install the generated
music's dependencies beyond numpy — the score is synthesised from scratch, so
there is no library to download and no licence to worry about.

Beyond ffmpeg, the Python side is only numpy, Pillow and soundfile. `setup.sh`
also installs `opencv-python-headless` if it can, because face detection is the
single most useful scoring signal on a trip full of people; the pipeline runs
without it.

## Layout

```
make_video.py          command line entry point
pipeline/
  config.py            every tunable in one place
  util.py              process helpers, probing, EXIF
  quality.py           the frame metrics
  ingest.py            library scanning and filtering
  livephoto.py         pairing stills to their iPhone motion halves
  analysis.py          per-shot scoring, video shot detection
  geo.py               GPS clustering, place names, sunrise/sunset
  context.py           the trip map: stops, chapters, screen time
  music.py             score synthesis and beat tracking
  arrange.py           song structure: sections across the timeline
  select.py            dedupe, selection, beat-aligned timing
  render.py            ffmpeg and PIL: compositing, Ken Burns, mixing
  report.py            the decision record
tools/
  make_test_media.py      generates a synthetic trip for testing
  make_audio_library.py   adds speech-like audio so the mix path can be tested
  verify_output.py        measures a rendered file against its plan
  sweep.sh                renders and verifies every option combination
  clean.sh                clears scratch space between runs
tests/                 220 tests, no media required
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
still_max_long_side     # cap on how large prepared stills are allowed to get
```

`--fit`, `--mood`, `--target`, `--size` and `--quality` cover most needs without
editing anything. `--quality 16` gives a larger, cleaner file; the default 18 is
a good balance.

## Licence

MIT — see [LICENSE](LICENSE).
