# Turning a personal media library into a finished video without a human editor

A research report for the Trip Recap Builder maintainer.

**Date:** 2 October 2026
**Scope:** consumer/prosumer auto-video products, academic work on highlight detection /
IQA / aesthetics / people-aware summarization / music-beat alignment, and open-source tools
whose actual source was read.
**Reader's starting point:** a deterministic Python + ffmpeg pipeline that reads EXIF/GPS,
scores frames on sharpness / exposure / colour / faces / motion, clusters GPS into
chapters, synthesises music on a known beat grid, cuts on the beat, applies Ken Burns,
mixes audio, and emits a human-readable decision report. It has `--dry-run`, a resolution
floor, a per-minute cap, a 64-bit dHash dedupe, downbeat phase chosen by onset energy,
7-way Ken Burns move distribution capped at 14% per move, −14 LUFS / −1 dBTP, and an
honest note that 4K video is an upscale rather than real detail.

**Therefore this report is written as a delta.** Section 2 lists what the pipeline already
does that most published work does not. Sections 3–6 are about what other people do that
the pipeline does not.

---

## 0. Confidence markers

| Marker | Meaning |
|---|---|
| **[verified]** | Read from primary source: paper abstract/full text fetched, official docs page, or actual repository source code / GitHub API. |
| **[reported]** | Vendor claim, press release, review article, or third-party blog. Not independently confirmed. Includes anything sourced from SEO-content marketing sites (flagged inline). |
| **[rumoured]** | Practitioner folklore, forum consensus, or video essays with no citable primary source. |

### Things I could not verify, stated plainly

- **"A Social-Aware Model for Selecting Photos" (CVPR 2017)** — cited widely in secondary
  sources as the basis for Google Photos' 2017 Memories feature. I searched the CVF CVPR 2017
  index (1.65 MB of paper titles, no match for `Social-Aware`, `Social Aware`, or
  `Selecting Photos`), arXiv, CrossRef and OpenAlex. **Not found.** I am not including it as
  evidence. [rumoured]
- **DOVER SRCC numbers (often quoted as 0.74–0.79)** — I read the official CVF abstract page
  for the ICCV 2023 paper. It states "state-of-the-art performance" but **contains no numbers**.
  Do not put DOVER SRCC figures in a design doc. [reported]
- **VADB GitHub repo has no detected license** (32 stars, last push 2026-02-13). Treat the
  dataset as not-licensed-for-reuse until that changes. [verified]
- **`paper-search` `search_semantic` returned empty arrays for every query** and
  `search_openalex` returned low-relevance noise. All academic verification below went through
  the arXiv API, CrossRef, Semantic Scholar's graph API, CVF Open Access, or PubMed Central
  directly. Anything I could not reach on one of those is marked unverified.

---

## 1. Top findings, ranked by relevance to this pipeline

### 1.1 Video-summarization evaluation is largely invalid, and *segmentation* dominates — not scoring

This is the single most important finding, and it inverts the obvious priorities.

> "We provide in-depth assessment of the evaluation pipeline for video summaries using two
> popular benchmark datasets: SumMe and TVSum. Surprisingly, we observe that **randomly
> generated summaries achieve comparable or better performance to the state-of-the-art. In
> some cases, the random summaries outperform even the reference summaries.** Moreover, it
> turns out that **the video segmentation, which is often considered as a fixed pre-processing
> method, has the most significant impact on the performance measure.**"
> — Otani, Nakashima, Rahtu, Heikkilä, *Rethinking the Evaluation of Video Summaries*,
> CVPR 2019 (project page; code at `mayu-ot/rethinking-evs`) **[verified]**

Their mechanism: summaries are generated with random importance scores and random segment
boundaries; the F1 scores those produce are the floor reachable **by chance**. Their
randomization test also shows the *segment length* effect — the knapsack formulation that
almost every summarizer uses penalises long segments, so short segments are overwhelmingly
likely to be selected. Change the segmentation, change the score, change nothing else.

**What this means here.** The pipeline's frame-scoring function — the largest and most
tuned body of code, with six weighted terms — is the component the literature says matters
*least* among the two things you can control. `analysis.py::_find_cuts()` (motion threshold
`MOTION_GOOD=12.0`, `MOTION_SHAKE=45.0`, `min_gap` from `min_shot_seconds=0.8`) is the
component with the largest measurable leverage. Anyone tuning this pipeline by A/B-ing
score weights is optimising the wrong knob.

It also means **no offline metric can tell you whether the output is watchable.** See §1.2.

### 1.2 Real film shot lengths fall in a narrow band and their *sequence* is near-1/f — your duration distribution is the wrong shape

Cutting, Brunick, DeLong, Iricinschi & Candan measured 160 English-language films from
1935–2010 (top-rated IMDb, five genres, ~1,100 shots and ~165,000 frames per film; frames
downsampled to 256×256). From the open-access full text (PMC3485803): **[verified]**

- Average shot length (ASL) was **~10 s in the 1930s–40s, falling to below 4 s after 2000.**
  Plotted log-scaled: r = −.75, t(158) = 14.3, p < .0001. Reliable across all five genres.
- Shot segmentation was done **by hand with computer assistance**, not automatically.
- **"increasingly, films have come to adopt near-1/f shot-length fluctuations as well"** —
  i.e. the serial pattern of shot durations approaches the ~1/f power-spectrum pattern of
  human reaction-time fluctuation measured in the lab (citing Cutting et al. 2010; Gilden 2001).
  Amplitude grows as √wavelength: self-similar structure at seconds, tens of seconds,
  minutes, tens of minutes.
- Films have also gotten **darker** (mean per-frame median luminance decreasing) and have
  **more motion** overall. Their Visual Activity Index is `1.0 − median r` over correlations
  of next-adjacent frame pairs.
- In contemporary films, **shorter shots have proportionally more within-shot motion**
  (r = −.46, t(158) = 6.62, p < .0001). In the 1935–1960 studio era there was **no** such
  relation — the two were independent.

**What this means here.** `Pipeline.photo_min/photo_max = 2.6/4.6 s` and
`video_min/video_max = 1.8/5.0 s`, allocated by `_base_weights()` which is monotonically
increasing in the item score. Three problems:

1. The band is too narrow and too high. Cutting et al.'s post-2000 ASL is **below 4 s** and
   that is a *mean*, over a distribution. A 90-second recap at 4 s/shot is 22 shots; that is
   a slideshow pace, not a montage pace.
2. **Duration is a monotone function of quality.** The montage systematically slows down over
   good material and hurries over ordinary material. Rhythm is read by viewers as meaning. If
   duration means "importance", the piece announces its own ranking instead of telling a story.
3. **There is no short-long alternation.** A 1/f shot-length sequence has power at every
   timescale, which means short runs are interrupted by long runs and vice versa. A uniform
   beat-locked cadence has power at exactly one timescale, and it reads as metronomic.

### 1.3 Cut on musical *accents*, not on beats — your downbeat phase is half the fix

`pipeline/music.py:222-233` chooses the 4/4 downbeat phase by maximising onset energy, then
treats every beat on that grid as equally good. FireRed-OpenStoryline's `select_bgm.py`
does something strictly better, and it is 60 lines of arithmetic you can read. **[verified]**

```python
# src/open_storyline/nodes/core_nodes/select_bgm.py
def _compute_accent_beats(y, sr, beat_frames, hop_length,
                          top_pct: float = 70.0,        # keep the top 30% of beats
                          min_sep_beats: int = 1,
                          use_percussive: bool = True,  # onset from percussive component
                          local_norm_win: int = 8,       # in beats
                          require_local_peak: bool = True):
    y_for_onset = librosa.effects.percussive(y) if use_percussive else y
    onset_env = librosa.onset.onset_strength(y=y_for_onset, sr=sr, hop_length=hop_length)

    strength = onset_env[clip(beat_frames)].astype(np.float64)

    # local normalization so loud sections don't dominate beat selection
    local_mean  = convolve(strength, ones(w)/w, mode="same")
    strength_norm = strength / (local_mean + 1e-8)

    thr  = percentile(strength_norm, 100.0 - top_pct)
    cand = where(strength_norm >= thr)[0]
    cand = cand[is_local_max(strength_norm)[cand]]   # no plateau runs
    # greedy min-separation suppression, strongest first
    ...
```

Five steps, each with a stated reason in the docstring: percussive separation → onset
strength at beat times → local normalisation over an 8-beat window → top-30% percentile →
local-max filter → minimum-separation suppression. It also emits `energy_mean`,
`energy_mean_db`, and `dynamic_range_db = p95 − p10` of RMS in dB as BGM descriptors.

**What this means here.** Your phase choice answers "which grid alignment is most musical?".
FireRed's filter answers "which *moments* inside that grid are musical?". You can keep
`downbeats` and add `accent_beats`; the cut allocator prefers accents and falls back to
beats when the accents run out. §5.1.

### 1.4 Nothing in the pipeline leaves you an editable timeline — and the best OSS tool does, in five formats

Your output is a rendered MP4 plus a prose report. That is a dead end for the one user who
is closest to a human editor and wants to fix three shots.

auto-editor, the closest open-source analogue (audio-driven deterministic cutter), ships:
**[verified]**

```
auto-editor example.mp4 --export premiere          # Premiere Pro XML
                            --export resolve        # DaVinci Resolve (FCPXML)
                            --export final-cut-pro  # FCPXML
                            --export shotcut        # ShotCut
                            --export kdenlive       # Kdenlive
                            --export clip-sequence  # independent media files
```
and `src/exports/otio.nim` implements OpenTimelineIO. It carries transitions through the
exports: `Cross Dissolve`/`Cross Fade` transition items in Premiere FCP7 XML, `SMPTE_Dissolve`
in OTIO, spine transitions in FCPXML, same-track transitions plus fade filters in ShotCut
and Kdenlive. It documents exactly where transitions *drop* (v1/v2 timeline formats have no
way to represent them; `clip-sequence` by design).

**What this means here.** Emitting FCPXML or OTIO alongside the MP4 is maybe 80 lines. It
converts your tool from "finishes the job" to "finishes the job and hands you the timeline
if you disagree with two shots." That is the single highest value-per-line change in this
report. §5.2.

### 1.5 Hard cuts with no handle are the cheap look, and a transition across a *small* cut is worse than no transition

auto-editor's transition documentation contains the most specific quality heuristic I found
in any of these repos: **[verified]**

> The full form is `dissolve:DURATION[:MIN-CUT]`. `MIN-CUT` — skip cuts whose *removed
> source interval* is shorter than this (default: `1sec`). **"Dissolving across a tiny
> silence trim reads as a stutter, so short cuts stay hard by default."** Use `0` to dissolve
> at every cut.
>
> Dissolves are linked: video cross-dissolves and audio cross-fades cover the same span.
> **A dissolve needs source material on both sides of the cut (the material that was cut out
> serves as the handle)**, so a transition may be shortened or skipped when a clip is too
> short to support it.

Plus: the timeline **fades in at the start and fades out at the end**, and it is explicit
that every export preserves that.

Two hard-won facts, both non-obvious: (a) the right unit for gating a transition is the
length of the *material you removed*, not the length of the clips you kept; (b) the handle
is the rejected material, so a clip with no handle adjacent cannot dissolve and the renderer
must be allowed to shorten or skip.

Meanwhile `xfade-easing` states the problem plainly: **[verified]**

> "Xfade is a FFmpeg video transition filter with many built-in transitions and an expression
> evaluator for custom transitions. However the progress rate is **linear, starting and
> stopping abruptly and proceeding at constant speed, therefore transitions lack interest.
> Easing inserts a progress envelope to smooth transitions in a natural way.**"

Its patched-FFmpeg variant exposes `easing=` (with CSS `cubic-bezier(...)`) and `reverse=`,
plus ported GLSL transitions. The stock-FFmpeg `transition=custom` variant "doesn't support
CSS easings, certain transitions, the reverse feature, full colour or textures" — so the
choice is: patch FFmpeg, or accept linear-rate stock transitions.

Your pipeline currently does **no video transitions at all** (`grep xfade` across `pipeline/`
returns nothing; only `afade` on audio, which is correct and already there). For a personal
archive of *stills*, hard cuts are arguably right — a dissolve between two photographs is a
dissolve between two unrelated images. See §4.6 for the case that it isn't.

### 1.6 Your single blended score is the wrong shape; DOVER's disentanglement is the alternative

Your `Analysis` config is one weighted sum: `w_sharpness 0.30`, `w_exposure 0.20`,
`w_faces 0.20`, `w_color 0.12`, `w_contrast 0.10`, `w_motion 0.08`. Total 1.00.

Wu, Zhang, Liao, Chen, Hou, Wang, Sun, Yan & Lin (ICCV 2023, pp. 20144–20154) took the other
route, and their finding is the reason: **[verified]**

> "the objective of the UGC-VQA problem is still ambiguous and can be viewed from two
> perspectives: **the technical perspective, measuring the perception of distortions; and the
> aesthetic perspective, which relates to preference and recommendation on contents.** To
> understand how these two perspectives affect overall subjective opinions in UGC-VQA, we
> conduct a large-scale subjective study... **The collected DIVIDE-3k confirms that human
> quality opinions on UGC videos are universally and inevitably affected by both aesthetic
> and technical perspectives.**"

They also ship **DOVER++**, "the first approach to provide reliable clear-cut quality
evaluations from a single aesthetic or technical perspective."

**What this means here.** `exposure + clipped + black + sharpness + contrast` *are* the
technical branch. `faces + color + motion` are arguably the aesthetic branch. They are not
commensurable — a correctly-exposed blurry photo scores high on one and low on the other,
and the blend hides that. Two branches with separate normalisation and separate weights, plus
DOVER-style single-branch scores for the report, would let a user say "this run was too
aesthetic, weight the technical side" without editing six numbers. §5.3.

### 1.7 Anti-repetition is done per-minute; the sharper rule is per-source-file

`max_per_minute: 2` caps bursts and adjacent seconds. That does not stop one 40-second
panorama video from contributing six 4-second excerpts spread across the whole 90 seconds —
which reads as *the same footage*, and is the specific failure MoneyPrinterTurbo wrote
`_prioritize_unique_source_clips()` to kill: **[verified]**

```python
# app/services/video.py — translated from the Chinese docstring
grouped = group all subclipped_items by item.source_file_path
for items in grouped.values():
    primary_item   = max(items, key=lambda item: item.duration)   # longest excerpt per source
    primary_items.append(primary_item)
    overflow_items.extend(item for item in items if item is not primary_item)
shuffle(primary_items); shuffle(overflow_items)
# stable sort on a cumulative source_usage counter, so round 2 prefers
# sources not yet seen; stability preserves randomness among equals
```

Their reasoning: preferring the *longest* excerpt per source avoids burning the good material
on a long timeline when only fragments of that source remain. The overflow tier is still
allowed to backfill — because failing to produce a video is worse than a repeated source.
Your `max_per_minute` is the right idea at a different granularity. §5.4.

---

## 2. What the pipeline already does that most of the literature does not

Listed so the recommendations above are read as additions, not replacements.

| Capability | Source in this repo | Why it is unusual |
|---|---|---|
| Supersampled Ken Burns (2×, chosen per run from source resolution) | `README.md` "Quality"; measured mean edge energy 5.402 → 5.688 (−4.6% vs −10.0%), 4× adds nothing (5.700) | Almost every "auto-slideshow" tool calls `zoompan` at 1× and eats the nearest-neighbour loss |
| Honest 4K reporting — states it is an upscale, quantifies near-Nyquist rise at +10% | `README.md` "About 4K" | No consumer product admits this |
| H.264 level derived from frame size + macroblock rate (4.0/4.2/5.1/5.2); GOP from fps | `README.md` "Quality" | A hardcoded `level=40` on a 4K file still *plays* on a tag-trusting decoder at 1080p |
| Cumulative-beat frame indexing so rounding never accumulates | `select.py:526-534`, "17ms at 30fps" | The standard bug is per-cut rounding, which drifts over a long timeline |
| 7 Ken Burns moves distributed so none exceeds 14% of timeline and no two adjacent shots repeat | `README.md` step 6 | Directly targets "one repeated trick" |
| Square-root pacing law on camera travel, modulated by music section | `README.md` "Paced camera moves" (2.4× more travel/s on short shots) | Auto-editor has `zoom:1..1.5` ramps but no per-shot travel scaling |
| Real sunrise/sunset per location; golden hour preferred over midday | `README.md` step 3 | Consumer products do "auto enhance", not physically-motivated time-of-day scoring |
| Chapter quotas by GPS clustering (5 h / 2.5 km) with a floor, plus per-chapter mean re-centring of weights | `README.md` step 3; `select.py:_base_weights` | `_base_weights` exists precisely to stop one strong chapter eating the quota — most implementations miss this |
| `--dry-run`, dHash dedupe, resolution floor, per-minute cap | `make_video.py:118`, `config.py` | auto-editor's equivalents are `--preview` and `--smooth` |
| −14 LUFS / −1 dBTP final target | `README.md` step 7 | Matches auto-editor's documented podcast target exactly — you are already at parity here |

---

## 3. Failure taxonomy: why generated montages look cheap

Each item is grounded in a source I read, a paper, or a code comment — not a blog listicle.
The low-quality sources named in §9 were consulted for leads only and are not cited here.

**3.1 Constant cadence.** Every shot the same length, or length monotonically tied to a
score. Cutting et al. measured real ASL at **below 4 s mean** post-2000 with a **near-1/f
distribution of shot durations** across all timescales — self-similar structure at seconds,
tens of seconds, minutes. Uniform beat-locked duration has energy at one timescale only, and
that is the metronome effect. **[verified]**

**3.2 The same source twice, far apart.** Not caught by burst/dedupe logic. This is a
*source-provenance* problem, not a near-duplicate-image problem; MoneyPrinterTurbo found it in
production and fixed it with per-source-file primary/overflow tiering. **[verified]**

**3.3 Cuts on any beat rather than on accents.** A beat grid has no accents; accents are what
a listener feels. FireRed's percentile + local-max + min-separation filter is 40 lines and
is the difference between "on the grid" and "on the music". **[verified]**

**3.4 Transitions at a constant rate.** Stock `xfade` interpolates linearly — "starting and
stopping abruptly and proceeding at constant speed, therefore transitions lack interest"
(xfade-easing's own words). Uniform linear dissolves read as a PowerPoint. **[verified]**

**3.5 A transition across the wrong cuts.** Dissolving across a cut that removed only 120 ms
of material "reads as a stutter" (auto-editor's own words). The gate is on *removed* interval,
not kept interval. **[verified]**

**3.6 Technical and aesthetic quality blended into one number.** A correctly-exposed blurry
frame and a sharp blurry-frame-of-a-boring-wallframe score identically under any single
number. DOVER's subjective study says human opinions are "universally and inevitably affected
by both" perspectives — and gives you a way to evaluate either one alone (DOVER++). **[verified]**

**3.7 No easing on anything that moves.** One camera move repeated is worse than three moves
alternating. The pipeline already solves this for Ken Burns (7 moves, ≤14% each, no adjacent
repeats). The remaining instances: chapter cards, transitions, title-in/title-out. If you
add any, use a damped overshoot rather than a linear ramp — MoneyPrinterTurbo's subtitle
spring is `scale = 1.0 − exp(−6·p)·cos(2.5π·p)` clamped, which needs no easing library. **[verified]**

**3.8 Faces detected but not identified.** Six frames of the same person at six different
scales passes a "does this contain a face" test. The people-aware literature treats *who* as
the signal: Aran & Gatica-Perez fuse audio-visual nonverbal cues to find dominant people in
group conversation (ICPR 2010, DOI 10.1109/icpr.2010.898) **[verified]**, and Takeuchi &
Sugimoto build both a personal-photo-library summarizer (ACM MIR workshop 2006, DOI
10.1145/1178677.1178707) and a user-adaptive variant of it (ACM IVR 2007, DOI
10.1145/1282280.1282349) — i.e. use the user's *own* library as the reference for what
matters to them. **[verified bibliographically; abstracts unavailable, so described by
title only]** `grep -i "embedding|identity|track|cluster" pipeline/analysis.py` returns
nothing today.

**3.9 Segment boundaries chosen for convenience.** The knapsack formulation penalises long
segments, so short segments win most selections; segmentation has "the most significant
impact on the performance measure" of anything in the pipeline (Otani et al.). If shot
boundaries are wrong, no scoring function rescues the result. **[verified]**

**3.10 Judged by an offline metric.** Random summaries "achieve comparable or better
performance to the state-of-the-art" on SumMe/TVSum. Any confidence interval computed against
such a metric is measuring noise. **[verified]**

**3.11 The tool finishes the job and stops.** No timeline out, no way to say "keep 78 of the
90 seconds." auto-editor exports seven formats; Remotion (61,442★) exists to be
programmable; lossless-cut (44,202★) exists to be non-destructive. Nothing in the pipeline
category exports an editable timeline. **[verified]**

**3.12 Transitions and handles not designed for.** A dissolve needs material on both sides.
If the only source is the material you cut, adjacent short clips cannot dissolve. A renderer
that assumes handles exist produces black frames, one-frame flashes, or silent-but-visible
transitions. auto-editor makes this explicit and lets the renderer **shorten or skip**.
**[verified]**

---

## 4. Open-source tools — verified repo stats and what the code actually does

All stats from `api.github.com/repos/{owner}/{repo}` on 2 Oct 2026. **[verified]**

| Repo | ★ | Forks | Lang | License | Last push | Notes |
|---|---:|---:|---|---|---|---|
| `harry0703/MoneyPrinterTurbo` | 127,967 | 20,025 | Python | MIT | 2026-10-01 | Text/topic → short video. ~50+ test files under `test/services/` |
| `calesthio/OpenMontage` | 62,171 | 7,931 | Python | AGPL-3.0 | 2026-09-06 | Created 2026-03-29. "12 production pipelines, 100+ tools, 700+ agent skill and production-knowledge files". 340 open issues |
| `remotion-dev/remotion` | 61,442 | 4,741 | TypeScript | NOASSERTION | 2026-10-01 | React-as-timeline |
| `mifi/lossless-cut` | 44,202 | 2,193 | TypeScript | GPL-2.0 | 2026-09-30 | The non-destructive reference point |
| `FireRedTeam/FireRed-OpenStoryline` | 3,457 | — | Python | Apache-2.0 | 2026-07-31 | Created 2026-02-07. 32 open issues. Best-read source in this report (§4.3) |
| `Breakthrough/PySceneDetect` | 5,214 | 523 | Python | BSD-3 | 2026-09-21 | Latest release v0.7.1 (2026-07-22). 64 open issues |
| `WyattBlue/auto-editor` | 5,403 | 674 | **Nim** | Unlicense | 2026-10-02 | **Rewritten from Python to Nim.** Created 2020-04-30. 0 open issues |
| `m1guelpf/auto-subtitle` | 2,286 | 367 | Python | MIT | **2024-07-12** | Stale ~2 years |
| `0xsline/OpenChatCut` | 2,080 | 314 | TypeScript | AGPL-3.0 | 2026-09-30 | Conversational editor, multi-track timeline, MCP, Remotion render |
| `m1guelpf/yt-whisper` | 1,445 | 144 | Python | MIT | **2024-01-16** | Stale ~2.7 years |
| `haltakov/natural-language-youtube-search` | 936 | 70 | Notebook | MIT | **2021-10-15** | Stale ~5 years |
| `MartinDelophy/ai-video-editor` | 885 | 114 | JavaScript | MIT | 2026-10-01 | "creators and AI agents edit the same real timeline" |
| `corvo007/MioSub` | 822 | 55 | TypeScript | AGPL-3.0 | 2026-07-18 | Subtitle-focused |
| `gzxx-2025/aid-studio` | 703 | 166 | Java | MIT | 2026-09-30 | AI comic/short-drama generation |
| `rendi-api/ffmpeg-cheatsheet` | 1,749 | 110 | — | — | 2026-04-29 | Categorised FFmpeg command reference |
| `zhouxiaoka/autoclip` | 9,097 | 1,670 | Python | MIT | 2026-10-01 | Podcast/interview/lecture → shorts, with subtitles + cover + copy |
| `Agent-Field/reels-af` | 115 | 37 | Python | Apache-2.0 | 2026-06-05 | Multi-agent reels, ~$0.1/reel self-claim |
| `scriptituk/xfade-easing` | 124 | — | C | MIT | 2026-09-09 | Patched-FFmpeg easings + GLSL transitions |

### 4.1 auto-editor — the closest analogue, and the most useful reference in this report

Its model: every moment on the timebase gets an integer **label** 0–255, and each label has
an **action**. Label 0 = inactive (default `cut`), label 1 = active (default `nil`). You can
define classes 2–255 with `--edit:N` / `--when:N`; **where classes overlap the higher label
wins.** **[verified]**

Methods that set label 1 (`src/analyze/`) **[verified]**:

| Method | Marks active when | Defaults |
|---|---|---|
| `audio` | loudest sample ≥ threshold | `threshold=0.04`, `stream=all` |
| `motion` | frame-to-frame change ≥ threshold | `threshold=0.02`, `stream=0`, `width=400`, `blur=9` |
| `blackdetect` | frame is mostly black | `threshold=0.98`, `pixel-black=0.10` |
| `subtitle`/`regex` | a subtitle line matches | `pattern` |
| `word` | a whole word appears in subtitles | `value` |

Thresholds accept **`dB`** (`audio:-19dB`, `motion:-19dB`) — video-editor-native units rather
than raw 0–1 floats. Combine with `or` / `and` / `xor` / `not`. `--edit "not audio:0.04"`
keeps the quiet parts.

**The motion algorithm, read from `src/analyze/motion.nim`** — and this is the important
part: **[verified]**

```
countDifferentPixels(a, b, len) -> int32     # differing GRAY pixels, SIMD
  arm64 : NEON   vld1q_u8 / vceqq_u8 / vaddlvq_u8
  x86   : SSE2   _mm_loadu_si128 / _mm_cmpeq_epi8 / _mm_movemask_epi8
  wasm  : WASM SIMD
  fallback: scalar loop
```

Frames are pulled through `format=gray`, one `motionness` value per timebase index (gaps
filled). Compare this to Cutting et al.'s VAI = `1.0 − median r` over correlations of
next-adjacent-frame pairs — a different estimator of the same underlying quantity, and a
perceptually-validated feature in the film-cinematography literature. Your
`MOTION_GOOD=12.0` / `MOTION_SHAKE=45.0` mean-abs-grey-delta constants are the same idea in
the same units. **The contribution worth stealing is the downscale-blur-differ recipe:
`width=400, blur=9`, gray, then SIMD popcount.** You get motion at a fraction of the cost and
it is defensible against the literature.

`src/analyze/blackdetect.nim`: per-frame ratio of pixels with luma ≤ `pixel-black`. Useful
for catching the start/end of a source video, where people hold a phone still and the frame
is not black but the content is dead.

**Pacing and truth-telling (the parts that matter most)** **[verified]**:

```
--margin 0.2sec              # default: pad each kept section 0.2s both sides
--margin 0.3s,1.5sec         # asymmetric — 1.5s after "avoids clipped words"
--smooth 0.2s,0.1s           # min-cut 0.2s, min-clip 0.1s (default); --smooth 0 = off
--preview                    # print what WOULD be cut, exit, render nothing
--when-active cut --when-inactive nil   # keep the silence, cut the speech = render the rejects
```

Two of these are directly stealable and neither is in the pipeline: **`--margin`** (a
few-frame handle on each side of every cut — the cheapest possible fix for "hard cuts feel
mechanical") and **rendering the rejects**. Your `--dry-run` is the peer of `--preview`;
you have no equivalent of the inverse, i.e. *a contact sheet or reel of what you threw
away and why*. For a pipeline whose entire value proposition is a human-readable decision
report, that is the obvious next artifact. §5.5.

Also worth knowing: `--transition dissolve:DURATION:MIN-CUT`, linked audio crossfade,
timeline head/tail fades, and per-export transition mapping (Premiere FCP7 XML transition
items; OTIO `SMPTE_Dissolve`; FCPXML spine transitions; ShotCut/Kdenlive same-track
transitions with fade filters). v1/v2 timeline formats silently drop transitions; the docs
say so.

And `--audio-normalize`: `peak:-3`, or full EBU R128 two-pass via ffmpeg `loudnorm` with
`i` (integrated, LUFS, default −24.0, range −70..5; common −23 EBU R128 / −16 streaming /
−14 podcasts), `lra` (default 7.0 LU, 1..50), `tp` (default −2.0 dBTP, −9..0), `gain`
(−99..99). Analysis pass measures, normalisation pass applies. **You are already at −14
LUFS / −1 dBTP — parity, no delta here.** Note only that auto-editor's `tp` default is −2
dBTP, 1 dB more conservative than yours; if you ever ship to a lossy transcode path, that
dB matters.

**Actions / ramps** (`skills/auto-editor-effects/SKILL.md`) — `zoom:1..1.5` is Ken Burns,
`zoom:1..1.5..1` is keyframed in-and-out, easing is `linear|in|out|inout`. `speed` preserves
pitch; `varispeed` does not (tape/vinyl). `add:path[:x:y:scale]` overlays a *layer* rather
than applying a per-frame effect, and `add` brings every stream the file holds (audio is
mixed in, not a second stream), so `add:./music.mp3,volume:0.4` is "background music, no
picture." Their `DEFAULT_RANDOM_SEED = 42` equivalent is worth copying if you want
reproducible shuffles — FireRed does exactly this in `plan_timeline.py`.

### 4.2 MoneyPrinterTurbo — four hard-won invariants in one file

`app/services/video.py` is 70,923 bytes. Four constants carry their own justification: **[verified]**

```python
_VIDEO_DURATION_SAFETY_MARGIN = 0.1
_MIN_MATERIAL_DIMENSION      = 480
_MIN_DIMENSION_TOLERANCE     = 10
_CLIP_PROCESSING_CONCURRENCY = 1     # default serial; each extra lane spawns an encoder task
```

- `_get_required_video_duration(audio_duration) = audio_duration + 0.1`. Their comment:
  "When video length is only *equal to* the audio length, FFmpeg may produce a final video
  slightly shorter because of frame-rate rounding." **The margin is a correctness fix, not
  padding.** You synthesise the score to fit the video, so you are safe in the easy direction
  — but this applies to any `--length` target you expose: if a user asks for 90 s and you
  target exactly 90 s of beats, you can render 89.97 s.
- `_MIN_DIMENSION_TOLERANCE = 10` exists because "WhatsApp compresses 9:16 to 478×850, two
  pixels under 480" and a hard 480 floor rejected all such material with
  `no valid materials found`. Your `ingest.py:74` has "Too small to be real content" — the
  question is whether it has the same tolerance band, and whether it can report *why* a file
  was rejected. Their `ingest.py:400-401` equivalent emits `"too small to use"` per item. §5.6.

Other named behaviours worth copying:

- `_apply_subtitle_spring_animation` / `_get_subtitle_spring_scale`:
  `scale = 1.0 − exp(−6·p)·cos(2.5π·p)`, clamped to `[_MIN,_MAX]`, applied identically to the
  colour frame **and** the alpha mask (their comment: applying it to only one composites the
  transparent region as a black outline on the first frame). Two-pass-safe, library-free,
  and it is the correct shape for anything that should "pop" — a Ken Burns ease-out, a card
  entry, a scale-up on a title.
- `subtitle_colors_are_indistinguishable(params)` — the renderer checks whether the chosen
  text colour is legible against the background and picks a different one. This is a real
  accessibility check, not a style preference.
- `subtitle_font_supports_text(font_path, text)` — glyph-coverage check with a Pillow sample
  render; CJK fallback breaks silently otherwise.
- `_escape_ffmpeg_concat_path` — concat-demuxer path escaping. A classic footgun when
  library paths contain `'` or `:`.
- `_write_videofile_with_codec_fallback` / `_disable_runtime_video_codec` — probe encoder
  availability at runtime and disable the failed codec permanently, so a second run doesn't
  repeat the failed encode.
- Transitions are `slide_out`, `zoom_in`, `zoom_out`, `shuffle`, implemented in **MoviePy**
  rather than `xfade`. Transitions are applied with duration `1`. Since the pipeline has no
  transitions at all, this is a "where to start" data point, not a model to copy.
- 50+ test files under `test/services/`, many named for specific regressions
  (`test_clip_speed.py`, `test_combine_clip_isolation.py`, `test_pause_concat_completeness.py`,
  `test_material_resolution...`, `test_media_atomic_concat.py`). A deterministic pipeline
  can and should have this.

### 4.3 FireRed-OpenStoryline — the agent-with-editing-rules design

File tree (207 paths) and three source files read. **[verified]**

`.storyline/skills/` contains `ai_transition_editing_skill`, `create_profile_style_skill`,
`default_editing_workflow_skill`, `speech_rough_cut_skill`, `subtitle_imitation_skill`.
The default workflow skill declares the canonical order, and marks which steps are
**mandatory**: **[verified]**

```
search_media (skip) → load_media (REQUIRED) → split_shots (skip) → understand_clips (skip)
→ filter_clips (skip) → group_clips (skip, but SHOULD RUN by default)
→ generate_script (skip) → element recommendation (skip, SHOULD RUN by default)
→ generate_voiceover (skip) → select_BGM (skip) → plan_timeline (REQUIRED) → render_video (REQUIRED)
```

`load_media` and `plan_timeline`/`render_video` are the only hard requirements. Everything
else degrades. That is the right failure posture for a pipeline, and it is what your
`--dry-run` plus graceful degradation already does.

Three defensive patterns in the code worth copying regardless of whether you ever add an
LLM: **[verified]**

1. `plan_timeline.py`: `DEFAULT_RANDOM_SEED = 42`,
   `BINARY_SEARCH_ITERATIONS = 50`, `RATIO_GROWTH_FACTOR = 2.0` / `RATIO_GROWTH_MAX = 10.0`,
   `SNAP_SAFETY_MAX_STEPS = 10_000`. A **binary search with an iteration cap and a bounded
   exponential probe** for the snap/stretch ratio, with a step ceiling so a pathological
   input cannot hang the run. Your beat snapping is exact (you generate the grid), so you
   don't need the ratio search — but the *shape* of the guard (iterations capped, growth
   bounded, hard step ceiling) is the right template for any search in the pipeline.
2. `filter_clips.py`: LLM called at `temperature=0.1, top_p=0.9, max_tokens=2048`, and
   `except: select_ids = input_clip_ids` with the message *"Failed to parse model output,
   using all clips."* Fail **open**, toward keeping content. A montage that drops material
   because a parse failed is worse than a montage with one bad shot.
3. `select_bgm.py`: librosa failure falls back to `ffmpeg -ac 1 -ar {sr} -vn` into a temp
   WAV, and if ffmpeg is also missing, raises a specific `RuntimeError` naming both causes.

`filter_clips.py` also does something your pipeline does not: it **injects each clip's
duration into the caption block before asking** (`_add_input_duration(clip_captions,
duration_lookup)`), so the selector can reason about pacing. Duration is the first thing a
montage editor needs and the last thing a captioner volunteers.

### 4.4 OpenMontage — the "production knowledge as files" bet

62,171★, AGPL-3.0, created 2026-03-29, and the README's own claim is "12 production
pipelines, 100+ tools, **700+ agent skill and production-knowledge files**", with a
"Paste A Video / start from a video you already love" entry point. **[verified]**

The bet worth noting is not the tooling — it is that **craft knowledge is stored as
versioned prose files the agent reads**, rather than baked into prompts. That is exactly
FireRed's `.storyline/skills/` shape. For a deterministic pipeline the analogue is
`docs/research/` plus the long explanatory docstrings already in `pipeline/arrange.py` —
which are unusually good (they explain *why* each fix exists, e.g. why `max_bars` is
capped). That is the same asset, already present.

Two cautions. 340 open issues five months after creation on a 62k★ repo is a very high
issue-per-star ratio and suggests high churn. And AGPL-3.0 — **do not copy code from it**
into an MIT-licensed pipeline. Read it, don't link it.

### 4.5 Others, briefly

- **`mifi/lossless-cut`** (44,202★, GPL-2.0) — the reference for what "keeps you in
  control" means without any AI: trim/cut at frame and timecode precision, export
  frame-accurate lossless segments, segment every file for fast scrubbing. It is also the
  fastest way to answer "can I just manually fix the three shots I don't like?"
- **`remotion-dev/remotion`** (61,442★) — React components as timeline. License reads
  `NOASSERTION` from the API, so check before embedding. The relevant idea for you is that
  the timeline is a *declarative data structure*, which is also what an OTIO/FCPXML export
  needs.
- **`MartinDelophy/ai-video-editor`** (885★, MIT) and **`0xsline/OpenChatCut`** (2,080★,
  AGPL-3.0) — both converge on the same stated premise: **one real timeline that a human
  and an agent both edit**, with OpenChatCut adding MCP + Remotion rendering. That is a
  live, third-party confirmation that "AI editor with an EDL out" is the shape people want.
- **`autoclip`** (9,097★, MIT) — longest-running serious OSS in this specific niche
  (created 2025-07-08, pushed 2026-10-01). Its scope statement names podcast / interview /
  lecture → Douyin / Xiaohongshu / TikTok / Reels / Shorts. **Not personal archives.**
- **Stale-but-instructive**: `auto-subtitle` (2 years), `yt-whisper` (2.7 years),
  `natural-language-youtube-search` (5 years). Star counts on these reflect 2022-era
  virality, not current quality. Useful as prior art; do not model your roadmap on them.

---

## 5. Techniques worth stealing

Each has: the source, what it is, why it works, and what it costs *you* specifically.

### 5.1 Accent-beat filtering → prefer accents when allocating cuts

**Source:** FireRed-OpenStoryline `select_bgm.py::_compute_accent_beats`, read from source. **[verified]**
**What:** percussive-separation onset strength at each tracked beat → local normalisation
over an 8-beat moving average → keep top-30% percentile → local-maximum filter →
greedy minimum-separation suppression.
**Why:** your `music.py:222-233` picks the *phase* of the downbeat grid by onset energy and
then treats every beat equally. Phase alignment makes cuts feel "on the grid"; accent
selection makes them feel "on the music." These are different problems and you have solved
only the first.
**Cost:** ~60 lines + a `librosa` dependency (you already detect onsets synthetically, so
this can run on your own onset envelope with no new dependency — you just need the
per-beat strength values you already compute to pick the phase).
**Risk:** low. Purely additive; `accent_beats` is a new list, existing beat logic untouched.
**Guard:** if the accent list is shorter than the number of shots needed, fall back to
beats. FireRed's own philosophy (fail toward keeping content).

### 5.2 Emit an editable timeline: FCPXML or OpenTimelineIO

**Source:** auto-editor `src/exports/{otio,fcp11,fcp7,kdenlive,mlt,shotcut,json}.nim` and
`docs/src/docs/transition.md`. **[verified]**
**What:** write the plan as an interchange format alongside the render.
**Why:** your user is *one human editor away* from wanting this. auto-editor exports seven
formats; you need one. OTIO is the better first choice (JSON, library-agnostic, and the
transitions map cleanly to `SMPTE_Dissolve`); FCPXML is the better second (Final Cut users
just double-click it).
**Cost:** ~80 lines for a minimal OTIO document — your timeline is already fully
determined: media references (source path), source ranges (the still's dwell or the
video's in/out), and one transition entry per join if you ever add them.
**Risk:** low. Purely additive; the MP4 still renders.
**Bonus:** OTIO is the cheapest possible answer to "can I fix shot 23?" and it makes the
`report.py` output machine-checkable.

### 5.3 Split the score into aesthetic and technical branches

**Source:** DOVER / DIVIDE-3k, ICCV 2023, pp. 20144–20154. **[verified]**
**What:** two separately-normalised scores with separately-configurable weights, instead of
one `sum(wᵢ·fᵢ)` over six terms.
**Why:** the terms are not commensurable. DOVER's subjective study found human quality
opinions are "universally and inevitably affected by both aesthetic and technical
perspectives" — which is an argument for *measuring both*, not for averaging them into one.
It also gives you DOVER++-style reporting: "this run: technical 0.81, aesthetic 0.44", which
tells a user what to change.
**Concretely:** technical = `{sharpness, exposure, clipped, black, contrast}`;
aesthetic = `{faces, color, motion}`. Normalise each within the library (rank-based, so a
dark trip does not collapse technical to zero), then combine.
**Cost:** ~40 lines in `analysis.py`, a config restructure, and a change to the report.
**Risk:** medium — it changes every score, so any existing thresholds need recalibration.
Do it behind a config flag.

### 5.4 Per-source-file primary/overflow tiering

**Source:** MoneyPrinterTurbo `app/services/video.py::_prioritize_unique_source_clips`,
read from source. **[verified]**
**What:** group candidate clips by source file; emit the **longest** excerpt per source
first; keep the rest as an overflow tier; stable-sort by a cumulative `source_usage` counter.
**Why:** `max_per_minute: 2` is the right idea at the wrong granularity. A single 40-second
panorama clip sliced into six 4-second excerpts across a 90-second piece reads as *the same
footage*, which is worse than two adjacent near-duplicates that dHash would catch.
**Cost:** ~25 lines in `select.py`.
**Risk:** low. Keep the overflow tier (their comment is explicit: backfilling beats failing
to produce a video).

### 5.5 Render the rejects

**Source:** auto-editor `--when-active cut --when-inactive nil`, documented as "see what
gets cut". **[verified]**
**What:** emit a contact sheet (or a short reel) of every rejected candidate with its score
components and the reason it lost.
**Why:** you already generate a human-readable decision report; this is its visual sibling
and it is what turns a black box into something a person can argue with. auto-editor
invented the inverse-mode flag; nobody in the montage-from-archives category does this.
**Cost:** ~60 lines (you already have the per-item scores and the frames).
**Risk:** none. Additive output.

### 5.6 `--margin`: a few frames of handle around every cut

**Source:** auto-editor, default `0.2 s` both sides, with asymmetric `--margin 0.3s,1.5sec`. **[verified]**
**What:** extend each kept section by a small pad, clamped so neighbours do not overlap.
**Why:** two effects. (a) On *video* material, a hard cut that lops a subject's motion
mid-gesture is the cheapest "AI edit" tell there is. (b) It gives every transition somewhere
to dissolve from, which is the handle requirement in §1.5.
**Cost:** ~10 lines in the concat builder; ~1 line in config.
**Risk:** low. Conflicts with exact beat alignment at the margins — clamp so the pad never
pushes a cut off its frame.

### 5.7 Gate transitions on *removed* interval, not kept interval

**Source:** auto-editor `dissolve:DURATION:MIN-CUT`, `MIN-CUT` default `1sec`. **[verified]**
**What:** only dissolve across joins where the discarded material between them was ≥ 1 s.
**Why:** their stated reason is that dissolving across a tiny trim "reads as a stutter."
This is a one-line rule that prevents the single most common amateur-looking montage effect.
**Cost:** one condition.
**Risk:** none. Applies whether or not you ever add transitions.

### 5.8 1/f-shaped shot durations, decoupled from quality

**Source:** Cutting et al. 2011, i-Perception 2(6):569–576, PMC3485803. **[verified]**
**What:** replace score-proportional duration with a power-law sample, then quantise to
whole beats and respect `min_shot_seconds`.
**Why:** three separate findings in one paper. (a) Post-2000 ASL is **below 4 s** — your
`photo_min/photo_max = 2.6/4.6` is slower than the mean of real film. (b) Shot-duration
sequences are **near-1/f**, i.e. self-similar across timescales — short runs must be
interrupted by long runs. (c) In modern film, **shorter shots carry proportionally more
within-shot motion** — so pair short durations with your high-motion items, which gives you
a *principled* coupling between duration and content that is not "good = long."
**Concretely:** sample `d ~ p(d) ∝ d^(-α)` for α ≈ 1 (adjust to taste), clamp to
`[photo_min, photo_max]`, round up to a whole number of beats. Remove `_base_weights` from
the *duration* path entirely — keep it for the *selection* path, where it belongs.
**Cost:** ~15 lines.
**Risk:** medium — it visibly changes every render. Ship it behind `--pace` with the
current behaviour as default, and A/B them side by side.

### 5.9 Cheap motion: downscale, blur, gray, SIMD popcount

**Source:** auto-editor `src/analyze/motion.nim` (NEON/SSE2/WASM-SIMD `countDifferentPixels`,
`format=gray`, `width=400`, `blur=9`) **[verified]**; perceptual validation of the underlying
quantity in Cutting et al.'s VAI **[verified]**.
**What:** compute motion as the count of differing gray pixels between consecutive
downscaled, blurred frames, one value per timebase index.
**Why:** your `MOTION_GOOD/MOTION_SHAKE` thresholds already work in these units; this is
the recipe that makes them cheap enough to compute for every frame of every clip.
**Cost:** moderate if you re-implement the SIMD; low if you keep numpy. `cv2.absdiff` +
`cv2.countNonZero` on a `width=400, blur=9` gray pair is within 2× of the hand-SIMD version.
**Risk:** low.

### 5.10 Library-free damped-overshoot easing

**Source:** MoneyPrinterTurbo `_get_subtitle_spring_scale`:
`scale = 1.0 − exp(−6·p)·cos(2.5π·p)`, clamped. **[verified]**
**What:** a spring/overshoot curve in four lines, no easing library.
**Why:** everything the pipeline moves right now moves linearly. A linear Ken Burns reads as
a slideshow; a decelerating one reads as a camera move. You already scale travel by a
square-root law — multiplying by an ease-out on top is the missing half of "paced camera
moves," and it costs one function.
**Cost:** ~10 lines. Apply to Ken Burns scale/position, and to any card or transition you
add later.
**Risk:** low. It *is* a visual change — A/B it.

### 5.11 Edge transitions: fade in at the head, out at the tail

**Source:** auto-editor transitions doc — "the timeline fades in at the start and fades out
at the end," preserved across every export. **[verified]**
**Why:** a piece that starts at full brightness and stops at full brightness has no
beginning or end. Yours currently fades *audio* (`render.py:634-635`,
`min(2.5, total*0.05)`) but not picture.
**Cost:** ~4 lines — a `fade=t=in` / `fade=t=out` on the video chain, reusing the durations
you already compute for audio.
**Risk:** none.

### 5.12 `--smooth`: floor both cuts and clips

**Source:** auto-editor `--smooth MIN_CUT,MIN_CLIP`, default `0.2s, 0.1s`. **[verified]**
**What:** enforce a minimum duration on every cut and every kept clip.
**Why:** you have `min_shot_seconds: 0.8` but it is used for *intra-source shot detection*
(`analysis.py:185, 250`), not for montage shot lengths. A 0.3-second flash of a frame in a
90-second piece is worse than not using it at all.
**Cost:** ~10 lines in `assign_timing`'s `lo` computation.
**Risk:** low.

### 5.13 Reject-and-report at the resolution floor, with tolerance

**Source:** MoneyPrinterTurbo `is_material_resolution_acceptable` — 480 nominal, **10 px
tolerance**, because real pipelines receive 478×850 from WhatsApp and a hard floor rejects
everything. **[verified]**
**What:** a hard floor *plus* a small tolerance band *plus* a per-item rejection reason in
the report.
**Why:** you have a floor (`ingest.py:74`); check it has the tolerance and that
`ingest.py:400-401`-equivalent messages reach the report. A silent discard is the worst
possible outcome: the user wonders why their 4-second video is missing.
**Cost:** ~10 lines.

### 5.14 Codec/encoder probe and permanent disable

**Source:** MoneyPrinterTurbo `_get_effective_video_codec`, `_ffmpeg_encoder_exists`,
`_disable_runtime_video_codec`, `_write_videofile_with_codec_fallback`. **[verified]**
**What:** probe available encoders at start; on failure, blacklist the codec for the rest of
the run *and* persist the finding.
**Why:** libx264 missing, VideoToolbox unavailable, `hevc_videotoolbox` refusing a profile —
these are the difference between "the tool crashed" and "the tool produced something".
**Cost:** ~20 lines.
**Risk:** none.

### 5.15 `--audio-normalize` reference values

**Source:** auto-editor `docs/src/docs/anorm.md`. **[verified]**
**What:** EBU R128 two-pass, `i` in LUFS, `lra` in LU, `tp` in dBTP.
**What for you:** you are at −14 LUFS / −1 dBTP (podcast target, streaming-class TP). auto-editor
defaults `tp` to **−2.0 dBTP**. Consider −2 dBTP so a downstream platform transcode has
headroom. That is the entire delta — but it is a real one.
**Cost:** one constant.

### 5.16 Determinism: seed every shuffle, and test the invariants

**Source:** FireRed `DEFAULT_RANDOM_SEED = 42` in `plan_timeline.py` **[verified]**;
MoneyPrinterTurbo's 50+ `test/services/` files, many named for specific regressions
**[verified]**.
**What:** seed every random decision; write a test per invariant you claim in the README.
**Why:** your README makes a lot of *measured* claims (5.402 → 5.688, −4.6% vs −10.0%, 2.4×
travel/s, 17 ms beat error). Those claims deserve tests, and every one of the techniques in
this section adds an invariant that deserves one.

---

## 6. Consumer / prosumer products

`Control retained?` = can you override the selection, and can you get an editable
timeline out. `Archive/trip fit?` = personal media library → finished video, **not**
spoken-word/social-media clipping.

| Name | Category | What it does | Control retained? | Archive/trip fit? | Maturity | Notes |
|---|---|---|---|---|---|---|
| **Google Photos — Highlight videos** | Cloud photo app | Builds themed highlight videos; replaced the old auto-**Movies**. You pick **people, places, themes and dates**; also offers 6-second Photo-to-video animations, Collage, and Remix | **Partial.** Theme/date/people selection is user-driven and there is an edit path into CapCut with Google Photos templates. **No EDL/XML export.** | **Best in class** — the only product here built for a personal library | Shipped consumer feature | Also offers hide-people / hide-photos controls **[reported]** |
| **Google Photos — Recap (2025)** | Cloud photo app | Gemini-generated year recap; opens in CapCut for editing with Google Photos templates | Partial; templates, not a timeline | Weak — annual, not library-driven | Announced 2025 **[reported]** | Marketing-led; evaluate on output, not on the demo |
| **Apple Photos + Apple Intelligence (June 2026)** | OS | Generated video *descriptions*, semantic search over camera clips, Home-app surfacing of "noteworthy clips" | N/A — this is search and surfacing, **not montage** | Not a montage tool | Shipped | Notable because it means on-device semantic video understanding is now a commodity **[reported]** |
| **Adobe Premiere — Eddie AI (Scripted Mode)** | Pro NLE | Free assistant; Scripted Mode assembles a rough cut from a script | **Yes** — lands in a real Premiere timeline you can finish | Weak — script-driven, assumes you supply the script | Shipping, covered by No Film School / Premiere Gal **[reported]** | The clearest "AI that hands you a timeline" shipping today |
| **Premiere 2025 AI features** | Pro NLE | Scene Edit Detection; Auto color / Lumetri "Auto"; Generative Extend; media intelligence | **Yes** — all in-timeline, non-destructive | Useful per-clip; not montage assembly | Shipping | The "Auto" button and Scene Edit Detection are the two most relevant primitives |
| **iMovie 10.4.3** | Consumer NLE | Manual + Magic Stew / style-based auto assembly; requires macOS 14.6+ | **Yes**, fully | Usable but manual; last release 2024-11-13 **[reported]** | Maintenance mode | Baseline for what "keeps control" means to a normal person |
| **LosslessCut** | OSS desktop | Frame-accurate lossless trim/cut/segment, no re-encode | **Total** | Excellent — the manual escape hatch | 44,202★, active | GPL-2.0 |
| **Opus Clip** | SaaS | Long video → vertical shorts; auto-reframe, captions, Virality Score (0–99) breaking down hook / pacing / topic shifts | **No EDL.** Reorder/splice in-app only; priced per source-minute | **No** — long-form spoken-word → social clips | Commercial | Auto-reframe genuinely good; B-roll insertion described as flaky; ~97% caption accuracy; "2.3× TikTok views for 80+ scores" is a **[reported]** vendor-adjacent marketing claim (ScaleReach) — treat as unverified **[reported]** |
| **CapCut (AI features)** | Consumer | Templates, auto-captions, text-to-video | Partial; template- and track-based | **No** — template library is its strength, archive assembly is not | Commercial, very large | Text-to-video is stock-footage assembly. Reviewers consistently report it "falls apart on visual consistency across cuts" **[reported]** (review-blog sourced) |
| **Descript** | Pro SaaS | Transcript-as-timeline editing; Studio Sound; Underlord; clip generation with Veo 3.1 / Sora / Kling; chapter markers | **Yes** — real timeline, and exports to other NLEs | **No** — fundamentally spoken-word / podcast oriented | Commercial, mature | The one product whose *editing model* is defensible; its domain assumption is just wrong for archives |
| **Kapwing (Kai agent)** | Consumer SaaS | Conversational editing; Kai learned background remove/blur and layer deletion by May 2026; works over >10 uploaded media per chat | Partial | Weak | Commercial | The >10-media change matters: an agent that can see a folder is closer to the archive case |
| **DJI LightCut** | — | **Discontinued. Updates and maintenance ended September 2026.** | — | — | **Dead** | Replaced by DJI Mimo / DJI Fly **[verified on dji.com/lightcut]** |
| **Premiere / Resolve / FCP / Kdenlive / ShotCut** | Pro NLEs | Receive auto-editor's exported XML/FCPXML/OTIO | **Total** | The destination | Mature | This is where a pipeline's output should be *aimed* |

**The category conclusion, stated plainly:** essentially the entire consumer "AI video"
category targets **social-media clips cut from spoken-word or long-form content**. Opus Clip,
autoclip, CapCut, Descript and Kapwing are all, at the core, transcript- or virality-driven.
**Not one of them solves "turn my trip photos into a video."** The two products that address
a personal library at all are Google Photos Highlight videos (partial control, no export)
and Premiere Eddie's Scripted Mode (full control, wrong domain assumption). LightCut, which
was the closest thing to a consumer "import your footage, get a cut" product, is dead as of
September 2026.

That is the market gap this pipeline is in.

---

## 7. Academic literature

Every row below was verified against a primary source (arXiv API, CrossRef, Semantic Scholar
graph API, CVF Open Access, or PubMed Central full text) on 2 Oct 2026.

| Paper / Year | Task | Method | Dataset | Headline metric | Verdict |
|---|---|---|---|---|---|
| **Otani, Nakashima, Rahtu, Heikkilä — Rethinking the Evaluation of Video Summaries, CVPR 2019** | *Meta-evaluation* of summarization benchmarks | Randomization test: summaries from random importance scores + random segment boundaries; vary segmentation | SumMe, TVSum | Random summaries **comparable to or better than SOTA**; sometimes beat the reference summaries. Segmentation has "the most significant impact on the performance measure" | **The most important paper in this report.** Read it before optimising anything. **[verified]** |
| **Cutting, Brunick, DeLong, Iricinschi, Candan — Quicker, faster, darker, i-Perception 2(6):569–576, 2011** (DOI 10.1068/i0441aap, PMC3485803) | Cinematographic measurement of 75 years of film | Hand-assisted shot segmentation; log shot length; Visual Activity Index `1.0 − median r` over adjacent-frame correlations; per-frame median luminance | 160 English-language films 1935–2010, 5 genres, ~1,100 shots / 165k frames each | ASL **~10 s (1930s–40s) → below 4 s (after 2000)**, r = −.75, t(158) = 14.3, p < .0001, all 5 genres. Shot-length fluctuation **near 1/f**, matching human RT fluctuation. Short shots ↔ more within-shot motion in modern film (r = −.46, p < .0001), **no** relation 1935–1960. Luminance decreasing | **Directly actionable.** The only source here with numbers a montage pipeline can be tuned against. Hand-segmented, so numbers are trustworthy. **[verified]** |
| **Wu, Zhang, Liao, Chen, Hou, Wang, Sun, Yan, Lin — DOVER / DIVIDE-3k, ICCV 2023, pp. 20144–20154** | UGC video quality assessment | Large-scale subjective study; **disentangled** aesthetic + technical branches; DOVER++; DOVER++ for single-perspective eval | DIVIDE-3k | "state-of-the-art performance" — **abstract contains no numbers**. Do not quote SRCC | The *architecture* is the finding: human opinions are "universally and inevitably affected by both" perspectives. **[verified]** |
| **Qiao, Zheng, Bo, Peng, Huang, Jiang, Wang, Chen, Zhou, Jin — VADB, arXiv:2510.25238 (2025-10-29)** | Video aesthetics | VADB-Net: dual-modal pre-training, two-stage | **10,490 videos**, annotated by **37 professionals**: overall + per-attribute aesthetic scores, free-text comments, objective tags | "outperforms existing VQA models in scoring tasks" — no numbers in abstract | Best available *annotation schema* to copy if you ever want learned scoring. 37 raters, not crowd noise. Repo `BestiVictory/VADB` has **no detected license**. **[verified]** |
| **Gygli, Grabner, Riemenschneider, van Gool — Creating summaries from user videos, ECCV 2014, pp. 505–520** | Personal/user-video summarization | Saliency + aesthetics + presence-of-people | SumMe (created here: 25 videos, ≥15 human summaries each) | The benchmark that Otani et al. showed is ill-posed | Still the canonical reference for the task, and the origin of the broken benchmark. **[verified]** |
| **Morère, Goh, Veillard, Chandrasekhar, Lin — Co-Regularized Deep Representations for Video Summarization, arXiv:1501.07738** | Keyframe summarization | CNN + restricted Boltzmann machines; co-regularized subject–scene association | Keyframe benchmarks | **Includes a user study against a real video-sharing site's in-use summarization.** "Our method consistently outperforms the baseline schemes for any given amount of keyframes **both in terms of attractiveness and informativeness**. The lead is even more significant for smaller summaries." | **The counterweight to Otani**: user studies with attractiveness *and* informativeness as separate axes do exist. Rare, but not absent. This is the evaluation template to copy. **[verified]** |
| **Zhang, Gong, Duan, van den Hengel, Liu — Let Your Video Listen to Your Music! (MVAA), arXiv:2506.18881 (2025-06-23)** | Beat–motion alignment | Two-step: insert keyframes at beat-aligned timestamps, then frame-conditioned diffusion inpainting for intermediate frames; pretrain on a small set + inference-time fine-tune | Not verified from the abstract | Beat-alignment metric + content-preservation metric + user study (per abstract) | **Different framing of your problem:** do not cut on the beat, *retime* onto it. Preserves continuous content. Right answer for real video where you would rather not drop frames. **[verified]** |
| **Mei, Yao, Jin — UBiSS, arXiv:2406.16301** | Bimodal (visual+textual) summarization | Unified model; **list-wise ranking objective**; NDCG_MS metric | BIDS (video, VM-summary, TM-summary) triplets; QVHighlights | "better than multi-stage summarization pipelines" | The **pairwise/list-wise ranking objective** is the stealable part — your selection problem is a ranking problem, not a per-item classification problem. **[verified]** |
| **Lee, Gong, Cho — Video Summarization with LLMs (LLMVS), arXiv:2504.11199** | Frame importance | M-LLM turns frames into captions; an LLM scores each caption's importance **in its local context**; a global attention pass refines across all captions | Standard benchmarks | "superiority… over existing ones" | The **local-then-global** two-stage scoring is a clean design if you ever want semantics. Your pipeline is deliberately deterministic; this is the note for if that changes. **[verified]** |
| **Qiu, Wu, Peng, Miao, Yang, Du — TVHighlights, CVPR 2026, pp. 9773–9783** | Highlight detection in film/TV | **LTV-HD**: (1) weakly-supervised pretraining on video-level labels, (2) LLM↔light-model iterative refinement; LLM emits noisy clip-level pseudo-labels, model learns noise-robustly, high-confidence predictions guide the LLM | **TVHighlights: 1,721 curated videos**, annotations from community behaviour (**no human labelling**) | "state-of-the-art performance on TVHighlights" | Good dataset, honest that labels are behavioural. But **no user study and no evidence a human would want to watch it.** Domain is movies, not personal archives. **[verified]** |
| **Wang, Xu, Zhang, Tang, Cheng — OmniShotCut, arXiv:2604.24762 (2026-04-27)** | Shot boundary detection | Shot-Query Transformer; relational prediction of shot ranges; **fully synthetic transition-synthesis pipeline** to get precise ground truth without manual labelling | OmniShotCutBench (new) | "effective and generality" | The synthetic-transition idea is the interesting part: **generate your own ground truth** for boundary detection rather than hand-labeling, exactly as Cutting et al. hand-labeled their 160 films. **[verified]** |
| **Takeuchi & Sugimoto — Video summarization using personal photo libraries, ACM MIR workshop 2006 (DOI 10.1145/1178677.1178707)**; and **— User-adaptive home video summarization using personal photo libraries, ACM IVR 2007 (DOI 10.1145/1282280.1282349)** | Personal-video summarization using the user's own photo library | Not described here | Not verified | Not verified | **Bibliographically verified only — abstracts unavailable.** Cited for the idea, which is exactly yours: use the user's own library as the reference for what matters to them. **[verified bibliographically]** |
| **Aran & Gatica-Perez — Fusing Audio-Visual Nonverbal Cues to Detect Dominant People in Group Conversations, ICPR 2010 (DOI 10.1109/icpr.2010.898)** | Who matters in a scene | Fuses audio + visual nonverbal cues | Group conversations | 17 citations | The reference point for "identify *which* person", which face detection alone cannot do. Meetings, not trips. **[verified]** |
| **Obrador, de Oliveira, Oliver — Supporting personal photo storytelling for social albums, ACM MM 2010 (DOI 10.1145/1873951.1874025)** | Personal photo narrative | Not verified | Not verified | 54 citations | Bibliographically verified only; cited as existing prior art in this exact problem space. **[verified bibliographically]** |
| KonIQ-10k / AVA / DPChallenge / Composition-Aware IAA (WACV 2020) / NIMA-era image-aesthetics work | *Image* aesthetics (listed for completeness, not recommended) | — | AVA: ~255k DPChallenge images, ~210 ratings/image, 1–10 scale | — | Image-aesthetics models do not transfer cleanly to video, and VADB's abstract explicitly names this as the barrier ("the temporal dynamics of video… hinder direct application of image-based methods"). Use as component priors only. **[verified for dataset facts]** |

### 7.1 Would a human actually want to watch it?

Scoring every paper on this axis, honestly:

- **Has a real subjective evaluation:** Morère et al. (2015) — attractiveness *and*
  informativeness vs. a live product's summarizer. That's it. One paper out of fourteen.
- **Has *some* user study:** MVAA (per its abstract).
- **Self-reported SOTA on an offline metric only:** DOVER, VADB, UBiSS, LLMVS,
  TVHighlights, OmniShotCut. Given Otani et al., an offline F1 number on these benchmarks is
  close to uninformative.
- **Measured against real films rather than a benchmark:** Cutting et al. Their data is
  descriptive, not competitive, which is exactly why it is trustworthy and useful.

**Bottom line:** the "would a human want to watch it" evidence in this literature is
essentially one paper. That is the strongest argument for §5.5 (render the rejects) and for
A/B renders by hand — and the strongest argument for not reading benchmark deltas as quality
signals.

---

## 8. Source quality: what I trusted and what I discarded

**Primary sources used:** arXiv API (`export.arxiv.org/api/query`), CrossRef REST,
Semantic Scholar graph API (`api.semanticscholar.org/graph/v1`), CVF Open Access
(`openaccess.thecvf.com`), PubMed Central full text, `raw.githubusercontent.com` for source
code, `api.github.com` for repo stats, official vendor documentation
(`support.google.com`, `apple.com/newsroom`, `dji.com`), and Reddit threads as
*practitioner* opinion only.

**Flagged and NOT used as evidence** — consulted for leads, nothing cited to them:
`scalereach.com` (Opus Clip "2.3× views"), `editorskeys.com`, `nemovideo.com`,
`stylefactoryproductions.com`, `aaapresets.com`, `nigelcamp.com` (the "ASL as pace control"
framing — I replaced it with Cutting et al.'s measured numbers), `blog.pic-time.com`,
`kapwing.com/resources` marketing posts, and YouTube craft videos. The "average shot length
is 4–6 s in most films" claim that circulates in those videos is **[rumoured]** and I did
not use it; Cutting et al. is the citable version and it says something different (below
4 s *post-2000*, ~10 s *pre-1960*).

**No Film School** and **Premiere Gal** were used only for what Eddie AI's Scripted Mode
does; the feature's existence should be re-verified in the product before being relied on.

**Two claims I could not resolve and did not paper over:** the alleged CVPR 2017 "Social-Aware
Model for Selecting Photos" paper (§0), and DOVER's SRCC figures (§0).

---

## 9. Recommended order of work

Ranked by (value to the finished film) ÷ (effort + risk):

1. **§5.11** edge fades — 4 lines, unambiguously an improvement.
2. **§5.6** `--margin` handles — 10 lines; prerequisite for §5.7.
3. **§5.8** decouple duration from quality, 1/f-shaped — 15 lines; the single largest
   *perceptual* change available. Ship behind `--pace`, A/B side by side.
4. **§5.1** accent-beat preference — 60 lines; the single largest *musical* change available.
5. **§5.5** render the rejects — 60 lines; turns the report from prose into something
   arguable, and is what makes §5.8's A/B honest.
6. **§5.2** OTIO/FCPXML export — 80 lines; converts the tool's failure mode from
   "unusable" to "almost right."
7. **§5.4** per-source tiering — 25 lines; fixes a specific, named, observed cheapness.
8. **§5.3** aesthetic/technical split — 40 lines but recalibrates everything; behind a flag.
9. **§5.12/5.13/5.14** min-clip floor, rejection tolerance + reasons, encoder probe — hygiene.
10. **§5.10** damped-overshoot easing on Ken Burns — 10 lines, visible.
11. **§5.16** seed everything, test every README claim.
12. Only then consider transitions at all (§5.7), and only with the handle requirement in
    §1.5 satisfied and `xfade-easing`'s linear-rate problem addressed (§5.10's curve, or a
    patched FFmpeg).

**And do not do these:** do not read benchmark F1 deltas as quality signals (§1.1); do not
copy OpenMontage code (AGPL-3.0) into an MIT pipeline; do not chase star counts on
`auto-subtitle` / `yt-whisper` / `natural-language-youtube-search`, which are 2–5 years
stale and reflect 2022 virality; do not copy DOVER SRCC numbers from a blog.

---

## 10. Sources

### Papers and datasets
1. Otani, Nakashima, Rahtu, Heikkilä. *Rethinking the Evaluation of Video Summaries.* CVPR 2019. Project page: https://mayu-ot.github.io/rethinking-evaluation-of-video-summaries · PDF: https://openaccess.thecvf.com/content_CVPR_2019/papers/Otani_Rethinking_the_Evaluation_of_Video_Summaries_CVPR_2019_paper.pdf · code: `mayu-ot/rethinking-evs`
2. Cutting, Brunick, DeLong, Iricinschi, Candan. *Quicker, faster, darker: Changes in Hollywood film over 75 years.* i-Perception 2(6):569–576, 2011. DOI https://doi.org/10.1068/i0441aap · full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC3485803/
3. Wu, Zhang, Liao, Chen, Hou, Wang, Sun, Yan, Lin. *Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives (DOVER / DIVIDE-3k).* ICCV 2023, pp. 20144–20154. https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Exploring_Video_Quality_Assessment_on_User_Generated_Contents_from_Aesthetic_ICCV_2023_paper.html
4. Qiao, Zheng, Bo, Peng, Huang, Jiang, Wang, Chen, Zhou, Jin. *VADB: A Large-Scale Video Aesthetic Database with Professional and Multi-Dimensional Annotations.* arXiv:2510.25238. https://arxiv.org/abs/2510.25238 · code: https://github.com/BestiVictory/VADB
5. Gygli, Grabner, Riemenschneider, van Gool. *Creating summaries from user videos.* ECCV 2014, pp. 505–520.
6. Morère, Goh, Veillard, Chandrasekhar, Lin. *Co-Regularized Deep Representations for Video Summarization.* arXiv:1501.07738. https://arxiv.org/abs/1501.07738
7. Zhang, Gong, Duan, van den Hengel, Liu. *Let Your Video Listen to Your Music!* (MVAA). arXiv:2506.18881. https://arxiv.org/abs/2506.18881
8. Mei, Yao, Jin. *UBiSS: A Unified Framework for Bimodal Semantic Summarization of Videos.* arXiv:2406.16301. https://arxiv.org/abs/2406.16301 · code: https://github.com/MeiYutingg/UBiSS
9. Lee, Gong, Cho. *Video Summarization with Large Language Models* (LLMVS). arXiv:2504.11199. https://arxiv.org/abs/2504.11199
10. Qiu, Wu, Peng, Miao, Yang, Du. *TVHighlights: LLM-Guided Human-Free Collaborative Training for Video Highlight Detection in Movies and TV Dramas.* CVPR 2026, pp. 9773–9783. https://openaccess.thecvf.com/content/CVPR2026/html/Qiu_TVHighlights_LLM-Guided_Human-Free_Collaborative_Training_for_Video_Highlight_Detection_in_CVPR_2026_paper.html
11. Wang, Xu, Zhang, Tang, Cheng. *OmniShotCut: Holistic Relational Shot Boundary Detection with Shot-Query Transformer.* arXiv:2604.24762. https://arxiv.org/abs/2604.24762
12. Takeuchi, Sugimoto. *User-adaptive home video summarization using personal photo libraries.* ACM IVR 2007. DOI https://doi.org/10.1145/1282280.1282349
13. Takeuchi, Sugimoto. *Video summarization using personal photo libraries.* ACM MIR workshop 2006. DOI https://doi.org/10.1145/1178677.1178707
14. Aran, Gatica-Perez. *Fusing Audio-Visual Nonverbal Cues to Detect Dominant People in Group Conversations.* ICPR 2010. DOI https://doi.org/10.1109/icpr.2010.898
15. Obrador, de Oliveira, Oliver. *Supporting personal photo storytelling for social albums.* ACM MM 2010. DOI https://doi.org/10.1145/1873951.1874025
16. Song, Vallmitjana, Stent, Jaimes. *TVSum: Summarizing Web Videos Using Titles.* CVPR 2015, pp. 5179–5187.

### Repositories — source read
17. `WyattBlue/auto-editor` — https://github.com/WyattBlue/auto-editor · `src/analyze/motion.nim`, `src/analyze/blackdetect.nim`, `src/exports/otio.nim`, `skills/auto-editor/SKILL.md`, `skills/auto-editor-effects/SKILL.md`, `docs/src/docs/transition.md`, `docs/src/docs/anorm.md` · docs https://auto-editor.com
18. `harry0703/MoneyPrinterTurbo` — https://github.com/harry0703/MoneyPrinterTurbo · `app/services/video.py` (`_prioritize_unique_source_clips`, `is_material_resolution_acceptable`, `_get_required_video_duration`, `_get_subtitle_spring_scale`, `_write_videofile_with_codec_fallback`, `_escape_ffmpeg_concat_path`)
19. `FireRedTeam/FireRed-OpenStoryline` — https://github.com/FireRedTeam/FireRed-OpenStoryline · `src/open_storyline/nodes/core_nodes/{plan_timeline.py,select_bgm.py,filter_clips.py}`, `.storyline/skills/default_editing_workflow_skill/SKILL.md`
20. `scriptituk/xfade-easing` — https://github.com/scriptituk/xfade-easing
21. `calesthio/OpenMontage` — https://github.com/calesthio/OpenMontage · https://www.openmontage.video
22. `Breakthrough/PySceneDetect` — https://github.com/Breakthrough/PySceneDetect · v0.7.1 (published 2026-07-22)
23. `mifi/lossless-cut` — https://github.com/mifi/lossless-cut
24. `remotion-dev/remotion` — https://github.com/remotion-dev/remotion
25. `0xsline/OpenChatCut` — https://github.com/0xsline/OpenChatCut
26. `MartinDelophy/ai-video-editor` — https://github.com/MartinDelophy/ai-video-editor
27. `zhouxiaoka/autoclip` — https://github.com/zhouxiaoka/autoclip
28. `rendi-api/ffmpeg-cheatsheet` — https://github.com/rendi-api/ffmpeg-cheatsheet
29. `m1guelpf/auto-subtitle` — https://github.com/m1guelpf/auto-subtitle
30. `m1guelpf/yt-whisper` — https://github.com/m1guelpf/yt-whisper
31. `haltakov/natural-language-youtube-search` — https://github.com/haltakov/natural-language-youtube-search
32. `corvo007/MioSub` — https://github.com/corvo007/MioSub
33. `gzxx-2025/aid-studio` — https://github.com/gzxx-2025/aid-studio
34. `Agent-Field/reels-af` — https://github.com/Agent-Field/reels-af

### Vendor documentation and product pages
35. Google Photos — Highlight videos / Highlights help: https://support.google.com/photos/answer/6128862
36. Google — 2025 Recap announcement: https://blog.google
37. Apple Intelligence, June 2026 — everyday experiences: https://www.apple.com/newsroom/2026/06/apple-intelligence-brings-powerful-ai-capabilities-into-everyday-experiences
38. DJI LightCut discontinuation: https://www.dji.com/lightcut
39. No Film School — Eddie AI / Premiere Gal review: https://nofilmschool.com/eddie-ai-premiere-gal-review
40. PySceneDetect documentation: https://www.scenedetect.com
41. OpenTimelineIO: https://opentimelineio.readthedocs.io

### Practitioner discussion (opinion only, not evidence)
42. r/aitubers — "Most AI video tools are solving the wrong problem" (the *control* argument)
43. r/VideoProfessionals — "AI videos look cheap and unprofessional" (symptom list, corroborating §3)