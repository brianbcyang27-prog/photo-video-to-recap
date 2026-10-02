# How people actually use AI to edit video — and what it means here

**Written:** 2 October 2026
**For:** the Trip Recap Builder maintainer
**Structure:** Layer 1 landscape survey → Layer 2 tool specifics (delegated to four
source reports) → Layer 3 prioritised gap analysis against this pipeline.

## The four source reports

| Report | Lines | Covers |
|---|---|---|
| [`broadcast-quality-audio.md`](broadcast-quality-audio.md) | 883 | Loudness standards, voice-enhancement AI, ducking, music fitting |
| [`automated-montage-techniques.md`](automated-montage-techniques.md) | 996 | Auto-highlight products, summarisation literature, open-source tools, failure taxonomy |
| [`nle-ai-in-resolve-and-premiere.md`](nle-ai-in-resolve-and-premiere.md) | 752 | Resolve 20→21.1 + Premiere 25→26.5 AI feature sets, measured quality, human/AI boundary |
| [`filter-quality-and-colour-harmonisation.md`](filter-quality-and-colour-harmonisation.md) | 1105 | ffmpeg filter quality, encode ladders, perceptual metrics, colour harmonisation |

Plus the pre-existing [`agentic-video-production.md`](agentic-video-production.md), which
asked a narrower question: should this pipeline adopt an agentic architecture? Its answer
(keep the render deterministic, agentise the judgement) held up under all four later
investigations.

**Evidence convention used throughout.** `[verified]` = read from a primary source — a
standards document, vendor manual, paper full text, or actual repository source.
`[reported]` = vendor marketing, press, or a single practitioner. `[rumoured]` = forum or
listicle folklore, no citable source. Where a commonly cited number turned out not to
exist, that is stated rather than substituted.

---

# Layer 1 — The landscape

## 1.1 Four architectures, and they are not variants of each other

Sorting every shipped tool by *where the AI sits relative to the timeline* gives four
groups. The groups differ in a way that matters more than their feature lists suggest.

### A. AI-enabled — assistance inside an existing timeline

DaVinci Resolve and Premiere Pro. You keep the timeline and the judgement; the model
accelerates specific tasks. This is the "copilot" model, and it dominates professional
practice because the editor stays accountable for the cut.

The Resolve 21 Reference Manual contains **225 occurrences of "Studio Version Only"**
`[verified]`. Practically the whole AI surface — the entire Fairlight AI set, Magic Mask,
Super Scale, UltraNR, the transcription engine, IntelliSearch, IntelliScript — is behind a
**$295 one-time licence** `[verified]`.

### B. AI-led — the transcript or the prompt is the interface

Descript, Opus Clip, autoclip, Kapwing, ChatCut. You edit text or issue instructions, and
the timeline follows.

**These are almost all spoken-word tools.** Opus Clip, autoclip, CapCut, Descript and
Kapwing are at the core transcript- or virality-driven `[verified]`. autoclip's own scope
statement names podcast / interview / lecture → Douyin / TikTok / Reels. *Not one of them
targets a personal media library.*

### C. Programmatic — deterministic code, no model in the render

This pipeline, and `auto-editor` (5,403★, now written in Nim), `PySceneDetect`,
`lossless-cut`. Selection and rendering are arithmetic. The "AI", where present, is
upstream analysis.

The decisive architectural evidence is in FireRed-OpenStoryline's own
`requirements.txt`: its render stack is `ffmpeg-python, moviepy, av, librosa` and its model
stack is `langchain, openai, mcp, sentence-transformers, faiss` `[verified]`. **The pixels
are made by the same class of deterministic renderer this project uses; the model
orchestrates above it.** Every serious surveyed system has this shape.

### D. Generative fill — new pixels from a prompt

Firefly, Veo, Kling, Runway, Luma. Adobe now exposes Veo, Kling, Runway and Luma *inside*
Premiere `[reported]`.

**This group is the weakest on evidence, and the vendors know it.** Adobe's own
"Generative Extend known issues" page is longer than its feature page, and the extended
clip cannot be transcribed, does not appear in Media Intelligence search, always uses
zero-based timecode, loses markers and logging metadata, cannot be collected by Project
Manager, and cannot round-trip through XML/AAF/EDL — for which Adobe writes
**"No workarounds exist for these limitations."** `[verified]` That single sentence is the
most useful thing Adobe has published about generative video in an NLE.

Not one generative feature in Resolve or Premiere has a published quality benchmark
`[verified]`. Resolve has no generative video feature at all.

## 1.2 The market gap, stated plainly

- Google Photos **Highlight videos** is the only consumer product built for a personal
  library. You select people, places, themes and dates. **No EDL or XML export.** `[verified]`
- Premiere **Eddie AI (Scripted Mode)** assembles a rough cut from a script you supply, and
  lands in a real timeline you can finish. Wrong domain assumption — it assumes you bring
  the narrative `[reported]`.
- **DJI LightCut**, the closest thing to a consumer "import your footage, get a cut"
  product, was **discontinued in September 2026** `[verified]`.
- On-device semantic video understanding is now a commodity — Apple ships generated video
  *descriptions* and semantic search over camera clips as of June 2026 `[reported]`. So the
  capability exists; it is not aimed here.

**Conclusion:** the consumer AI-video category serves social clips cut from spoken word.
Turning a personal photo library into a finished film is unoccupied. That is the space
this pipeline is in, and the absence of competition is not evidence that the problem is
easy — it is more likely evidence that the quality bar is high and the market is small.

## 1.3 The five findings that generalise

These held up across all four source reports. They are the reason to read all four.

### 1. No offline metric can tell you whether the output is watchable

Otani et al. (CVPR 2019) built a randomization test on SumMe and TVSum:

> "randomly generated summaries achieve comparable or better performance to the
> state-of-the-art. In some cases, the random summaries outperform even the reference
> summaries. Moreover, it turns out that **the video segmentation, which is often
> considered as a fixed pre-processing method, has the most significant impact on the
> performance measure.**" `[verified]`

Their own mechanism note: the knapsack formulation most summarizers use penalises long
segments, so short segments win most selections.

Across fourteen surveyed papers, exactly **one** (Morère et al. 2015) used a real
subjective evaluation with attractiveness *and* informativeness as separate axes.

**Implication:** benchmark F1 deltas are not quality signals. The only "would a human watch
this" evidence in the literature is one paper. This is the strongest argument for rendering
the rejects, A/B-ing by hand, and distrusting any number that claims to measure taste.

### 2. Hardware-conditional output is the failure mode nobody plans for

Premiere's Object Mask **silently falls back to an older detection model on Intel Macs and
Windows with older AMD drivers** `[verified]`. Same project, different masks, different
machine.

Blackmagic moved scripting behind the $295 Studio tier, framing it as: *"The Python API was
being used to hack studio features into the free version."* `[reported]`

Adobe on its own AI Assistant: *"It's not recommended to do client work just yet."*
`[reported]`

**Implication:** this pipeline's offline, machine-independent, deterministic property is
not a limitation to apologise for. Across four reports it is the one thing none of these
products can offer. It should be defended explicitly, not quietly traded away.

### 3. Generative and learned features are hardware-, credit- and jurisdiction-gated

Firefly is metered in generative credits (Standard 2,000 → Premium 50,000/month
`[verified]`), blocked in Russia, Belarus and China, limited for K-12 and some enterprise
CCE v3 licences, and Separate Crosstalk requires biometric opt-in and is unavailable for
files uploaded from Illinois or Texas `[verified]`. Resolve's Motion Debblur costs **up to
60 seconds of compute per second of footage** `[verified]`. Voice Convert requires 8 GB
VRAM to activate at all and takes hours `[verified]`.

**Implication:** if any of this is ever adopted, the report should carry an **operation
cost table** — expected wall-clock per feature per minute of media — and no feature's
output may depend on which machine ran it.

### 4. The folklore correction

The most-repeated audio numbers in this field are wrong or unverifiable:

| Claim | Verdict |
|---|---|
| "YouTube normalises to −14 LUFS" | **No LUFS figure exists anywhere in YouTube's documentation.** What *is* official is that YouTube applies dynamic volume adjustment plus a viewer-toggleable **Stable volume** and **Voice boost** `[verified]` |
| "Netflix is −31 LKFS" | −31 is the **Dolby dialnorm reference**. Netflix's spec is −27 LKFS ±2 LU, dialog-gated, max −2 dBTP `[verified]` |
| "Podcasts are −16/−19 LUFS" | Derived from AES TD1008 (−18 LUFS ±1), not a platform rule `[derived]` |
| "Premiere's frame interpolation" | Measured at **VMAF 34.01** by MSU, roughly half of RIFE's 66.33, and *below* two-line frame averaging `[reported]` — see the caveat in §2.4 |

The only broadcaster-published music level is the BBC's: *"When the final mix is complete
the BBC recommends taking the music down 4db."* `[verified]`

### 5. The semantic gap is real, and the commercial tools can't close it either

This pipeline has no vision model. It computes Laplacian variance, Sobel energy,
exposure, colourfulness, Haar face rectangles, an eye-region brightness proxy
(`eye_region.std() / 42.0`), and a dHash `[verified from source]`. It cannot tell a
cathedral from a car park.

The people-aware literature treats *who* as the signal, not *whether*: Takeuchi &
Sugimoto built a personal-photo-library summariser specifically for this problem (ACM MIR
2006, ACM IVR 2007) `[verified bibliographically]`.

**But note what the commercial tools achieve:** Adobe states Media Intelligence *"supports
English language searches only"* and analyses downsampled stills so it *"won't pick up as
well on small details or fast motion"* `[verified]` — which are precisely travel footage's
failure modes. Resolve's IntelliSearch publishes no recall figure `[verified]`.

**Implication:** this pipeline's GPS clustering, EXIF and clip geometry are already a
*better* retrieval index than either product's, and they are language-independent and
deterministic. If learned retrieval is ever added it must be a **recall booster over** the
deterministic index, never the index itself.

---

# Layer 2 — Tool specifics

Delegated. Each source report carries full feature tables, per-claim confidence markers and
sources. Highlights that bear on this pipeline:

## 2.1 Audio

- **Whole-programme normalisation, not per-segment.** See §3 — this is a live bug.
- **No published dB threshold exists for when enhancement sounds robotic.** The mechanism
  is *musical noise*: spectral subtraction leaves isolated spectral peaks, a time-frequency
  artefact. Every commercial product ships a strength slider whose documented meaning is
  *"more noise removal buys more artefacts and less dialogue"* — iZotope RX 11 and Adobe
  Podcast both say this in their own docs `[verified]`. **The vendor's slider is the
  artefact threshold, because the trade cannot be expressed as a level.**
- **Speech enhancement is the wrong tool here.** Trained on speech-with-clean-reference; on
  real recordings PESQ falls to 1.46/4.5 with a **63.52% SNR reduction** `[verified]`.
  Room tone and wind are "everything else" from the model's point of view. RX ships
  De-wind *separately* from Dialogue Isolate `[verified]` — if the speech model handled
  wind, that module would be redundant. On ambience-only shots, running one is the highest
  artefact-risk action available.
- **Beat tracking is ~80% accurate** (Essentia's AMLt calibration) `[verified]`. So 80% of
  cuts on the beat sounds intentional; 100% sounds mechanical.
- **Do not use Demucs.** 7.52 dB SDR is audible artefact, plus PyTorch weights. The
  guidance is to *generate thinner music* rather than separate thicker music.

## 2.2 Selection and pacing

- **Real film shot lengths:** ASL ~10 s in the 1930s–40s falling to **below 4 s after
  2000**, across 160 hand-segmented films and 5 genres; shot-length sequences approach
  **near-1/f** fluctuation, matching human reaction-time spectra `[verified, PMC3485803]`.
  Short shots also carry proportionally more within-shot motion in modern film
  (r = −.46, p < .0001) — which gives a *principled* coupling between duration and content
  that is not "good = long."
- **Cut on accents, not beats** — percussive onset strength → local normalisation over an
  8-beat window → top-30% percentile → local-max filter → min-separation suppression.
  ~50 lines, readable in `FireRed-OpenStoryline/src/open_storyline/nodes/core_nodes/select_bgm.py`
  `[verified]`.
- **DOVER's disentanglement** (ICCV 2023): human quality opinions are *"universally and
  inevitably affected by both"* technical and aesthetic perspectives, and the two are not
  commensurable `[verified]`. Single blended score is the wrong shape — even before
  semantics.
- **Transitions gate on the *removed* interval, not the kept one.** auto-editor: dissolving
  across a trim shorter than the removed material *"reads as a stutter"* `[verified]`.
- **Nothing in this category exports an editable timeline** — except auto-editor, which
  ships seven formats including OpenTimelineIO `[verified]`.

## 2.3 Professional NLEs

- **Premiere Remix** lands *"within 5 seconds of the target duration, typically within 1
  second"* `[verified]`. **Resolve Music Editor** *"does not use time compression/expansion"*
  and Blackmagic's own advice is to fix timing afterwards with Elastic Wave `[verified]`.
  **This pipeline's synthesised beat grid beats both, and is now citable.**
- **Premiere Enhance Speech outputs only a mono downmix** of a stereo clip `[verified]`.
  Decisive for a recap with ambience beds.
- **Resolve's own recommended Voice Isolation operating point is 70–80** on the Amount
  control — the vendor sanctioning a narrow range before artefacts `[verified]`.
- **Resolve's Ducker defaults: 2.7 dB duck, 15 ms lookahead, 10 ms rise, 150 ms hold,
  750 ms recovery** — and it is documented as *non-compressing* sidechain `[verified]`.
- **MCP is a 2027 integration, not a 2026 dependency.** Resolve 21.1's MCP server is
  Studio-only and two months old; Adobe has no protocol surface, only a beta chat `[verified]`.
  Adobe also publishes an `llms.txt` and a documentation-search MCP — and its MCP server
  shares **no capability** with the REST API `[verified]`.

## 2.4 Programmatic quality

- **THE BUG — see §3.1.** Written as BT.601, read as BT.709, on every still in every render.
- **ffmpeg's quality filters pair frames by timestamp, not order.** With mismatched
  timebases they report a near-lossless encode as **VMAF 67.03 instead of 99.94** `[verified]`.
  Any verifier must normalise both legs. Directly relevant if encoder-fidelity checks are
  ever added.
- `accurate_rnd` and `full_chroma_int` are **bit-exact no-ops** on ffmpeg ≥ 5.x `[verified]`.
- Moving `preset` off `veryfast` costs 2.4× the time for −0.025 VMAF — measured, negative
  `[verified]`.
- **Colour harmonisation is the largest perceptual lever available:** matching chroma
  μ to the library median in LAB a\*/b\* with λ ∈ [0.5, 0.7] measured median pairwise
  ΔE76 **75.5 → 45.9 (39%)** across 35 levels of drift, beating σ-matching `[verified]`.
  Never touch L\*; never match σ.

### A caveat on the one external benchmark

MSU's Video Frame Interpolation Benchmark is the only place a commercial NLE is measured
against open weights. Its participant list does include `Adobe Premiere Pro`,
`Frame Averaging` and `RIFE` `[verified]`, and the page footer reads `04 Oct 2022`
`[verified]` — so it is a ~2022 build, not 26.5, and Resolve is not in the table.

**The specific scores could not be independently confirmed:** the leaderboard is
JavaScript-rendered and returned empty on fetch. The figures (Premiere VMAF 34.01 /
PSNR 21.93 / SSIM 0.78 vs RIFE 66.33 / 27.15 / 0.914) are internally coherent but should be
treated as `[reported]`. The *structure* of the finding — that NLE optical flow underperforms
open weights — is sound regardless.

---

# Layer 3 — Prioritised gap analysis

Merged and de-duplicated from all four source reports, ranked by expected perceived quality
gain per unit of implementation cost for **this** pipeline: deterministic, offline, macOS,
numpy/Pillow/soundfile only, 154 pure-logic tests plus a 21-check rendered-file verifier.

## Tier 0 — bugs. Not improvements. Fix regardless.

### 3.1 The colour matrix is the largest defect in the pipeline — and the smallest fix

**Verified twice, independently:**

```
ffprobe on a real rendered 4K segment:
  color_range=unknown  color_space=unknown
  color_transfer=unknown  color_primaries=unknown
```

ffmpeg's default RGB→YUV matrix is bit-for-bit BT.601 — the default conversion's SHA-256 is
identical to explicit `out_color_matrix=bt601` and different from `bt709`.

So the pipeline renders every still with BT.601 luma coefficients, ships it untagged, and
every player resolves "unknown" as BT.709 for HD. **Written as 601, read as 709.**

| | per-photo median ΔE76 |
|---|---|
| Pipeline as shipped | **18.98** (worst 105.69) |
| Fixed | **2.51** (worst 4.01) |
| Regressions | **0 of 40** |

Against a just-noticeable-difference of ~2.3 ΔE76. For 1080p the correct matrix is BT.709.
The residual 2.51 is 8-bit 4:2:0 chroma subsampling — the floor, not a defect.

**Fix:** add `:out_color_matrix=bt709` to the `scale` inside each still chain, plus
`-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv` on output.
Roughly ten lines.

**Precise scope — this is a still-path defect, not a global one.** The two paths fail
differently, and the distinction matters:

- **Stills (`render.py:351`)** — PIL hands ffmpeg 24-bit RGB, so the RGB→YUV conversion
  happens *inside the encode* and the wrong matrix is **baked into the pixels**. This is
  the measured ΔE76 18.98 case, and it matches the report's measurement scope exactly
  (40 library photos).
- **Video (`render.py:388-404`)** — input is already YUV, so `scale` resamples without a
  matrix conversion. Nothing is baked. The source clips are untagged too
  (`color_space=unknown` on the `test_media` probes), but modern phone video is BT.709 and
  a player's HD guess matches it, so this path is usually benign. It still needs the tags
  so correctness stops depending on the reader guessing.

One extra reason to tag every segment uniformly: the final mux is `-c copy` concat
(`render.py:786`), so partially-tagged segments produce a file with internally
inconsistent colour signalling. Uniform tags make that impossible rather than unlikely.

Add a verifier check that fails on `color_space=unknown`, so it cannot regress.
The verifier already has the calibration discipline for this — break the control, confirm
it fails, keep the fix.

### 3.2 Per-segment `loudnorm` is lifting your room tone by ~19 dB

`pipeline/render.py:537` runs `loudnorm=I=-18:TP=-1.5:LRA=11` on **each segment** before
concatenation. Three consequences:

1. With no `measured_*` inputs this is **single-pass dynamic mode**, which re-gains over
   time — on a 3-second ambience clip that is textbook pumping `[verified, ffmpeg docs]`.
2. The repo measures Live Photo room sound *"around −37 dB mean"*. Normalising ambience to
   −18 LUFS is a **~+19 dB lift** on a noise floor. Amplifying and gating a room's noise
   floor is the classic route to pumping hiss.
3. It destroys shot-to-shot loudness relationships. A quiet market and a loud market both
   arrive at −18.

**Fix:** normalise the **assembled programme**, once, two-pass, feeding `measured_I` /
`measured_LRA` / `measured_TP` / `measured_thresh` back so `loudnorm` takes its linear path.

### 3.3 `alimiter` is sample-peak limiting, not true-peak

`render.py:658,664` use `alimiter=limit=0.891` (−1.001 dBFS *sample* peak) and
`limit=0.97`. `alimiter` does lookahead limiting with no oversampling, so **neither branch
guarantees the −1 dBTP that EBU R128, ATSC A/85, AES TD1008 and Netflix all specify** —
and AAC encoding creates inter-sample peaks above the sample peak.

`level=0` correctly disables auto-level. Keep that. **Fix:** measure with
`ebur128=peak=true` *after* encode, and treat the number as measured rather than assumed.

### 3.4 Ducking: fixed threshold, no knee, and a pumping risk

`render.py:655`: `sidechaincompress=threshold=0.035:ratio=6:attack=18:release={…}:makeup=1`.

- `threshold=0.035` is a **fixed constant** — so duck depth varies with however hot the TTS
  ran this render. Derive it from the narration stem's measured short-term loudness.
- `knee` is **unset → 2.82843**, a near-hard knee on a music bed at 6:1 is an audible step.
  Set 4–8.
- `ratio=6`, `attack=18` are in the recommended bands. Keep.

**The upgrade is architectural and free:** because the narration is *synthesised*, the
phrase boundaries are already known. Emit the sidechain control signal directly from the
phrase list instead of detecting speech in real time. That makes pumping **structurally
impossible** rather than tuned away, and removes the fragile threshold constant.

**Cross-report tension, resolved.** The audio report recommends 6–12 dB of reduction;
Resolve's own Ducker defaults to **2.7 dB** with "2.0–5.0 works best" `[verified]`; the BBC
says take the music down 4 dB. These do not actually conflict: Resolve's Ducker is
documented as **non-compressing** sidechain — a fixed level shift, not a ratio. A 6:1
compressor at 12 dB over threshold yields 10 dB. **Neither figure transfers to a
`sidechaincompress` chain without knowing which mechanism is being emulated.** Note also
that Resolve's 750 ms recovery sits above the audio report's 300–600 ms band; that is the
parameter to test.

### 3.5 `setrange=limited` is currently a bit-exact no-op — keep it anyway

The `format=yuv420p,setrange=limited` at the end of every still chain produces a
**bit-identical** output with and without it `[verified, SHA-256]`.

**Do not remove it.** `render.py:886-894` records that the PNG intermediate exists partly
*because* a JPEG has no colour-range concept, so ffmpeg decodes it full-range and the
segment comes out `yuvj420p` — which the concat demuxer then propagates to the whole film.
`setrange=limited` is the guard against exactly that, and with `-c copy` concat
(`render.py:786`) a single range mismatch poisons the finished file.

The point is narrower than "the filter is useless": it is load-bearing in *intent* and
inert in *effect* on the current PNG path. The missing half is that nothing declares the
range on the **output**, which is why §3.1's `-color_range tv` tag matters — the container
should not depend on a reader inferring it.

## Tier 1 — instruments. Build before changing anything else.

The Otani finding says you cannot measure whether an edit is good. That makes measurement
infrastructure the precondition for every other change.

| Change | Cost | Why first |
|---|---|---|
| **Encoder-fidelity checks in the verifier** — `libvmaf` + `ssim` + `psnr` against a `-qp 0` reference, with `settb=AVTB,setpts=N/(FPS*TB)` on **both** legs | ~60 lines | Highest leverage per line. Without it, a near-lossless encode reports VMAF 67.03 instead of 99.94 `[verified]`. **Do not skip the normalisation** — the metrics pair frames by timestamp |
| **Verifier check that fails on `color_space=unknown`** | ~10 lines | Locks in 3.1 against regression, using the existing calibrated-control discipline |
| **Render the rejects** — a contact sheet of every rejected candidate with its score components and the reason it lost | ~60 lines | `--dry-run` exists; its inverse does not. This is what makes any pacing change A/B-able honestly |
| **Deterministic frame/index comparison discipline** | design rule | Never let a measurement pass or fail on a frame index it did not compute itself. `verify_output.py` already decodes rawvideo into numpy for orientation and detail checks — apply the same standard to encoder fidelity |

## Tier 2 — high gain, low cost, no new dependencies

| Change | Cost | Gain |
|---|---|---|
| **Tag BT.709** (3.1) | ~10 lines | 7.6× ΔE76 improvement |
| **Whole-programme two-pass `loudnorm`** (3.2) | ~30 lines | Removes a +19 dB noise-floor lift |
| **Accent-beat preference** — percussive onset strength → local norm → top-30% percentile → local-max → min-separation | ~50 lines | Biggest *musical* gain available. **No new dependency:** `_onset_envelope` already exists |
| **Cut handles** — ±2–3 frames around each beat, cut point still beat-locked | trivial | The most-replicated trick in automated editing. Zero-handle cuts read as jump cuts |
| **Edge fades on picture** — you fade audio (`render.py:634`) but not video | 4 lines | A piece starting and stopping at full brightness has no beginning or end |
| **Per-source-file primary/overflow tiering** — longest excerpt per source first, others as overflow | ~25 lines | Fixes a named cheapness: one 40 s panorama sliced into six excerpts reads as the same footage, and `max_per_minute` cannot see it |
| **Encoder probe with permanent disable** | ~20 lines | The difference between "the tool crashed" and "the tool produced something" |
| **Upload profile** — `+faststart`, YouTube GOP, BT.709 tags, 384k stereo AAC | ~15 lines | Four documented deviations from YouTube's official spec |
| **Title-card pass** — verify the font loads, assert local contrast behind text, step size down to fit | ~60 lines | The verifier's legibility test counts near-white pixels — **a white title on a bright sky passes it** |

## Tier 3 — the perceptual changes. Ship behind flags, A/B by hand.

These visibly change every render and cannot be validated by any metric in existence. Use
Tier 1's contact sheet and side-by-side renders.

| Change | Cost | Why |
|---|---|---|
| **Chroma harmonisation in LAB a\*/b\*** — match μ to the library or per-chapter median, λ ∈ [0.5, 0.7], clip gain to [0.85, 1.18], never touch L\*, never match σ | ~120 lines | Largest perceptual lever. Measured median pairwise ΔE76 **75.5 → 45.9** across 35 levels of drift `[verified]` |
| **Decouple duration from score, 1/f-shaped** — sample `d ~ d^-α`, clamp, quantise to whole beats | ~15 lines | ASL is monotone in score today, which announces the ranking instead of telling a story. Real films are near-1/f and post-2000 ASL is below 4 s. **Keep `_base_weights` for selection, remove it from the duration path** |
| **Re-weight `min_sharpness`** — normalise by local contrast, restrict to face boxes when faces exist, use a percentile of `\|Laplacian\|` rather than its variance | ~30 lines | `w_sharpness = 0.30` is the largest-weighted criterion and the one with the most documented failure modes: depth of field, texture density, watermarks, noise all move it independently of focus |
| **Rolling-mean scene detection** instead of a fixed `scene_threshold` | ~60 lines | Handheld motion is phone footage's dominant failure mode and produces false cuts |
| **Library-free damped overshoot easing** on Ken Burns — `1 − e^(−6p)·cos(2.5πp)`, clamped | ~10 lines | Every move is currently linear. A linear Ken Burns reads as a slideshow; a decelerating one reads as a camera move. The missing half of "paced camera moves" |
| **Aesthetic/technical score split** — technical `{sharpness, exposure, clipped, black, contrast}`, aesthetic `{faces, color, motion}`, separately normalised | ~40 lines | They are not commensurable. Behind a flag; recalibrates every score |

## Tier 4 — product shape

| Change | Cost | Why |
|---|---|---|
| **Emit OpenTimelineIO or FCPXML** alongside the MP4 | ~80 lines | Converts the tool's failure mode from "unusable" to "almost right." Your timeline is already fully determined: source path, source range, one entry per join. OTIO is the better first choice (JSON, library-agnostic, transitions map to `SMPTE_Dissolve`) |
| **A `bpm` + `loop` + `seed` contract on the music spec** | small | Adobe's Generate Music already lets you type a BPM and set Loop `[reported]`. Adopt the *contract* so a future offline music model drops straight in — without taking the credit-metered cloud service |
| **Accept an EDL on input**, not just emit one | — | Today `trip_recap.edl.json` is output-only; `make_video.py` has no `--edl` input path, so "hand-edit the EDL" does not re-render. This is what OTIO would fix |
| **Optional licensed music catalog** | — | See §4 |

## Explicitly not recommended

| Thing | Why |
|---|---|
| Any NLE generative feature | Metered, cloud, geo-blocked, and Adobe's own page says *"No workarounds exist for these limitations"* |
| NLE frame interpolation | Open weights beat it on every metric, and it breaks frame-exact beat arithmetic |
| AI upscaling by default | No public benchmark exists for *any* commercial upscaler. Blackmagic warns High Sharpness amplifies grain. If added: hero shots only, keep the original, log the decision |
| Learned denoise by default | You cannot A/B a slider you cannot scrub. Gate on a measured noise metric with a hard threshold and log it |
| Speech enhancement on ambience-only segments | Maximum suppression of a signal with nothing to preserve. Highest-artefact action available, and this project has many ambience-only shots |
| Demucs / source separation | 7.5 dB SDR is audible. Generate thinner music instead |
| Tracking benchmark F1 as quality | Otani et al. |
| `minterpolate` for slow motion | Invents frames; breaks the cumulative-beat arithmetic that the whole timing model rests on |
| `deshake` / `vidstabtransform` | Phone footage is already stabilised; double stabilisation looks like float |
| A CLIP or "aesthetic score" | No validated metric exists for "looks professionally edited." Would be false precision |
| Raising `ZOOM_SS_CAP` above 2 | Measured: 4× adds nothing on real photos (5.700 vs 5.688 edge energy) at twice the cost |
| Moving `preset` off `veryfast` | Measured: 2.4× time for −0.025 VMAF |
| `accurate_rnd` / `full_chroma_int` | Bit-exact no-ops on ffmpeg ≥ 5.x |

---

# 4. The semantic question, and the music question

## 4.1 Understanding what the footage is about

Confirmed as the largest *unaddressed* gap, but not the largest total gap — Tier 0's colour
and audio bugs are larger and free.

The right architecture is **semantics as an input feature, not a decision-maker.** A
captioning pass emits text per shot; the existing deterministic pipeline consumes that text
as one more column in scoring. The renderer never sees a token, so all 170 tests stay valid.
This is the shape FireRed uses and the shape the pre-existing `agentic-video-production.md`
already recommended.

Constraints learned from FireRed's implementation `[verified]`:
- Call at `temperature=0.1, top_p=0.9`; on parse failure **fail open** — `"Failed to parse
  model output, using all clips."` A montage that drops material because a parse failed is
  worse than a montage with one bad shot.
- **Inject each clip's duration into the caption block before asking.** Duration is the
  first thing a montage editor needs and the last thing a captioner volunteers.
- Keep the deterministic score as a floor and a hard retention floor so the model refines
  but never guts the cut.

If LLMVS's local-then-global two-stage scoring is ever adopted — captions scored in local
context, then refined globally across all captions `[verified]` — it slots into this seam
without changing the architecture.

## 4.2 Supplying your own pop music

**The beat-following already exists and is well made.** `--music-track FILE`, or drop files
in `music/`. The path runs spectral-flux onset detection → autocorrelation tempo with a
log-normal prior at 120 BPM and an octave check → an **Ellis-style dynamic-programming beat
tracker**. `music.py:621` deliberately does *not* quantise a supplied track's tempo — it
keeps the real tempo and snaps the timeline to the real beats, because quantising it
"anyone can hear." Better than most tools manage.

**Sourcing: not YouTube.** Downloading from YouTube violates its Terms of Service, and this
repo is MIT-licensed and public — so shipping a downloader means publishing a tool whose
documented workflow is "download copyrighted pop music." That is a distribution decision,
not a technicality.

The licensed catalog APIs are better on the merits anyway:

| | Epidemic Sound | Artlist Enterprise |
|---|---|---|
| BPM range | `bpmMin`/`bpmMax` | `bpmMin`/`bpmMax` |
| Energy | `mood` (there is an `energetic` mood id) | `categoryIds` → Mood |
| Video theme | — | **`categoryIds` → Video Theme, incl. "Travel"** |
| Natural language | `/v0/tracks/search` takes free text — *"high energy track for a workout"* | `query` |
| Vocals | `vocalType` (`NONE` = instrumental) | `vocalType` |
| Licensed file | stream endpoint | Download API mints signed MP3/WAV URLs |
| Access | API key; free tier exists (`tierOption: "FREE"` in responses) | Account-manager gated |

Free alternatives: **YouTube's own Audio Library** (free, licensed for video — the
legitimate YouTube-sourced option), Jamendo, Free Music Archive, Kevin MacLeod.

Epidemic Sound documents the exact integration pattern wanted here: let an LLM write the
search query from the video's mood, then call the search endpoint. **`--mood` becomes a
search term instead of a synth parameter.**

**What "more energetic" actually requires,** in order of impact:

1. **Accent beats, not beats.** Tier 2. No new dependency.
2. **Decouple duration from score.** Tier 3. `_base_weights` currently makes duration
   monotone in quality, which is the opposite of punchy.
3. **Drive sections off the track's measured RMS envelope** (`dynamic_range_db = p95 − p10`)
   rather than the synth's assumed shape.

**The honest trade:** synthesised music gives an exact grid — the pipeline's main musical
advantage. A real pop track means ~80% beat accuracy and real tempo drift. That trade is
worth it *for energy*, and 80% is sufficient, because a grid that hits every beat exactly
sounds mechanical rather than good.

---

# 5. Recommended order

1. **§3.1 BT.709 tagging** + a verifier check that fails on `color_space=unknown`.
2. **§3.2 whole-programme two-pass loudnorm.** Removes a +19 dB noise-floor lift.
3. **§3.3/§3.4 true-peak measurement, ducking knee, gate-driven sidechain.**
4. **Tier 1 instruments** — encoder fidelity with both legs normalised, and render the
   rejects. Nothing after this point is safe to change without them.
5. **Tier 2** in listed order — accent beats, cut handles, edge fades, per-source tiering.
6. **Tier 3** behind flags, A/B by hand against the contact sheet.
7. **Tier 4** — OTIO export, then the music catalog decision.
8. **Only then** semantics, as an input feature with the deterministic score as its floor.

The governing principle, unchanged from the pre-existing note and now supported by all four
investigations: **an agent belongs above a verified deterministic pipeline, improving the
judgement calls — never inside the render, and never inside a measurement.**

---

## What remains unverified

Stated rather than smoothed over, because the standard here is that a test never shown to
fail is not evidence:

- MSU's frame-interpolation **scores** (participant list and page date verified; the
  JavaScript-rendered table could not be read on fetch).
- The alleged CVPR 2017 "Social-Aware Model for Selecting Photos," widely cited as the basis
  for Google Photos' 2017 Memories — **not found** in the CVF 2017 index, arXiv, CrossRef or
  OpenAlex. Not used as evidence.
- DOVER's SRCC figures — the ICCV 2023 abstract states "state-of-the-art performance" and
  contains **no numbers**. Do not quote SRCC from a blog.
- No public benchmark exists for Resolve Super Scale vs Topaz, Resolve UltraNR vs Neat
  Video, Voice Isolation vs Enhance Speech, Auto Reframe accuracy, or transcription WER.
- Apple Music's real normalisation target is `[reported]` at ≈−16; Apple publishes nothing.
- Exact Resolve/Premiere ducking and Voice Isolation numbers come from vendor manuals read
  through a browser engine, because Adobe's help pages block direct fetches.
