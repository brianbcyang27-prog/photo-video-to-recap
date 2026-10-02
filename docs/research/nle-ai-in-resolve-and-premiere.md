# What professional editors actually use AI for in DaVinci Resolve and Premiere Pro

**A research report for the Trip Recap Builder maintainer.**

**Date:** 2 October 2026
**Scope:** the shipped AI feature sets of DaVinci Resolve 20 → 21.1 and Adobe Premiere Pro
25.x → 26.5.2, their quality evidence, what practitioners report about them, and the
human/AI boundary (including MCP).
**Reader's starting point:** a deterministic Python + ffmpeg pipeline that reads EXIF/GPS,
scores frames, clusters GPS into chapters, knows its beat grid *exactly*, cuts on the beat,
applies Ken Burns, mixes to −14 LUFS / −1 dBTP, and emits a human-readable decision report.
**Therefore this report is written as a delta.** It answers one question: *is there anything
in either NLE's AI stack that a deterministic pipeline should adopt, import, or call out to?*

**Method.** Primary sources only where possible: the Resolve 21 Reference Manual (4,444 pp),
the Resolve 20 and Resolve 21 New Features Guides, the Resolve 21.1 release notes, and the
per-feature Adobe Help pages for Premiere (fetched 2 Oct 2026, each carrying a "Last updated"
date). Practitioner evidence is quoted with a link and marked at its true confidence level.

---

## 0. Confidence markers

| Marker | Meaning |
|---|---|
| **[verified]** | Read from a primary source: vendor manual/release note/help page, or an independent research benchmark. |
| **[reported]** | Vendor marketing claim, press release, review, or a single named practitioner. |
| **[rumoured]** | Forum consensus, comment-thread colour, SEO listicle, or video-essay assertion with no citable source. Flagged inline as such. |

Note on sources: Blackmagic's own New Features Guides carry no `[verified]` caveats because
they *are* the primary source — but they are also vendor copy, so any performance adjective
("dramatically better results", "incredible results") is **[reported] by definition** even
though the existence of the feature is **[verified]**.

### Things I could not verify, stated plainly

- **No public benchmark exists** for Resolve Super Scale vs Topaz Video AI, Resolve UltraNR vs
  Neat Video, Resolve Voice Isolation vs Premiere Enhance Speech vs Adobe Podcast, Premiere
  Auto Reframe accuracy, CineFocus depth-map accuracy, Magic Mask v2 vs Premiere Object Mask,
  or AI Audio Assistant vs a human mix. I searched; the MSU benchmark collection (the most
  likely home for such numbers) publishes a Video Frame Interpolation Benchmark that
  *does* include Adobe Premiere Pro, but its Video Super-Resolution Benchmark contains only
  academic models (PSRT, RVRT, VRT, BasicVSR++ …) — **no commercial upscalers at all**.
  Everything in §5 marked "no public benchmark found" means I could not find one, not that
  one does not exist. [verified as absence-of-finding]
- **Word-error-rate for NLE transcription is not published** by either vendor. Resolve and
  Premiere both ship a "transcription engine"/"Speech to Text" with no accuracy figure, no
  language-count claim tied to a number, and no benchmark. [verified]
- **No MOS/WER/quality score is published for Speech Generator or Voice Convert.** [verified]
- **Reddit and the Blackmagic forum are not directly readable from this environment** (Reddit
  returns a bot challenge on `www` and a 496-char stub on `old`; `forum.blackmagicdesign.com`
  returns HTTP 202 with 0 bytes). All practitioner quotes below were captured from indexed
  search snippets of those threads, which preserve the quoted text and the thread URL but not
  the surrounding argument. I mark these **[rumoured]** where the snippet is the only
  evidence, even when the sentence itself is unambiguous. [verified limitation]
- **`helpx.adobe.com` blocks unauthenticated fetches** (403 Akamai; Cloudflare challenge even
  through a text proxy). I read the pages through a real browser engine instead, which is why
  per-feature "Last updated" dates are trustworthy.

---

## 1. Executive summary — the shape of the market

Three facts dominate everything else, and all three come from the vendors themselves.

**1. The two products have completely different AI business models, and that difference
predicts every feature's maturity.**

DaVinci Resolve Studio is a **one-time $295 licence** with no metering, no credits, no cloud
requirement, and no per-use consent dialog. **[verified]** Every AI feature except Face
Refinement and Object Removal runs locally on the Neural Engine; Blackmagic's own What's New
page states the DaVinci AI Neural Engine "is fully supported in Apple M series and Snapdragon
X Elite". **[verified]** Blackmagic has also claimed the Snapdragon X Elite boosts Neural Engine
AI performance "by up to 4.7× on Windows computers". **[reported]** Because an editor pays once
and can run a feature a thousand times, they will happily run a slow, imperfect tool and correct
it by hand.

Adobe's generative features are **cloud services metered in generative credits**, gated behind
a sign-in, and blocked in some jurisdictions. Adobe's own Generative Extend page states it is
"unavailable to some users" in Russia, Belarus and China; that "users who are part of K-12 and
educational organizations may experience limited or no access"; and that "some enterprise
users, particularly those with a CCE v3 licence, don't have Firefly services enabled".
**[verified]** Consent is separate: Separate Crosstalk "may involve creating and processing
biometric information, including … voice embeddings", requires explicit opt-in, and is
"isn't available on files uploaded from Illinois or Texas due to legal restrictions".
**[verified]** That is a product-shaped tax on exactly the small-creator use case.

**2. Almost everything genuinely useful is old, local, non-generative, and unglamorous.**
The Resolve features practitioners actually reach for are transcription, beat detection,
Magic Mask, Voice Isolation, Super Scale and UltraNR — all of which are analysis or
restoration, not generation. Premiere's equivalents are Media Intelligence, Auto Reframe,
Object Mask, Enhance Speech, Auto Color and Remix. **Not one generative feature in either
product has a published quality benchmark.**

**3. Adobe's own documentation undercuts its generative features harder than any critic
could.** Premiere's "Generative Extend known issues" page is longer than its feature page.
The extended clip "cannot be transcribed using Speech to Text", "does not appear in Media
Intelligence search results", "always use[s] zero-based timecode", loses markers, loses user
logging metadata, cannot be collected by Project Manager, cannot go to Audition, and cannot
carry modified Audio Channels / Color / Interpret Footage / Timecode / VR Properties — for
which Adobe writes **"No workarounds exist for these limitations."** **[verified]** That last
sentence is the most useful thing Adobe has said about generative video in an NLE.

---

## 2. DaVinci Resolve 20 → 21.1: the AI feature set

Resolve 20 introduced the bulk of the audio AI; Resolve 21 (New Features Guide dated April
2026) added the visual AI and the Photo page. Resolve 21.1 (September 2026) added the MCP
server and moved scripting behind Studio.

### 2.1 Tier gating, stated exactly

The Resolve 21 Reference Manual contains **225 occurrences of the string "Studio Version
Only"** in its TOC and chapter headings. **[verified]** Practically all of the AI surface is
Studio-gated, including the entire Fairlight AI set, the entire Resolve FX AI set, the colour
AI set, Magic Mask, Super Scale, UltraNR, the transcription engine, IntelliSearch, IntelliScript
and SmartSwitch.

Resolve Studio 21 is listed at **$295** on Blackmagic's product page. **[verified]** Blackmagic
adds the Neural Engine as the Studio differentiator: Studio "adds DaVinci Neural Engine for
automatic AI region tracking, stereoscopic tools, more Resolve FX filters, more Fairlight FX
audio plugins and advanced HDR grading". **[verified]**

### 2.2 Audio / Fairlight (inherited from Resolve 20)

| Feature | What it actually does | Gating | Vendor-stated limits |
|---|---|---|---|
| **Voice Isolation** | AI model "trained for any type of human voice, male or female, young or old"; removes background noise, music-under-dialogue, HVAC, jackhammers | Studio | Mono/stereo/dual-mono only; **"isn't supported on tracks with more than two channels"**; real-time but **not** on live audio input. Manual p1146. **[verified]** |
| **Dialogue Leveler** | Detects dialogue, rides loud areas down, lifts soft ones, drops non-dialogue background "without the typical 'pumping'" | Studio | Same channel limits as Voice Isolation. Manual p1147. **[verified]** |
| **Dialogue Separator** | Independent level control over dialogue / background / reverberant ambience | Studio | **"currently a mono-only plugin"** — stereo must be split to dual-mono and processed twice. Manual p1150 / p4093. **[verified]** |
| **Music Remixer** | Stem separation into Vocals / Drums / Bass / Guitar / Other via Neural Engine | Studio | 5 fixed stems; no per-stem timing. Manual p1148. **[verified]** |
| **Music Editor** | Retimes music to a target duration | Studio | **"works effectively with beat-driven music (pop, dance, etc.). It is not intended for free form, ambient or non-beat driven material"** and **"does not use time compression/expansion or alter the pitch"** — it repeats/removes sections. Returns **four** candidate edits. Manual p1148–1149. **[verified]** |
| **Beat Detector** | Marks beats on a timeline audio clip; snap to them | Studio | **"will only work effectively with beat-driven music, and with 4/4 or 3/4 time signatures at this time."** Manual p1153. **[verified]** |
| **Dialogue Matcher** | Capture a tone profile from a reference clip, apply to another (level/EQ/reverb) | Studio | Fully automatic; requires a captured profile first. Manual p1153. **[verified]** |
| **Audio Assistant** | Categorises + colour-codes tracks, auto-mixes, applies Voice Isolation / De-Ess / ducking, writes automation, adds mastering plugins to hit a chosen **Delivery Standard** loudness | Studio | Requires correct Track Categories first or results are wrong; fully undoable. Manual p1154 / p4045. **[verified]** |
| **Ducker** | Non-compressing sidechain ducking from up to multiple source tracks | Studio | Documented defaults: 2.7 dB duck (2.0–5.0 "works the best"), 15 ms lookahead, 10 ms rise, 150 ms hold, 750 ms recovery. Manual p1150–1152. **[verified]** |
| **Voice Convert** | Drives one voice model with another recording's performance; keeps sync, inflection, pitch variation, emotion | Studio | **"requires your system to have a minimum of 8GB of video ram before it will activate"**; needs ~10 min of clean source; **Better mode "takes about 3x the time as faster, and 10 minutes or so of material can take a few hours"**. Manual p1160–1163. **[verified]** |
| **Speech Generator** | Text → spoken-word clip; 4 built-in licensed voices or a custom voice | Studio | **"The recording only needs to be 10-20 seconds long. In fact, longer recordings are truncated to the first 20 seconds"**; best on short paragraphs — "if the text you input is too long, the generated voice can start to drift into strange territory after a while". R21 Guide p43 / manual p1164. **[verified]** |
| **AI IntelliCut** | *(Resolve 20 name)* — a Fairlight **menu grouping**, not a model: Remove Silence, Checkerboard to New Tracks, Create ADR Cues | Studio | Resolve 21's manual no longer uses the "IntelliCut" label; the three tools survive individually (Create ADR Cues p3919, Remove Silence p3945, Checkerboard p3946). R20 Guide p90. **[verified]** |
| **AI Animated Subtitles** | Word-level animated caption styling | Studio | Manual p1281. **[verified]** |

Resolve's own recommended Voice Isolation operating point is the single most quotable number
in the audio stack: **"Values between 70 and 80 work well for natural results while strongly
isolating the source"** (Amount control, where 50 is the roughly-equal mix). Manual p1146.
**[verified]** This is the vendor sanctioning a *narrow* range before artefacts appear.

### 2.3 Visual / colour / effects

| Feature | What it does | Gating | Vendor-stated limits |
|---|---|---|---|
| **Super Scale** | 2×/3×/4× and 2–4× **Enhanced** upscaling, plus Sharpness and Noise Reduction controls. "actually increases the source resolution of the clip being processed" | Studio | **"processor-intensive … turning this on will likely prevent real-time playback"**; must bake via Optimized Media or cache; Sharpness High **"also sharpens grain and noise in the image to an undesirable extent at the default settings"**. Manual p296–297. **[verified]** |
| **UltraNR** | AI spatial NR that "provides intelligently targeted noise reduction based on the machine learning of real world video noise patterns, rather than relying on a specific mathematical formula" | Studio | Analyze a flat patch; **"An NR Radius of Medium should provide suitable quality for most images"**; Small→fast, Large→slow and better. Manual p3585–3586. **[verified]** |
| **Magic Mask v2** | Underlying AI "entirely redesigned"; no separate Person/Object modes; clicks replace strokes; adds a per-frame paint tool | Studio | Paint strokes "won't track with the mask" — fine for a few frames, not for a fix across a shot. Manual p3339–3341. **[verified]** |
| **CineFocus** (new in 21) | Virtual lens: click a point in the viewer to focus, AI Depth Map drives defocus; shaped/anamorphic bokeh, chromatic + spherical aberration, curved lens, **Remove and Replace Grain**, Automatic Edge Handling | Studio | Depth-map blur artefact repair requires manual work: "Blur: … can help if it contains jagged edges … but **can cause artifacts at image object edges. This option is only recommended for repair of a difficult depth map.**" Manual p3599–3605. **[verified]** |
| **Motion DeBlur** (new in 21) | Removes slow-shutter motion blur; renders new media into the Media Pool | Studio | **"a very resource intensive operation that can take up to 60 seconds per second of footage to analyze"**; "Avoid very fast camera movement"; better with the subject near frame centre; optional extra GPU memory for quality. R21 Guide p57–59. **[verified]** |
| **UltraSharpen** (new in 21) | Single control; sharpens "only where it thinks it is needed"; also improves resolution of low-res footage | Studio | One knob, no mask, no temporal control. R21 Guide p56. **[verified]** |
| **Face Refinement / Face Age Transformer / Face Reshaper / Blemish Removal** (new in 21) | Skin, age, face shape, blemish | Studio | Manual p3674–3700. **[verified]** |
| **Depth Map**, **Relight**, **Detail Recovery**, **Optical-flow retiming (Speed Warp / "AI Speed Warp Metal")** | Depth for comps; relight; grain/detail recovery; AI slow-motion | Studio | Speed Warp results "will vary according to the content of the clip"; **"AI Speed Warp Metal (macOS only) … should dramatically speed up the DaVinci Neural Engine Speed Warp operations."** Manual p3700s, p4575. **[verified]** |

### 2.4 Media organisation and text-based editing

- **IntelliSearch** — content search over the Media Pool. Two model packages, **Faster** and
  **Better**; "Better mode uses a larger model and **can take significantly longer to analyze
  your clips**". Both are "equally quick once analysis is complete". Search modes: All, Visual
  Only, Transcript Only, Metadata. Includes a Face Gallery and named-face management. Manual
  p409–411. **[verified]**
- **IntelliSearch for the new Photo page** — R21 lets you search AI-analysed *photographs* "based
  on content, objects, people, colors". R21 Guide p8. **[verified]** Directly relevant: a travel
  recap is photo-heavy.
- **Transcription / Audio Transcription / Detect Speakers / Text-Based Editing** — Manual p1018,
  p1024. Word-level timecode is the enabler for everything else. **[verified]**
- **IntelliScript** — builds a timeline from a `.txt` or `.fdx` script against selected clips.
  The manual's caveat is the important part: **"IntelliScript chooses the 'best' take by
  matching the spoken dialog only. It does not use any good take metadata or make judgments
  based on anything other than dialog matching."** It also requires **"just the plain spoken
  text that has full stops (.) after sentences. Standard script formatting with scene
  descriptions, parentheticals, and character names will likely cause transcription issues."**
  Manual p1027–1028. **[verified]**
- **IntelliScript Supports Final Draft Imports** — new in R21. R21 Guide p80. **[verified]**
- **AI Slate Finder / AI Slate ID** — new in R21, reads camera slates. R21 Guide p55. **[verified]**
- **Scene Cut Detection** on the timeline and in the Cut Timeline (Studio). Manual p550, p709. **[verified]**
- **Multicam SmartSwitch** (Studio). Manual p1070. **[verified]**

---

## 3. Adobe Premiere Pro 25.x → 26.5.2: the AI feature set

Premiere 26.5 (September 2026) is the shipping version; 26.5.1 and 26.5.2 are point releases
from September 2026. **[verified]** Note that Adobe's help centre has been restructured —
URLs moved from `/premiere-pro/using/…` to `/premiere/desktop/…`, and several pages now carry
an **"Explore the beta app"** banner and refer to **"Adobe Premiere (beta)"**, which reads as a
promotion of the old Premiere Beta into the shipping help docs. **[reported]**

### 3.1 Tier gating, stated exactly

Premiere's AI splits into three very different classes, and conflating them is the most common
error in coverage of this product:

| Class | Where it runs | Billing | Examples |
|---|---|---|---|
| **Local ML** | On your machine | Included in subscription | Media Intelligence, Object Mask, Auto Reframe, Auto Color, Auto Ducking, Enhance Speech, Speech to Text, Translate Captions (translation is cloud), AI Assistant (cloud LLM) |
| **Cloud generative (credits)** | Adobe's cloud | **Generative credits** | Generative Extend, Generative Media Tool, Generate Music, Sound Effects, Soundscape |
| **Cloud voice separation (consent-gated)** | Adobe's cloud | Included, but biometric opt-in | Separate Crosstalk |

Generative credit plans, from Adobe's credits FAQ: **Firefly Standard 2,000 / Pro 4,000 /
Pro Plus 10,000 / Premium 50,000 credits per month.** Premium video generation is billed at
**20 credits per second at 540p 24 fps**; Adobe's published table lists the 540p row directly
and the 720p (50/s) and 1080p (100/s) rows are consistently reported by third parties.
**[verified]** for the tiers and the 540p rate; **[reported]** for the 720p/1080p rates.
Credit add-ons are sold at 2,000 for $9.99, 7,000 for $29.99, 10,000 for $49.99 and 50,000 for
$199.99 per month. **[reported]**

### 3.2 Feature detail (all "Last updated" dates as fetched 2 Oct 2026)

- **Auto Color** — Adobe Sensei ML applies basic corrections (exposure, white balance, contrast)
  and **re-organises the Basic Correction sliders** so you can see what it did; an **Intensity**
  slider scales the whole intervention. **"Auto Color works on source footage, including LOG
  footage and footage with LUTs applied."** (updated 18 Aug 2026) **[verified]**
- **Color mode** — a whole grading workspace with a Clip Grid, Color Controls panel, Color
  monitor with Sequence/Clip playback modes and Solo mode. Requires Color Management. **Note:
  "Starting in the upcoming Premiere 27.0 release, the Lumetri panel is being retired"** while
  the Lumetri Color *effect* remains. (updated 24 Aug 2026) **[verified]**
- **Style modules** — composable looks built from Color & Contrast, Detail and Stylize modules.
  One revealing restriction: **"Zones are not available within Color & Contrast modules …
  because they're so sensitive to alterations in image tonality created by operations
  happening earlier in the grade."** (updated 18 Aug 2026) **[verified]**
- **Object Mask** — AI click-to-select a person or object, then track. 26.5 added **Sharp and
  Smooth** mask modes: Sharp is binary edges, Smooth is for hair/fur/translucency and **"does
  not support the mask expansion parameter"**. 26.5 also **"now recognizes significantly more
  selectable objects per scene"** with more accurate boundaries on irregular shapes — but
  **"On Intel-based Macs and on Windows systems with older AMD graphics drivers, Object
  Masking automatically falls back to the previous detection model."** First use downloads
  models; Object Masking is **"temporarily unusable"** until they land. (updated 9 Sep 2026)
  **[verified]**
- **Enhance Speech** — one click in Essential Sound; background task; a **Mix Amount** slider
  blends enhanced against original. Hard limits: **"supports only mono and stereo files …
  doesn't support multichannel audio or nested sequences. When you enhance a stereo audio
  clip, Enhance Speech outputs only a mono downmix."** (updated 7 Jan 2026) **[verified]**
- **Repair / Clarity panels** — non-AI but shipped alongside: Reduce Noise, Reduce Rumble (<80 Hz),
  DeHum (50/60 Hz), DeEss, Reduce Reverb, plus Dynamics / EQ / Vocal Enhancer. (updated Jan 2026
  / Aug 2025) **[verified]**
- **Dynamic Auto Ducking** — writes computed keyframes into an **Amplify effect** on the target
  clip; parameters are Duck Against, Sensitivity, Duck Amount, Fade Duration, Fade Position.
  Caveat: **"selecting the Generate Keyframes button again overwrites all manual changes to
  the keyframes."** (updated 7 Jan 2026) **[verified]**
- **Remix** — fits music to a target duration by cutting and looping. Documented accuracy:
  **"The Remix tool finds a version of your music within 5 seconds of the target duration,
  typically within 1 second."** Accuracy depends on original tempo, available transition points
  and number of distinct sections. Segments and Variations parameters trade edit count against
  flexibility. **Intros/outros are always preserved unchanged. And, critically: "Edits may
  combine lines from different verses. Lyrical continuity isn't preserved across cuts."**
  (updated 18 Aug 2026) **[verified]**
- **Separate Crosstalk** — separates overlapping speakers into individual tracks. **Cloud only.**
  Consent text: "This may involve creating and processing biometric information, including
  numerical representations of voice characteristics called **voice embeddings**." Adobe
  "uses this information only to provide Separate Crosstalk. It isn't stored beyond providing
  the feature, and Adobe does not use your content to train AI models." **"Separate Crosstalk
  isn't available on files uploaded from Illinois or Texas due to legal restrictions."**
  (updated 24 Sep 2026) **[verified]**
- **Media Intelligence / Search panel** — **"Your footage, analysis data, and searches never
  leave your computer and are never seen by Adobe. The AI models for analysis and search are
  installed locally … and don't require the internet to work."** Two honest limits:
  **"Currently, visual search supports English language searches only"** and **"The AI model
  analyzes a smaller version of your footage as a series of still images, so it won't pick up
  as well on small details or fast motion."** (updated 2 Jun 2026) **[verified]**
- **Speech to Text / Text-Based Editing / Translate Captions** — transcription, transcript-driven
  cutting, pause detection, speaker naming/removal, caption styling, and caption translation
  via **"cloud service and third-party translation models (Google Translate and Microsoft
  Translator)"**. 26.5 added **"a new Voice Activity Detection model for speech-to-text"**.
  **[verified]**
- **Generative Extend** — add frames to one end of a clip. Documented media envelope:
  360p–4K UHD, any aspect ratio, 12–60 fps but **"extensions above 30 fps are generated at
  30 fps"**, 8/10/16-bit source but **"extensions are generated in 8-bit"**, SDR or HDR source
  but **"extensions are generated in SDR"**. **"Generative Extend cannot create or extend
  spoken dialog"**; **"clips containing music are not eligible"**; mono/stereo only. Output is
  H.264 MP4 or .wav in the *Captured and Generated* scratch disk. (updated 9 Sep 2026) **[verified]**
- **Generative Media Tool** — text-to-video with optional reference frames from the sequence,
  text-to-sound-effect with **"Sync to video"** or **"Record guide"**, partner models, and a
  **Generate soundscape** mode that "uses the visible content" of the selected range to drive
  ambient audio. (updated 9 Sep 2026) **[verified]**
- **Generate Music (beta)** — **this is the feature most relevant to a beat-synced pipeline.**
  Select a timeline range, prompt for mood/style/instruments, then **"Set the Tempo to Auto BPM
  or enter a specific BPM. Select Loop if you want the generated music to repeat."** It
  generates multiple variations for A/B against picture, saves every generation in a Generation
  history, is instrumental-only ("does not generate vocals or lyrics"), and consumes generative
  credits. (updated 25 Aug 2026) **[verified]**
- **Premiere AI Assistant (beta)** — see §7.
- **Paper Edit** — new in 26.5: select multiple transcript lines and build a sequence from them.
  **[verified]**

---

## 4. Master comparison table

`Tier gating` = what you must own/pay for. `Quality verdict` = best available evidence.
`Careful editor?` = would a working professional put this in a client cut.

| Feature | Platform | What it does | Tier gating | Quality verdict | Would a careful editor actually use it? |
|---|---|---|---|---|---|
| **Super Scale** | Resolve | AI upscale 2–4× + Enhanced; Sharpness/NR controls | Studio, $295, local | Vendor: convincing illusion; High sharpness also amplifies grain **[verified]**; practitioner split, no benchmark **[rumoured]** | As a last resort before delivering a punch-in on low-res footage — not as a general path |
| **Topaz Video AI** (context) | third party | Dedicated AI restore/denoise/interpolate | separate paid licence | No public benchmark found **[verified absence]**; "miles better" **[rumoured]** | Yes, for archival and for genuinely unusable footage |
| **UltraNR** | Resolve | Learned spatial NR; analyse a flat patch | Studio, local | "by far the best results for excessively noisy footage" **[reported]**; practitioners: over-aggressive, slow, detail loss vs Neat Video **[rumoured]** | Yes, but always with a Show Patch on a real flat region and a detail check |
| **Neat Video** (context) | third party | GPU NR plugin | paid plugin | "retains a lot more detail" **[rumoured]** | Yes, when detail retention is the brief |
| **CineFocus** | Resolve 21 | Virtual lens + AI depth map; rack focus, shaped bokeh, grain repair | Studio, local | No benchmark; vendor warns depth-map Blur "can cause artifacts at image object edges" **[verified]** | **Yes — and this is the sleeper feature for a photo-heavy travel recap** |
| **Magic Mask v2** | Resolve | Click-to-mask person/object, track, paint fix-ups | Studio, local | "dramatically better results" is vendor copy **[reported]**; reportedly handles occlusions/reflections | Yes — the most-used Resolve AI feature in practice |
| **Object Mask** | Premiere | Click-to-select + track; Sharp/Smooth modes | Subscription; model download on first use | 26.5 expanded object coverage **[verified]**; **falls back to an older model on Intel Macs / old AMD drivers** → non-deterministic across machines **[verified]** | Yes, now; the hardware-dependent fallback is a real reproducibility trap |
| **Motion DeBlur** | Resolve 21 | AI slow-shutter deblur, renders new media | Studio, local | **up to 60 s of compute per second of footage**; "avoid very fast camera movement" **[verified]** | Only on hero shots; never in batch |
| **UltraSharpen** | Resolve 21 | Single-knob selective sharpening | Studio, local | No benchmark; one control, no masking **[verified]** | Occasionally, on a slightly soft shot |
| **Voice Isolation** | Resolve | Learned dialogue extraction with an Amount mix | Studio, local; mono/stereo only | Vendor-blessed range **70–80** **[verified]**; metallic artefact reported beyond it **[rumoured]** | Yes — this is the best-known AI audio tool in either NLE |
| **Enhance Speech** | Premiere | One-click dialogue cleanup + Mix Amount | Subscription; **stereo→mono downmix** | **25 s to process a 5 s clip on under-specced hardware** (~5× realtime) **[verified]**; hallucination reports **[rumoured]** | Yes for location sound, but never on a stereo bed — you will lose the stereo image |
| **Separate Crosstalk** | Premiere | Cloud speaker separation into tracks | Subscription + **explicit biometric consent**; blocked for IL/TX uploads | No benchmark **[verified absence]** | Yes for two-person panels; the consent flow is a client-comms problem |
| **Ducker / Auto Ducking** | both | Sidechain ducking; Premiere writes Amplify keyframes | Studio / subscription | Not AI in the modern sense; deterministic parameters, documented defaults **[verified]** | Yes — this is the correct way to solve the problem both vendors' AI audio also solve |
| **Music Remixer (stem split)** | Resolve | 5-way Neural Engine stem separation | Studio | No benchmark **[verified absence]** | Yes, for clearing a vocal out of a bed |
| **Music Editor** | Resolve | Retime music to a target duration | Studio | **Beat-driven only; no time-stretch; no pitch change; 4 candidates; "timings can be quite close, sometimes they may not be exact"** **[verified]** | As a *proposal generator*, never as the final cut |
| **Beat Detector** | Resolve | Beat markers + snapping | Studio | **Beat-driven music in 4/4 or 3/4 only** **[verified]** | Yes — if your music is 4/4 or 3/4 |
| **Remix** | Premiere | Fit music to target duration | Subscription | **Within 5 s of target, typically within 1 s**; preserves intros/outros; breaks lyrical continuity **[verified]** | Same as Music Editor: a proposal, not a cut |
| **Generate Music** | Premiere (beta) | Prompt + **Auto BPM or a typed BPM** + Loop | Subscription + **generative credits**, cloud | New (Aug 2026); no benchmark **[verified absence]** | Yes, and it is the most promising thing here for your use case — but generate offline and import |
| **Audio Assistant** | Resolve | Auto-categorise, auto-mix, auto-loudness to a Delivery Standard | Studio | No benchmark vs human mix **[verified absence]** | As a first pass on a big assembly you will re-mix anyway |
| **IntelliScript** | Resolve | Timeline from a script | Studio | **Chooses "best" take by dialogue match only; no good-take metadata** **[verified]** | For script-heavy narrative only — never for anything where performance matters |
| **Transcription / Speech to Text** | both | Word-level timecode | Studio / subscription | **No published WER from either vendor** **[verified]** | Yes — it is the enabler for everything else, and you should still verify the transcript |
| **IntelliSearch** | Resolve | Content search; Faster/Better models; face gallery | Studio; large model download | Better mode "can take significantly longer to analyze" **[verified]** | Yes for large libraries |
| **Media Intelligence** | Premiere | Local content search, transcripts, metadata, markers | Subscription; **local only** | **English-only visual search; analyses downsampled stills, so weak on small detail and fast motion** **[verified]** | Yes for retrieval, with the English and fast-motion caveats understood |
| **Auto Color** | Premiere | Sensei ML exposure/WB/contrast, reorganised sliders, Intensity | Subscription | No benchmark **[verified absence]** | Yes as a starting point on LOG; verify on skin tones |
| **Color mode / Style modules** | Premiere | Dedicated grading workspace; composable look modules | Subscription; beta-badged | No benchmark; zones disabled inside Color & Contrast modules **[verified]** | Yes for non-specialists; specialists will stay in Lumetri until 27.0 |
| **Auto Reframe** | Premiere | Reframe to 9:16 / 1:1 / 16:9; custom target resolution | Subscription | No accuracy benchmark **[verified absence]** | As a first pass, always |
| **Optical-flow retiming / Speed Warp** | Resolve | AI slow-motion | Studio | "results will vary according to the content of the clip" **[verified]**; see §5 for the one benchmark that does exist — it is not flattering | Only when nothing else will do |
| **Generative Extend** | Premiere | Add seconds to one end of a clip | Subscription + **credits**, cloud, geo-blocked, K-12/CCE-limited | 8-bit SDR output, music ineligible, one side only, breaks markers/metadata/timecode; **"No workarounds exist"** for several **[verified]** | Occasionally, on a talking head — never in a system that must round-trip XML |
| **Generative Media Tool** | Premiere | Text/reference-frame video, SFX, soundscapes | Subscription + **credits**, cloud | No benchmark; ~4 months old **[verified absence]** | No, not yet |
| **Separate Speech Generator / Voice Convert** | Resolve | TTS from text; voice swapping | Studio; **8 GB VRAM floor**; ~3 h model training | No MOS/WER **[verified absence]** | Speech Generator: yes for scratch VO. Voice Convert: only where you have explicit rights |
| **MCP server** | Resolve 21.1 Studio | Native MCP; assistants drive the project | **Studio only; Python scripting moved to Studio in 21.1** | Too new to have field reports **[verified that it exists]** | Watch; don't depend on it |
| **AI Assistant** | Premiere (beta) | Conversational project/timeline automation | Subscription; public beta | Adobe: **"It's not recommended to do client work just yet"** **[verified]** | No — not yet |

---

## 5. Measured numbers

Every number below is either from a vendor document (with the vendor's own wording) or from a
named independent benchmark. Where a commonly-cited comparison has **no** public benchmark, I
say so instead of substituting a forum opinion.

### 5.1 Frame interpolation — the one place a real benchmark exists

The **MSU Video Frame Interpolation Benchmark** (MSU Graphics & Media Laboratory) compares 8+
algorithms on a mixed gaming + real-life dataset with **more than 400 subjective participants**,
scored on Subjective, PSNR, SSIM, VMAF, LPIPS and MS-SSIM. Dataset: "From 30 fps".
**[verified]** — full leaderboard at `https://videoprocessing.ai/benchmarks/video-frame-interpolation.html`

| Rank | Algorithm | Subjective | PSNR | SSIM | VMAF | LPIPS | MS-SSIM | FPS |
|---|---|---|---|---|---|---|---|---|
| 1 | Chronos-SloMo-v2 | 2.05 | 27.57 | 0.917 | 65.88 | 0.053 | 0.939 | 4.35 |
| 2 | **RIFE** | 1.99 | 27.15 | 0.914 | **66.33** | 0.039 | 0.939 | 27.3 |
| **3** | **Adobe Premiere Pro** | **1.49** | **21.93** | **0.78** | **34.01** | 0.085 | **0.885** | not reported |
| 4 | XVFI (S_tst=3) | 1.38 | 27.35 | 0.913 | 63.47 | 0.061 | 0.933 | 5.4 |
| 5 | Frame Repeating | 1.31 | 23.06 | 0.75 | 29.84 | 0.097 | 0.771 | n/a |
| 6 | Super-SloMo | 1.11 | 26.69 | 0.904 | 61.35 | 0.068 | 0.924 | 3.1 |
| 7 | CAIN | 1.07 | 27.44 | 0.919 | 64.81 | 0.087 | 0.931 | 35.2 |
| 8 | Frame Averaging | 0 | 24.18 | 0.788 | 36.17 | 0.087 | 0.830 | n/a |
| 9 | FILM | not rated | 28.11 | 0.928 | 68.68 | 0.033 | 0.948 | not reported |
| 10 | CURE | not rated | 27.15 | 0.911 | 62.87 | 0.051 | 0.956 | not reported |

What this means: **Premiere Pro's frame interpolation scores roughly half of RIFE's VMAF
(34.01 vs 66.33) and about 3 dB worse PSNR**, and it barely beats naive frame repeating on
VMAF. Frame Averaging — a two-line ffmpeg filter — actually edges it on VMAF. **[verified]**

**Caveats I will not hide:** the benchmark page carries a 2022 date, so "Adobe Premiere Pro" is
a ~2022 build, not 26.5; DaVinci Resolve is *not* in the table; and no NLE has published its
own numbers. Treat this as strong evidence about the *class* of optical-flow interpolation
shipped in NLEs, not as a measurement of today's builds. **[verified]**

**No public benchmark found** for: Resolve Optical Flow / Speed Warp, Topaz Apollo or Chronos,
Resolve Super Scale vs Topaz Video AI, or any commercial upscaler.

### 5.2 Compute and throughput

| Number | Source | Marker |
|---|---|---|
| Motion DeBlur: **"up to 60 seconds per second of footage"** to analyse | Resolve 21 New Features Guide p58 | **[verified]** |
| Voice Convert: **requires a minimum of 8 GB of video RAM before it will activate** | Resolve 21 Manual p1161 | **[verified]** |
| Voice Convert: Better mode ≈ **3×** Faster mode; ~10 min of source → **"a few hours"** | Resolve 21 Manual p1163 | **[verified]** |
| Speech Generator: custom voice sample is **truncated to the first 20 seconds** | Resolve 21 Guide p44 | **[verified]** |
| Enhance Speech on under-specced hardware: **"taking 25 seconds to process a 5-second clip"** (~5× realtime) | Adobe, "Enhance Speech is running slow" | **[verified]** |
| Enhance Speech recommended hardware for speed: **macOS 13 + M1**, or a recommended Windows GPU | Adobe, Enhance Speech technical requirements | **[verified]** |
| Resolve Neural Engine device selection: Auto / specific GPU; **"AI Speed Warp Metal (macOS only) … should dramatically speed up"** Speed Warp | Resolve 21 Manual p100, p4575 | **[verified]** |
| Premiere 26.5 Object Mask: falls back to the **previous** detection model on **Intel-based Macs and Windows with older AMD graphics drivers** | Adobe, Object masking in Premiere | **[verified]** |
| Blackmagic: Snapdragon X Elite boosts Neural Engine AI performance "**by up to 4.7×** on Windows computers" | Jon Peddie Research, reporting Blackmagic | **[reported]** |

### 5.3 Accuracy numbers the vendors actually publish

- **Premiere Remix: "finds a version of your music within 5 seconds of the target duration,
  typically within 1 second."** Adobe. **[verified]** This is the only honest accuracy figure
  either vendor publishes for any music-fitting feature.
- **Resolve Music Editor: "While timings can be quite close, sometimes they may not be exact."
  If you need exact timing, Blackmagic points you at the Elastic Wave tool.** Resolve 21
  Manual p1149. **[verified]** The vendor is telling you the feature does not meet a frame
  contract.
- **Resolve Voice Isolation: "Values between 70 and 80 work well for natural results."**
  Manual p1146. **[verified]**
- **Resolve Ducker: 2.7 dB default, "most of the time, a value between 2.0 dB and 5.0 dB works
  the best"; 15 ms lookahead; 10 ms rise; 150 ms hold; 750 ms recovery.** Manual p1152. **[verified]**
- **Premiere Enhance Speech:** stereo clips are **downmixed to mono**. **[verified]**

### 5.4 Licensing / metering numbers

- **DaVinci Resolve Studio 21: $295, one-time.** Blackmagic product page. **[verified]**
- **Firefly plans: Standard 2,000 / Pro 4,000 / Pro Plus 10,000 / Premium 50,000 credits per
  month.** Adobe credits FAQ. **[verified]**
- **Premium video generation: 20 credits/second at 540p 24 fps** (Adobe FAQ, 540p row);
  50/s at 720p and 100/s at 1080p are consistently reported but were not on the row I could
  read directly. **[verified]** / **[reported]**

### 5.5 Explicit "no public benchmark found"

| Comparison | Status |
|---|---|
| Resolve Super Scale vs Topaz Video AI | No public benchmark found. MSU's Video Super-Resolution Benchmark contains only academic models (PSRT, RVRT, VRT, BasicVSR++, BasicVSR, RBPN, DBVSR, iSeeBetter, LGFN, DynaVSR-R, HAT, TMNet, COMISR, RSDN) — no commercial tools. **[verified absence]** |
| Resolve UltraNR vs Neat Video | No public benchmark found |
| Resolve Voice Isolation vs Premiere Enhance Speech vs Adobe Podcast | No public benchmark found |
| Premiere Auto Reframe subject-tracking accuracy | No public benchmark found |
| Resolve CineFocus depth-map accuracy | No public benchmark found |
| Magic Mask v2 vs Premiere Object Mask | No public benchmark found |
| AI Audio Assistant vs a human mix | No public benchmark found |
| Transcription word-error rate (either vendor) | No public benchmark found |
| Speech Generator / Voice Convert MOS or WER | No public benchmark found |

---

## 6. What editors report doesn't work

All quotes here are from indexed search snippets of threads I could not open in full (see §0), so
they carry their thread URL and are marked at the confidence their evidence actually supports.
The signal is in the *direction* of the comments, which is consistent across independent threads.

**Upscaling.** The longest-running Resolve thread is a straight split.
> "Topaz VideoEnhance AI is miles better. Not even on the same league. I own both."
> — [r/davinciresolve, `u9b69a`](https://www.reddit.com/r/davinciresolve/comments/u9b69a/how_is_resolve_studio_for_upscaling_compared_to/) **[rumoured]**

> "I prefer resolve RTX upscale faster than topaz. I prefer topaz for something like fake frames/
> smooth motion."
> — [r/davinciresolve, `1jynx10`](https://www.reddit.com/r/davinciresolve/comments/1jynx10/new_dr_superscale_ai_vs_topaz_video_ai/) **[rumoured]**

Against that, an older Resolve 17 thread reports the opposite failure mode for Topaz:
> "the topaz upscale looks like a 50s lens wide open. it adds glow to the whole image that makes
> no sense and isn't desireable."
> — [r/davinciresolve, `mbk0u7`](https://www.reddit.com/r/davinciresolve/comments/mbk0u7/resolve_17_super_scale_vs_topaz_video_enhance_ai/) **[rumoured]**

Blackmagic's own manual corroborates the shared failure mode without naming a competitor:
Sharpness set to High "also sharpens grain and noise in the image to an undesirable extent at
the default settings". **[verified]**

Note that Topaz Labs publishes a comparison page positioning Topaz Video against Resolve. That
is competitor content marketing and is **[reported]** at best. **[rumoured]** for its
characterisations.

**Denoise.** UltraNR is the most criticised feature in the Resolve AI set.
> "I've found the AI NR to be way too aggressive, and really slow to process, but the regular
> manual NR is pretty good."
> — [r/davinciresolve, `1q3p7cn`](https://www.reddit.com/r/davinciresolve/comments/1q3p7cn/the_resolve_ai_ultranr_noise_reducer_is_insane/) **[rumoured]**

> "Resolve does a good job but in areas where there are fine details it blurs/smooths it out
> too much where neat video retains a lot more detail."
> — [r/davinciresolve, `1nf45h6`](https://www.reddit.com/r/davinciresolve/comments/1nf45h6/do_resolve_noise_reduction_plugins_match_neat/) **[rumoured]**

Blackmagic's own forum thread, from the UltraNR era:
> "In the comparison, the new UltraNR, perhaps because it is still in beta, is slower and with
> loss of detail in some areas than Neat Video."
> — [Blackmagic forum, `t=106578`](https://forum.blackmagicdesign.com/viewtopic.php?f=21&t=106578) **[rumoured]**

**Masking.** Premiere's Object Mask splits the room just as sharply.
> "As Object Mask doesn't work for me, it doesn't save any time for me."
> — [r/premiere, `1tcbgus`](https://www.reddit.com/r/premiere/comments/1tcbgus/the_object_mask_tool_is/) **[rumoured]**

> "The new object mask is a life saver for me! I got used to the new workflow almost
> immediatly, so i have no regret from the old version!"
> — [r/premiere, `1vrodzr`](https://www.reddit.com/r/premiere/comments/1vrodzr/do_you_prefer_the_new_masking_method_in_premiere/) **[rumoured]**

There are also live threads titled "Object Mask Tool not working at all" (`1uuqkxl`) and
"Latest version object mask taking forever" (`1we0n33`) — consistent with a feature whose
quality is version-dependent and machine-dependent, which the official page's own hardware
fallback clause explains. **[rumoured]** for the titles, **[verified]** for the fallback clause.

**Audio.** Practitioner comparisons of Resolve Voice Isolation against Premiere Enhance Speech
and Adobe Podcast cluster around the same complaint: artefacts and, in Premiere's case, the
silent stereo→mono collapse. **[rumoured]** No benchmark exists either way (§5.5).

**Adobe's own known-issues page is the strongest negative evidence in the report.** For
Generative Extend, Adobe documents that extended clips: cannot be transcribed with Speech to
Text; do not appear in Media Intelligence results; cannot have speed changes applied; cannot be
multicam; can only be extended on one side; **fail outright with "Can't complete" if the clip
contains music**; cannot carry >2 channels; lose all markers; lose user-entered logging metadata
(Shot, Scene, Good Take); **always reset to zero-based timecode**; cannot have Audio Channels,
Color, Interpret Footage, Timecode or VR Properties modified; have an inaccessible source color
space; cannot be collected by Project Manager; and are not fully supported in XML, AAF, EDL,
OMF or OTIO. **"No workarounds exist for these limitations."** **[verified]**

---

## 7. The human/AI boundary, and the MCP angle

### 7.1 Where the boundary actually sits in both products

Every AI feature in both NLEs is placed on one of three sides, and the placement is consistent:

**Fully automatic, and intended to be fully automatic.**
Audio Assistant's whole point is that "The initial fader levels really don't matter as Audio
Assistant will handle all fader adjustments". Object Mask and Magic Mask generate a mask and
let you repaint. IntelliScript generates a complete timeline. The Generative Media Tool and
Generate Music generate and insert clips. **[verified]**

**Automatic proposal, human decision.** Resolve's Music Editor returns **four** candidate
retimes and leaves the user to pick; Premiere's Remix returns variations and tells you to
verify against the target; Premiere's Generate Music generates multiple variations "so you can
compare different options directly against your video". **[verified]**

**Assistive only.** Audio Assistant can be "completely undo"ne. AI Assistant is designed around
an **Always ask permission / Auto approve** switch and states that "All actions are added to
Premiere's Undo and History panels". Premiere's Object Mask is explicitly framed as removing
the need "to leave your editing environment to use After Effects". **[verified]**

The tell for which category a feature is in is whether Adobe/Blackmagic gives you a way to
inspect what it did. Resolve: Show Patch (where UltraNR sampled), Magic Mask's Click list,
CineFocus's Depth/Focus/Aperture views, Music Editor's cut indicators. Premiere: Auto Color's
reorganised sliders, Essential Sound's progress and Mix Amount, the Object Mask hover-preview
and Sharpness control. Features without an inspection affordance are the ones to distrust.

### 7.2 Adobe's stance on the boundary is unusually blunt

Premiere AI Assistant is in **public beta**, and Adobe's own overview page says:
> "**It's not recommended to do client work just yet.** For the best experience, work with
> duplicates of your projects, or import new media into a fresh project, rather than testing on
> work you can't afford to lose."
> **[verified]**

That is the vendor telling you the boundary is not yet crossed. Its capability list is also
narrower than the marketing implies: create and organise bins, move and rename clips, apply
colour labels, generate transcripts, detect slates, add markers, create stringouts, build rough
assemblies, place organised media on a timeline. It "only modifies project and timeline content.
It does not change your workspace layout or open panels on your behalf." **[verified]**

### 7.3 MCP is the actual 2026 inflection, and it is paywalled

From the **DaVinci Resolve 21.1 release notes**:
> "Advanced scripting now requires DaVinci Resolve Studio."
> "Built in Python support for console scripting."
> "**Native MCP server for interacting with AI assistants.**"
> **[verified]** — the note text as reproduced on Blackmagic's own forum
> ([`t=239823`](https://forum.blackmagicdesign.com/viewtopic.php?f=21&t=239823)) and corroborated
> independently by Cined, ProVideoCoalition and RedSharkNews. Blackmagic's own framing of the
> scripting change: "The Python API was being used to hack studio features into the free
> version." **[reported]**

Practical consequences:

1. **MCP is Studio-only.** Resolve 21.1's native MCP server ships with the $295 licence, and the
   free edition's Python surface was removed. **[verified]**
2. **The API surface moved with it.** The 21.1 release notes list a Studio-only scripting
   section; a community server claims 37 tools and "100% API coverage", and a third-party
   `adobe_premiere_pro_mcp` exists on GitHub. **I could not read the shipped API reference
   directly, so tool counts and coverage percentages are [rumoured].** Do not plan against them.
3. **Premiere has no MCP story.** Its automation surface is an in-app beta chat, not an
   addressable protocol. That is a structural asymmetry: Resolve can be *driven* by an agent;
   Premiere can only be *advised* by one. **[verified]**
4. **Both vendors now restrict automation in the same direction.** Resolve gates scripting
   behind Studio; Premiere gates generation behind credits, consent and regions. For a
   deterministic pipeline, the automation surface you can rely on is the one you own.

---

## 8. Top 10 findings that would change this pipeline's roadmap

Ranked by expected value to a beat-synced, deterministic travel-recap builder.

### 1. Do not adopt any NLE generative feature. They are metered, cloud, geo-blocked and structurally lossy.
Premiere's Generative Extend outputs 8-bit SDR from any source, cannot touch clips with music,
resets timecode to zero, drops markers and logging metadata, and cannot be collected by Project
Manager or round-tripped through XML/AAF/EDL without Render-and-Replace first. Resolve has no
generative video feature at all. A recap must round-trip deterministically; nothing here can.
**Action:** if you ever want generative fill, generate it offline, save it to `media/`, and
let the pipeline consume it as an ordinary asset.

### 2. Your beat grid is already better than both NLEs' music fitting, and both vendors admit it.
Premiere Remix is "within 5 seconds of the target duration, typically within 1 second". Resolve
Music Editor "does not use time compression/expansion", works only on beat-driven 4/4 or 3/4
material, and Blackmagic's own tip is to fix the timing afterwards with Elastic Wave. Your
pipeline *knows* its grid exactly and quantises to whole frames.
**Action:** no change needed — but this is now a defensible, documentable design decision. Say
so in the README, with the two vendor numbers as the citation.

### 3. Generate Music's typed-BPM contract is the one generative feature with a real timing contract — steal the interface, not the service.
Premiere lets you set **"Auto BPM or enter a specific BPM"** with a loop toggle, generates
variations, and saves a generation history. That is exactly the contract a beat-synced pipeline
wants. It is also beta, credit-metered and cloud.
**Action:** adopt the *contract* — an explicit `bpm` + `loop` + `seed` on your music spec, and
a generation-history concept — so a future offline music model drops straight in.

### 4. NLE frame interpolation is measurably weak; open weights are not.
MSU's benchmark puts Adobe Premiere Pro at **VMAF 34.01 / PSNR 21.93 / SSIM 0.78**, versus
**RIFE at 66.33 / 27.15 / 0.914** — and barely ahead of frame averaging. If you ever need
slow-motion or frame-rate conversion, RIFE/FILM in your own pipeline beats the NLE on every
metric and is reproducible.
**Action:** if frame interpolation is on the roadmap at all, implement it behind the same
pluggable-quality-gate as everything else, defaulting to off. Never use the NLE's.

### 5. AI upscaling has no benchmark and a split practitioner verdict — so treat it as a hero-shot-only, original-preserving operation.
MSU publishes no commercial upscalers at all. Blackmagic warns High Sharpness amplifies grain
and noise. Practitioners are split roughly evenly between "miles better" for Topaz and "adds
glow that isn't desirable".
**Action:** keep Lanczos/ffmpeg as the pipeline's scaler. If AI upscale is added, gate it to
hand-picked hero shots, always keep the original, and record the decision in the report — the
same discipline you already apply to the 4K-upscale honesty note.

### 6. AI denoise in an NLE is a quality regression you cannot A/B. Prefer deterministic NR and threshold it.
UltraNR is widely reported as over-aggressive, slow, and detail-destroying versus Neat Video —
and in a pipeline you cannot scrub the slider, so you would be committing to it blind.
**Action:** stay with `hqdn3d`/`atadenoise`. If you add learned NR, gate it on a measured
noise metric (e.g. variance in a flat region) with a hard threshold and log the decision —
never apply it unconditionally.

### 7. Voice isolation is real, but only in a narrow, vendor-sanctioned envelope — and one of the two implementations destroys your stereo image.
Resolve: Amount **70–80**, mono/stereo/dual-mono only, never >2 channels. Premiere: mono/stereo
only, and **"Enhance Speech outputs only a mono downmix"** of a stereo clip. For a travel recap
with ambience beds, that difference is decisive.
**Action:** if you add dialogue cleanup, make it a stem-aware, sidechain-based operation on a
mono dialogue stem, never a whole-mix replace. Keep the bed.

### 8. Content search is English-only and blind to fast motion and small detail — exactly the failure modes of travel footage.
Premiere's own words: **"visual search supports English language searches only"** and **"analyzes
a smaller version of your footage as a series of still images, so it won't pick up as well on
small details or fast motion."** Resolve's IntelliSearch has the same Faster/Better split with
no published recall figure.
**Action:** your GPS clustering, EXIF and clip-geometry metadata are already a better retrieval
index than either product's, and are language-independent and deterministic. Do not regress to
a learned index. If you want learned retrieval, keep it as a *recall* booster over your
deterministic index, never as the index.

### 9. MCP is real but paywalled and two months old — build your own surface now, treat NLE MCP as a later integration target.
Resolve 21.1 shipped a native Studio-only MCP server and moved Python scripting behind Studio;
Premiere has no protocol surface, only a beta chat that Adobe says not to use on client work.
Community servers exist (37-tool claims, 100% API coverage claims) but I could not verify any
of it against the shipped reference.
**Action:** expose your pipeline as a small, versioned tool surface you own — the MCP shape is
free and gives you an agent integration later without a vendor dependency. Do not budget time
against Resolve MCP in 2026; budget it as a 2027 integration.

### 10. Budget render time per AI operation explicitly, and treat hardware as part of the output contract.
Motion DeBlur is **up to 60 s of compute per second of footage** and needs 8 GB VRAM-adjacent
headroom; Voice Convert model training takes **hours** and needs 8 GB VRAM to even activate;
Enhance Speech runs **5× realtime on under-specced hardware**; and Premiere's Object Mask
silently **falls back to an older model on Intel Macs and old AMD drivers**, meaning the same
project produces different masks on different machines.
**Action:** your report already prints a decision trail. Extend it with an **operation cost
table** — expected wall-clock per feature per minute of media — and never let a feature's
output depend on which machine ran it. That last one is the generalisable lesson: several of
these tools are hardware-conditional by design, which is fatal for a reproducible pipeline and
tolerable only for a human sitting in front of the result.

---

## 9. Sources

### Primary — Blackmagic Design

1. DaVinci Resolve 21 Reference Manual (4,444 pp) — local copy `resolve21_manual.pdf`; page
   references above are to this document. https://www.blackmagicdesign.com/products/davinciresolve (manual link)
2. DaVinci Resolve 21 New Features Guide (April 2026) — local copy `resolve21_newfeatures.pdf`.
   https://www.blackmagicdesign.com/products/davinciresolve/whats-new
3. DaVinci Resolve 20 New Features Guide — local copy `resolve20_newfeatures.pdf`.
   https://www.blackmagicdesign.com/products/davinciresolve (archive)
4. DaVinci Resolve 21.1 release notes (text reproduced on Blackmagic's forum) —
   https://forum.blackmagicdesign.com/viewtopic.php?f=21&t=239823
5. DaVinci Resolve / Studio product page ($295 Studio 21) — https://www.blackmagicdesign.com/products/davinciresolve
6. DaVinci Resolve What's New (Neural Engine "fully supported in Apple M series and Snapdragon X Elite") — https://www.blackmagicdesign.com/products/davinciresolve/whatsnew
7. Blackmagic forum, "Resolve Noise Reduction VS. Neat Video…which is better?" —
   https://forum.blackmagicdesign.com/viewtopic.php?f=21&t=106578

### Primary — Adobe (all fetched 2 Oct 2026; "Last updated" dates are Adobe's)

8. Premiere "What's new" (release history 25.0 → 26.5) — https://helpx.adobe.com/premiere/desktop/whats-new/whats-new.html
9. Premiere release notes (26.5.2, 26.5.1, 26.5) — https://helpx.adobe.com/premiere/desktop/whats-new/release-notes.html
10. Premiere AI models preferences — https://helpx.adobe.com/premiere/desktop/get-started/preferences-and-settings/ai-models-preferences.html
11. Use Auto Color (18 Aug 2026) — https://helpx.adobe.com/premiere/desktop/correct-color/add-color-effects/use-auto-color.html
12. Color mode basics (24 Aug 2026) — https://helpx.adobe.com/premiere/desktop/correct-color/color-mode-fundamentals/color-mode-basics.html
13. Available style modules (18 Aug 2026) — https://helpx.adobe.com/premiere/desktop/correct-color/color-mode-fundamentals/available-style-modules.html
14. Object masking in Premiere (9 Sep 2026) — https://helpx.adobe.com/premiere/desktop/add-video-effects/work-with-masks/object-masking.html
15. Apply Sharp and Smooth modes in Object Masking — https://helpx.adobe.com/premiere/desktop/add-video-effects/work-with-masks/apply-sharp-and-smooth-modes-in-object-masking.html
16. Apply Enhance Speech (7 Jan 2026) — https://helpx.adobe.com/premiere/desktop/add-audio-effects/adjust-volume-and-levels/enhance-speech.html
17. Enhance Speech technical requirements — https://helpx.adobe.com/premiere/desktop/add-audio-effects/adjust-volume-and-levels/enhance-speech-technical-requirements.html
18. "Enhance Speech is running slow" — https://helpx.adobe.com/premiere/desktop/troubleshooting/audio-issues/enhance-speech-is-running-slow.html
19. Repair dialogue — https://helpx.adobe.com/premiere/desktop/add-audio-effects/adjust-volume-and-levels/repair-dialogue.html
20. Improve dialogue clarity — https://helpx.adobe.com/premiere/desktop/add-audio-effects/adjust-volume-and-levels/improve-dialogue-clarity.html
21. Automatically duck audio — https://helpx.adobe.com/premiere/desktop/add-audio-effects/adjust-volume-and-levels/automatically-duck-audio.html
22. Remix tool considerations in Premiere (18 Aug 2026) — https://helpx.adobe.com/premiere/desktop/add-audio-effects/apply-audio-effects/remix-tool-considerations-in-premiere.html
23. Separate overlapping speakers — https://helpx.adobe.com/premiere/desktop/add-audio-effects/basic-audio-editing/separate-overlapping-speakers.html
24. About cloud-based AI voice features (24 Sep 2026) — https://helpx.adobe.com/premiere/desktop/edit-projects/edit-with-generative-ai/about-cloud-based-ai-voice-features.html
25. Generative Extend overview (9 Sep 2026) — https://helpx.adobe.com/premiere/desktop/edit-projects/edit-with-generative-ai/generative-extend-overview.html
26. Generative Extend known issues (2 Apr 2026) — https://helpx.adobe.com/premiere/desktop/troubleshooting/limitations-and-known-issues/generative-extend-known-issues.html
27. Generative Media Tool overview (9 Sep 2026) — https://helpx.adobe.com/premiere/desktop/edit-projects/edit-with-generative-ai/generative-media-tool-overview.html
28. Generative Media Tool FAQ (9 Sep 2026) — https://helpx.adobe.com/premiere/desktop/edit-projects/edit-with-generative-ai/generative-media-tool-faq.html
29. Generate Music (beta) overview (25 Aug 2026) — https://helpx.adobe.com/premiere/desktop/edit-projects/edit-with-generative-ai/generate-music-overview.html
30. Add generated music to your timeline (25 Aug 2026) — https://helpx.adobe.com/premiere/desktop/edit-projects/edit-with-generative-ai/add-generated-music-to-your-timeline.html
31. Search for media using AI-powered Media Intelligence (2 Jun 2026) — https://helpx.adobe.com/premiere/desktop/organize-media/file-organization/search-for-media-using-ai-powered-media-intelligence.html
32. Auto Reframe overview (15 Apr 2026) — https://helpx.adobe.com/premiere/desktop/add-video-effects/commonly-used-effects/auto-reframe-overview.html
33. Translate captions (2 Jun 2026) — https://helpx.adobe.com/premiere/desktop/add-text-images/insert-captions/translate-captions.html
34. Premiere AI Assistant (beta) overview (18 Jun 2026) — https://helpx.adobe.com/premiere/desktop/premiere-ai-assistant/overview.html
35. Premiere AI Assistant (beta) FAQ (18 Jun 2026) — https://helpx.adobe.com/premiere/desktop/premiere-ai-assistant/assistant-faq.html
36. Generative credits FAQ (plan tiers, credits/second) — https://helpx.adobe.com/creative-cloud/apps/generative-ai/generative-credits-faq.html
37. Generative credits add-on pricing — https://www.adobe.com/ai/overview/generative-credits.html
38. Adobe blog, Jan 2026 — new AI video editing tools / Object Mask tracking speed — https://blog.adobe.com/en/publish/2026/01/20/new-ai-powered-video-editing-tools-premiere-major-motion-design-upgrades-after-effects

### Primary — independent benchmark

39. MSU Video Frame Interpolation Benchmark (MSU Graphics & Media Laboratory) — https://videoprocessing.ai/benchmarks/video-frame-interpolation.html
40. MSU Video Super-Resolution Benchmark (no commercial upscalers present) — https://videoprocessing.ai/benchmarks/video-super-resolution.html

### Secondary / trade press (used only for corroboration, marked [reported] in text)

41. Cined, "DaVinci Resolve 21.1 Released – AI Assistant Integration via MCP" — https://www.cined.com/davinci-resolve-21-1-released-ai-assistant-integration-via-mcp-individual-hdr-trims-and-python-scripting-moves-to-studio/
42. ProVideoCoalition, "DaVinci Resolve 21.1 is a huge update" — https://www.provideocoalition.com/davinci-resolve-21-1-is-a-huge-update/
43. RedSharkNews, "DaVinci Resolve 21.1 new features and release" — https://www.redsharknews.com/davinci-resolve-21.1-new-features-release
44. Adobe blog on Premiere Beta Object Mask (community announcements) — https://community.adobe.com/announcements-732/now-released-new-masking-tools-311921
45. Jon Peddie Research, "There's new Resolve magic in the air" (reports Blackmagic's "up to 4.7x" Snapdragon X Elite Neural Engine claim) — https://www.jonpeddie.com/news/theres-new-resolve-magic-in-the-air/

### Community MCP projects (claims unverified — [rumoured])

46. `samuelgursky/davinci-resolve-mcp` — https://github.com/samuelgursky/davinci-resolve-mcp
47. `saadk408/davinci-resolve-lua-mcp` (free-edition bridge via in-app Lua) — https://glama.ai/mcp/servers/saadk408/davinci-resolve-lua-mcp
48. `hetpatel-11/adobe_premiere_pro_mcp` — https://github.com/hetpatel-11/adobe_premiere_pro_mcp

### Practitioner threads (read via indexed search snippets only — [rumoured])

49. r/davinciresolve `u9b69a` Super Scale vs Topaz — https://www.reddit.com/r/davinciresolve/comments/u9b69a/how_is_resolve_studio_for_upscaling_compared_to/
50. r/davinciresolve `1jynx10` Resolve RTX vs Topaz — https://www.reddit.com/r/davinciresolve/comments/1jynx10/new_dr_superscale_ai_vs_topaz_video_ai/
51. r/davinciresolve `mbk0u7` Resolve 17 Super Scale vs Topaz (glow complaint) — https://www.reddit.com/r/davinciresolve/comments/mbk0u7/resolve_17_super_scale_vs_topaz_video_enhance_ai/
52. r/davinciresolve `1q3p7cn` UltraNR "insane" (too aggressive, slow) — https://www.reddit.com/r/davinciresolve/comments/1q3p7cn/the_resolve_ai_ultranr_noise_reducer_is_insane/
53. r/davinciresolve `1nf45h6` Resolve NR vs Neat Video detail loss — https://www.reddit.com/r/davinciresolve/comments/1nf45h6/do_resolve_noise_reduction_plugins_match_neat/
54. r/davinciresolve `1hdnl4i` Topaz vs Resolve Studio, $300 comparison — https://www.reddit.com/r/davinciresolve/comments/1hdnl4i/is_topaz_ai_or_resolve_studio_a_better_use_of_300/
55. r/premiere `1tcbgus` "The Object Mask tool is…" ("doesn't save any time for me") — https://www.reddit.com/r/premiere/comments/1tcbgus/the_object_mask_tool_is/
56. r/premiere `1vrodzr` new masking method ("life saver for me") — https://www.reddit.com/r/premiere/comments/1vrodzr/do_you_prefer_the_new_masking_method_in_premiere/
57. r/premiere `1onq6ys` "New Premiere Pro Object Mask is a Game Changer" — https://www.reddit.com/r/premiere/comments/1onq6ys/new_premiere_pro_object_mask_is_a_game_changer/
58. r/premiere thread "Does Auto Reframe do anything" (post id `1tiropk`) — thread slug not captured;
   find via Reddit search for the post id. Auto Reframe is undocumented as to subject-tracking
   accuracy in any case (§5.5).
59. r/davinci_resolve `1wafb08` Resolve 21.1 release notes discussion — https://www.reddit.com/r/davinciresolve/comments/1wafb08/davinci_resolve_211_release_notes/
60. Blackmagic forum `t=166116` SuperScale vs Topaz (>800% render-time difference claim) — https://forum.blackmagicdesign.com/viewthread.php?t=166116

### Sources deliberately excluded or downgraded

- **Topaz Labs "Topaz Video vs DaVinci Resolve"** — https://www.topazlabs.com/learn/topaz-labs-vs-davinci-resolve — vendor
  content marketing for a competitor; **[reported]** at best.
- **kunalganglani.com, filmora/cryptoscobra listicles, cutsio.com, aiarty.com, unifab.ai** — SEO
  content marketing or affiliate pages. One of them publishes an "accuracy / time saved" table
  for Resolve 21 AI features; I did **not** use it, because the numbers are unsourced and the
  page format is lead-generation.