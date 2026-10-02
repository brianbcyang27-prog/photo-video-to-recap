# Broadcast-quality audio in automated video pipelines

Research report for the travel-recap generator in this repo (`python3` +
`ffmpeg`, numpy/Pillow/soundfile only, offline, no downloaded models at ship
time).

Written 2026-10-02. Every claim carries an evidence tag:

| Tag | Meaning |
| --- | --- |
| `[verified]` | Read directly from a standards body's own document, a platform's own support/developer documentation, official tool documentation, or a peer-reviewed paper I fetched the text of. |
| `[derived]` | Arithmetic on `[verified]` parameter semantics. No new assumptions. |
| `[reported]` | Consistent across multiple independent secondary sources, but no primary document found. Cited as folklore-tier. |
| `[rumoured]` | Circulated in forums, tutorials, or cheat sheets, with no primary support found, and in at least one case contradicted by primary sources. |

## 1. The short answer

1. **YouTube publishes no loudness target.** There is no LUFS figure in
   YouTube Help, the YouTube upload spec, the format spec, or the API docs.
   "YouTube −14 LUFS" is folklore. `[rumoured]`
2. **Netflix is −27 LKFS ±2 LU dialog-gated, max true peak −2 dBTP.** The
   "−31 LKFS" figure everybody quotes is the Dolby Digital decoder dialnorm
   reference, not a Netflix spec. `[verified]`
3. **Apple Podcasts publishes no loudness spec at all.** `[verified]`
4. **"Podcast −16 / −19 LUFS"** is an extrapolation from AES TD1008 (−18 LUFS
   for speech, with a ±2 LU tolerance that reaches −16), not a platform rule.
   `[derived]` from `[verified]` AES TD1008.
5. **Speech enhancement models degrade on real recordings**, are trained for
   speech not ambience, and treat music as interference. On the actual travel
   target — room tone, wind, street noise — they are the wrong tool. `[verified]`
6. **Music should sit ~2–3 LU above where speech would sit, but 4 dB lower than
   the final mix**, per BBC's own mixing guide. `[verified]`
7. For this repo specifically: the two loudest problems are not missing AI. They
   are (a) per-segment single-pass `loudnorm` on 3-second ambient clips, and
   (b) sample-peak limiting where true-peak limiting was intended. See §9.

## 2. Units, because the folklore conflates them

| Unit | What it is | Source |
| --- | --- | --- |
| **LUFS** | Gated integrated programme loudness. Relative loudness where 0 LUFS = full scale. | ITU-R BS.1770 `[verified]` |
| **LKFS** | Numerically identical to LUFS. ATSC's spelling. | ITU-R BS.2434 report reproduces ATSC A/85 in LKFS alongside R128 in LUFS in the same table `[verified]` |
| **LRA** | Loudness range, from the short-term distribution. | ITU-R BS.1770 `[verified]` |
| **dBTP** | Inter-sample *true peak*. Can exceed 0 dBFS on the sample peak after lossy encoding. | EBU R128 `[verified]` |
| **RMS / mean dB** | Not loudness. Not gated, not K-weighted. Do not use for delivery compliance. | — |

The critical distinction for this pipeline: **LUFS is a gated programme
measurement, dBTP is an inter-sample peak ceiling, and they constrain different
things.** R128 requires *both* −23 LUFS integrated and max −1 dBTP. Cutting to
−1 dBFS sample peak does not satisfy −1 dBTP, because AAC encoding creates
inter-sample peaks above the sample peak.

## 3. Loudness targets

### 3.1 Standards bodies (primary, authoritative)

| Body / doc | Target | Tolerance | Max true peak | Measurement | Tag |
| --- | --- | --- | --- | --- | --- |
| **EBU R128** (v3, June 2014) | −23 LUFS | ±0.5 LU (live programmes excepted) | −1 dBTP | Full programme | `[verified]` |
| **EBU R128 S1** (short form: adverts, promos) | −23 LUFS | ±0.5 LU | −1 dBTP | — | `[verified]` |
| **EBU R128 S1 short-term** | Short-term loudness must not exceed **−18 LUFS** | — | — | 3 s window | `[verified]` |
| **ITU-R BS.1770-5** | Measurement algorithm, not a target. Approved 2023-11-22, superseding BS.1770-4. | — | — | K-weighting + gating | `[verified]` |
| **ITU-R BS.1864** | −24 LKFS for international exchange of digital TV programmes | — | — | — | `[verified]` |
| **ATSC A/85:2026-07**, Annex M Table M.1 | Long form, no metadata: **−24 LKFS**; short form: **−24 LKFS**; with metadata: dialnorm value governs | ±2 LU (via BS.2434 table) | **−2 dBTP** | Anchor element recommended; full-mix conditionally permitted | `[verified]` |
| **ITU-R BS.2434** summary table | North America A/85: −24 ±2 LKFS, −2 dBTP. Europe R128: −23 ±0.5 LUFS, max short-term −18 LUFS, −1 dBTP. Japan TR-B32: −24 ±1 LKFS, −1 dBTP. Australia OP-59: −24 ±1 LKFS, −2 dBTP. Short form up to −16 ±1 LKFS/LUFS, −1 dBTP. | — | — | — | `[verified]` |
| **AES TD1004.1.15-10** | Streaming / network file playback: range **−16 LKFS to −20 LKFS**; should not exceed **−16 LUFS** | — | — | — | `[verified]` |
| **AES TD1008 v3.13** (2021-09-24), Table 1 | Assorted content, speech measurable: **−18 LUFS** | **+1 LU** | **−1 dBTP at codec input** | **Dialog** Integrated Loudness | `[verified]` |
| **AES TD1008**, Table 1 | Speech *not* measurable: −18 LUFS | **+2 LU** | −1 dBTP | Format-specific → Table 2 | `[verified]` |
| **AES TD1008**, Table 1 | Music, track-normalized: **−16 LUFS** | **+0.2 LU** | — | Integrated loudness | `[verified]` |
| **AES TD1008**, Table 1 | Music, album-loudest-track (on-demand services): **−14 LUFS** | **+0.2 LU** | — | Integrated loudness | `[verified]` |
| **AES TD1008**, Table 2 | Format-specific: News/Talk **−18**, Pop music **−16**, Mixed format **−17**, Sports **−17**, Drama **−18** | — | — | — | `[verified]` |
| **AES TD1008** blend formula | `Distribution Loudness = −16 − [2 × (SpeechPercent / 100)]` LUFS | — | — | — | `[verified]` |
| **AES TD1008** interstitials | −18 LUFS | +0.2 LU | — | — | `[verified]` |
| **AES TD1008** virtual assistant | −18 LUFS (integrated loudness of assistant's voice preceding volume control) | n/a | — | — | `[verified]` |
| **AES TD1006.1.17-10 / AES71-2018** (OTT/OVD) | With no prior arrangement, follow regional broadcast delivery/exchange recs. Use an anchor element (e.g. dialog) for integrated measurement in lieu of full-programme. | — | "Advisable to author content with a true peak level below 0 dBFS" | — | `[verified]` |
| **AES TD1009** | Improving Dialogue Intelligibility in Media — exists, listed by TC-BOD. Not relied on here. | — | — | — | `[verified]` |

Two clauses in AES TD1008 matter more than any single number and are almost
never quoted:

> "Therefore, if operationally workable, the listener experience can be
> improved by normalizing music 2 or 3 LU higher than speech." `[verified]`

> "It is recommended that as the audio capabilities of devices advance and
> loudness metadata becomes more widely supported, industry experts and
> stakeholders reconvene to further revise this document by lowering its
> Distribution Loudness recommendations by 6 LU." `[verified]`

`[derived]` If AES eventually publishes −24 speech / −22 music, the current
−16/−14 figures were already optimistic.

### 3.2 Platforms

| Platform | What it officially says | Tag |
| --- | --- | --- |
| **Netflix** | Dialog-gated. **−27 LKFS ±2 LU**, measured per ITU-R BS.1770. Max true peak **≤ −2 dBTP**. Bed/object limiters must be at **−2.3 dBTP or lower**. Sound Mix Specifications v1.6. | `[verified]` |
| **Spotify** | Normalizes to **−14 dB LUFS per ITU 1770** at playback. Modes: Loud / Normal **−14** / Quiet **−19**. The web player and some third-party devices bypass normalization. | `[verified]` |
| **Apple Music** | Sound Check "adjusts the loudness between different songs to play at the same volume." **No number published.** The −16 LUFS figure is widely reported and almost certainly correct in practice, but Apple has never published it. | `[verified]` for the absence, `[reported]` for −16 |
| **YouTube** | **No LUFS/LKFS target published anywhere.** See §4.1. | `[verified]` for the absence |
| **Apple Podcasts** | The RSS / podcast technical-requirements page contains **no loudness spec**. | `[verified]` for the absence |
| **TikTok / Instagram / Meta** | No official loudness target found in platform documentation. | `[rumoured]` for any number |

Note the direction-of-travel: Apple Music quiet (−16), YouTube quiet, Spotify
loud (−14), Netflix dialog-anchored and much quieter (−27). A single master
cannot be correct for all of them. See §9 for why −16 LUFS is the right choice
for *this* project despite YouTube.

## 4. The folklore corrections

### 4.1 YouTube: the −14 LUFS claim

**Verdict: `[rumoured]`. YouTube publishes no loudness target, and I could not
find one in any official source.**

What I checked, and what came back:

| Official source | Result |
| --- | --- |
| `support.google.com/youtube` full-text search for "LUFS", "loudness normalization", "audio volume normalization", "volume normalization" | No article containing "LUFS" or "LKFS" |
| **Video & audio quality enhancements** (`answer/16619284`) | **Contains no number, but is decisive on behaviour** — see below |
| Video and audio formatting specifications (`answer/4603579`) | No loudness content; bitrate/codec only (MPEG-2 Layer II or AC-3 ≥128 kbps; AAC ≥128 kbps) |
| Fix poor audio quality (`answer/6082335`) | Music-label partners only; file-format advice |
| Stream 5.1 surround sound audio on YouTube (`answer/13440750`) | AAC/AC-3/EAC3, 48 kHz, 384 kbps; no loudness |
| YouTube performance FAQ (`answer/141805`) | Discovery/ranking only |

What the official enhancement page *does* say, verbatim `[verified]`:

> "To improve the viewing experience, YouTube may automatically apply visual
> and/or audio enhancements to your content... Audio enhancements can help
> improve a viewer's audio experience. These enhancements may include:
> Automatic adjustments to volume levels and sound mixing; A more comfortable
> listening experience with **Stable volume**, an enhancement that balances the
> range between quiet and loud parts of a video, reducing variations in sound;
> **Voice boost**, which makes it easier to hear dialogue by reducing background
> sound and highlighting speech."

> "Viewers can manage video audio on their device including turning Stable
> volume or Voice boost on or off in their player settings."

That is the whole of the official story, and it is more useful than the folklore:

- YouTube **does** process your audio volume. It is not a raw passthrough.
- The processing is **viewer-toggleable**. Your mix is not the only thing
  heard; the listener may add dynamic-range reduction and dialogue emphasis on
  top of it. `[verified]`
- YouTube ships a **"Voice boost"** feature whose stated function is "reducing
  background sound and highlighting speech." `[verified]` If you ship an
  aggressively denoised mix, a viewer with Voice boost on is hearing your
  processing applied twice — once by you, once by the platform.

Consequences for this project, `[derived]`:

1. Mastering **hotter** than roughly −14 LUFS buys nothing except that YouTube
   will turn it down; every dB of crush is permanent and the gain is not.
2. Mastering **quieter** than the platform target costs real perceived loudness
   if the platform only attenuates. Historically, YouTube and Apple Music were
   both observed to attenuate only and not boost. `[reported]`
3. Because platform normalisation is *dynamic* (Stable volume), a mix with a
   high loudness range gets squashed on playback. Delivering a mix that already
   has controlled dynamic range means Stable volume does almost nothing — which
   is the safest place to be. `[derived]`

So the practical answer is not "hit −14". It is: **do not exceed −16 LUFS, and
do not exceed −1.5 dBTP.** That is the target this project should use, and the
justification is in §9, not in a YouTube spec that does not exist.

### 4.2 Netflix: −31 is a Dolby dialnorm artefact

**Verdict: the −31 LKFS figure is `[rumoured]` and wrong as a Netflix spec.**

Netflix's Sound Mix Specifications v1.6 specify dialog-gated **−27 LKFS ±2
LU** with max true peak **≤ −2 dBTP**, and bed/object limiters at **−2.3 dBTP
or lower**. `[verified]`

−31 LKFS is the reference level used by the **Dolby Digital / AC-3 (and AAC)
decoder's Dialogue Normalization (dialnorm) metadata**, not a delivery spec.
It is what a decoder compares a stream's dialog level against in order to
apply gain on playback. Content authored *for* a platform that reads dialnorm
is often delivered much louder than −31 and gets turned down by the decoder
instead. Quoting "Netflix −31" conflates a codec's playback normalisation
reference with a distributor's mixing requirement.

### 4.3 Podcasts: −16 / −19

**Verdict: `[derived]` from AES TD1008, not a platform rule.**

- AES TD1008 assorted-with-measurable-speech target: **−18 LUFS, +1 LU upper
  tolerance**, so the acceptable band is **−18 to −17 LUFS**. `[verified]`
- Where speech is not measurable, the tolerance widens to **+2 LU** and the
  target becomes format-specific (Table 2): News/Talk **−18**, Mixed **−17**,
  Pop **−16**, Drama **−18**, Sports **−17**. `[verified]`
- **−16 LUFS** for a stereo podcast is therefore the top of the mixed-format
  band, not a distinct standard. `[derived]`
- **−19 LUFS** for mono is a loudness-matching argument: BS.1770 K-weighting
  and the channel-summing behaviour mean a mono file reads lower on a meter
  than the same content summed to stereo, so mono material needs a negative
  offset to *sound* equally loud. That mechanism is real; the exact −19 figure
  is `[reported]`, and note that ffmpeg's `loudnorm` has a dedicated
  `dual_mono` option for exactly this compensation. `[verified]` (option
  existence)

Since this project ships **stereo video**, the podcast numbers are not
operative anyway. See §9.

## 5. What the evidence actually says about voice enhancement

### 5.1 Peak benchmark scores — and why they do not transfer

| Model | Metric | Score | Test set | Tag |
| --- | --- | --- | --- | --- |
| ZipEnhancer (S) | WB-PESQ **3.69** | DNS Challenge 2020 official test set, **without reverberation** | `[verified]` |
| MP-SENet | WB-PESQ **3.60** | same | `[verified]` |
| FRCRN | covered in the same comparison | same | `[verified]` |
| DeepFilterNet | PESQ 2.81 / CSIG 4.14 / CBAK 3.31 / COVL 3.46 / STOI 0.942 | Voicebank+DEMAND test set | `[verified]` |
| DeepFilterNet2 | PESQ 3.08 / CSIG 4.30 / CBAK 3.40 / COVL 3.70 / STOI 0.943 | same | `[verified]` |
| DeepFilterNet3 | PESQ 3.17 / CSIG 4.34 / CBAK 3.61 / COVL 3.77 / STOI 0.944 | same | `[verified]` |
| DeepFilterNet RTF | 0.19 single-threaded notebook CPU (DF3); 0.04 on a Core-i5 (DF2) | — | `[verified]` |

These are good numbers **on speech mixed with noise, where a clean speech
reference exists**. That is the entire qualification, and the challenge
organisers said it first:

> "While the performance is good on the synthetic test set, often the model
> performance degrades significantly on real recordings." — INTERSPEECH 2020
> DNS Challenge paper `[verified]`

> "Testing the developed models on the synthetic test set gives a heuristic on
> model performance, but it is not enough to ensure good performance when
> deployed in real-world conditions." `[verified]`

A 2025 comparative evaluation on genuinely real-world corpora makes the
magnitude concrete:

| Dataset | Condition | Result | Tag |
| --- | --- | --- | --- |
| VPQAD | Real-world: cafeterias, laboratories, 50 participants | CMGAN PESQ **1.00 → 1.46** | `[verified]` |
| VPQAD | real-world | CMGAN average **SNR reduction 63.52 %** | `[verified]` |
| VPQAD | real-world | U-Net PESQ **1.36** | `[verified]` |
| SpEAR | synthetic | U-Net PESQ **1.14** | `[verified]` |
| Clarkson | real schools, children, non-soundproof rooms | PESQ averages **1.76–3.28** depending on subset | `[verified]` |

A PESQ of 1.46 out of 4.5 is *bad*. The paper's own reading of why:

> "aggressive noise reduction may introduce artifacts that affect the
> naturalness of the enhanced speech, making it less suitable for applications
> prioritizing the listening experience." `[verified]`

### 5.2 The artefact threshold question

**There is no published dB threshold at which enhancement "sounds robotic."**
No standard body, and no paper I found, defines one. Anyone quoting you a
number is quoting folklore. `[verified]` for the absence.

What *does* exist is the mechanism, and it is consistent across every source:

**Spectral subtraction and spectral gating leave isolated spectral peaks behind
— "musical noise."** `[verified]` This is the canonical artefact of the whole
class, and it is a *time-frequency local* artefact, not a level threshold. The
trading relationship is stated directly in the literature:

> "the annoying artifact in spectral subtraction is the universal phenomenon
> need to be solved. A lot of modifications of the basic suppression rules have
> been proposed to solve the musical noise, but these techniques only reduce
> the musical noise partly at the sacrifice of the audible clearness." —
> INTERSPEECH 2007 `[verified]`

In practice the "robotic" threshold is reached when **three conditions line
up**, and any one of them alone is enough to make it obvious `[derived]`:

1. **Wideband, aperiodic content is present** — wind, water, footsteps, cloth,
   room reverb. Enhancement attenuates it per-bin, and the residual is
   discontinuous, which the ear reads as synthetic.
2. **The signal has no speech in it for long stretches.** A model has nothing
   to preserve, so it applies maximum suppression to everything.
3. **Low SNR.** At low SNR the model cannot distinguish noise from speech, so
   it removes speech.

The commercial tools confirm the trade-off in their own documentation, which is
the most honest statement of it available:

iZotope RX 11 **Dialogue Isolate** `[verified]`:

> "Higher values will instruct the separation algorithm to broadly define what
> it categorizes as noise and reverb in the input signal. This can result in
> more significant noise reduction, **at the cost of introducing artifacts and
> potentially reducing dialogue clarity**."

> "Lower values will instruct the separation algorithm to narrowly define what
> it categorizes as noise and reverb... This can result in more noise being
> included in the rendered signal, but **can help to maintain dialogue
> clarity**."

Adobe Podcast **Enhance Speech** `[verified]`:

> "The results often depend on the speaker's audibility and the amount of
> background noise."

> "But no setup can completely remove unwanted artifacts or noises from your
> recordings."

So every commercial product ships a strength control whose documented meaning
is "more noise removal buys more artefacts and less dialogue." **That control
is the artefact threshold.** There is no dB value; the vendor's slider is the
answer, and it exists because the trade cannot be expressed as a level.

### 5.3 Does enhancement work on room tone, wind and street noise?

**Mixed, and specifically worse than the folklore suggests.**

**Against it — wind and room tone:**

- Speech enhancement models are trained to *preserve speech* and *suppress
  everything else*. Room tone and wind are exactly "everything else." `[derived]`
- iZotope RX ships **De-wind** as a *separate module* from **Dialogue Isolate**
  `[verified]`. If the speech model handled wind, De-wind would be redundant.
  RX's Dialogue Isolate documentation explicitly names what it *is* for:
  "constant or non-stationary background noise, such as hiss, crowds, traffic,
  footsteps, weather, or other noise with highly variable characteristics"
  `[verified]` — weather and traffic are in scope, but the documentation
  recommends a *different* module for the stationary cases:
  > "For stationary noise, such as hiss, buzz, line noise, etc., Dialogue
  > Isolate should produce good results, **but we also suggest trying the
  > Spectral De-noise module** if you are having trouble achieving acceptable
  > results with Dialogue Isolate." `[verified]`
- On pure ambience with no speech, a speech model has no anchor. `[derived]`

**For it — traffic and street:**

- Traffic *is* in the documented target list, and the DNS corpus explicitly
  includes traffic-type noise and real open-office/conference-room recordings
  collected on two different microphones. `[verified]`
- Adobe explicitly sells "Remove background noise" for voiceovers, and
  YouTube ships a "Voice boost" feature doing the same thing. `[verified]`
- **But the measured effect on real out-of-distribution recordings is small**:
  PESQ 1.46 on VPQAD, and a 63.52 % *reduction* in SNR. `[verified]`

**The verdict for this project:** `[derived]`

| Content | Enhancement verdict |
| --- | --- |
| Narration / TTS voice over silence | **Strongly positive.** This is the training distribution. |
| Narration over travel ambience (distant chatter, street) | **Positive if restrained.** Set the strength low; the noise-reduction ceiling buys you nothing you can hear. |
| Narration over wind | **Negative.** Wind is broadband and aperiodic; you get musical noise, and the wind gusts get gated on and off. Use spectral denoising, or better, a high-pass and leave it. |
| Room tone with no narration (a segment that is only ambience) | **Actively harmful.** Maximum suppression, nothing preserved, pure artefact. **Do not run it.** |

That last row is the decisive one for a travel recap: this project has a large
fraction of shots that are ambience-only (the repo already notes Live Photo
room sound "measured around −37 dB mean"). Running a speech enhancer over an
ambience-only shot is the single highest-artefact-risk action available.

### 5.4 What enhancement does to music beds

**It destroys them, and it is designed to.**

The training distribution problem is documented explicitly `[verified]`:

> "While subsets of large-scale SE datasets include singing voice as part of the
> target vocal signals, the associated interference conditions typically remain
> dominated by acoustic distortions such as environmental noise or
> reverberation, **rather than structured musical accompaniment**."
> — *Teaching Speech Enhancement Models to Sing*, 2026

> "the background in singing recordings is often **harmonically correlated
> music, which poses a different challenge than the typically uncorrelated
> background noise in speech scenarios**." — SingVERSE, 2025 `[verified]`

`[derived]` A music bed is harmonic, correlated with itself, and shares
frequency content with voice (both are "the harmonic thing in the midrange").
A speech enhancer's decision boundary for "speech" will land on parts of the
music. The result is not a subtle EQ change; it is holes, warble, and
pumping as the mask flickers between syllables.

This is why the commercial products all ship **separate** music handling rather
than folding it into the enhancer: iZotope RX has a distinct **Music Rebalance**
module with its own neural stem separation `[verified]`; Adobe Podcast markets
"**Adjust speech, music, and ambience** for a more natural sound" as three
independent controls `[verified]`; YouTube ships "Voice boost" as a *viewer-
toggleable separate feature* from "Stable volume" `[verified]`.

`[derived]` **The architectural conclusion, which is also the practical one:**
every serious product treats dialogue / music / ambience as three separate
stems, because no single model can treat them with one policy. Any pipeline
that runs a speech enhancer over a mixed voice+music bus is fighting the
design of every commercial product in the space.

## 6. Music ducking

### 6.1 How much

The only broadcaster-published number I found, and it is primary `[verified]`:

> **BBC Best Practice Guide — Sound Mixing v1.0.1 (2018), §3 Music:**
> "BBC research has demonstrated that reducing music levels a little in the mix
> allows people across the audience demographic to hear dialogue better,
> including those with certain types of hearing loss. **When the final mix is
> complete the BBC recommends taking the music down 4db.**"

And the same guide's diagnosis of *why* under-level dialogue is worse than a
hot music cue `[verified]`:

> "If the dialogue level is lower the viewer turns up the volume and then when
> a loud section of the mix happens (usually music) it becomes unbearably
> loud, so the viewer turns the volume down. This goes on throughout the
> programme and this is what we will call **'volume surfing'**."

And from the research behind it — BBC R&D WHP185 `[verified]`, which is the
actual hearing-loss study, not advice:

> "lowering the level of the left and right channels by **6dB** gave a
> statistically significant improvement in reported dialogue clarity for
> hearing impaired listeners"

`[derived]` Reconciling these: **4 dB is the whole-mix music trim; 6 dB is the
measured intelligibility improvement from lowering *background* generally.**
They are the same physics at different scopes, and both point the same way.

### 6.2 The music-to-speech relationship

`[derived]`, from the verified AES TD1008 clause that normalising music 2–3 LU
higher than speech improves the listener experience:

| Programme element | Distribution loudness | Relation |
| --- | --- | --- |
| Speech / narration | −18 LUFS (AES TD1008) | anchor |
| Music bed, programme | −15 to −16 LUFS | **+2 to +3 LU** above speech |
| Interstitial / sting | −18 LUFS | same as speech, because it is short and standalone |

`[verified]` The historical analogue is in EBU Technical Review 297
(Spikofski & Klar, 2004), which reports the established DVB practice of
levelling speech to 0 dB and music "to between **−8 dB and −4 dB**" for
speech-led broadcast. So 4–8 dB under the dialogue peak, in the pre-LUFS era,
is the long-standing broadcast answer.

### 6.3 Ducking depth, attack, release

There is **no standard** for these. What exists is tool parameter ranges and
the physics of pumping.

The mechanism, from ffmpeg's own `acompressor` documentation `[verified]`:

> "If a signal is compressed too much it may sound dull or 'dead' afterwards
> or **it may start to 'pump'** (which could be a powerful effect but can also
> destroy a track completely)."

Pumping in a ducking context has a specific cause: **the release is shorter
than the gap between syllables.** The bed comes back up during the consonant
between words, so you hear a rhythmic amplitude modulation on the music locked
to speech prosody, not to the music's own tempo. `[derived]`

The published range that matters — ffmpeg `sidechaincompress` parameter ranges,
all `[verified]` from the FFmpeg Filters documentation §8.105:

| Parameter | Range | Default |
| --- | --- | --- |
| `level_in` | 0.015625 – 64 | 1 |
| `threshold` | 0.00097563 – 1 (linear amplitude) | 0.125 |
| `ratio` | 1 – 20 | 2 |
| `attack` | 0.01 – 2000 ms | 20 ms |
| `release` | 0.01 – 9000 ms | 250 ms |
| `makeup` | 1 – 64 | 1 |
| `knee` | 1 – 8 | 2.82843 |

`acompressor` §8.2 has identical ranges plus `detection` (`rms` default, `peak`
optional), `link` (`average` default, `max` optional), and `mix` (0–1, default
1).

`[derived]` Recommended ducking parameters, from those ranges plus the pumping
physics above:

| Parameter | Recommend | Rationale |
| --- | --- | --- |
| `threshold` | set from the **narration stem's own** loudness, not a fixed constant | A fixed threshold means the duck depth depends on how hot the TTS happened to be this run |
| `ratio` | **6:1 – 10:1** | Below ~4:1 the depth varies audibly with speech level; above ~12:1 the transition from ducked to un-ducked becomes a step |
| `attack` | **10 – 30 ms** | Long enough that the bed does not move on the consonant, short enough that it is already down under the first vowel |
| `release` | **300 – 600 ms** | **This is the pumping control.** Must exceed the median inter-syllable gap. Below ~150 ms you pump; above ~800 ms the bed lags the last word audibly |
| `knee` | 4 – 8 (soft) | A hard knee on music is an audible step; soft knee at high ratio is indistinguishable |
| `makeup` | 1 (off) | Ducking and gain compensation in the same filter is how you accidentally end up louder than you started |

**How many dB of reduction is right?** `[derived]` With `ratio=6:1` and a
threshold set at the narration's short-term loudness, a speech crest 12 dB over
threshold produces 10 dB of reduction. With a soft knee the effective range is
roughly **6 – 12 dB**, and **that is the right answer**: it is deep enough to
keep the bed out of the way of the voice, shallow enough that the bed does not
disappear between words. Going to 20 dB+ does not sound better; it sounds like
the music is being switched off.

### 6.4 The anti-pumping technique that matters more than the numbers

`[derived]`, and this is the technique that separates a scripted pipeline from
a live DJ rig:

**Ducking on a gate, not on a live compressor.** Because this pipeline
generates the narration, it *knows where the speech is*. It does not need to
detect it in real time. So:

1. Run a gate on the narration stem to get speech regions.
2. Build a control signal that is a **binary or slow-ramped gate**, not the raw
   narration waveform.
3. Feed that to the sidechain.

The benefit is that the music can only move between **known** speech boundaries,
never mid-phrase, so pumping is structurally impossible rather than merely
tuned away. The cost is that gate thresholds have to be right.

`[verified]` ffmpeg provides the primitives: `silenceremove` (§8.108) with
`start_threshold` / `start_duration` / `start_silence` for region detection, and
`asendcmd` (used in the `afftdn` §8.23 example) for command-driven processing
at exact times.

A practical simplification for this repo: `[derived]` **if the narration is
synthesised, you already know the phrase boundaries.** Emit a control signal
directly from the TTS segment list, one duck envelope per phrase, and skip
detection entirely.

## 7. Music fitting

### 7.1 Time-stretch quality

| Tool | Range / options | Tag |
| --- | --- | --- |
| **ffmpeg `atempo`** (§8.65) | Tempo must be in **[0.5, 100.0]**. "**tempo greater than 2 will skip some samples rather than blend them in.** If for any reason this is a concern it is always possible to daisy-chain several instances of atempo." | `[verified]` |
| **ffmpeg `rubberband`** (§8.104) | Requires `--enable-librubberband`. Parameters: `tempo`, `pitch`, `transients` (crisp/mixed/smooth), `detector` (compound/percussive/soft), `phase` (laminar/independent), `window` (standard/short/long). | `[verified]` |
| **ffmpeg `asetrate`** (§8.53) | Resample-based pitch/tempo. | `[verified]` |
| **Premiere Remix** | "AI-powered Remix… Retime your music to match the length of your edited video." Adobe help last updated 2026-08-18. | `[verified]` |

`[derived]` The `atempo` caveat is the important one and it is the *reason*
`rubberband` exists. Phase-vocoder time-stretching degrades transients: kick
drums and hats get smeared, and a travel recap with cuts on the beat sounds
worse after a 1.9× naive stretch than before. `transients=crisp` plus
`phase=independent` are the two settings that address this specifically.

`[derived]` Practical guidance:

- Keep `|tempo − 1| < 0.10` and `atempo` is fine. This is the "generate a bed
  at approximately the right length" case.
- Beyond ~10 % stretch, either enable librubberband, or **do not stretch at all**
  — generate a bed near the target length and crossfade. Stretching is the tool
  for the last 3 %, not the first 30 %.

### 7.2 Source separation for music beds

If you need to strip vocals or rebalance an existing bed `[verified]`:

| Model | Extra data | Mean SDR (dB) | Drums | Bass | Other | Vocals | Tag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **HT Demucs** | none | **7.52** | 7.94 | 8.48 | 5.72 | 7.93 | `[verified]` |
| **Hybrid Demucs** | none | 7.64 | 8.12 | 8.43 | 5.65 | 8.35 | `[verified]` |
| **HT Demucs** | 800 songs | **9.00** | 10.08 | 10.39 | 6.32 | 9.20 | `[verified]` |
| Hybrid Demucs | 800 songs | 8.34 | 9.31 | 9.13 | 6.18 | 8.75 | `[verified]` |
| BSRNN | none | 8.24 | 9.01 | 7.22 | 6.70 | 10.01 | `[verified]` |
| Spleeter | 2500 songs | 5.91 | 6.71 | 5.51 | 4.55 | 6.86 | `[verified]` |
| D3Net | 1500 songs | 6.68 | 7.36 | 6.20 | 5.37 | 7.80 | `[verified]` |
| SCNet | none | 9.00 | 10.51 | 8.82 | 6.76 | 9.89 | `[verified]` |
| SCNet-large | none | 9.69 | 10.98 | 9.49 | 7.44 | 10.86 | `[verified]` |
| Demucs (original paper) | none | 6.3 | — | — | — | — | `[verified]` |
| Demucs (original paper) | 150 songs | 6.8 | — | — | — | — | `[verified]` |

(MUSDB18-HQ, BSSEval v4 / `museval` SDR. 0 dB SDR = no improvement over the
mixture.) The original paper notes separation of 1 minute of 44.1 kHz audio takes
0.8 s on a V100. `[verified]`

`[derived]` **This is a strong argument against Demucs in this repo**, for two
reasons that have nothing to do with quality:

1. **7.5 dB SDR is not clean.** That is audible artefact on a bed that is
   supposed to sit 10 dB under narration. A separated bed will sound worse than
   an unseparated one played quietly.
2. It is a PyTorch model with weights to download. The repo's dependency
   contract is numpy/Pillow/soundfile, offline, no model downloads at ship
   time.

If you need thinner music, **generate thinner music** rather than separating
thicker music.

### 7.3 Beat detection accuracy

| Tool | Architecture / detail | Accuracy | Tag |
| --- | --- | --- | --- |
| **librosa `beat.beat_track`** | Ellis dynamic-programming beat tracker. Three stages: (1) measure onset strength, (2) estimate tempo from onset correlation, (3) pick peaks in onset strength consistent with the estimated tempo. `tightness` default **100**, `trim` default **True**, `start_bpm` default **120.0**, `hop_length` default **512**. | Not published here | `[verified]` |
| **madmom `DBNDownBeatTrackingProcessor`** | Recurrent network emitting downbeat activations, decoded by a Dynamic Bayesian Network as an HMM over states `(bar, beat)`. Has a `beats_per_bar` parameter. | Not published here | `[verified]` |
| **Essentia `BeatTrackerMultiFeature`** | Multi-feature onset detection (beat emphasis function, spectral flux via modified information gain, 2048/512), candidates by TempoTapDegara, selection by TempoTapMaxAgreement. **Requires 44 100 Hz input.** `minTempo` range 40–180 (default 40), `maxTempo` range 60–250 (default 208), confidence output range [0, 5.32]. | **confidence in (1.5, 3.5] → "good confidence, accuracy around 80 % in AMLt measure"** | `[verified]` |
| **madmom `evaluation.beats`** | Metrics: F-measure (default window **0.07 s**), P-score, Cemgil accuracy (default sigma **0.04 s**), Goto accuracy (default threshold **0.175**), information gain. | — | `[verified]` |

`[verified]` madmom's own documentation flags a trap that applies to any
comparison you read: *"beat detections and annotations are not quantised before
being evaluated with F-measure... Hence these evaluation functions DO NOT report
the exact same results/scores"* as Goto & Muraoka's reference implementation.

`[derived]` The 80 % AMLt figure is the number to hold in mind. A beat grid is
**right roughly four times out of five**, and the failure mode is a
systematically doubled or halved tempo rather than random scatter. So:

- **80 % of cuts landing on a beat sounds intentional. 100 % would sound
  mechanical.**
- Never trust a single global tempo estimate. `librosa` and Essentia both
  support per-frame or per-region tempo, and a global BPM across a
  genre-crossing bed will be wrong for most of it.
- Guard the case where detection fails outright: `librosa` returns **0 BPM and
  an empty beat list** if no onset strength is detected, so `phase` and
  `stretch` must have a defined fallback. `[verified]` the return value;
  `[derived]` the requirement.

## 8. Tool comparison tables

### 8.1 Voice enhancement

| Tool | Method | Measured evidence | Offline / no-download | Tag |
| --- | --- | --- | --- | --- |
| **ffmpeg `afftdn`** (§8.23) | FFT spectral denoising. `nr` 0.01–97 dB (default **12**), `nf` −80…−20 dB (default **−50**), `rf` residual floor −80…−20 dB (default **−38**), `ad` adaptivity 0–1 (default 0.5), `tn` track noise floor, `tr` track residual, `nl` noise link (none/min/max/average, default min), `bm` band multiplier 0.2–5 (default 1.25), `sn` sample-noise capture via `asendcmd`, **`gs` gain_smooth 0–50 (default 0)** — "Useful to reduce random music noise artefacts." | Classical DSP, no ML benchmark | **Yes. Zero new dependencies.** | `[verified]` |
| **ffmpeg `anlmdn`** (§8.54) | Non-negative log-spectral-matrix denoiser. | — | Yes | `[verified]` |
| **ffmpeg `arnndn`** (§8.50) | RNNoise via ONNX. **Requires a model file** (`model` is mandatory). `mix` −1…1 (default 1); negative values let you hear the removed noise. | RNNoise lineage | No — needs a model | `[verified]` |
| **ffmpeg `highpass`/`lowpass`/`adeclick`/`deesser`/`silenceremove`** | Filtering, not enhancement | — | Yes | `[verified]` |
| **noisereduce** (Python) | Spectral gating. Stationary and non-stationary modes. v3 adds a PyTorch implementation; documented as markedly faster. | — | Adds a dependency | `[verified]` |
| **DeepFilterNet 1/2/3** | ERB-scaled spectral-envelope gains + complex deep filtering, 48 kHz, 20 ms windows, 10 ms hop, 40 ms total latency, 5-tap frequency-domain filter below 4.8 kHz. RTF 0.19 (DF3, 1 thread CPU), 0.04 (DF2). | PESQ 2.81 / 3.08 / 3.17 on Voicebank+DEMAND | No — PyTorch + models | `[verified]` |
| **ZipEnhancer / MP-SENet / FRCRN** | SOTA benchmark models | WB-PESQ 3.69 / 3.60 on DNS 2020 official test set, no reverb | No | `[verified]` |
| **Resemble Enhance** | Denoiser + enhancer, MIT licence, trained at 44.1 kHz. Low commit activity. | — | No | `[verified]` |
| **iZotope RX 11 Dialogue Isolate** | Neural separation into dialogue / noise / reverb, three independent gain controls; Good-Real-time and Best-Offline modes; reverb removal added in RX 11. | Vendor-documented strength↔artefact trade-off | No — commercial, offline-capable | `[verified]` |
| **Adobe Podcast Enhance Speech** | Cloud model. v1 and v2 both shipped; Adobe advises A/B testing a ≤30 s clip. | — | **No — cloud only.** | `[verified]` |
| **Premiere Enhance Speech** | Commercial, in-app | — | No | `[verified]` |
| **DaVinci Resolve 18.1 Voice Isolation** | Neural Engine, **Studio-only** licence tier | — | No | `[verified]` |
| **YouTube "Voice boost"** | Platform-side, viewer-toggleable | — | n/a | `[verified]` |

### 8.2 Ducking / dynamics

| Tool | Parameters | Tag |
| --- | --- | --- |
| **ffmpeg `sidechaincompress`** | `threshold` 0.00097563–1 (def 0.125), `ratio` 1–20 (def 2), `attack` 0.01–2000 ms (def 20), `release` 0.01–9000 ms (def 250), `makeup` 1–64 (def 1), `knee` 1–8 (def 2.82843), `level_in` 0.015625–64 (def 1), `mode` downward/upward. All settable as runtime **commands**. | `[verified]` |
| **ffmpeg `acompressor`** | Same ranges, plus `detection` rms/peak (def rms), `link` average/max (def average), `mix` 0–1 (def 1). | `[verified]` |
| **ffmpeg `alimiter`** | `limit` (def 1), `attack` def 5 ms, `release` def 50 ms, `asc` automatic sigma, `asc_level`, `level` **auto-level — default ENABLED, "normalizes audio back to 0dB if enabled"**. Lookahead limiter; the delay it produces equals the attack time. | `[verified]` |
| **ffmpeg `loudnorm`** | `I` −70.0…−5.0 (def **−24.0**), `LRA` 1.0–50.0 (def **7.0**), `TP` −9.0…+0.0 (def **−2.0**), `measured_I/LRA/TP/thresh`, `offset`, `linear` (def true), `dual_mono` (def false), `print_format`, `stats_file`. Single-pass and double-pass. **In dynamic mode the stream is upsampled to 192 kHz for accurate true-peak detection.** | `[verified]` |
| **ffmpeg `dynaudnorm`** | Frame length 10–8000 ms (def **500**), Gaussian window 3–301 odd (def **31**). Evens out section-to-section level *without* dynamic-range compression — "It will retain 100% of the dynamic range within each section." | `[verified]` |
| **ffmpeg `ebur128`** (§20.10) | Measurement, not processing. Exports `I`, `LRA`, `lra_low`, `lra_high`, `sample_peak`, `true_peak`. `peak=true` oversamples for true-peak accuracy (needs libswresample); `metadata=1` injects per-100 ms `lavfi.r128.*` keys; `dualmono` + `panlaw` (def **−3.01 dB**). | `[verified]` |
| **Premiere Auto Ducking / Auto Loudness** | Commercial one-click | `[verified]` |
| **BBC: take the music down 4 dB after the mix is complete** | The reference figure | `[verified]` |
| **AES TD1008: normalise music 2–3 LU higher than speech** | The reference relationship | `[verified]` |

### 8.3 Music fitting

| Tool | Fit method | Notes | Tag |
| --- | --- | --- | --- |
| **Generate at the right length** | Ask the music generator for the target duration | No artefacts, no dependency, no processing. The only method with no quality ceiling. | `[derived]` |
| **ffmpeg `atempo`** | Phase-vocoder time-stretch, range 0.5–100.0 | Above 2.0 it skips samples; chain instances | `[verified]` |
| **ffmpeg `rubberband`** | High-quality time-stretch/pitch, transient- and phase-aware | Needs `--enable-librubberband` (not in every build) | `[verified]` |
| **ffmpeg `apad` + `atrim` + `afade`** | Pad to length, trim, crossfade | The repo already does this; the fade-out is what stops a hard audio cut | `[verified]` |
| **Demucs / HTDemucs** | Neural 4-stem separation | MUSDB18-HQ SDR 7.52 (no extra data) / 9.00 (800 extra). PyTorch + weights. | `[verified]` |
| **Premiere Remix** | AI music retiming to clip length | Cloud/desktop app | `[verified]` |
| **librosa `beat.beat_track`** | Ellis DP beat tracking | ~80 % accuracy (AMLt, via Essentia's confidence calibration) | `[verified]` |
| **madmom `DBNDownBeatTrackingProcessor`** | RNN + DBN-HMM, **downbeat**-aware | `beats_per_bar` parameter; useful if you want cuts on bar 1 | `[verified]` |
| **Essentia `BeatTrackerMultiFeature`** | Multi-feature onset + TempoTap | Requires 44.1 kHz; confidence output is calibrated | `[verified]` |

## 9. What to adopt

Ranked by **quality gain ÷ dependency weight**, under this repo's constraints:
numpy/Pillow/soundfile only, ffmpeg as a system binary, offline, no model
downloads at ship time.

Dependency weight key: **0** = already available (ffmpeg / numpy / soundfile),
**1** = one pure-Python or pure-C package, **3** = heavy (PyTorch + weights).

### Adopt now — the audit of what is already there

Before adding anything, four of this repo's current settings are wrong on the
evidence above. These are the highest-value changes in the report.

**1. Per-segment `loudnorm=I=-18` on 3-second ambient clips — replace with a
whole-programme two-pass normalise. (weight 0, large gain)**

`pipeline/render.py:537` runs `loudnorm=I=-18:TP=-1.5:LRA=11` on **each
segment**, before concatenation.

Three problems `[derived]`:

- `loudnorm` with no `measured_*` inputs is **single-pass / dynamic mode**.
  The FFmpeg docs state dynamic mode upsamples to 192 kHz for true-peak
  accuracy and, unlike linear mode, it *re-gains over time* — which on a 3 s
  clip of varying ambience is exactly the pumping behaviour. `[verified]` for
  the mode semantics.
- Normalising ambience to −18 LUFS is a **~+19 dB lift** on the room tone the
  repo itself measures at "around −37 dB mean" (`pipeline/render.py:554-556`).
  Amplifying and then gating the noise floor of a room is the classic route to
  pumping hiss.
- It **destroys the loudness relationships between shots**. A quiet market and
  a loud market both arrive at −18. `[derived]`

Replace with: normalise the **assembled programme**, once, in two passes, with
`measured_I` / `measured_LRA` / `measured_TP` / `measured_thresh` fed back so
`loudnorm` takes its **linear** path. Use `print_format=json` + `stats_file` to
get the measurements. This is documented ffmpeg functionality, weight 0. `[verified]`

**2. `alimiter` is sample-peak limiting, not true-peak. (weight 0, moderate gain)**

`pipeline/render.py:658` and `:664` use `alimiter=limit=0.891:level=0` and
`alimiter=limit=0.97:level=0`.

- `limit=0.891` is **−1.001 dBFS** sample peak `[derived]`; `limit=0.97` is
  **−0.265 dBFS** sample peak `[derived]`.
- The FFmpeg `alimiter` documentation describes lookahead limiting with no
  oversampling `[verified]`, whereas `loudnorm` in dynamic mode explicitly
  upsamples to 192 kHz for true-peak accuracy `[verified]`.
- `level=0` correctly disables the auto-level that would normalise back to
  0 dB — good, and worth keeping.

`[derived]` Neither branch guarantees −1 dBTP after AAC encoding, which is what
EBU R128, ATSC A/85, AES TD1008 and Netflix all actually specify. If the
programme final is to claim a true-peak number, it must be **measured** after
encoding, not limited before it. Use `ebur128` (§20.10) in analysis mode, not
`alimiter`:

```
ffmpeg -i out/final.mp4 -af ebur128=peak=true -f null -
```

`[verified]` `ebur128` `peak` is a flag that can be cumulated: `sample` gives
`SPK`, `true` gives `TPK` and per-frame `FTPK` and "if enabled, the peak lookup
is done on an over-sampled version of the input stream for better peak
accuracy." It requires a build with libswresample. It exports `I`, `LRA`,
`lra_low`, `lra_high`, `sample_peak` and `true_peak` as read-only values, and
with `metadata=1` injects per-100 ms-frame `lavfi.r128.*` keys. It also has
`dualmono` and `panlaw` (**default −3.01 dB**), which is the same mono/stereo
compensation as `loudnorm`'s `dual_mono`.

**3. Ducking parameters are in range but should be gate-driven. (weight 0, moderate gain)**

Current: `sidechaincompress=threshold=0.035:ratio=6:attack=18:release=380:makeup=1`.

| Parameter | Current | Assessment |
| --- | --- | --- |
| `threshold=0.035` | −29.12 dBFS `[derived]` | `[derived]` A **fixed constant**. This is the fragile one — duck depth varies with how hot the TTS ran this render. Derive it from the narration stem's measured short-term loudness instead. |
| `ratio=6` | In the recommended 6–10 band (§6.3) | Keep. |
| `attack=18` | In the recommended 10–30 ms band | Keep. |
| `release=380` | In the recommended 300–600 ms band | Keep — and note the existing comment about the 380 ms flush tail is a genuine `sidechaincompress` internal-buffering behaviour, correctly handled. |
| `makeup=1` | Off | Keep. |
| `knee` | **unset → default 2.82843** | `[derived]` A near-hard knee on a music bed is an audible step at 6:1. Set 4–8. |

The upgrade: because the narration is synthesised, emit the sidechain control
signal from the phrase boundaries directly (§6.4) instead of from the raw
narration waveform. This makes pumping structurally impossible rather than
tuned away, and removes the fragile `threshold` constant. Weight 0.

**4. Add `highpass` on the narration bus. (weight 0, small-moderate gain)**

`[derived]` Phone and laptop TTS mics, and phone camera mics, put their
lowest useful content at 100 Hz and below; wind and handling noise are
everywhere below 200 Hz. A `highpass` at 80–100 Hz removes rumble that eats
headroom, that no amount of later loudness normalisation can recover, and that
competes with the low end of the music bed. Nobody misses it.

### Adopt next

| # | Change | Weight | Gain |
| --- | --- | --- | --- |
| 5 | `afftdn` with a **measured** noise profile instead of speech enhancement | **0** | High on noisy street clips; zero risk of the "no speech → maximum suppression" failure mode |
| 6 | Master limiter on the final programme at −1.5 dBTP, after encoding-aware measurement | **0** | Moderate; removes the "true peak depends on the mood" failure the current comment describes |
| 7 | `dynaudnorm` with a long frame length on the **ambience-only** segments | **0** | Moderate; evens shot-to-shot level *without* touching within-shot dynamics (documented: "retain 100% of the dynamic range within each section") |
| 8 | Enable `rubberband` in the ffmpeg build for the music bed; use `transients=crisp`, `phase=independent` when `|stretch − 1| > 0.10` | **0** (if the build has it) | Moderate; the `atempo`-above-2.0 sample-skipping caveat makes naive stretch audible |

`afftdn` detail `[verified]`: use `asendcmd` to capture a real noise profile
from a noise-only moment of the clip rather than guessing `nf`:

```
asendcmd=0.0 afftdn sn start,asendcmd=0.4 afftdn sn stop,afftdn=nr=20:nf=-40
```

and set `gs` (gain_smooth, 0–50) above 0 — the FFmpeg docs say it is there
specifically to "reduce random music noise artefacts," which is the artefact
class from §5.2. Start around `gs=3`.

### Adopt only if you are willing to pay for it

| # | Change | Weight | Verdict |
| --- | --- | --- | --- |
| 9 | DeepFilterNet / RNNoise (`arnndn`) for the **narration stem only** | **3** | Justified *only* for narration, never for ambience, never on the music bed. RTF 0.19 on one CPU thread is acceptable. Still a model download. |
| 10 | Demucs / HTDemucs for bed rebalancing | **3** | **Not recommended.** 7.5 dB SDR is audible artefact, and a separated bed under narration sounds worse than a quiet unseparated one. Generate thinner music instead. |
| 11 | iZotope RX Dialogue Isolate (Best/Offline) | **3** + commercial | The best-in-class tool for the actual job, and its documented strength↔artefact trade is exactly the honest version of what §5.2 describes. Worth a human-in-the-loop stage, not worth automating. |
| 12 | Any cloud enhancer (Adobe Podcast Enhance Speech, etc.) | breaks offline | Adobe's own docs say results "depend on the speaker's audibility and the amount of background noise," and "no setup can completely remove unwanted artifacts." Also v1/v2 both still exist, so results are not stable across versions — that alone disqualifies it from a reproducible pipeline. |

### Explicitly do not do

- **Do not chase a YouTube LUFS number.** It does not exist. Deliver at a
  conservative measured target and let the platform do what it does. `[verified]`
- **Do not run any speech enhancer on ambience-only segments.** Maximum
  suppression of a signal with nothing to preserve is the highest-artefact
  action available, and this project has a lot of ambience-only shots.
- **Do not normalise segments individually.** Normalise the programme.
- **Do not set the music bed 20 dB under the narration.** 6–12 dB is the
  range; deeper does not sound better, it sounds switched off. `[derived]`
- **Do not use beat tracking as a hard constraint.** ~80 % AMLt accuracy means
  roughly one in five placements is wrong; a mix that hits every beat exactly
  is more suspicious, not less.

### The target this project should actually adopt

`[derived]`, from the verified evidence above, and justified explicitly:

| Parameter | Value | Why this and not something else |
| --- | --- | --- |
| **Integrated programme loudness** | **−16 LUFS** | Netflix is −27 and irrelevant here. YouTube publishes nothing. Apple Music's real behaviour is reported as ≈−16 `[reported]`. −16 is the loudest target that is safe against *every* platform's downward normalisation, including YouTube's own "Stable volume" `[verified]` on that platform's behaviour. −14 would gain at most 2 LU of headroom and risk attenuation by an unspecified normaliser. |
| **Max true peak** | **−1.5 dBTP** | Stricter than Netflix's −2, inside EBU R128's −1 dBTP, and inside ATSC A/85's −2 dBTP with margin. Measured after encode. |
| **Loudness range** | Do not constrain above **LRA 9** | YouTube's Stable volume compresses whatever range it finds `[verified]`; arriving pre-compressed means it does nothing and you keep your dynamics. |
| **Music bed** | −15 to −16 LUFS programme level, 4 dB down after the mix completes | AES TD1008's "+2–3 LU over speech" `[verified]`, applied to a −16 programme, lands the bed at −13 to −14; the BBC's 4 dB post-mix trim `[verified]` takes it to −17 to −18. |
| **Ducking** | 6–10:1, attack 15–25 ms, release 350–500 ms, knee 4–8, gate-driven | §6.3, driven from known phrase boundaries. |
| **Ambience-only segments** | `dynaudnorm`, `loudnorm` off, enhancement off | §5.3. |
| **Narration segments** | `highpass` 80–100 Hz → optional `afftdn` with measured profile → programme-level two-pass `loudnorm` | §9 items 1, 4, 5. |

## 10. Sources

### Standards bodies

- EBU R 128 (v3, June 2014) — https://tech.ebu.ch/docs/r/r128.pdf
- EBU R 128 S1 (short-form content) — cited via the BBC technical specification, which requires compliance
- EBU Tech 3343 / 3344 — cited by AES TD1008
- EBU Technical Review 297, "Levelling and Loudness" (Spikofski & Klar, Jan 2004) — https://tech.ebu.ch/docs/techreview/trev_297-spikofski_klar.pdf
- ITU-R BS.1770-5 (approved 2023-11-22, in force) — https://www.itu.int/rec/R-REC-BS.1770-5-202311-I/en
- ITU-R BS.1864 — cited by ITU-R BS.2434
- ITU-R Report BS.2434, "Loudness in Internet delivery of broadcast-originated content" — https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BS.2434-2018-PDF-E.pdf
- ATSC A/85:2026-07, Annex M — https://www.atsc.org/wp-content/uploads/2026/07/A85-2026-07-Annex-M.pdf
- AES TD1004.1.15-10 — https://aes.org/wp-content/uploads/2024/01/AESTD1004_1_15_10.pdf
- AES TD1008 v3.13 (2021-09-24) — https://aes.org/wp-content/uploads/2024/01/20210924_TD1008_v3.13.pdf
- AES TD1006.1.17-10 / AES71-2018 — https://aes.org/wp-content/uploads/2024/01/AESTD1006_1_17_10.pdf
- AES TD1009, Improving Dialogue Intelligibility in Media (listed, not relied on)
- AES77-2023 (standard form of the TD1008 streaming recommendations)

### Broadcasters

- BBC Best Practice Guide — Sound Mixing v1.0.1 (2018) — https://www.bbc.com/backstage/downloads/audiomixguidelines.pdf — **the "take the music down 4db" source**
- BBC Radio Technical Specification v1.7 (2022-03-28) — https://www.bbc.co.uk/commissioning/radio/documents/technicalspecificationradiojuly2022_v01.7.pdf
- BBC Radio technical specification (web) — https://www.bbc.co.uk/commissioning/radio/making-content/technical-specification
- BBC R&D White Paper WHP185 — http://downloads.bbc.co.uk/rd/pubs/whp/whp-pdf-files/WHP185.pdf — **the 6 dB hearing-loss clarity study**
- BBC Scotland AS-11 technical delivery standards — https://downloads.bbc.co.uk/scotland/commissioning/TechnicalDeliveryStandardsBBCFile.pdf

### Platform documentation

- Netflix Sound Mix Specifications Best Practices v1.6 — https://partnerhelp.netflixstudios.com/hc/en-us/articles/360001794307-Netflix-Sound-Mix-Specifications-Best-Practices-v1-6
- Spotify loudness normalization — https://support.spotify.com/us/artists/article/loudness-normalization/
- Apple Support, "If music in Apple Music sounds quiet" (Sound Check) — https://support.apple.com/en-us/109331
- Apple Podcasts RSS technical requirements — https://podcasters.apple.com/support/823-podcast-requirements
- **YouTube Help, "Video & audio quality enhancements"** — https://support.google.com/youtube/answer/16619284 — **the decisive YouTube citation: no number, but documents volume adjustment, Stable volume, and Voice boost**
- YouTube Help, "Video and audio formatting specifications" — https://support.google.com/youtube/answer/4603579
- YouTube Help, "Fix poor audio quality" — https://support.google.com/youtube/answer/6082335
- YouTube Help, "Stream 5.1 surround sound audio on YouTube" — https://support.google.com/youtube/answer/13440750
- YouTube Help, "Disclosing use of GenAI content" (explicitly exempts "audio repair" and "voice or audio repair" from disclosure) — https://support.google.com/youtube/answer/14328491

### Tool documentation

- FFmpeg Filters Documentation — `loudnorm` §8.97, `afftdn` §8.23, `acompressor` §8.2, `sidechaincompress` §8.105, `alimiter` §8.31, `atempo` §8.65, `asetrate` §8.53, `rubberband` §8.104, `silenceremove` §8.108, `silencedetect` §8.107, `dynaudnorm` §8.85, `deesser` §8.82, `anlmdn`, `arnndn` §8.50, `adeclick`, `highpass` §8.94, `lowpass` §8.98, `volume` §8.120, `volumedetect` §8.121, `ebur128` §20.10 — https://ffmpeg.org/ffmpeg-filters.html
- iZotope RX 11 Manual, Dialogue Isolate — https://docs.izotope.com/rx11/en/dialogue-isolate.html
- iZotope RX 9 help (module comparison, Music Rebalance) — https://s3.amazonaws.com/izotopedownloads/docs/rx9/en/print/index.html
- Adobe Podcast FAQ (Enhance Speech limitations) — https://helpx.adobe.com/podcast/adobe-podcast-faq.html
- Adobe Podcast technical requirements — https://helpx.adobe.com/podcast/technical-requirements.html
- Adobe "What is Enhance Speech?" (v1/v2 guidance) — https://podcast.adobe.com/en/guides/what-is-enhance-speech
- Adobe Premiere feature page (Remix, Enhance Speech, Auto Ducking) — https://main--cc--adobecom.aem.live/cc-shared/fragments/products/premiere/features/audio
- noisereduce (PyPI) — https://pypi.org/project/noisereduce
- Resemble Enhance — https://github.com/resemble-ai/resemble-enhance
- DeepFilterNet — https://github.com/Rikorose/DeepFilterNet

### Libraries

- librosa `beat.beat_track` / `librosa.beat` — https://librosa.org/doc/latest/api/beat.html
- madmom `evaluation.beats` (source: `FMEASURE_WINDOW = 0.07`, `CEMGIL_SIGMA = 0.04`, `GOTO_THRESHOLD = 0.175`) — https://madmom.readthedocs.io/en/v0.16/_modules/madmom/evaluation/beats.html
- Essentia `BeatTrackerMultiFeature` (confidence calibration, 44.1 kHz requirement, tempo ranges) — https://essentia.upf.edu/reference//std_BeatTrackerMultiFeature.html

### Papers

- Reddy et al., "The INTERSPEECH 2020 Deep Noise Suppression Challenge: Datasets, Subjective Testing Framework, and Challenge Results" — https://www.isca-archive.org/interspeech_2020/reddy20_interspeech.pdf
- Reddy et al., "The INTERSPEECH 2020 Deep Noise Suppression Challenge" (arXiv:2005.13981) — https://arxiv.org/pdf/2005.13981
- Schrödinger et al., "DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement" (INTERSPEECH 2023; DF1/2/3 Voicebank+DEMAND numbers) — https://www.isca-archive.org/interspeech_2023/schroter23b_interspeech.pdf
- ZipEnhancer / MP-SENet benchmark comparison (WB-PESQ 3.69 / 3.60, DNS 2020 official test set) — https://arxiv.org/pdf/2509.16979
- "A Comparative Evaluation of Deep Learning Models for Speech Enhancement in Real-World Noisy Environments" (VPQAD/SpEAR/Clarkson results) — https://arxiv.org/pdf/2506.15000
- SingVERSE, real-world singing voice enhancement benchmark — https://arxiv.org/html/2509.20969v1
- Bereuter, Plumbley, Sontacchi, "Teaching Speech Enhancement Models to Sing" (2026) — https://arxiv.org/pdf/2607.11630
- Huang & Plumbley, "An Optimal Speech Enhancement under Uncertainty on Spectral Estimation" (INTERSPEECH 2007; musical-noise artefact trade-off) — https://www.isca-archive.org/interspeech_2007/huang07_interspeech.pdf
- Défossez et al., "Demucs: Deep Extractor for Music Sources" (arXiv:1909.01174) — https://arxiv.org/abs/1909.01174
- Défossez et al., "Music Source Separation in the Waveform Domain" (arXiv:1911.13254; MUSDB SDR 6.3/6.8) — https://arxiv.org/pdf/1911.13254
- Li et al., "SCNet" (arXiv:2401.13276; the MUSDB18-HQ comparison table used for the Demucs SDR figures) — https://arxiv.org/html/2401.13276v1
- Ellis, "Beat Tracking by Dynamic Programming," J. New Music Research 36.1 (2007) — cited by librosa
- Goto & Muraoka, "Issues in evaluating beat tracking systems," IJCAI-97 — cited by madmom
- Davies, Degara & Plumbley, "Evaluating Methods for Musical Audio Beat Tracking Algorithms" — cited by madmom

### Folklore-tier (cited only as folklore, never as a source of a number)

- "LUFS Targets per Platform" 2026 cheat sheet — https://cdn.prod.website-files.com/64e8910adc5a63966a68acea/6a26c78286c3131865cdcf7e_6a26c7804f9891c35618c56b_lufs-targets-per-platform-cheatsheet.pdf — **used in this report only as the example of what folklore looks like.** Its YouTube −14, Apple Music −16, podcast −16/−19 and Netflix −27 rows are all either unverified or (for YouTube) contradicted by the absence of any official figure.
- Gearspace / productionadvice / mixertube forum threads on YouTube "Stats for nerds" volume normalisation — the source of the widely repeated "YouTube ≈ −11 to −14 LUFS" claims, based on reading YouTube's own per-video stats rather than documentation.
- MeterPlugs blog on YouTube's 2019 move to ITU BS.1770 measurement — useful colour on *when* YouTube changed measurement, not a published target.