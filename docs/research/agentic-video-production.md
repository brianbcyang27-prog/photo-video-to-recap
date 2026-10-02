# Research notes: AI agents for high-quality video production

Written after surveying both the academic literature and the shipped open-source
systems, to decide whether this pipeline should adopt an agentic architecture.
The short answer: **not by replacing what works.** The evidence points at
keeping deterministic rendering and using agents only for the decisions that
are genuinely judgement calls.

## What the shipped systems actually do

The decisive evidence came from reading `requirements.txt` of the most mature
project found, **FireRed-OpenStoryline** (FireRedTeam, ~3.5k stars,
open-sourced 2026-02). Its render stack is:

    ffmpeg-python, moviepy, av, librosa

and its "AI" stack is:

    langchain, langchain_openai, mcp, openai, sentence-transformers, faiss

That split is the whole answer. The pixels are made by ffmpeg/moviepy/av — the
same class of deterministic render this project uses. The model is used for
*orchestration and content understanding*, not for rendering. Every other
serious project surveyed (Crayotter, velocut, deepvideo, openscene,
StoryCraft) has the same shape: a real timeline engine underneath, an LLM
driving it from above.

MAViS (arXiv 2508.08487v5) makes the same architectural point in the
generative setting: it orchestrates specialised agents across stages (script →
shot design → keyframe → animation → audio) but keeps deterministic generation
tools underneath, and explicitly designs for "capability limitations of current
generative models."

## So: keep the render deterministic. Agentise the judgement.

This project already has the right skeleton — ingest, analyse, select, arrange,
render, verify — and 186 tests proving the render is reliable. An agent that
replaces the renderer would trade a verified, reproducible film for a
non-deterministic one and would break the whole verification approach that has
been built up. The evidence does not support that trade.

Where an LLM *would* earn its place, in descending order of value:

### 1. Content understanding (highest value)
FireRed's `understand_clips` and `filter_clips` prompts have an LLM read a
caption of each clip plus an aesthetic score, then choose what to keep — but
*with a hard retention constraint* (keep > 80% of clips) and a rule to
prefer higher `aes_score`/duration among near-duplicate descriptions. This
project currently selects on metadata only (time, place, sharpness, light).
An LLM pass that reads captions and identifies near-duplicate visual
descriptions — the "a hundred near-identical beach shots" problem — is the one
place a model clearly beats arithmetic. Constraint: always keep the
deterministic score as a floor, and keep the >80% retention rule so the model
can refine but never gut the cut.

### 2. Editing-skill archiving (cheap, proven)
FireRed's most-adopted feature is "save your whole editing workflow as a
Skill." A `recap_style.json` that captures pacing, pacing law, mood-to-music
mapping, and pacing aggression — replayable across libraries — is a natural fit
for the existing `--mood` flag and costs almost nothing.

### 3. Conversational refinement (useful, heavier)
Re-cutting with natural language ("drop the harbour shots", "put the sunset
last"). FireRed does this over MCP. It is genuinely nice but it turns a
turnkey one-shot tool into a session server — a real product change, not a
quality fix. Defer.

### 4. AI transitions (do not adopt)
FireRed's own README warns AI transitions are costly, unpredictable, and
source-dependent. This project's motion system already produces clean
transitions via beat-aligned cuts; a diffusion transition between two real
photos would be slower, non-deterministic, and rarely better. Decline.

### 5. LLM-generated music prompt (marginal)
Currently a deterministic synth. Not worth an LLM dependency.

## What the research changed about the plan

Nothing was thrown away. But two real findings did redirect effort:

- The pacing work (per-shot camera-move speed, energy-reactive) came directly
  out of the "cinematic language" thread (Camera Artist, ACDC). It is
  deterministic and testable, which fits the architecture the evidence
  endorses. Implemented and verified (2.4x short-vs-long move speed).
- A length-aware A/V tail mismatch was found and, via a stashed control
  render, proven **pre-existing** (identical 0.17s offset without the pacing
  change). It is a music-grid/picture-lock issue, not a pacing regression, and
  is a separate candidate for a future fix.

## Recommendation

Adopt the hybrid the evidence shows, incrementally and behind a flag:

1. **Now (done):** deterministic pacing, verified. The right place to start.
2. **Next:** an optional `--understand` LLM pass that reads clip captions to
   flag near-duplicate groups and nudge selection — always bounded by the
   deterministic score and a hard retention floor. The model refines; it does
   not decide. This is where the money is.
3. **Later, if wanted:** `recap_style.json` skill-archived pacing presets.
4. **Skip:** LLM rendering, AI transitions, session-server chat editing — all
   trade the verified, reproducible, turnkey property this project is built on.

The governing principle: an agent should sit *above* a verified deterministic
pipeline, improving the judgement calls, never *inside* the render, where it
would cost reproducibility and testability.
