# Fresh Internet Theory
Submitted for: [Hack NC State 2026](https://hackncstate2026.devpost.com/)
Track: Siren’s Call 

The Dead Internet is here. We built a reality shield.

**Fresh Internet Theory** is a Chrome extension that sits on top of Facebook Reels and helps people decide whether what they’re watching is likely AI-generated or not. We combine two signals:
- **Crowd intelligence** (real users voting `ai` / `not_ai`)
- **Twelve Labs video analysis** (artifact-level evidence + confidence)

The goal is simple: give people a way to sanity-check viral reels before fear, outrage, or fake scandals spread.

## Inspiration
We kept seeing the same pattern: a shocking reel goes viral, people react emotionally, and by the time someone fact-checks it, the damage is already done.  
In the **Siren’s Call** theme, that felt like the core problem: misinformation is no longer just text-based; it is increasingly visual, fast, and emotionally manipulative.

We wanted a tool that works **inside the scrolling loop itself**, not a separate fact-checking website nobody opens in time.

## Problem
People scrolling short-form video often have:
- no time for deep verification,
- no trusted signal in the UI,
- no easy way to contribute to community truth.

This creates real social harm:
- public panic from fake events,
- reputational damage from synthetic scandal videos,
- manipulation through highly believable AI-generated content.

Our solution matters because it inserts trust signals where decisions are actually made: at the reel level, in real time, during consumption.

## What it does
We didn't just “run an AI detector.”  
We took a different route: **federated human + AI consensus**, with explainability.

Our approach:
- instant on-platform overlay for every reel,
- user voting loop that continuously improves signal quality,
- forensic-style AI evidence output (not just a binary label),
- a hybrid verdict strategy that can prioritize people when crowd signal is strong.

## How we built it

![Flow chart](https://i.imgur.com/ImYS158.png)

This is fully coded and running end-to-end (localhost only)

### Extension (`extension/`)
- `content_script.js` injects an overlay on Facebook Reels.
- It fetches current reel state and shows:
  - AI / Not-AI vote counts
  - verdict badge
  - evidence snippets when available
- Users can vote `ai` or `not_ai` directly in the overlay.

### Backend (`backend/`, FastAPI + Valkey)
The extension uses:
- `GET /reels/{reel_id}`: fetch reel state, votes, and analysis
- `POST /reels/{reel_id}/vote`: save user vote
- `POST /reels/{reel_id}/post`: enqueue analysis for unknown reel

### Analysis Worker (`12labs/`)
- `reel_downloader.py`: downloads reel video
- `upload.py`: uploads/indexes video in Twelve Labs
- `worker.py`: runs AI analysis, extracts evidence, writes results back to datastore

### Verdict Algorithm (Crowd + AI Analysis)

Right now, we persist raw AI analysis for auditability.  
For final verdicting, our intended/active policy is a **crowd-first, reliability-weighted hybrid**:

Let:
- $v_i \in \{0,1\}$ be user vote (`1 = ai`, `0 = not_ai`)
- $w_i$ be reliability weight for voter $i$
- $p_a$ be AI probability (`confidence/100`)

Human consensus score:

$$
p_h = \frac{\sum_i w_i v_i}{\sum_i w_i}
$$

Effective human volume:

$$
n_{\text{eff}} = \sum_i w_i
$$

Consensus margin:

$$
m = |p_h - 0.5|
$$

Decision:

$$
p_f =
\begin{cases}
p_h, & n_{\text{eff}} \ge N_{\min} \ \wedge\ m \ge \delta \\
\alpha(n_{\text{eff}})\,p_a + (1-\alpha(n_{\text{eff}}))\,p_h, & \text{otherwise}
\end{cases}
$$

with $\alpha(n_{\text{eff}})=e^{-n_{\text{eff}}/\kappa}$.

Interpretation:
- if there are **enough reliable human votes with clear agreement**, trust humans;
- if not, rely more on AI;
- as crowd signal grows, AI influence decays.

This is aligned with established crowdsourcing literature: [model annotator reliability](https://arxiv.org/abs/2409.12218), avoid blind majority vote, and use additional labels strategically when uncertainty is high.

### Result format
The analysis includes:
- `is_likely_ai_generated`
- `confidence`
- `key_evidence` with categories/timestamps/rationales
- `user_prior_assessment`

### Current demo status
- End-to-end flow works locally.
- AI analysis matched majority user feedback **~95%** of the time in our testing set.

## Challenges we ran into
- Balancing weights between user votes and AI confidence.
- Making UX feel real-time while analysis pipelines are asynchronous.
- Handling Twelve Labs API rate limits during bursty workloads.

## What we learned
- Detection trust is more about **explanations** than raw scores.
- Queue-based systems are essential for AI-heavy moderation workloads.
- Crowd signal quality improves dramatically when you model reliability, not just vote count.

## What's next for Fresh Internet Theory
This can scale with more engineering:
- queue-based architecture already handles asynchronous processing and API rate limits,
- worker model can be horizontally scaled,
- similarity reuse in the worker can reduce repeated analysis cost for near-duplicate reels,
- datastore schema already keeps votes + analysis + timestamps in one canonical reel record.

Next production steps:
- anti-brigading controls (rate limits, reputation, device/account trust),
- identity-aware worker reliability updates,
- moderation dashboard and abuse monitoring,
- cloud deployment + observability.

## Teamwork & Presentation

- **[Ashwin Kumar](https://linkedin.com/in/ashwinkumarmv)**: backend APIs, datastore design, schema normalization
- **[Sidharth Shambu](https://linkedin.com/in/sshambu)**: Twelve Labs worker pipeline, Chrome extension integration

Collaboration setup:
- rapid WhatsApp loops,
- shared TODO lists for task ownership,
- Tailscale for seamless local environment sharing during integration.
