# Wednesday — the graph remembers what a loop encloses

**[Open interactive laboratory](https://anttiluode.github.io/Wednesday/)** · [Frozen protocol](PROTOCOL.md) · [Observed caveats](LIMITATIONS.md) · [JSON receipts](results/pilot.json)

Wednesday is a falsifiable bridge between the *experience-written geometry* of [GraphNN](https://github.com/anttiluode/GraphNN) and the *paired operator interrogations / collision controls* of [NextLevel](https://github.com/anttiluode/NextLevel) and [SihtiGrapImageModel](https://github.com/anttiluode/SihtiGrapImageModel). It is **not** a new Transformer, a learned model or a new theorem. Its first job is to pin down exactly what closed-loop phase buys us.

## What you can do in the browser

- Generate separate 7×7 A and B binary worlds and write both into one two-channel, 8×8 vertex lattice of uint8 edge phases.
- Select any face. Watch its two open-path phase measurements and their closed-loop difference. Toggle vertex coordinates arbitrarily while the loop value remains the same.
- Rewrite one bit in A or B. With the compensated writer, all other loops remain identical. With the cheaper one-edge writer, an adjacent face is corrupted. Track changed faces **and actual edge-byte writes**.
- Inject independent phase shot noise, interrogate a selected face, and reconstruct both worlds with noisy probes; optionally allocate exactly 196 paired shots via uniform or confidence-directed sequential reads. The same potential shot samples are used for both policies.
- Inspect the held-out pilot's measured success and failure rates, and the much stronger 13-byte exact indexed memory.

No build or node modules needed. `index.html` is a fully client-side visualization.

## Core equation

An oriented edge stores an 8-bit modular phase `φ_uv`. The clockwise phase of a face is

```
Φ = h_top + v_right - h_bottom - v_left   (mod 256)
```

A stored bit maps to `Φ=64` (+π/2) or `Φ=192` (−π/2). Two independent paths between the same face corners provide `z_topright` and `z_leftbottom`, and the paired response `z_topright*conj(z_leftbottom)` reveals `exp(iΦ)` exactly. Under vertex rephasing `φ_uv -> φ_uv + α_v - α_u`, individual open paths change, the closed-loop phase does not. This is standard U(1) gauge algebra; we do not claim novelty for the identity.

The more interesting cost: writing one face's flux by changing its right edge also changes the next face's flux. To avoid overwriting the neighbor, the writer updates the *suffix of downward edges in that row*. One logical bit write costs **1–7 edge updates**; otherwise collateral forgetting occurs. The browser lets you see the failure.

## Pilot, seed 41, 100 trials per condition

See the full [receipts](results/pilot.json) and [protocol](PROTOCOL.md). Two task distributions (independent bits / procedural shapes) × two gauge frames (fixed / random). Each trial stores 98 facts (49 A, 49 B). For the paired-only gauge test, path noise is zero and the shared endpoint disturbance has σ=1.5 rad. For active-read testing, independent path shot-noise σ=0.85 rad times a per-cell heterogeneity factor.

| Condition, independent facts | Accuracy |
|:--|--:|
| Exact loop decode after random rephasing | 100.0% |
| One open path, shared phase disturbance | 49.5% |
| Two paths, shared disturbance, no independent noise | 100.0% |
| Noisy paired read, **196 total shots** distributed uniformly | 87.1% |
| Noisy paired read, **196 total shots** allocated by confidence | 90.4% |
| Indexed 98-bit exact table, **13 bytes** | **100.0%** |

The adaptive +3.3 percentage-point difference here is under one **engineered** shot-noise model and a **hand-coded** heuristic, not a learned interrogation policy. Under other noise distributions it may vanish or reverse. The exact indexed baseline is both smaller and more accurate. **A graph's role here is interrogation geometry and robust coordinate-invariant transport, not compression of 98 random bits.**

## Run the reproducible experiment

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --seed 41 --trials 100 --out results/pilot.json
python -m http.server 8000
```

Then open http://localhost:8000/ to interact. On GitHub Pages the static site is served directly. Python 3.10+ and NumPy only. `run.py` runs on CPU and logs four scenario/coordinate combinations with standard errors. CI repeats unit tests and a short pilot to check the code doesn't break.

## What would make a Gate 1?

1. Let a *learned* image writer change transport coefficients as in GraphNN, with fewer external addresses.
2. Ask for changed details in A without erasing B, then test four/eight history objects and structured relationship queries.
3. Train a **feedback-adaptive** interrogation policy. After the first path response, choose the next probe. Compare against equal shot/compute direct access, conventional indexed memory and vector codecs.
4. Include physically relevant edge noise, gains, non-unitarity and noise-model shifts; avoid presenting exact gauge invariance as real-device performance.

Wednesday starts by separating facts that are exact mathematical identities from effects that require data. The value is having a visible apparatus for both.
