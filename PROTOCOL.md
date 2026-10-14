# Wednesday Gate 0 — frozen after implementation, October 14, 2026

**Exploratory project.** We chose the model and controls before the recorded 100-trial sweep, but the idea and code were developed together; this is **not preregistration**. The `results/pilot.json` output is reproducible. See also `LIMITATIONS.md`.

## Question and null

Can **closed-loop phase** store two sets of independently updateable visual facts under arbitrary local U(1) coordinate transformations, without a mismatch between write and read addressing? Does a feedback-dependent reader improve accuracy when independent shot noise makes some measurements ambiguous?

- **Known mathematical property / positive control:** phase around a closed loop is invariant to vertex gauge rephasing, modulo quantization. This does not establish novel physics or novel information capacity.
- **Negative compression control:** a public 98-bit universe can be stored exactly in **13 bytes**, while our two-channel graph requires **224 bytes**. Direct indexing wins on cost and noiseless accuracy.
- **Update interference control:** changing one shared edge can change a second face's loop value; compensating updates to all rightward suffix vertical edges isolate the intended face but touch more bytes.
- **Active-read test:** under the same pre-generated noisy measurements and a total budget of 196 paired shots for 98 facts, a hand-designed confidence sampler may or may not outperform the two-shots-each uniform allocator.

## Fully specified representation

- 8×8 vertices, 8×7 rightward and 7×8 downward directed edges, **112 edges per phase channel**, two channels for A/B.
- Each edge carries one `uint8` modular-turn phase, with values 0–255 and angle `2πcode/256`; 224 **actual bytes** of inference-time memory.
- 7×7 fundamental faces per channel (49 each), one fact per face. Bit 1 corresponds to clockwise flux code 64 (+π/2); bit 0 corresponds to code 192 (−π/2).
- Clockwise face flux: `h[r,c] + v[r,c+1] - h[r+1,c] - v[r,c] (mod 256)`.
- Two *open* paths from top-left to bottom-right corners of selected face: `p1=h_top+v_right`, `p2=v_left+h_bottom`. The paired complex ratio reads `p1-p2=face flux`. **The query knows its face coordinates and memory channel.** Two paths cost four oriented edge traversals, plus readout. No single shared remote detector, no learned route selection.
- A vertex rephase adds `α_destination - α_source` to each oriented edge, modulo 256. Invariance is exact in this discrete simulation. No independent physical edge perturbations are cancelled this way.
- Writer is given the `(channel, face row, face column, new bit)` fact. It computes the existing face flux and adds the desired difference to **all downward edges of that face row at columns greater than the target column**. This is **1–7 edge writes** per fact. Changing only the immediate rightward edge is the naive one-byte ablation; for interior columns it corrupts the neighboring face. Only the compensated writer preserves other face fluxes.

## Data, measurement noise and evaluation

- Deterministic `numpy.random.default_rng(seed)` generator, starting at seed 41; 100 independent environments per scenario/condition. **Do not treat 98 facts within one environment as independent trials.**
- Two kinds of input: independent Bernoulli(0.5) 7×7 bit maps in channels A/B; and procedural rectangles with local salt. Neither the geometry nor policies learn from this distribution.
- Random vertex phases are sampled independently, once after image writes, for both channels. Each trial measures both open and paired response with shared random endpoint phase noise standard deviation 1.5 radians; **zero independent path noise** in this positive-control subtest. Gauge rephasing and shared endpoint phase cancel exactly in paired response.
- For noisy read allocation, fixed total of `98*2=196` paired shots, candidate max 6 shots per cell, independent per-path Gaussian noise `sigma=0.85 × cell_scale` radians, `cell_scale ~ Uniform(0.55,1.45)` held fixed for the trial. A fresh common endpoint term `Normal(0,1.5)` is shared across both paths of each shot; it cancels in the ratio. All potential 6 shots per cell are **pre-generated once**; both policies receive the same per-cell shot sequences, ensuring paired comparisons.
- Each item gets one initial shot. Uniform allocates the remaining 98 shots to least-sampled items (thus exactly two per item). Adaptive greedily gives the next shot to the available item with lowest `abs(sum(sin(ratio_shots)))/sqrt(number_of_shots)`, never observing ground-truth bits, capped at 6 shots per item. **This is a hand-designed uncertainty heuristic, not learned active sensing.**
- Predictions are the sign of the accumulated `sin(ratio)`; exact indexed 13B baseline always returns correct known stored bit. Report per-trial mean accuracy, trial-level standard error, correct face count, collateral changed-face counts, and edge writes. All measurements of structural facts are finite and quantized.

## Reproduction

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --seed 41 --trials 100 --sigma 0.85 --shots-per-item 2 --out results/pilot.json
python run.py --seed 97 --trials 250 --sigma 1.2 --shots-per-item 2 --out results/highnoise.json
```

`index.html` is a zero-dependency GUI for the same algebra and experiments, with individually sampled shots. It is **not bit-for-bit seeded against the Python implementation**, because it uses browser `Math.random`. Use Python for reproducible receipts. Serve locally with `python -m http.server 8000` to load its JSON receipt, or view GitHub Pages.

## Pass / fail distinction

- Mathematical controls: face loop values do not change under arbitrary `uint8` vertex rephasings; exact read recovers both channels; correctly compensated rewrite changes exactly one face; naive one-edge overwrite may corrupt another face.
- Empirical question: compare adaptive vs uniform under repeated independent trials and different noise distributions. If a learned policy is added later, it must win on **unseen** noise models and question distributions, with read budget and hidden query-access costs accounted for.
- This experiment **cannot establish** superiority to attention, learned graph networks, an indexed memory, or any general AI architecture. Memory coding, updates and geometry are hand-engineered. GraphNN's trained graph transport is a separate project.
