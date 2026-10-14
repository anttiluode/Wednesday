# Interpretation and threats to validity

1. **Nothing here is trained.** The writer is a known-basis gauge-fixed construction, and the reader chooses a known face. Confidence-directed repetition is deterministic hand-coded active sampling.
2. **The gauge intervention is mathematical.** Independent edge noises do *not* cancel as a gauge, and a physical device would have readout loss, nonunitarity, shot noise, phase drift, timing, and calibration issues. The ideal unitary algebra cannot prove a hardware implementation.
3. **A 13-byte exact bitset solves our chosen question perfectly.** The 224-byte phase graph uses 17.2× the inference memory to store the same 98 bits. Shared architecture code and probe path addressing are additional resources, not accounted as stored bytes.
4. **All queries know their `(channel,row,column)` addresses.** The experiment does not infer object correspondences, pixel identities, or future-query policy from raw images. Real GraphNN does, in part, learn a writer from images; Wednesday does not yet.
5. **The two-channel separation is explicit.** A and B are stored in distinct phase edge arrays. This is not selective coexistence of images within the *same* channel or automatic compression of shared scene structure.
6. **No image semantics.** The 7×7 masks are binary, not natural images, continuous color, object identities, or unforeseen geometries.
7. **Two routes are privileged.** The source is a selected face and two detector endpoints are chosen by that face, rather than two ports fixed for an entire lattice. If the reader can access raw edges, it can directly compute the same flux; the paired measurement does not create information.
8. **A naive update is intentionally wrong.** The one-edge control is only to exhibit shared-edge collateral effects and is not a state-of-the-art memory baseline.
9. **Adaptive budget differs from training a routing network.** Policies use identical pre-sampled shot arrays (common random numbers). Noise heterogeneity is fixed within each trial. Other noise laws, cost models, or more powerful uniform estimators can change the comparison.
10. **Gauge redundancy is enormous.** 112 edges - 64 vertices + 1 global degree = 49 independent face fluxes per channel. No gauge invariant compression occurs automatically by adding redundant edges. A canonical basis of 49 code bytes/channel exists, and since only two phase values are ever used, 98 *bits* are sufficient.
11. **No hardware timing/FLOPs matched.** Cost notes charge four edge traversals per two-path read and count edge-byte changes in writes, but Python runtime includes bookkeeping and learned alternatives are not compared.

## What would make the next test scientifically different?

A genuinely learned, state-dependent writer and interactive reader that observe small scenes **through a partial interface**, adapt probes after feedback, and retain identity-specific changed details without free addressing. Pair it with exact indexing (when available), a learned vector codec, a GraphNN direct decoder, and a phase-transport model. Require independent seeds, held-out structure shifts, varied noise, inference-byte counts and total measurements. That is an appropriate possible Gate 1, not a result here.
