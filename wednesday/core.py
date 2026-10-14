"""Wednesday: gauge-covariant discrete U(1) transport memory.

This is a physically *inspired* graph algebra, not an optical apparatus or
trained network. The 8x8 square lattice has 112 oriented edges and 49 faces.
Two uint8 edge-phase channels (A/B) occupy 224 bytes in total.
No hidden facts are available to the reader; writer has the requested bit.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

SIDE = 8
FACES = SIDE - 1
CHANNELS = 2
EDGE_BYTES = CHANNELS * 2 * SIDE * FACES  # 224
BITS = CHANNELS * FACES * FACES            # 98
INDEXED_BYTES = (BITS + 7) // 8            # 13
BIT_0 = 192     # -pi/2
BIT_1 = 64      # +pi/2


def byte_of(bit: int) -> int:
    if bit not in (0, 1):
        raise ValueError('bit must be 0 or 1')
    return BIT_1 if bit else BIT_0


@dataclass
class Lattice:
    """Directed edge phases in uint8 turns, h rightward, v downward."""
    h: np.ndarray     # [2,8,7]
    v: np.ndarray     # [2,7,8]

    @classmethod
    def empty(cls) -> 'Lattice':
        return cls(np.zeros((CHANNELS, SIDE, FACES), dtype=np.uint8),
                   np.zeros((CHANNELS, FACES, SIDE), dtype=np.uint8))

    def copy(self) -> 'Lattice':
        return Lattice(self.h.copy(), self.v.copy())

    def bytes(self) -> bytes:
        return self.h.tobytes() + self.v.tobytes()

    def check(self):
        assert self.h.dtype == self.v.dtype == np.uint8
        assert self.h.shape == (2, 8, 7) and self.v.shape == (2, 7, 8)
        assert len(self.bytes()) == EDGE_BYTES

    def face(self, channel: int, row: int, col: int) -> int:
        """Clockwise U(1) holonomy of one face, modulo 256."""
        self._valid(channel, row, col)
        return (int(self.h[channel, row, col]) + int(self.v[channel, row, col+1])
                - int(self.h[channel, row+1, col]) - int(self.v[channel, row, col])) % 256

    @staticmethod
    def _valid(channel: int, row: int, col: int):
        if channel not in (0, 1) or not (0 <= row < FACES and 0 <= col < FACES):
            raise ValueError('outside memory faces')

    def read_bit(self, channel: int, row: int, col: int) -> int:
        return int(np.sin(self.face(channel, row, col) * (2*np.pi/256)) > 0)

    def write(self, channel: int, row: int, col: int, bit: int) -> int:
        """Flux-preserving rewrite. Modifies a suffix of downward edges.

        For a row of adjacent plaquettes, changing the right edge of one face
        also changes the left edge of its neighbor. Updating the full suffix
        is the compensating write that leaves ALL other plaquettes intact.
        Returns the number of edge bytes written (not a hidden zero-cost write).
        """
        self._valid(channel, row, col)
        desired = byte_of(bit)
        delta = (desired - self.face(channel, row, col)) % 256
        self.v[channel, row, col+1:] = (self.v[channel, row, col+1:].astype(np.int16) + delta) % 256
        return FACES - col

    def naive_write(self, channel: int, row: int, col: int, bit: int) -> int:
        """One-edge update: cheaper, but usually corrupts the next face."""
        self._valid(channel, row, col)
        delta = (byte_of(bit) - self.face(channel, row, col)) % 256
        self.v[channel, row, col+1] = (int(self.v[channel, row, col+1]) + delta) % 256
        return 1

    def write_image(self, channel: int, image: np.ndarray) -> int:
        if image.shape != (FACES, FACES) or not np.isin(image, [0, 1]).all():
            raise ValueError('expected a 7x7 binary image')
        return sum(self.write(channel, row, col, int(image[row, col]))
                   for row in range(FACES) for col in range(FACES))

    def read_image(self, channel: int) -> np.ndarray:
        return np.array([[self.read_bit(channel, r, c) for c in range(FACES)]
                         for r in range(FACES)], dtype=np.uint8)

    def rephase(self, phases: np.ndarray):
        """Vertex gauge transform W_uv -> W_uv + alpha_v - alpha_u.

        alpha is a uint8 tensor [2,8,8]. Both open paths change, while their
        relative phase and all closed holonomies are exactly invariant.
        """
        if phases.shape != (CHANNELS, SIDE, SIDE) or phases.dtype != np.uint8:
            raise ValueError('expected uint8 [2,8,8] gauge coordinates')
        h = (self.h.astype(np.int16) + phases[:,:,1:].astype(np.int16) - phases[:,:,:-1]) % 256
        v = (self.v.astype(np.int16) + phases[:,1:,:].astype(np.int16) - phases[:,:-1,:]) % 256
        self.h, self.v = h.astype(np.uint8), v.astype(np.uint8)

    def open_paths(self, ch: int, r: int, c: int) -> tuple[int, int]:
        self._valid(ch, r, c)
        return ((int(self.h[ch,r,c]) + int(self.v[ch,r,c+1])) % 256,
                (int(self.v[ch,r,c]) + int(self.h[ch,r+1,c])) % 256)

    def probe(self, ch: int, r: int, c: int, rng: np.random.Generator,
              sigma: float = 0., common_sigma: float = 0.) -> dict:
        """Two paths between opposite face corners, not fixed-port probes.

        Shot noise is independent for each path. Common endpoint phase noise
        is identical across the two readings and cancels in their difference.
        """
        a, b = self.open_paths(ch,r,c)
        shared = rng.normal(0, common_sigma)
        angle_a = 2*np.pi*a/256 + shared + rng.normal(0, sigma)
        angle_b = 2*np.pi*b/256 + shared + rng.normal(0, sigma)
        ratio = (angle_a - angle_b + np.pi) % (2*np.pi) - np.pi
        return {'single':int(np.sin(angle_a)>0),
                'paired':int(np.sin(ratio)>0),
                'ratio_angle':float(ratio),
                'single_angle':float((angle_a+np.pi)%(2*np.pi)-np.pi),
                'phase_code':self.face(ch,r,c)}


def procedural_world(rng: np.random.Generator, structured: bool = True) -> np.ndarray:
    """Two 7x7 channels. Independent or shapes + local salt."""
    if not structured:
        return rng.integers(0, 2, (2, FACES, FACES), dtype=np.uint8)
    array = np.zeros((2,FACES,FACES), dtype=np.uint8)
    for ch in range(2):
        for _ in range(2):
            top = int(rng.integers(0,5)); left = int(rng.integers(0,5))
            height = int(rng.integers(1,4)); width = int(rng.integers(1,4))
            array[ch,top:top+height,left:left+width] ^= 1
        salt = rng.random((FACES,FACES)) < .07
        array[ch] ^= salt.astype(np.uint8)
    return array


def shot_arrays(world: np.ndarray, rng: np.random.Generator,
                maxshots: int = 6, sigma: float = .85,
                common_sigma: float = 1.5, heterogeneity: bool = True):
    """Pre-generate identical potential physical shots for all policies.

    No gauge transform is needed here since paired phase cancels it exactly.
    Each cell has fixed but unobserved shot noise scale (heterogeneity).
    """
    if heterogeneity:
        scales = rng.uniform(.55, 1.45, world.shape)
    else:
        scales = np.ones_like(world, dtype=float)
    shape = world.shape + (maxshots,)
    common = rng.normal(0,common_sigma,shape)
    first = rng.normal(0,sigma,shape)*scales[...,None] + common
    second = rng.normal(0,sigma,shape)*scales[...,None] + common
    code = np.where(world==1, np.pi/2, -np.pi/2)
    angle = (code[...,None]+first-second+np.pi)%(2*np.pi)-np.pi
    return np.sin(angle).reshape(-1,maxshots), scales


def allocate_shots(sines: np.ndarray, budget: int, policy: str = 'uniform', cap: int = 6):
    """A deliberately simple confidence-directed sequential reader.

    A first shot is mandatory per item. Additional shot decisions only use
    already observed signals. The matched uniform reader alternates items.
    This is *hand-designed*, NOT a learned reinforcement-learning policy.
    """
    n, maxshots = sines.shape
    if cap > maxshots or budget < n or budget > n*cap:
        raise ValueError('invalid shot budget / cap')
    counts = np.ones(n, dtype=np.int32)
    totals = sines[:,0].copy()
    for step in range(budget-n):
        if policy == 'uniform':
            candidates = np.flatnonzero(counts < cap)
            idx = int(candidates[np.argmin(counts[candidates])])
        elif policy == 'confidence':
            available = counts < cap
            # smaller abs(score)/sqrt(N) => less confident
            scores = np.abs(totals)/np.sqrt(counts)
            scores[~available] = np.inf
            idx = int(scores.argmin())
        else:
            raise ValueError(policy)
        totals[idx] += sines[idx,counts[idx]]
        counts[idx] += 1
    return (totals > 0).astype(np.uint8), counts


def one_trial(seed: int, *, structured: bool, gauge: bool = True,
              sigma: float = .85, budget_per_item: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    world = procedural_world(rng, structured)
    memory = Lattice.empty()
    write_edges = sum(memory.write_image(ch, world[ch]) for ch in range(CHANNELS))
    before = [memory.face(ch,r,c) for ch in range(2) for r in range(7) for c in range(7)]
    if gauge:
        memory.rephase(rng.integers(0,256,(2,8,8),dtype=np.uint8))
    after = [memory.face(ch,r,c) for ch in range(2) for r in range(7) for c in range(7)]
    exact_correct = float(np.mean(memory.read_image(0)==world[0]) + np.mean(memory.read_image(1)==world[1]))/2
    # For a fair paired/open-path comparison use paired and single from the SAME shots.
    single=[]; paired=[]
    for ch in range(2):
        for r in range(7):
            for c in range(7):
                p = memory.probe(ch,r,c,rng,0.,1.5)
                single.append(int(p['single']==int(world[ch,r,c])))
                paired.append(int(p['paired']==int(world[ch,r,c])))
    # Always preallocate identical noisy paired observations for both policies.
    shots, scales = shot_arrays(world,rng,sigma=sigma)
    budget = BITS * budget_per_item
    answer_u, counts_u = allocate_shots(shots,budget,'uniform')
    answer_c, counts_c = allocate_shots(shots,budget,'confidence')
    truth = world.reshape(-1)
    # Test a real counterfactual and collateral corruption, after random gauge.
    r=int(rng.integers(0,7)); c=int(rng.integers(0,6)); ch=int(rng.integers(0,2))
    target=1-int(world[ch,r,c]); preserved=memory.copy(); naive=memory.copy()
    preserving_cost=preserved.write(ch,r,c,target)
    naive_cost=naive.naive_write(ch,r,c,target)
    changed=[(z,y,x) for z in range(2) for y in range(7) for x in range(7)
             if preserved.face(z,y,x)!=memory.face(z,y,x)]
    changed_naive=[(z,y,x) for z in range(2) for y in range(7) for x in range(7)
                   if naive.face(z,y,x)!=memory.face(z,y,x)]
    return {'structured':structured,'gauge':gauge,'gauge_invariant':before==after,
            'exact':exact_correct,'single_one_shot':float(np.mean(single)),
            'paired_one_shot':float(np.mean(paired)),
            'noisy_uniform':float(np.mean(answer_u==truth)),
            'noisy_confidence':float(np.mean(answer_c==truth)),
            'uniform_shots':int(counts_u.sum()),'confidence_shots':int(counts_c.sum()),
            'write_edges_initial':write_edges,'preserving_rewrite_edges':preserving_cost,
            'naive_rewrite_edges':naive_cost,'preserved_changed_faces':len(changed),
            'naive_changed_faces':len(changed_naive),
            'read_write_payload_bytes':len(memory.bytes()),
            'exact_indexed_bytes':INDEXED_BYTES,
            'confidence_shot_std':float(counts_c.std())}


def sweep(seed: int=41, trials: int=200, sigma: float=.85, budget_per_item: int=2):
    if trials<1:
        raise ValueError('need at least one trial')
    from statistics import mean,stdev
    records=[]
    metrics=('exact','single_one_shot','paired_one_shot','noisy_uniform','noisy_confidence',
             'preserved_changed_faces','naive_changed_faces','preserving_rewrite_edges',
             'write_edges_initial','confidence_shot_std')
    for structured in (False,True):
        for gauge in (False,True):
            runs=[one_trial(seed+1000003*i+1009*structured+13*gauge,
                            structured=structured,gauge=gauge,
                            sigma=sigma,budget_per_item=budget_per_item) for i in range(trials)]
            records.append({'scenario':'shapes' if structured else 'independent',
                            'gauge':'random_vertex_phases' if gauge else 'fixed_frame',
                            'trials':trials,
                            'all_loop_invariant':all(x['gauge_invariant'] for x in runs),
                            'budget_shots_per_trial':BITS*budget_per_item,
                            'metrics':{k:{'mean':round(mean(x[k] for x in runs),6),
                                          'trial_stderr':round(stdev(x[k] for x in runs)/np.sqrt(trials),6) if trials>1 else None}
                                       for k in metrics}})
    return {'seed':seed,'trials':trials,'sigma':sigma,'budget_per_item':budget_per_item,
            'graph_bytes':EDGE_BYTES,'exact_indexed_bytes':INDEXED_BYTES,
            'bits_per_pair':BITS,'protocol':'exploratory: NOT pretrained, NOT a learned policy',
            'records':records}
