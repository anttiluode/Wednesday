import unittest
import numpy as np
from wednesday.core import (Lattice,EDGE_BYTES,INDEXED_BYTES,byte_of,
                            procedural_world,shot_arrays,allocate_shots,one_trial,sweep)

class GaugeMemoryTests(unittest.TestCase):
    def test_budgets(self):
        m=Lattice.empty();m.check()
        self.assertEqual((len(m.bytes()),EDGE_BYTES,INDEXED_BYTES),(224,224,13))
    def test_two_channel_independent_values(self):
        m=Lattice.empty();a=np.random.default_rng(19).integers(0,2,(2,7,7))
        for ch in range(2):m.write_image(ch,a[ch])
        for ch in range(2):np.testing.assert_array_equal(a[ch],m.read_image(ch))
    def test_exact_gauge_invariance(self):
        rng=np.random.default_rng(45);m=Lattice.empty()
        for ch in range(2):m.write_image(ch,rng.integers(0,2,(7,7)))
        initial=np.array([[ [m.face(ch,r,c) for c in range(7)] for r in range(7)] for ch in range(2)])
        paths=m.open_paths(0,3,3)
        m.rephase(rng.integers(0,256,(2,8,8),dtype=np.uint8))
        self.assertNotEqual(paths,m.open_paths(0,3,3))
        final=np.array([[ [m.face(ch,r,c) for c in range(7)] for r in range(7)] for ch in range(2)])
        np.testing.assert_array_equal(initial,final)
    def test_rephase_twice_still_exact(self):
        rng=np.random.default_rng(666);m=Lattice.empty()
        m.write(0,3,2,1)
        for _ in range(20):m.rephase(rng.integers(0,256,(2,8,8),dtype=np.uint8))
        self.assertEqual(m.face(0,3,2),64)
    def test_rewrite_preserves_all_neighbors_after_gauge(self):
        rng=np.random.default_rng(4);m=Lattice.empty()
        for ch in range(2):m.write_image(ch,rng.integers(0,2,(7,7)))
        m.rephase(rng.integers(0,256,(2,8,8),dtype=np.uint8))
        before=np.array([m.face(ch,r,c) for ch in range(2) for r in range(7) for c in range(7)])
        oldbit=m.read_bit(0,2,3);cost=m.write(0,2,3,1-oldbit)
        after=np.array([m.face(ch,r,c) for ch in range(2) for r in range(7) for c in range(7)])
        self.assertEqual(cost,4)
        self.assertEqual(np.sum(before!=after),1)
        self.assertEqual(m.read_bit(0,2,3),1-oldbit)
    def test_naive_rewrite_damages_neighbor(self):
        m=Lattice.empty();m.write(0,2,3,1)
        original=m.face(0,2,4)
        m.naive_write(0,2,3,0)
        self.assertNotEqual(m.face(0,2,4),original)
        self.assertNotIn(m.face(0,2,4),(64,192))
    def test_ratio_matches_loop_but_open_path_does_not(self):
        rng=np.random.default_rng(6);m=Lattice.empty();m.write(1,2,1,1)
        a,b=m.open_paths(1,2,1)
        self.assertEqual((a-b)%256,m.face(1,2,1))
        m.rephase(rng.integers(0,256,(2,8,8),dtype=np.uint8))
        x,y=m.open_paths(1,2,1)
        self.assertEqual((x-y)%256,64)
        self.assertNotEqual((a,b),(x,y))
    def test_common_phase_cancels(self):
        rng=np.random.default_rng(4);m=Lattice.empty();m.write(0,0,0,1)
        for _ in range(100):self.assertEqual(m.probe(0,0,0,rng,0,3)['paired'],1)
    def test_noise_policies_equal_budget(self):
        rng=np.random.default_rng(3);w=procedural_world(rng)
        signals,_=shot_arrays(w,rng)
        for policy in ('uniform','confidence'):
            ans,c=allocate_shots(signals,196,policy)
            self.assertEqual(c.sum(),196)
            self.assertEqual(len(ans),98)
    def test_determinism(self):
        self.assertEqual(one_trial(2,structured=False),one_trial(2,structured=False))
        self.assertEqual(sweep(2,3),sweep(2,3))
    def test_invalid(self):
        m=Lattice.empty()
        with self.assertRaises(ValueError):m.write(2,0,0,1)
        with self.assertRaises(ValueError):m.write(0,0,0,2)
        with self.assertRaises(ValueError):m.rephase(np.zeros((8,8),dtype=np.uint8))

if __name__=='__main__':unittest.main()
