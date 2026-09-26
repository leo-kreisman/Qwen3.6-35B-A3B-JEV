import tempfile
import unittest
import os
from pathlib import Path
import numpy as np
from tiles import slice_tile, BLOCK, DirectReader
from simulate import lru
from analyze import compare


class BundleTests(unittest.TestCase):
    def test_comparison_rejects_incomplete_or_nonfinite_results(self):
        good={'answers':[{'id':'x','probabilities':[0.6,0.4],'logits':[1.0,0.0]}]}
        for logits in ([1.0], [float('nan'),0.0], [float('inf'),0.0]):
            bad={'answers':[{'id':'x','probabilities':[0.6,0.4],'logits':logits}]}
            with self.assertRaises(ValueError):compare(good,bad)
        with self.assertRaises(ValueError):compare(good,{'answers':good['answers']*2})

    def test_direct_unaligned_request_returns_exact_bytes(self):
        data=bytes(range(256))*32
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.bin'
            with p.open('wb') as f:
                f.write(data);f.flush();os.fsync(f.fileno())
            with DirectReader(p) as reader:
                reader.seek(17)
                self.assertEqual(reader.read(7000),data[17:7017])
                self.assertEqual(reader.physical_bytes,8192)
                self.assertEqual(reader.capacity,8192)

    def test_quant_blocks_roundtrip_gate_and_down(self):
        rng=np.random.default_rng(17)
        for q in (12,13,14):
            block,_=BLOCK[q];columns=512;rows=512
            raw=rng.integers(0,256,rows*2*block,dtype=np.uint8).tobytes()
            gate=b''.join(slice_tile(raw,(columns,rows),q,j,256) for j in (0,256))
            self.assertEqual(raw,gate)
            down=[np.frombuffer(slice_tile(raw,(columns,rows),q,j,256,True),dtype=np.uint8).reshape(rows,block) for j in (0,256)]
            self.assertEqual(raw,np.concatenate(down,axis=1).tobytes())

    def test_no_cut_inside_quantization_block(self):
        with self.assertRaises(ValueError):slice_tile(b'',(512,512),12,128,256,True)

    def test_weighted_lru_eviction_and_oversize(self):
        demands=[(0,[1]),(1,[1]),(0,[1]),(2,[0]),(0,[1])]
        r=lru(demands,{0:2,1:3,2:20},5)
        self.assertEqual(r['expert_payload_bytes'],25)
        self.assertEqual(r['hits'],2)
        r=lru(demands,{0:2,1:3,2:20},3)
        self.assertEqual(r['expert_payload_bytes'],27)


if __name__=='__main__':unittest.main()
