"""Independent format checks; run with python3 -m unittest discover -s scripts/tiered."""
import ctypes
import importlib.util
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('tiered_build', HERE/'build.py')
build=importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)

class CodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        so=Path(cls.tmp.name)/'codec.so'
        subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(HERE/'codec.cpp'),'-o',str(so)],check=True)
        cls.lib=ctypes.CDLL(str(so))
        cls.lib.jev_pack.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
        cls.lib.jev_expand.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_independent_decode_and_metadata(self):
        books=[(0,5,10,15),(1,5,9,13),(2,6,10,14),(3,6,9,12),
               (4,6,8,10),(5,7,9,11),(6,8,10,12),(4,7,10,13),
               (0,3,6,9),(6,9,12,15),(0,2,4,6),(9,11,13,15),
               (2,5,8,11),(4,7,10,15),(0,5,8,11),(5,8,11,14)]
        raw=random.Random(7).randbytes(144*100)
        packed=ctypes.create_string_buffer(81*100)
        decoded=ctypes.create_string_buffer(len(raw))
        self.assertEqual(self.lib.jev_pack(raw,len(raw),packed),1)
        self.assertEqual(self.lib.jev_expand(packed,len(packed),decoded,len(raw)),1)
        expected=bytearray()
        for b in range(100):
            p=packed.raw[b*81:(b+1)*81]; r=raw[b*144:(b+1)*144]
            self.assertEqual(p[:16],r[:16])
            c=books[p[16]]; expected.extend(p[:16])
            for byte in p[17:]:
                q=[c[(byte>>shift)&3] for shift in (0,2,4,6)]
                expected.extend((q[0]|q[1]<<4,q[2]|q[3]<<4))
            # A codebook with max nibble error 2 is always available.
            out=decoded.raw[b*144:(b+1)*144]
            error=sum(((x>>shift&15)-(y>>shift&15))**2
                      for x,y in zip(r[16:],out[16:]) for shift in (0,4))
            self.assertLessEqual(error,256*4)
        self.assertEqual(decoded.raw,expected)

    def test_bad_lengths_and_codebook(self):
        out=ctypes.create_string_buffer(144)
        self.assertEqual(self.lib.jev_pack(bytes(143),143,out),0)
        self.assertEqual(self.lib.jev_expand(bytes(80),80,out,144),0)
        bad=bytearray(81);bad[16]=255
        self.assertEqual(self.lib.jev_expand(bytes(bad),81,out,144),0)

    def test_hot_profile_and_missing_layer_rejected(self):
        p=Path(self.tmp.name)/'trace.csv'
        p.write_text('layer,expert,dropped\n0,2,0\n0,2,0\n0,1,0\n0,0,0\n')
        hot,_=build.load_profile([p],4,{0},.75)
        self.assertEqual(hot[0],{0,2})
        with self.assertRaises(ValueError): build.load_profile([p],4,{0,1},.8)
        p.write_text('layer,expert,dropped\n0,2,1\n')
        with self.assertRaises(ValueError): build.load_profile([p],4,{0},.8)

if __name__=='__main__': unittest.main()
