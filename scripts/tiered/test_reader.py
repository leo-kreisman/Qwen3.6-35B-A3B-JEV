"""Native reader gates against synthetic extents, with actual direct I/O."""
import ctypes
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
from test_codec import build, HERE

class ReaderTests(unittest.TestCase):
    def test_native_reader_and_rejection(self):
        root=HERE.parents[1]
        vendor=root/'vendor/BigMoeOnEdge'
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); exe=d/'reader-test'; so=d/'codec.so'
            subprocess.run(['g++','-O2','-std=c++17',str(HERE/'reader_test.cpp'),
                            '-I'+str(vendor/'core/src/moe'),str(vendor/'build/core/libbmoe_core.a'),
                            '-pthread','-o',str(exe)],check=True)
            subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(HERE/'codec.cpp'),'-o',str(so)],check=True)
            lib=ctypes.CDLL(str(so))
            lib.jev_pack.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
            lib.jev_expand.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
            source=d/'source'; source.write_bytes(random.Random(9).randbytes(1<<20))
            packed=ctypes.create_string_buffer(81); expanded=ctypes.create_string_buffer(144)
            self.assertEqual(lib.jev_pack(source.read_bytes()[:144],144,packed),1)
            self.assertEqual(lib.jev_expand(packed,81,expanded,144),1)
            reference=d/'reference';reference.write_bytes(expanded.raw)
            header=build.HEADER.pack(build.MAGIC,source.stat().st_size,build.fingerprint(source),1)
            entry=build.ENTRY.pack(0,4096,144,81,1)
            valid=header+entry+bytes(4096-len(header)-len(entry))+packed.raw+bytes(4096-81)
            pack=d/'pack';pack.write_bytes(valid)
            def run(mode): subprocess.run([str(exe),str(pack),str(source),str(reference),mode],check=True)
            run('accept')
            for blob in [valid[:20], b'BADMAGIC'+valid[8:],
                         header+build.ENTRY.pack(0,8192,144,81,1)+valid[72:],
                         header+build.ENTRY.pack(1,4096,144,81,1)+valid[72:]]:
                pack.write_bytes(blob); run('reject')
            pack.write_bytes(valid)
            with source.open('r+b') as f: f.write(b'changed')
            run('reject')

if __name__=='__main__': unittest.main()
