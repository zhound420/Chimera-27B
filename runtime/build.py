#!/usr/bin/env python3
"""Fetch a pinned llama.cpp snapshot, apply Chimera's patch, and build locally."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent
REV = '3cf03257f219afbe7334045ff7c6a06ac68c627d'
ARCHIVE_SHA256 = 'c076d7534afa0e5d0ec2a0d425b11e791c16f3de0d727221aea071cef156a280'

def run(args, env):
    print(' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), env=env, check=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=HERE/'upstream')
    p.add_argument('--build', type=Path, default=HERE/'build')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--cuda-root', type=Path)
    p.add_argument('--architectures', default='native')
    p.add_argument('--cpu-only', action='store_true')
    p.add_argument('--legacy-cpu', action='store_true', help='Disable AVX2, FMA, and F16C for the original E5-v1 test host.')
    p.add_argument('--cc', default=os.environ.get('CC', 'gcc'))
    p.add_argument('--cxx', default=os.environ.get('CXX', 'g++'))
    p.add_argument('--ggml-prefix', type=Path, help='Advanced: reuse a GGML SDK built from exactly the pinned revision.')
    a = p.parse_args()
    source, build = a.source.resolve(), a.build.resolve()
    env = dict(os.environ)
    if a.cuda_root:
        cuda = a.cuda_root.resolve()
        env['PATH'] = str(cuda/'bin')+os.pathsep+env['PATH']
        env['LD_LIBRARY_PATH'] = str(cuda/'lib64')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    stamp = source/'.chimera-patch.json'
    patch_hash = hashlib.sha256((HERE/'chimera.patch').read_bytes()).hexdigest()
    expected = {'revision':REV, 'patch_sha256':patch_hash}
    if source.exists():
        if not stamp.exists() or json.loads(stamp.read_text()) != expected:
            raise RuntimeError('Existing source directory is not this patched revision; choose a new --source directory.')
    else:
        source.parent.mkdir(parents=True, exist_ok=True)
        archive = source.parent/('llama-'+REV+'.tar.gz')
        if not archive.exists():
            urllib.request.urlretrieve('https://codeload.github.com/ggml-org/llama.cpp/tar.gz/'+REV, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
            raise RuntimeError('Upstream archive checksum mismatch')
        temporary = source.parent/(source.name+'-extract')
        temporary.mkdir(exist_ok=False)
        try:
            with tarfile.open(archive) as bundle:
                bundle.extractall(temporary, filter='data')
            (temporary/('llama.cpp-'+REV)).rename(source)
        finally:
            shutil.rmtree(temporary)
        run(['patch','--batch','--forward','-d',source,'-p1','-i',HERE/'chimera.patch'],env)
        stamp.write_text(json.dumps(expected,indent=2)+'\n')
    args = ['cmake','-S',source,'-B',build,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_C_COMPILER='+a.cc,'-DCMAKE_CXX_COMPILER='+a.cxx,
            '-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_TOOLS=ON',
            '-DLLAMA_BUILD_SERVER=ON','-DLLAMA_BUILD_UI=OFF','-DLLAMA_BUILD_APP=OFF',
            '-DLLAMA_BUILD_MTMD=ON','-DLLAMA_CURL=OFF','-DLLAMA_OPENSSL=OFF']
    if a.ggml_prefix:
        args += ['-DLLAMA_USE_SYSTEM_GGML=ON','-DCMAKE_PREFIX_PATH='+str(a.ggml_prefix.resolve())]
        if a.cuda_root: args += ['-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,'+str(cuda/'lib64')]
    else:
        args += ['-DGGML_CUDA='+('OFF' if a.cpu_only else 'ON'),'-DGGML_CUDA_NCCL=OFF']
        if not a.cpu_only:
            args += ['-DCMAKE_CUDA_ARCHITECTURES='+a.architectures,'-DCMAKE_CUDA_HOST_COMPILER='+a.cxx]
            if a.cuda_root: args += ['-DCUDAToolkit_ROOT='+str(cuda),'-DCMAKE_CUDA_COMPILER='+str(cuda/'bin/nvcc')]
    if a.legacy_cpu:
        args += ['-DGGML_NATIVE=ON','-DGGML_AVX2=OFF','-DGGML_FMA=OFF','-DGGML_F16C=OFF']
    run(args,env)
    run(['cmake','--build',build,'--parallel',str(a.jobs),'--target','llama-server','llama-quantize'],env)
    if os.name == 'posix' and os.uname().sysname == 'Linux':
        run([a.cxx,'-shared','-fPIC','-O2','-std=c++17','-I'+str(source/'ggml/include'),
             HERE/'mmvq_cutoff.cpp','-ldl','-o',build/'libchimera_mmvq.so'],env)
    (build/'chimera-build.json').write_text(json.dumps({**expected,'source':str(source),'configure_command':list(map(str,args))},indent=2)+'\n')
    print('Built patched server: '+str(build/'bin/llama-server'),flush=True)

if __name__ == '__main__': main()
