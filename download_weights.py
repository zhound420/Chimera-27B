#!/usr/bin/env python3
"""Download only the selected weights from the pinned, verified HF release."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--precision',choices=['q4','q8','both'],default='q4')
    p.add_argument('--donor',action='store_true',help='Also fetch the fitted FP32 donor for reconstruction.')
    p.add_argument('--directory',type=Path,default=ROOT)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args()
    release=json.loads((ROOT/'huggingface-release.json').read_text())
    manifest=json.loads((ROOT/'weights-manifest.json').read_text())
    files=['mmproj-Qwen3.8-27B-BF16.gguf']
    if a.precision in ['q4','both']:files.append('Chimera-27B-v0.1-Q4_K_M.gguf')
    if a.precision in ['q8','both']:files.append('Chimera-27B-v0.1-Q8_0.gguf')
    if a.donor:files.append('reconstruction/shared-aligned-fp32.npz')
    destination=a.directory.resolve()
    if not a.verify_only:
        command=['hf','download',release['repo'],*files,'--revision',release['revision'],'--local-dir',str(destination)]
        if a.dry_run:command.append('--dry-run')
        subprocess.run(command,check=True)
    if a.dry_run:return
    for name in files:
        path=destination/name;expected=manifest[name]
        if path.stat().st_size!=expected['bytes']:raise RuntimeError('Size mismatch: '+name)
        digest=hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda:source.read(16*1024**2),b''):digest.update(chunk)
        if digest.hexdigest()!=expected['sha256']:raise RuntimeError('Checksum mismatch: '+name)
        print('Verified '+name,flush=True)

if __name__=='__main__':main()
