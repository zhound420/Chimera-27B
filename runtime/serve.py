#!/usr/bin/env python3
"""Launch Chimera without changing system services. Linux/CUDA reference profiles."""
import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--precision', choices=['q4','q8'], default='q4')
    p.add_argument('--model-dir', type=Path, default=HERE.parent)
    p.add_argument('--build', type=Path, default=HERE/'build')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8081)
    p.add_argument('--context', type=int, default=131072)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--device', default=None, help='Q4 language device; default CUDA0.')
    p.add_argument('--vision-device', default=None, help='Default CUDA1 for Q4; CUDA0 for Q8.')
    p.add_argument('--tensor-split', default='0.40,0.60', help='Q8 two-GPU split.')
    p.add_argument('--draft-max', type=int, default=None, help='Default 5 for Q4; 0 for Q8.')
    p.add_argument('--mmvq-cutoff', type=int, choices=range(1,9), default=None,
                   help='Optional Linux Ampere (cc 8.6) tuning shim; use 2 to reproduce the 69 TPS configuration.')
    p.add_argument('--donor-scale', type=float, default=0.1)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    build, models = a.build.resolve(), a.model_dir.resolve()
    server = build/'bin/llama-server'
    manifest = build/'chimera-build.json'
    if not manifest.is_file(): raise RuntimeError('Build with runtime/build.py first; a stock runtime can silently ignore the donor.')
    if not server.is_file(): raise FileNotFoundError(server)
    model = models/('Chimera-27B-v0.1-'+('Q4_K_M' if a.precision=='q4' else 'Q8_0')+'.gguf')
    projector = models/'mmproj-Qwen3.8-27B-BF16.gguf'
    for path in [model,projector]:
        if not path.is_file(): raise FileNotFoundError(path)
    draft = a.draft_max if a.draft_max is not None else (5 if a.precision=='q4' else 0)
    vision = a.vision_device or ('CUDA1' if a.precision=='q4' else 'CUDA0')
    args = [str(server),'-m',str(model),'--mmproj',str(projector),'--mmproj-device',vision,
            '--alias','chimera-27b-v0.1','--host',a.host,'--port',str(a.port),'-ngl','99',
            '--ctx-size',str(a.context),'--parallel','1','-b','256','-ub','128',
            '--flash-attn','on','--cache-type-k','q8_0','--cache-type-v','q8_0',
            '--threads',str(a.threads),'--threads-batch',str(a.threads),'--fit','off',
            '--no-context-shift','--reasoning-format','deepseek',
            '--override-kv','qwen35.hybrid.scale=float:'+str(a.donor_scale),
            '--chat-template-kwargs',json.dumps({'reasoning_effort':'medium'})]
    if a.precision == 'q4': args += ['--split-mode','none','--device',a.device or 'CUDA0','--main-gpu','0']
    else: args += ['--split-mode','layer','--tensor-split',a.tensor_split]
    if draft: args += ['--spec-type','draft-mtp','--spec-draft-n-max',str(draft)]
    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = str(a.threads)
    env['GGML_CUDA_GRAPH_OPT'] = '0'
    if a.mmvq_cutoff:
        shim = build/'libchimera_mmvq.so'
        if not shim.is_file(): raise FileNotFoundError(shim)
        env['LD_PRELOAD'] = str(shim)+(':'+env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
        env['HYBRID_MMVQ_CUTOFF'] = str(a.mmvq_cutoff)
    print(json.dumps({'command':args,'donor_scale':a.donor_scale,'mmvq_cutoff':a.mmvq_cutoff},indent=2),flush=True)
    if not a.dry_run: os.execvpe(args[0],args,env)

if __name__ == '__main__': main()
