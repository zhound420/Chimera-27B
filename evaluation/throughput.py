import argparse,json,statistics,time,urllib.request
from pathlib import Path

def request(path,data):
 req=urllib.request.Request(API+path,json.dumps(data).encode(),{'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=300) as response:return json.load(response)

def benchmark(draft,label=None):
 probes=[]
 for prompt in ['def binary_search(items, target):\n    low, high = 0, len(items)\n    while low < high:\n',
                'def parse_int(text):\n    """Parse a signed decimal integer."""\n']:
  probes.append(request('/completion',dict(prompt=prompt,n_predict=96,temperature=0,seed=42,return_tokens=True,cache_prompt=False)))
 prompts=[
 'Implement a Python LRU cache with constant-time get/put, capacity zero, and explain the invariants. Include example tests.',
 'Implement a Python incremental UTF-8 decoder for byte chunks with strict errors, and explain how it handles incomplete sequences. Include example tests.',
 'Implement a stable topological sort for string nodes, including neighbor-only nodes, duplicate edges, and cycle detection. Explain its complexity and include tests.']
 samples=[]
 for prompt in prompts:
  start=time.monotonic()
  result=request('/v1/chat/completions',dict(model=ALIAS,messages=[{'role':'user','content':prompt}],max_tokens=768,
   temperature=1.0,top_p=.95,top_k=20,min_p=0.0,seed=42,reasoning_effort='medium'))
  samples.append(dict(wall_seconds=time.monotonic()-start,response=result))
 speed=[s['response']['timings']['predicted_per_second'] for s in samples]
 report=dict(draft_max=draft,median_decode_tps=statistics.median(speed),samples=samples,greedy_probes=probes,
  note='768-token throughput probes may be truncated; these are not quality scores.')
 (R/((label or f'q4-draft-{draft}')+'-speed.json')).write_text(json.dumps(report,indent=2)+'\n')
 return report


if __name__ == '__main__':
 p=argparse.ArgumentParser();p.add_argument('--api',default='http://127.0.0.1:8081');p.add_argument('--model',default='chimera-27b-v0.1');p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--draft-max',type=int,default=5)
 a=p.parse_args();API=a.api.rstrip('/');ALIAS=a.model;R=a.output_dir;R.mkdir(parents=True,exist_ok=True)
 if (R/'throughput-speed.json').exists():raise FileExistsError('Use a fresh output directory')
 print(json.dumps(benchmark(a.draft_max,'throughput'),indent=2))
