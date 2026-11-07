"""Development quality screen; never switches models or counts as final promotion."""
import argparse
import base64
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('hybrid_bench_common', ROOT/'benchlib.py')
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)

EXTRA = [
    ('json_patch', 'Implement apply_patch(document, operations), supporting only top-level JSON object keys and operations add, replace, remove, test. Each op has op, path and optionally value. Paths begin / and encode ~ as ~0 and / as ~1. Reject invalid escapes or nested paths with ValueError. Empty-string key is valid as path /. add may overwrite; replace/remove/test require an existing key. Failed test or invalid operation raises ValueError. Return a deep copy, never mutate document, operations, or nested values. Python standard library only.',
     """import copy
d={'a':{'x':[1]},'a/b':3,'~':4,'':9}; old=copy.deepcopy(d)
o=apply_patch(d,[{'op':'replace','path':'/a~1b','value':8},{'op':'remove','path':'/~0'},{'op':'test','path':'/','value':9}])
assert o['a/b']==8 and '~' not in o and d==old
o['a']['x'].append(2); assert d==old
ops=[{'op':'add','path':'/z','value':{'x':[]}}];o=apply_patch(d,ops);o['z']['x'].append(2);assert ops[0]['value']=={'x':[]}
for ops in [[{'op':'test','path':'/','value':0}],[{'op':'replace','path':'/missing','value':1}],[{'op':'remove','path':'/a/x'}],[{'op':'add','path':'/a~2','value':0}]]:
 try: apply_patch(d,ops)
 except ValueError: pass
 else: raise AssertionError('invalid patch accepted')
assert d==old"""),
    ('ttl_cache', 'Implement TTLCache(capacity, clock), with put(key,value,ttl), get(key,default=None), and len(cache). clock() gives monotonic seconds. TTL starts on put; expires when now >= expiry. get does not refresh TTL but makes live entries most recently used. put replaces and refreshes. On overflow evict least recently used live entry after removing expired entries. ttl<=0 removes the key. capacity=0 stores nothing. len excludes expired entries. Standard library only.',
     """now=[0];c=TTLCache(2,lambda:now[0]);c.put('a',1,10);c.put('b',2,2)
now[0]=2;assert c.get('b','missing')=='missing';assert len(c)==1
c.put('c',3,20);assert c.get('a')==1;c.put('d',4,20);assert c.get('c') is None
now[0]=10;assert c.get('a') is None;assert len(c)==1
c.put('d',5,1);now[0]=11;assert c.get('d') is None
c.put('x',1,3);c.put('x',2,0);assert len(c)==0
z=TTLCache(0,lambda:0);z.put('a',1,20);assert len(z)==0"""),
    ('utf8_stream', 'Implement decode_chunks(chunks), accepting an iterable of bytes chunks and returning the combined UTF-8 string. Decode incrementally with strict error handling: raise UnicodeDecodeError on invalid sequences or incomplete final sequences. Empty chunks and arbitrary split boundaries must work. Do not concatenate all input bytes first; standard library only.',
     """s='A€𐍈é中文\\n';raw=s.encode();assert decode_chunks([])==''
for i in range(len(raw)+1):assert decode_chunks([raw[:i],b'',raw[i:]])==s
assert decode_chunks(bytes([b]) for b in raw)==s
for bad in [b'\\xff',b'\\xe2\\x82',b'\\xc0\\xaf',b'\\xed\\xa0\\x80']:
 try:decode_chunks([bad])
 except UnicodeDecodeError:pass
 else:raise AssertionError('invalid UTF-8 accepted')"""),
    ('dependency_levels', 'Implement dependency_levels(graph). graph maps string task names to lists of prerequisite task names. Include prerequisite-only names. Return sorted lists of tasks by earliest runnable parallel batch: tasks in a batch depend only on earlier batches. Deduplicate repeated prerequisites. Raise ValueError for cycles including self cycles. Do not mutate input; empty graph returns [].',
     """g={'build':['parse','parse'],'test':['build'],'docs':[],'parse':['fetch']}
assert dependency_levels(g)==[['docs','fetch'],['parse'],['build'],['test']]
assert g['build']==['parse','parse'];assert dependency_levels({})==[]
assert dependency_levels({'a':['b'],'c':['b']})==[['b'],['a','c']]
for g in [{'x':['x']},{'a':['b'],'b':['a']},{'ok':[],'x':['y'],'y':['x']}]:
 try:dependency_levels(g)
 except ValueError:pass
 else:raise AssertionError('cycle accepted')"""),
    ('overlap_search', 'Implement find_all(text, pattern) returning starting indices of every substring occurrence, including overlaps. Empty pattern matches all len(text)+1 boundaries. Work on Python strings and Unicode. Use O(len(text)+len(pattern)) time; no regex or external packages.',
     """import random
assert find_all('aaaa','aa')==[0,1,2];assert find_all('abc','')==[0,1,2,3]
assert find_all('ééé','éé')==[0,1]
r=random.Random(294)
for _ in range(200):
 s=''.join(r.choice('abé') for _ in range(r.randrange(30)));p=''.join(r.choice('abé') for _ in range(r.randrange(8)))
 expected=[i for i in range(len(s)+1) if s.startswith(p,i)];assert find_all(s,p)==expected
assert len(find_all('a'*100000,'a'*1000))==99001"""),
]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='chimera-27b-v0.1')
    p.add_argument('--label',default='swift-development')
    p.add_argument('--api',default='http://127.0.0.1:8081')
    p.add_argument('--effort',default=None)
    p.add_argument('--temperature',type=float,default=1.0)
    p.add_argument('--max-tokens',type=int,default=16384)
    p.add_argument('--split',choices=['development','heldout'],default='development')
    p.add_argument('--coding-only',action='store_true')
    p.add_argument('--case-names',default=None)
    p.add_argument('--lock',type=Path,default=ROOT/'evaluation.lock')
    a=p.parse_args()
    if a.effort is None:
        a.effort='medium'
    out=ROOT/'baselines'/a.label
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        raise FileExistsError('Use a fresh run label to preserve previous results')
    B.ROOT,B.MODEL,B.API=out,a.model,a.api
    base_request=B.request
    def effort_request(*args,**kwargs):
        kwargs['reasoning_effort']=a.effort
        kwargs.update(temperature=a.temperature,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repeat_penalty=1.0)
        return base_request(*args,**kwargs)
    B.request=effort_request
    cases=B.CODE_CASES+EXTRA
    if a.case_names:
        selected=set(a.case_names.split(','))
        if not selected.issubset({x[0] for x in cases}):raise ValueError('Unknown case filter')
        cases=[x for x in cases if x[0] in selected]
    fixture=json.dumps(cases,indent=2)
    (out/'development-fixture.json').write_text(fixture+'\n')
    result={'model':a.model,'reasoning_effort':a.effort,'split':a.split,'fixture_sha256':hashlib.sha256(fixture.encode()).hexdigest(),
            'max_tokens':a.max_tokens,
            'sampling':{'temperature':a.temperature,'top_p':0.95,'top_k':20,'min_p':0.0,'seed':42},
            'promotion_evidence':False,'codes':{},'other':{},
            'timing_note':'Concurrent user activity and background downloads are possible; not a controlled TPS benchmark.'}
    def save():
        result['coding_passed']=sum(v.get('passed',False) for v in result['codes'].values())
        result['coding_total']=len(cases)
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    def check_model():
        with urllib.request.urlopen(a.api+'/v1/models',timeout=10) as response:
            names=[m['id'] for m in json.load(response)['data']]
        if a.model not in names:raise RuntimeError('Serving model changed; refusing to mislabel results')
    with a.lock.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_model()
        for name,prompt,tests in cases:
            check_model()
            started=time.monotonic()
            try:
                answer=B.ask(name,prompt+'\nReturn the implementation in one Python fenced code block.',max_tokens=a.max_tokens)
                blocks=re.findall(chr(96)*3+r'(?:python)?\s*\n(.*?)'+chr(96)*3,B.content(answer),re.S)
                finish=answer['choices'][0].get('finish_reason')
                passed=finish!='length' and bool(blocks) and B.sandbox(blocks[0]+'\n'+tests)
                result['codes'][name]={'passed':passed,'finish_reason':finish,'usage':answer.get('usage'),
                                      'timings':answer.get('timings'),'wall_seconds':time.monotonic()-started}
            except Exception:result['codes'][name]={'passed':False,'error':traceback.format_exc()}
            save()
            print(json.dumps({'case':name,**result['codes'][name]}),flush=True)
            last=list(result['codes'].values())[-2:]
            if len(last)==2 and all(x.get('finish_reason')=='length' for x in last):
                result['early_stopped']='Two consecutive requests exhausted the output budget; remaining cases not evaluated.'
                result['complete']=True
                result['full_suite_completed']=False
                save()
                print(json.dumps(result),flush=True)
                return
        if a.coding_only:
            result['complete']=True;save();print(json.dumps(result),flush=True);return
        try:
            check_model()
            original=B.request
            def budgeted(*args,**kwargs):
                kwargs['max_tokens']=max(kwargs.get('max_tokens',0),8192)
                return original(*args,**kwargs)
            B.request=budgeted
            result['other']['tool_retry']=B.tool_check('tool-retry')
            image=base64.b64encode((ROOT/'fixtures/vision-test.png').read_bytes()).decode()
            answer=B.request('vision',[{'role':'user','content':[
                {'type':'text','text':'Read the order code and total. Identify the color of the square, circle, and triangle.'},
                {'type':'image_url','image_url':{'url':'data:image/png;base64,'+image}}]}])
            result['other']['vision']=answer['choices'][0].get('finish_reason')!='length' and all(
                x in B.content(answer).lower() for x in ['vision-7429','42.50','red','blue','green'])
        except Exception:result['other']['error']=traceback.format_exc()
        result['complete']=True
        save()
        print(json.dumps(result),flush=True)

if __name__=='__main__':main()
