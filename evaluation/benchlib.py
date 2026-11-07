"""Portable subset of the exact original request and execution harness."""

import json, resource, subprocess, time, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
MODEL='chimera-27b-v0.1'
API='http://127.0.0.1:8081'

def request(label, messages, max_tokens=384, timeout=1800, **extra):
    body = {'model': MODEL, 'messages': messages, 'max_tokens': max_tokens,
            'temperature': 0, 'seed': 42, 'reasoning_effort': 'high',
            'chat_template_kwargs': {'enable_thinking': True},
            'strata_checkpoint': False, 'stream': False, **extra}
    start = time.monotonic()
    req = urllib.request.Request(API + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    if 'error' in result or not result.get('choices'):
        raise RuntimeError(str(result)[:1000])
    with (ROOT / 'requests.jsonl').open('a') as log:
        log.write(json.dumps({'label': label, 'wall_seconds': time.monotonic() - start,
                              'response': result}) + '\n')
    return result

def ask(label, prompt, **kwargs):
    return request(label, [{'role': 'user', 'content': prompt}], **kwargs)

CODE_CASES = [
    ('merge', 'Implement merge_intervals(intervals) for closed intervals on the real number line with integer endpoints. Merge exactly when next_start <= current_end. Shared endpoints merge: [1,3] and [3,4] become [1,4]. Consecutive integer endpoints do NOT merge: [1,4] and [5,7] stay separate. Return sorted lists. Do not mutate input; handle empty input.',
     'a=[[5,7],[1,3],[3,4]]; assert merge_intervals(a)==[[1,4],[5,7]]; assert a==[[5,7],[1,3],[3,4]]\nassert merge_intervals([])==[]\nassert merge_intervals([[1,10],[2,3],[10,11]])==[[1,11]]'),
    ('cache', 'Implement class LRUCache(capacity) with get(key) returning -1 for missing, and put(key,value). Evict least recently used on overflow. Handle capacity zero. Use O(1) operations.',
     'c=LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1)==1; c.put(3,3); assert c.get(2)==-1; c.put(1,8); c.put(4,4); assert c.get(3)==-1; assert c.get(1)==8\nz=LRUCache(0); z.put(1,2); assert z.get(1)==-1'),
    ('toposort', 'Implement topological_sort(graph) where graph maps nodes to lists of outgoing neighbors. Include neighbor-only nodes. Return a valid order, raising ValueError on any cycle. Handle empty graph.',
     'g={"a":["b","c"],"b":["d"],"c":["d"]}; o=topological_sort(g); assert set(o)==set("abcd"); p={x:i for i,x in enumerate(o)}; assert all(p[a]<p[b] for a,bs in g.items() for b in bs)\nassert topological_sort({})==[]\ntry:\n topological_sort({"a":["b"],"b":["a"]})\nexcept ValueError:\n pass\nelse:\n raise AssertionError("cycle not detected")'),
]

def sandbox(code):
    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    args = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session',
            '--ro-bind', '/usr', '/usr', '--symlink', 'usr/bin', '/bin',
            '--symlink', 'usr/lib', '/lib', '--symlink', 'usr/lib64', '/lib64',
            '--tmpfs', '/tmp', '--proc', '/proc', '--dev', '/dev',
            '/usr/bin/python3', '-I', '-S', '-']
    with (ROOT / 'code-execution.log').open('a') as log:
        result = subprocess.run(args, input=code, text=True, stdout=log,
                                stderr=subprocess.STDOUT, timeout=15, preexec_fn=limits)
    return result.returncode == 0

def content(result):
    return result['choices'][0]['message'].get('content') or ''

def tool_check(label):
    tools = [{'type': 'function', 'function': {'name': 'read_ticket',
              'description': 'Read a ticket; retry exactly once if a temporary error is returned.',
              'parameters': {'type': 'object', 'properties': {'ticket_id': {'type': 'string'}},
                             'required': ['ticket_id'], 'additionalProperties': False}}}]
    messages = [{'role': 'user', 'content': 'Use read_ticket to read ticket TASK-42. Retry once on a temporary error. Then tell me the exact repository path and issue from the tool response.'}]
    for i, tool_result in enumerate([{'error': 'temporary failure; retry once'},
                                    {'path': 'src/cache/normalize.py', 'issue': 'empty key collision'}]):
        result = request(f'{label}-tool-{i}', messages, max_tokens=768,
                         reasoning_budget_tokens=256, tools=tools)
        message = result['choices'][0]['message']
        calls = message.get('tool_calls') or []
        if len(calls) != 1 or calls[0]['function']['name'] != 'read_ticket':
            return False
        if json.loads(calls[0]['function']['arguments']) != {'ticket_id': 'TASK-42'}:
            return False
        messages += [message, {'role': 'tool', 'tool_call_id': calls[0]['id'],
                               'content': json.dumps(tool_result)}]
    result = request(label + '-tool-final', messages, max_tokens=768,
                     reasoning_budget_tokens=256, tools=tools)
    answer = content(result).lower()
    return 'src/cache/normalize.py' in answer and 'empty key collision' in answer
