#!/usr/bin/env python3
"""Re-execute saved coding answers with the original tests in Bubblewrap."""
import argparse
import json
from pathlib import Path
import re
import tempfile
import benchlib

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    a=p.parse_args();comparisons=[]
    with tempfile.TemporaryDirectory(prefix='chimera-replay-') as temporary:
        benchlib.ROOT=Path(temporary)
        for result_file in sorted(a.results.glob('*/result.json')):
            folder=result_file.parent
            report=json.loads(result_file.read_text())
            fixtures={x[0]:x for x in json.loads((folder/'development-fixture.json').read_text())}
            responses={x['label']:x['response'] for x in map(json.loads,(folder/'requests.jsonl').read_text().splitlines())}
            for name, recorded in report['codes'].items():
                response=responses.get(name)
                if response is None: raise RuntimeError('Missing response for '+str(folder/name))
                blocks=re.findall(r'```(?:python)?\s*\n(.*?)```',benchlib.content(response),re.S)
                passed=response['choices'][0].get('finish_reason')!='length' and bool(blocks) and benchlib.sandbox(blocks[0]+'\n'+fixtures[name][2])
                comparisons.append({'run':folder.name,'case':name,'recorded':recorded['passed'],'replayed':passed})
    print(json.dumps(comparisons,indent=2))
    if not all(row['recorded']==row['replayed'] for row in comparisons): raise SystemExit('A recorded result did not reproduce.')
    print('All '+str(len(comparisons))+' saved coding outcomes reproduced. This is a replay, not fresh model sampling.')

if __name__=='__main__':main()
