#!/usr/bin/env python3
"""Rebuild the evaluated graft from a converted Qwen base and the calibrated donor."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import gguf

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--donor',type=Path,default=Path(__file__).resolve().parent/'shared-aligned-fp32.npz')
    a=p.parse_args()
    partial=a.output.with_suffix('.partial.gguf')
    if a.output.exists() or partial.exists(): raise FileExistsError('Choose a fresh output path.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.output.parent).free < a.base.stat().st_size+100*1024**2:
        raise RuntimeError('Insufficient space for the rebuilt GGUF.')
    reader=gguf.GGUFReader(a.base)
    if reader.fields['general.architecture'].contents() != 'qwen35': raise ValueError('Expected a Qwen3.8-27B qwen35 GGUF.')
    if any(k.startswith('qwen35.hybrid.') for k in reader.fields): raise ValueError('Base already contains hybrid metadata.')
    donor=np.load(a.donor,allow_pickle=False)
    for name,shape in [('w1',(2304,5120)),('w3',(2304,5120)),('w2',(5120,2304))]:
        if donor[name].shape != shape: raise ValueError('Unexpected donor shape: '+name)
    writer=gguf.GGUFWriter(partial,'qwen35')
    for k,v in reader.fields.items():
        if k.startswith('GGUF.') or k in ['general.architecture','general.name']:continue
        writer.add_key_value(k,v.contents(),v.types[0],v.types[1] if len(v.types)>1 else None)
    writer.add_string('general.name','EXPERIMENTAL Qwen27 + DeepSeek separate shared-expert branch; ALIGNED')
    writer.add_int32('qwen35.hybrid.layer',0)
    writer.add_uint32('qwen35.hybrid.base_width',17408)
    writer.add_uint32('qwen35.hybrid.donor_width',2304)
    writer.add_float32('qwen35.hybrid.scale',0.1)
    writer.add_float32('qwen35.hybrid.clamp',10.0)
    writer.add_string('hybrid.provenance.qwen_revision','37561816e66b5b9153b1b64c10968c18c1e6f4a0')
    writer.add_string('hybrid.provenance.deepseek_revision','1bae0e1088993b9b82de08f59de5be5a34e42170')
    writer.add_string('hybrid.status','Experimental separate additive branch; not quality validated')
    for t in reader.tensors: writer.add_tensor(t.name,t.data,raw_dtype=t.tensor_type)
    for name,key in [('gate','w1'),('up','w3'),('down','w2')]:
        writer.add_tensor(f'blk.0.ffn_{name}_shexp.weight',gguf.quantize(donor[key],gguf.GGMLQuantizationType.Q8_0),raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file(progress=True);writer.close()
    partial.rename(a.output)
    check={t.name:t for t in gguf.GGUFReader(a.output).tensors}
    for t in reader.tensors:
        assert t.tensor_type==check[t.name].tensor_type and np.array_equal(t.shape,check[t.name].shape)
        if t.name in ['blk.0.ffn_gate.weight','blk.0.ffn_up.weight','blk.0.ffn_down.weight']:
            assert np.array_equal(t.data,check[t.name].data)
    print(a.output)

if __name__=='__main__':main()
