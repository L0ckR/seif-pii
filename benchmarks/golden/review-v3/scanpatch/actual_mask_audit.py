"""Compare actual alphanumeric mask coverage on frozen full Scanpatch cases."""
import argparse,json
from pathlib import Path
import pyarrow.parquet as parquet
p=argparse.ArgumentParser();p.add_argument('--after',type=Path,required=True);a=p.parse_args()
root=Path('/home/lockr/projects/seif-pii-golden-improvements')
before=root/'output/golden-improvements/external-scanpatch/final-v2/candidate-predictions.jsonl'
rows=parquet.read_table(root.parent/'seif-pii/output/external-bench-next/scanpatch-test.parquet').to_pylist()
old={v['id']:v['predictions'] for v in map(json.loads,before.read_text().splitlines())}
new={v['id']:v['predictions'] for v in map(json.loads,(a.after/'candidate-predictions.jsonl').read_text().splitlines())}
result={}
for profile in ('rules','hybrid'):
 changed=[];boundary=[]
 for i,row in enumerate(rows):
  key=f'test_{i:04d}';text=row['text']
  def positions(spans):return {j for _,s,e in spans for j in range(s,e) if text[j].isalnum()}
  if positions(old[key][profile])!=positions(new[key][profile]):changed.append(key)
  if old[key][profile]!=new[key][profile]:boundary.append(key)
 result[profile]={'cases':len(rows),'actual_changed_masks':changed,'entity_boundary_changes':boundary}
 assert not changed
with (a.after/'actual-mask-validation.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
