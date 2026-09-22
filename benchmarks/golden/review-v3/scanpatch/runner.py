"""Frozen post-diagnostic regression from 4e46c86 to 1219108; no new model calls."""
import hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path('/home/lockr/projects/seif-pii-golden-improvements')
BASE=ROOT/'output/golden-improvements/external-scanpatch'
OUT=ROOT/'output/generalized-fixes-v3/scanpatch-release'
spec=importlib.util.spec_from_file_location('scan_runner',ROOT/'output/golden-improvements/scanpatch_frozen_regression.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
r.disable_network();OUT.mkdir(exist_ok=False)
source_sha=r.snapshot_candidate(ROOT,OUT/'candidate-source')
expected=json.loads((ROOT/'output/generalized-fixes-v3/golden-release.json').read_text())['source_sha256']
assert all(expected[p]==h for p,h in source_sha.items())
r.run_worker(BASE,OUT/'candidate-source',OUT/'candidate-predictions.jsonl')
support,rows,cache,evidence=r.score_evidence_and_rows(BASE,r.MAIN)
before=r.read_predictions(BASE/'final-v2/candidate-predictions.jsonl');after=r.read_predictions(OUT/'candidate-predictions.jsonl')
scopes=r.make_scopes(support,rows,cache,{'before':before,'after':after})
old=json.loads((BASE/'final-v2/report.json').read_text())
for name,s in scopes.items():
 for profile in ('rules','hybrid'):
  assert s['systems']['before_'+profile]==old['scopes'][name]['systems']['current_'+profile]
transitions={}
def chars(spans):return {(t,i) for t,s,e in spans if t in support.COMMON for i in range(s,e)}
for profile in ('rules','hybrid'):
 counts={'changed_cases':0,'lost_tp_characters':0,'new_fp_characters':0,'gained_tp_characters':0,'removed_fp_characters':0,'cases_with_lost_tp':0,'cases_with_new_fp':0}
 for row in rows:
  k=row['id'];t=row['text'];gold=chars(row['merged_gold'])
  a=chars(support.merge_adjacent(t,support.coarsen(before[profile][k],support.LOCAL_MAP)))
  b=chars(support.merge_adjacent(t,support.coarsen(after[profile][k],support.LOCAL_MAP)))
  lost=(a&gold)-(b&gold);fp=(b-gold)-(a-gold)
  counts['changed_cases']+=before[profile][k]!=after[profile][k];counts['lost_tp_characters']+=len(lost);counts['new_fp_characters']+=len(fp)
  counts['gained_tp_characters']+=len((b&gold)-(a&gold));counts['removed_fp_characters']+=len((a-gold)-(b-gold))
  counts['cases_with_lost_tp']+=bool(lost);counts['cases_with_new_fp']+=bool(fp)
 transitions[profile]=counts
report={'schema_version':1,'before_revision':'4e46c86a6f6c7bf91e9525968981ae47d7984aba','after_revision':'12191083eea106edf85c7bbc6e60a140faf1b4c2','stage':'Post-diagnostic regression on previously used data; not a fresh holdout','network_calls':0,'new_model_calls':0,'previous_metrics_reproduced':True,'source_sha256':source_sha,'input_sha256':{'corpus':r.sha(r.MAIN/'output/external-bench-next/scanpatch-test.parquet'),'ner_cache':r.sha(r.MAIN/'output/external-bench-next/scanpatch-predictions-first.jsonl'),'baseline_predictions':r.sha(BASE/'final-v2/candidate-predictions.jsonl')},'cases':532,'primary_cases':164,'profile':'same frozen PERSON+LOCATION with original mappings','transitions':transitions,'scopes':scopes,'limitations':['Previously inspected single terminal-initial punctuation mismatch remains; no source-specific punctuation rule introduced.','The displayed typed-character metric counts punctuation; actual mask preserves punctuation.','This is offline quality replay and does not measure HTTP RPS.']}
r.save_json(OUT/'report.json',report);r.save_json(OUT/'summary.json',r.summarize_scopes(scopes));r.save_json(OUT/'validation.json',{'source_matches_golden_final':True,'previous_metrics_exactly_reproduced':True,'predictions_byte_identical':r.sha(BASE/'final-v2/candidate-predictions.jsonl')==r.sha(OUT/'candidate-predictions.jsonl'),'transitions':transitions})
print(json.dumps({'transitions':transitions,'primary':r.summarize_scopes(scopes)['primary_common_whole_cases']},indent=2))
