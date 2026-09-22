"""Offline detector CPU timing; deliberately excludes HTTP and model inference."""
import argparse
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--dataset',type=Path,required=True)
parser.add_argument('--cache',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.source.resolve()))
from seif.detector import Span,detect,merge_ner_candidates
cases=[json.loads(line) for line in args.dataset.read_text().splitlines()]
cache={r['case_id']:r for r in map(json.loads,args.cache.read_text().splitlines())}
model={cid:[Span(e['start'],e['end'],e['entity_type'],e['score'],'frozen-ner') for e in row['entities']] for cid,row in cache.items()}
def call(row,hybrid):
    result=detect(row['text'])
    return merge_ner_candidates(row['text'],result,model[row['case_id']]) if hybrid else result
for row in cases:
    call(row,True)
profiles={}
for profile in ('rules','hybrid'):
    samples=[]; totals=[]
    for repeat in range(9):
        order=list(cases);random.Random(20260922+repeat).shuffle(order)
        start=time.perf_counter_ns()
        for row in order:
            before=time.perf_counter_ns();call(row,profile=='hybrid')
            samples.append((time.perf_counter_ns()-before)/1000)
        totals.append((time.perf_counter_ns()-start)/1e9)
    ordered=sorted(samples)
    profiles[profile]={'calls':len(samples),'mean_us':statistics.mean(samples),'median_us':statistics.median(samples),'p95_us':ordered[int(.95*(len(ordered)-1))],'p99_us':ordered[int(.99*(len(ordered)-1))],'max_us':max(samples),'median_corpus_seconds':statistics.median(totals)}
long_inputs={
    'benign_approximately_100k_chars':('Обычное предложение без личных сведений. '*3000)[:100000],
    'repeated_address_prefixes':('Доставьте в Условную область, город Примерово, ул. Тихая, д. 3. '*1600),
    'repeated_issuer_prose':('Чек выдан покупателю. ОВД района принимает посетителей. '*1800),
}
long_results={}
for name,text in long_inputs.items():
    timings=[]
    for _ in range(3):
        start=time.perf_counter_ns();spans=detect(text);timings.append((time.perf_counter_ns()-start)/1e9)
    long_results[name]={'characters':len(text),'median_seconds':statistics.median(timings),'max_seconds':max(timings),'spans':len(spans)}
result={'schema_version':1,'python':platform.python_version(),'gil_enabled':sys._is_gil_enabled() if hasattr(sys,'_is_gil_enabled') else None,'dataset_sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest(),'source_sha256':{str(p.relative_to(args.source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((args.source/'seif').glob('*.py'))},'profiles':profiles,'large_payloads':long_results,'limitations':['Offline CPU timing; cached NER, no HTTP, Redis or network model calls.','Sequential local measurements on shared hardware; not service RPS or SLA validation.']}
with args.output.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({'profiles':profiles,'large_payloads':long_results}))
