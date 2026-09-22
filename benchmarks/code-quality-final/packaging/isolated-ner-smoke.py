import hashlib,json,subprocess,tempfile,shutil,sys
from pathlib import Path
root=Path.cwd()
files=['scripts/ner_service.py','seif/__init__.py','seif/async_callbacks.py']
with tempfile.TemporaryDirectory(prefix='seif-ner-image-') as directory:
 target=Path(directory)
 for name in files:
  (target/name).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target/name)
 code="""import sys,json,inspect
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from fastapi.testclient import TestClient
import fastapi,starlette
from unittest.mock import patch
from scripts import ner_service
from seif import async_callbacks
assert str(ner_service.__file__).startswith(sys.argv[1])
assert str(async_callbacks.__file__).startswith(sys.argv[1])
class Analyzer:
 def analyze(self,**kwargs): return []
def no_worker(*args,**kwargs): raise AssertionError('Unexpected framework threadpool dispatch')
app=ner_service.create_app(ner_service.NerSettings(demo=True), analyzer_factory=Analyzer)
with TestClient(app) as client, patch('starlette._exception_handler.run_in_threadpool',no_worker):
 assert client.get('/health').status_code==200
 response=client.post('/analyze',json={'text':1});assert response.status_code==422
 response=client.get('/missing');assert response.status_code==404
 response=client.post('/analyze',json={'text':'Example synthetic text'});assert response.status_code==200
print(json.dumps({'status':'PASS','python':sys.version,'fastapi':fastapi.__version__,'starlette':starlette.__version__,'checks':['isolated_container_imports','health','validation_422','http_404','analyze_200','no_handler_threadpool']}))
"""
 result=subprocess.check_output(['/tmp/seif-presidio-313/bin/python','-I','-c',code,str(target)],cwd=target,text=True)
 report=json.loads(result);report['source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
 Path('output/code-quality/isolated-ner-smoke.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report))
