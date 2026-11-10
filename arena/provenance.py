import hashlib,json,platform,sys,importlib.metadata
from pathlib import Path
from common import ROOT,RUNS

report={'python':sys.version,'platform':platform.platform(),'packages':{},'sources':{}}
for p in ['numpy','scipy','scikit-learn','lightgbm','pandas','pyarrow']:
    try:report['packages'][p]=importlib.metadata.version(p)
    except importlib.metadata.PackageNotFoundError:pass
sys.path.insert(0,str(ROOT/'.techarena_deps'))
for p in ['torch','implicit']:
    report['packages'][p]=importlib.metadata.version(p)
for name in ['INSTRUCTIONS.md','starter_solution.py','train.csv','evaluation.csv','example_submission.csv']:
    with open(ROOT/name,'rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    original=Path('<LOCAL_HOME>/Downloads')/name
    with open(original,'rb') as f:oh=hashlib.file_digest(f,'sha256').hexdigest()
    assert h==oh,f'Workspace source differs from supplied original: {name}'
    report['sources'][name]={'sha256':h,'bytes':(ROOT/name).stat().st_size,'matches_supplied_download':True}
(RUNS/'provenance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
