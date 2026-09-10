"""Check training features and logits against the exported Rust model."""
import argparse,json,subprocess
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--targets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(1)
m=json.loads(a.model.read_text())
net=torch.nn.Sequential(torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,1))
for layer,raw in zip((net[0],net[2],net[4]),m['layers']):
 layer.weight.data.copy_(torch.tensor(raw['weights']));layer.bias.data.copy_(torch.tensor(raw['bias']))
probe=Path(__file__).resolve().parent/'target/release/score'
raw=subprocess.check_output([str(probe),str(a.model),str(a.targets)],text=True)
native=[json.loads(x) for x in raw.splitlines()]
rows=[json.loads(x) for x in a.targets.read_text().splitlines()]
assert len(rows)==len(native)
max_error=0.;actions=0;argmax_mismatches=0
with torch.no_grad():
 for r,n in zip(rows,native):
  assert r['id']==n['id']
  x=(torch.tensor(r['features'])-torch.tensor(m['mean']))/torch.tensor(m['scale'])
  expected=net(x).flatten();actual=torch.tensor(n['scores'])
  assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
  max_error=max(max_error,float((expected-actual).abs().max()))
  actions+=len(actual)
  if expected.argmax()!=actual.argmax():
   # Exact/tiny ties may have different floating-point reduction ordering.
   assert abs(float(expected.max()-expected[actual.argmax()]))<1e-4
   argmax_mismatches+=1
assert max_error<1e-4,max_error
report=dict(model=str(a.model),positions=len(rows),actions=actions,nativeFeaturesMatched=True,maxAbsoluteLogitError=max_error,nearTieArgmaxMismatches=argmax_mismatches,tolerance=1e-4)
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
