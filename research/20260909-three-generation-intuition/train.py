"""Fine-tune the parent, preserving architecture and normalization exactly."""
import argparse, copy, hashlib, json, random
from pathlib import Path
import torch
p=argparse.ArgumentParser()
p.add_argument('--parent',type=Path,required=True)
p.add_argument('--targets',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
torch.set_num_threads(1)
torch.manual_seed(290909)
base=json.loads(a.parent.read_text())
net=torch.nn.Sequential(torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,1))
for layer,raw in zip((net[0],net[2],net[4]),base['layers']):
 layer.weight.data.copy_(torch.tensor(raw['weights']));layer.bias.data.copy_(torch.tensor(raw['bias']))
parent=copy.deepcopy(net).eval()
mean,scale=torch.tensor(base['mean']),torch.tensor(base['scale'])
rows=[];seen=set()
for path in a.targets:
 for line in path.read_text().splitlines():
  r=json.loads(line)
  if r['completedDepth']<=r['sourceCompletedDepth']:continue
  # State-disjoint partitions: heldout states never become training examples.
  r['sourceFile']=str(path)
  rows.append(r)
heldout_states={r['state'] for r in rows if r['seed']%5==0}
prepared=[]
for r in rows:
 if r['state'] in seen:continue
 seen.add(r['state'])
 x=(torch.tensor(r['features'])-mean)/scale
 prepared.append((r,x,r['state'] in heldout_states))
train=[r for r in prepared if not r[2]];test=[r for r in prepared if r[2] and r[0]['sourceFile']==str(a.targets[0]) and r[0]['seed']%5==0]
assert train and test
optimizer=torch.optim.AdamW(net.parameters(),lr=0.0003,weight_decay=0.002)
def metrics(model,examples):
 with torch.no_grad():
  correct=top3=0;loss=0
  for r,x,_ in examples:
   scores=model(x).flatten(); t=r['target']
   correct+=int(scores.argmax()==t);top3+=int(t in scores.topk(min(3,len(scores))).indices)
   loss+=float(torch.nn.functional.cross_entropy(scores[None,:],torch.tensor([t])))
 return dict(roots=len(examples),top1=correct/len(examples),top3=top3/len(examples),xent=loss/len(examples))
before=metrics(net,test)
losses=[]
for epoch in range(20):
 random.Random(290909+epoch).shuffle(train)
 total=0
 for r,x,_ in train:
  logits=net(x).flatten()
  with torch.no_grad(): old=parent(x).flatten().softmax(0)
  loss=torch.nn.functional.cross_entropy(logits[None,:],torch.tensor([r['target']]))+0.2*torch.nn.functional.kl_div(logits.log_softmax(0),old,reduction='sum')
  optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);optimizer.step()
  total+=float(loss.detach())
 losses.append(total/len(train))
result=copy.deepcopy(base)
for raw,layer in zip(result['layers'],(net[0],net[2],net[4])):
 raw['weights']=layer.weight.detach().tolist();raw['bias']=layer.bias.detach().tolist()
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result)+'\n')
report=dict(parentSha256=hashlib.sha256(a.parent.read_bytes()).hexdigest(),modelSha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),train=metrics(net,train),heldoutBefore=before,heldoutAfter=metrics(net,test),losses=losses,epochs=20,lr=0.0003,klWeight=0.2,normalizationFrozen=True)
a.output.with_suffix('.report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
