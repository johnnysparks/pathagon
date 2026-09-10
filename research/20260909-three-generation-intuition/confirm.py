"""Apply the registered selection rule, then independently confirm a qualifying finalist.

Run only after run.py exits successfully. Replays are verified before reporting a
strength gate; promotion still requires representative replay and latency review.
"""
import hashlib,json,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent; ROOT=P.parents[1]; W=P/'workspace'
BIN=P/'target/release/pathagon-three-generation-intuition'
ZERO=ROOT/'data/models/pathfinder-action-transition-v4-xent/transition-policy.json'
SEEDS={'g0':59091000,'unguided':59092000}
# A successful full verification is required, regardless of prior marker files.
subprocess.run([sys.executable,str(P/'verify.py')],check=True)
from summarize import arena
screen=json.loads((P/'results.json').read_text())['arenas']
eligible=[]
for g in range(1,4):
 rates=[screen[f'g{g}-vs-{op}']['pointRate'] for op in SEEDS]
 if min(rates)>=.53:eligible.append((min(rates),sum(rates)/2,g))
selection=dict(rule=json.loads((P/'protocol.json').read_text())['confirmationSelection'],
               freshSeedBases=SEEDS,eligibleGenerations=[x[2] for x in eligible],
               generation=max(eligible)[2] if eligible else None)
selection_path=P/'confirmation-selection.json'
if selection_path.exists():assert json.loads(selection_path.read_text())==selection
selection_path.write_text(json.dumps(selection,indent=2)+'\n')
if not eligible:
 print('No generation qualified for independent confirmation.',flush=True)
 sys.exit(0)
g=max(eligible)[2];model=W/f'g{g}-model.json'
model_hash=hashlib.sha256(model.read_bytes()).hexdigest()
model_metadata={k:json.loads(model.read_text())[k] for k in ['featureOrder','model','schemaVersion']}
previous_seeds={json.loads(line)['seed'] for f in W.glob('*.jsonl')
                if f.name.endswith('-collection.jsonl') or f.name in [n+'.jsonl' for n in screen]
                for line in f.read_text().splitlines()}
base=json.loads(ZERO.read_text());weights=dict(path=241,material=112,capture=887,structure=40,threat=154,edge=74)
results={};audits={};confirmation_seeds=set()
for op,seed in SEEDS.items():
 name=f'confirmation-g{g}-vs-{op}';path=W/(name+'.jsonl');stamp=W/(name+'.done.json')
 cmd=list(map(str,[BIN,'--model',model,'--output',path,'--candidate-id',name,'--games',400,'--workers',4,'--seed',seed,'--max-plies',196,'--opening-random-plies',4,'--depth',5,'--beam',256,'--nodes',256000,'--deadline-ms',0]))
 if op=='g0':cmd+=['--opponent-model',str(ZERO)]
 if stamp.exists():assert json.loads(stamp.read_text())['command']==cmd
 else:
  print(name,'started',flush=True);start=time.monotonic()
  with (W/(name+'.log')).open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  stamp.write_text(json.dumps(dict(command=cmd,seconds=time.monotonic()-start),indent=2)+'\n')
  print(name,'finished',flush=True)
 assert hashlib.sha256(model.read_bytes()).hexdigest()==model_hash
 rows=[json.loads(s) for s in path.read_text().splitlines()];assert len(rows)==400
 for index,r in enumerate(rows):
  candidate='light' if index%2==0 else 'dark';reference='dark' if candidate=='light' else 'light'
  assert r['seed']==seed+index//2 and r['seed'] not in previous_seeds
  assert r['agents'][candidate]==name
  assert r['agents'][reference]==('reference-model' if op=='g0' else 'unguided-control')
  assert r['config']['boardSize']==7 and r['config']['reservePerPlayer']==14 and r['config']['maxPlies']==196
  assert r['plies']==len(r['moves'])
  assert r['agentSpecifications'][candidate]['parameters']['transitionPolicyModel']==model_metadata
  reference_parameters=r['agentSpecifications'][reference].get('parameters') or {}
  assert ('transitionPolicyModel' in reference_parameters)==(op=='g0')
  if op=='g0':assert reference_parameters['transitionPolicyModel']=={k:base[k] for k in model_metadata}
  for side in ['light','dark']:
   m=r['agentSpecifications'][side]['manifest']
   assert m['depth']==5 and m['beam']==256 and m['nodeBudget']==256000 and m['evaluatorWeights']==weights
  if index%2:
   assert [m['action'] for m in r['moves'][:4]]==[m['action'] for m in rows[index-1]['moves'][:4]]
 new_seeds={r['seed'] for r in rows};assert not new_seeds & confirmation_seeds;confirmation_seeds.update(new_seeds)
 output=subprocess.check_output([str(P/'target/release/audit'),'--arena',str(path)],text=True)
 audits[op]=json.loads(output);assert audits[op]['games']==400
 (W/(name+'.audit.json')).write_text(output)
 results[op]=arena(path)
 h=hashlib.sha256()
 for f in sorted((ROOT/'pathagon/engine-rs/src').rglob('*.rs')):
  h.update(str(f.relative_to(ROOT)).encode());h.update(f.read_bytes())
 assert h.hexdigest()==json.loads((P/'protocol.json').read_text())['engineSourceSha256']
 failed=[anchor for anchor,r in results.items() if r['pointRate']<.53 or r['pairedBootstrap95'][0]<=.5]
 report=dict(generation=g,modelSha256=model_hash,arenas=results,nativeAudits=audits,
             allAnchorsComplete=len(results)==2,
             strengthGatePassed=len(results)==2 and all(r['pointRate']>=.53 and r['pairedBootstrap95'][0]>.5 for r in results.values()),
             strengthGateFailedAgainst=failed,
             promotionDecision='not-promoted-strength-gate-failed' if failed else 'pending',
             deploymentLatency='not-run-strength-gate-failed' if failed else 'pending',
             promotionPending=[] if failed else ['representative replay review','deployed latency gate'])
 (P/'confirmation-results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
