"""Verify finished campaign against the frozen protocol, not completion markers alone."""
import hashlib,json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1];W=P/'workspace'
protocol=json.loads((P/'protocol.json').read_text())
h=hashlib.sha256()
for f in sorted((R/'pathagon/engine-rs/src').rglob('*.rs')):
 h.update(str(f.relative_to(R)).encode());h.update(f.read_bytes())
assert h.hexdigest()==protocol['engineSourceSha256'],'engine changed'
zero=R/'data/models/pathfinder-action-transition-v4-xent/transition-policy.json'
models=[zero]+[W/f'g{g}-model.json' for g in range(1,4)]
base=json.loads(zero.read_text())
for i,p in enumerate(models[1:],1):
 m=json.loads(p.read_text());report=json.loads(p.with_suffix('.report.json').read_text())
 assert {k:v for k,v in m.items() if k!='layers'}=={k:v for k,v in base.items() if k!='layers'}
 assert [(len(l['weights']),len(l['weights'][0]),len(l['bias'])) for l in m['layers']]==[(32,32,32),(32,32,32),(1,32,1)]
 assert report['parentSha256']==hashlib.sha256(models[i-1].read_bytes()).hexdigest()
 assert report['modelSha256']==hashlib.sha256(p.read_bytes()).hexdigest()
 assert report['modelSha256']!=report['parentSha256']
 assert report['epochs']==20 and report['lr']==0.0003 and report['klWeight']==0.2
 subprocess.run([str(R/'.venv-pathagon-gnn/bin/python'),str(P/'parity.py'),'--model',str(p),'--targets',str(W/f'g{i}-labels.jsonl'),'--output',str(W/f'g{i}-parity.json')],check=True)
weights=dict(path=241,material=112,capture=887,structure=40,threat=154,edge=74)
training_seeds=set();eval_seeds=set();audits={}
names=[f'g{g}-collection' for g in range(1,4)]+['g0-vs-unguided']+[f'g{g}-vs-{op}' for g in range(1,4) for op in ['g0','unguided']]
for name in names:
 path=W/(name+'.jsonl');rows=[json.loads(l) for l in path.read_text().splitlines()]
 collection=name.endswith('collection');assert len(rows)==(80 if collection else 100)
 marker=json.loads((W/(name+'.done.json')).read_text())
 command=marker['command']; model_path=command[command.index('--model')+1]
 generation=int(name[1]);expected_model=models[max(0,generation-1) if collection else generation]
 assert Path(model_path)==expected_model
 if collection or name.endswith('vs-g0'):
  expected_opponent=expected_model if collection else zero
  assert Path(command[command.index('--opponent-model')+1])==expected_opponent
 else:assert '--opponent-model' not in command
 assert command[command.index('--deadline-ms')+1]=='0'
 for index,r in enumerate(rows):
  candidate_side='light' if index%2==0 else 'dark'
  reference_side='dark' if candidate_side=='light' else 'light'
  assert r['agents'][candidate_side]==name
  expected_reference='reference-model' if collection or name.endswith('vs-g0') else 'unguided-control'
  assert r['agents'][reference_side]==expected_reference
  assert r['agentSpecifications'][candidate_side]['parameters']['transitionPolicyModel']['model']==base['model']
  reference_parameters=r['agentSpecifications'][reference_side].get('parameters') or {}
  assert ('transitionPolicyModel' in reference_parameters)==(expected_reference=='reference-model')
  assert r['config']['boardSize']==7 and r['config']['reservePerPlayer']==14 and r['config']['maxPlies']==196
  assert r['plies']==len(r['moves'])
  for side in ['light','dark']:
   m=r['agentSpecifications'][side]['manifest']
   assert m['depth']==(4 if collection else 5) and m['beam']==256 and m['nodeBudget']==(32000 if collection else 256000)
   assert m['evaluatorWeights']==weights
  (training_seeds if collection else eval_seeds).add(r['seed'])
 if not collection:
  for i in range(0,len(rows),2):
   a,b=rows[i:i+2];assert a['seed']==b['seed']
   assert [m['action'] for m in a['moves'][:4]]==[m['action'] for m in b['moves'][:4]]
 # Independently replay the authoritative game transitions and termination.
 output=subprocess.check_output([str(P/'target/release/audit'),'--arena',str(path)],text=True)
 audit=json.loads(output);assert audit['games']==len(rows)
 (W/(name+'.audit.json')).write_text(output)
 audits[name]=audit
assert not training_seeds & eval_seeds
for g in range(1,4):
 rows=[json.loads(l) for l in (W/f'g{g}-labels.jsonl').read_text().splitlines()]
 assert rows and all(r['teacherDepth']==6 and r['teacherNodeBudget']==512000 for r in rows)
 assert all(len(r['features'])==len(r['actions']) and 0<=r['target']<len(r['actions']) and all(len(f)==32 for f in r['features']) for r in rows)
 assert all(r['seed'] in training_seeds for r in rows)
 collection={r['seed']:r for r in (json.loads(l) for l in (W/f'g{g}-collection.jsonl').read_text().splitlines())}
 assert all(r['sourceCompletedDepth']==collection[r['seed']]['moves'][r['ply']]['completedDepth'] for r in rows)
 assert any(r['features'][0][9]==1 and r['completedDepth']>r['sourceCompletedDepth'] for r in rows), 'No eligible relocation supervision'
reuse=json.loads((W/'g1-source-reuse.json').read_text())
for name,digest in reuse['sourceFiles'].items():
 assert hashlib.sha256((W/name).read_bytes()).hexdigest()==digest
assert protocol['version']==2
assert 'strictly exceeds' in protocol['labeling']['eligibility']
result=dict(engineFrozen=True,architectureAndNormalizationFrozen=True,nativeFeatureAndLogitParityVerified=True,threeSequentialWeightUpdates=True,collectionAndDeploymentBudgetsFrozen=True,pairedOpeningsVerified=True,trainingEvaluationSeedsDisjoint=True,nativeAudits=audits)
(P/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print('Verified three sequential generations, fixed budgets, paired evaluations, and all replay terminations.')
