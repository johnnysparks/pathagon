"""Resumable, fixed-protocol campaign. Never treats an incomplete file as done."""
import concurrent.futures as cf
import hashlib,json,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=P.parents[1]
W=P/'workspace'
BIN=P/'target/release/pathagon-three-generation-intuition'
LABEL=P/'target/release/label'
PY=ROOT/'.venv-pathagon-gnn/bin/python'
ZERO=ROOT/'data/models/pathfinder-action-transition-v4-xent/transition-policy.json'
W.mkdir(exist_ok=True)
def run(name,cmd):
 stamp=W/(name+'.done.json')
 if stamp.exists():
  meta=json.loads(stamp.read_text())
  assert meta['command']==list(map(str,cmd)),name
  return
 print(name,'started',flush=True)
 start=time.time()
 with (W/(name+'.log')).open('w') as log:
  subprocess.run(list(map(str,cmd)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 stamp.write_text(json.dumps(dict(command=list(map(str,cmd)),seconds=time.time()-start),indent=2))
 print(name,'finished',round(time.time()-start,1),'seconds',flush=True)
def arena(name,model,opponent,games,seed,depth,nodes,independent=False):
 cmd=[BIN,'--model',model,'--output',W/(name+'.jsonl'),'--candidate-id',name,'--games',games,'--workers',4,'--seed',seed,'--max-plies',196,'--opening-random-plies',4,'--depth',depth,'--beam',256,'--nodes',nodes,'--deadline-ms',0]
 if opponent:cmd+=['--opponent-model',opponent]
 if independent:cmd+=['--independent-seeds']
 run(name,cmd)
 return W/(name+'.jsonl')
parents=[ZERO];targets=[]
for generation in range(1,4):
 name=f'g{generation}'
 games=arena(name+'-collection',parents[-1],parents[-1],80,39090900+generation*1000,4,32000,True)
 lines=games.read_text().splitlines()
 assert len(lines)==80
 # Split whole games; source identity and seed partition survive sharding.
 jobs=[]
 for shard in range(4):
  input_path=W/f'{name}-games-{shard}.jsonl'
  input_path.write_text('\n'.join(lines[shard::4])+'\n')
  output=W/f'{name}-labels-{shard}.jsonl'
  jobs.append((f'{name}-label-{shard}',[LABEL,'--model',parents[-1],'--input',input_path,'--output',output,'--depth',6,'--nodes',512000]))
 with cf.ThreadPoolExecutor(max_workers=4) as pool:
  list(pool.map(lambda pair:run(*pair),jobs))
 target=W/f'{name}-labels.jsonl'
 game_by_seed={r['seed']:r for r in map(json.loads,lines)}
 annotated=[]
 for shard in range(4):
  for line in (W/f'{name}-labels-{shard}.jsonl').read_text().splitlines():
   row=json.loads(line)
   source_depth=game_by_seed[row['seed']]['moves'][row['ply']]['completedDepth']
   assert row.get('sourceCompletedDepth',source_depth)==source_depth
   row['sourceCompletedDepth']=source_depth
   annotated.append(json.dumps(row))
 target.write_text('\n'.join(annotated)+'\n')
 targets.append(target)
 output=W/f'{name}-model.json'
 run(name+'-train',[PY,P/'train.py','--parent',parents[-1],'--targets',*targets,'--output',output])
 parents.append(output)
# Do not inspect arenas to choose training or alter subsequent generations.
arena('g0-vs-unguided',ZERO,None,100,49090000,5,256000)
for gen in range(1,4):
 arena(f'g{gen}-vs-g0',parents[gen],ZERO,100,49091000,5,256000)
 arena(f'g{gen}-vs-unguided',parents[gen],None,100,49092000,5,256000)
run('summarize',[PY,P/'summarize.py'])
print('ALL THREE GENERATIONS AND FIXED ARENAS COMPLETE',flush=True)
