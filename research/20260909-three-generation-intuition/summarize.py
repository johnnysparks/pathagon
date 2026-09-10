"""Compact campaign evidence, with paired-seed bootstrap intervals."""
import collections,hashlib,json,random,statistics
from pathlib import Path
P=Path(__file__).resolve().parent; W=P/'workspace'

def arena(path):
 rows=[json.loads(s) for s in path.read_text().splitlines()]
 points=[];pairs=collections.defaultdict(list);color=collections.defaultdict(list)
 nodes=collections.defaultdict(int);depths=collections.Counter()
 for i,r in enumerate(rows):
  side='light' if i%2==0 else 'dark'
  point=0.5 if r['winner'] is None else float(r['winner']==side)
  points.append(point);pairs[r['seed']].append(point);color[side].append(point)
  assert r['config']['maxPlies']==196
  for m in r['moves']:
   nodes['candidate' if m['player']==side else 'control']+=m['nodes']
   depths[m['completedDepth']]+=1
 pairpoints=[statistics.mean(v) for v in pairs.values()]
 assert all(len(v)==2 for v in pairs.values())
 rng=random.Random(290909)
 boot=sorted(statistics.mean(rng.choices(pairpoints,k=len(pairpoints))) for _ in range(10000))
 return dict(games=len(rows),wins=points.count(1),losses=points.count(0),draws=points.count(.5),pointRate=statistics.mean(points),pairedBootstrap95=[boot[250],boot[9749]],byColor={k:statistics.mean(v) for k,v in color.items()},terminations=dict(collections.Counter(r['reason'] for r in rows)),uniqueTrajectories=len({json.dumps([m['action'] for m in r['moves']],sort_keys=True) for r in rows}),nodes=dict(nodes),gameWallSeconds=dict(mean=statistics.mean(r['researchWallSeconds'] for r in rows),max=max(r['researchWallSeconds'] for r in rows)))
report={'arenas':{},'generations':{}}
for g in range(1,4):
 games=[json.loads(x) for x in (W/f'g{g}-collection.jsonl').read_text().splitlines()]
 labels=[json.loads(x) for x in (W/f'g{g}-labels.jsonl').read_text().splitlines()]
 phase_coverage={}
 for phase,flag in [('placement',0),('relocation',1)]:
  subset=[r for r in labels if int(r['features'][0][9])==flag]
  phase_coverage[phase]=dict(selected=len(subset),eligible=sum(r['completedDepth']>r['sourceCompletedDepth'] for r in subset))
 report['generations'][str(g)]=dict(phaseCoverage=phase_coverage,games=len(games),terminations=dict(collections.Counter(r['reason'] for r in games)),uniqueTrajectories=len({json.dumps([m['action'] for m in r['moves']],sort_keys=True) for r in games}),relocationPlies=sum(m['action']['kind']=='relocate' for r in games for m in r['moves']),labels=len(labels),eligibleLabels=sum(r['completedDepth']>r['sourceCompletedDepth'] for r in labels),completedDepths=dict(collections.Counter(r['completedDepth'] for r in labels)),changedTeacherActions=sum(r['changedAction'] for r in labels),training=json.loads((W/f'g{g}-model.report.json').read_text()))
for name in ['g0-vs-unguided']+[f'g{g}-vs-{op}' for g in range(1,4) for op in ['g0','unguided']]:
 report['arenas'][name]=arena(W/(name+'.jsonl'))
(P/'results.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Three-generation screening results','', '| Match | W–L–D | Points | Paired bootstrap 95% |','|---|---:|---:|---:|']
for name,r in report['arenas'].items():
 lo,hi=r['pairedBootstrap95'];lines.append(f"| {name} | {r['wins']}–{r['losses']}–{r['draws']} | {r['pointRate']:.1%} | {lo:.1%}–{hi:.1%} |")
lines+=['','Intervals resample opening pairs; repeated trajectories and opponent specialization can still limit generalization. These are screening results, not automatic promotion evidence.']
lines += ['', 'The independent finalist results and final promotion decision are in [CONFIRMATION.md](CONFIRMATION.md).']
(P/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(report,indent=2))
