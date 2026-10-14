"""Deterministic Wednesday controlled-gauge memory pilot."""
import argparse
import json
from pathlib import Path
from wednesday.core import sweep

p=argparse.ArgumentParser()
p.add_argument('--seed',type=int,default=41)
p.add_argument('--trials',type=int,default=200)
p.add_argument('--sigma',type=float,default=.85)
p.add_argument('--shots-per-item',type=int,default=2)
p.add_argument('--out',type=Path,default=Path('results/pilot.json'))
a=p.parse_args()
if not 1<=a.shots_per_item<=6 or a.sigma<0:
    p.error('shots-per-item must be 1..6; sigma >=0')
res=sweep(a.seed,a.trials,a.sigma,a.shots_per_item)
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(res,indent=2)+'\n')
for row in res['records']:
    m=row['metrics']
    print(row['scenario'],row['gauge'],
          'exact',m['exact']['mean'],'single',m['single_one_shot']['mean'],
          'pair',m['paired_one_shot']['mean'],
          'uniform',m['noisy_uniform']['mean'],
          'confidence',m['noisy_confidence']['mean'],
          'rewrite:',m['preserved_changed_faces']['mean'],'vs naive',m['naive_changed_faces']['mean'])
print('saved',a.out)
