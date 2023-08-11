import json
from experiment import Experiment
from interval_models import MDPSpec

filename = 'SC_maze_5'
with open('data/input/cfgs/'+filename+'.json') as f:
   load_file = json.load(f)

for filename in load_file:
   for i, cfg in enumerate(load_file[filename]):
      if i == 0:
         continue
      cfg['policy'] = 'qumdp'
      cfg['a_loss'] = 'kld'
      cfg['train_deterministic'] = False
      cfg['specification'] = MDPSpec.Rminmax.value
      exp = Experiment(f"MAZE-SC-5-{i}-{cfg['name']}", cfg, 30)
      try:
         exp.execute(True)
      except Exception as e:
         print("Run failed!", e)
         print(e, file=open(f"./{exp.name}.exception", "w"))
