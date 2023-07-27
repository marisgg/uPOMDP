import json
from experiment import Experiment

filename = 'SC_mazes'
with open('data/input/cfgs/'+filename+'.json') as f:
   load_file = json.load(f)

for filename in load_file:
   for cfg in load_file[filename]:
      for policy in ["qumdp", "umdp"]:
         cfg["policy"] = policy
         exp = Experiment(cfg["name"] + policy, cfg, 100)
         exp.execute(False)
