import json
from experiment import Experiment

filename = 'SC_mazes'
with open('data/input/cfgs/'+filename+'.json') as f:
   load_file = json.load(f)

for filename in load_file:
   for cfg in load_file[filename]:
      for loss in ['mse', 'cce']:
         for policy in ["qumdp", "umdp"]:
            cfg["policy"] = policy
            cfg['a_loss'] = loss
            exp = Experiment("LP_TEST_" + cfg["name"] + policy + loss, cfg, 100)
            exp.execute(False)
