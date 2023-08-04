import json
from experiment import Experiment

filename = 'SC_maze_5'
with open('data/input/cfgs/'+filename+'.json') as f:
   load_file = json.load(f)

for filename in load_file:
   for cfg in load_file[filename]:
      for loss in ['kld']:
         for policy in ["qumdp", "umdp"]:
            cfg["policy"] = policy
            cfg['a_loss'] = loss
            exp = Experiment("LP_TEST_" + cfg["name"] + policy + loss, cfg, 4)
            exp.execute(False)
