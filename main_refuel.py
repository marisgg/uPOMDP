import json
from experiment import Experiment
from interval_models import MDPSpec

filename = 'refuel'
with open('data/input/cfgs/'+filename+'.json') as f:
   load_file = json.load(f)
   cfg = load_file[filename][0]

for cfg in load_file[filename]:
   for policy in ["qumdp"]:
      cfg["policy"] = policy
      cfg['a_loss'] = 'kld'
      cfg['train_deterministic'] = False
      cfg['specification'] = MDPSpec.Rminmax.value
      exp = Experiment(cfg["name"] + "_Large_" + policy, cfg, 4)
      exp.execute(False)
