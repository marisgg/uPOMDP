import json
from experiment import Experiment
from interval_models import MDPSpec

filename = 'avoid'
with open('data/input/cfgs/'+filename+'.json') as f:
   load_file = json.load(f)

for filename in load_file:
   for cfg in load_file[filename]:
         for policy in ["qumdp"]:
            cfg["policy"] = policy
            cfg['a_loss'] = 'kld'
            cfg['train_deterministic'] = False
            cfg['specification'] = MDPSpec.Rminmax.value
            exp = Experiment("rocks_test" + cfg["name"] + policy, cfg, 4)
            exp.execute(False)
