import json
from experiment import Experiment
from interval_models import MDPSpec

filename = 'obstacle'
with open('data/input/cfgs/'+filename+'.json') as f:
   load_file = json.load(f)
   cfg = load_file[filename][0]

for cfg in load_file[filename]:
   cfg['a_loss'] = 'kld'
   cfg['policy'] = 'qumdp'
   cfg['train_deterministic'] = False
   cfg['specification'] = MDPSpec.Rminmax.value
   exp = Experiment(cfg["name"] + "_WIP", cfg, 100)
   exp.execute(False)
