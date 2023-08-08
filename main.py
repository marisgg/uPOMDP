import json
from experiment import Experiment
from interval_models import MDPSpec

filename = 'intercept'
with open('data/input/cfgs/'+filename+'.json') as f:
   load_file = json.load(f)

for filename in load_file:
   for cfg in load_file[filename]:
      cfg['a_loss'] = 'cce'
      cfg['train_deterministic'] = True
      cfg['specification'] = MDPSpec.Rminmax.value
      exp = Experiment("Tester-" + cfg["name"], cfg, 1)
      exp.execute(False)
