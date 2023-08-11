import json
from experiment import Experiment

from interval_models import MDPSpec

filename = 'intercept'
with open('data/input/cfgs/'+filename+'.json') as f:
   load_file = json.load(f)

for filename in load_file:
   for cfg in load_file[filename]:
      cfg['a_loss'] = 'kld'
      cfg['train_deterministic'] = False
      cfg['specification'] = MDPSpec.Rminmax.value
      exp = Experiment("INTERCEPT-" + cfg["name"], cfg, 30)
      try:
         exp.execute(True)
      except Exception as e:
         print("Run failed!", e)
         print(e, file=open(f"./{exp.name}.exception", "w"))
