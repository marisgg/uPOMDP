import json
from experiment import Experiment
from interval_models import MDPSpec

filename = 'evade'
with open('data/input/cfgs/'+filename+'.json') as f:
   load_file = json.load(f)

for filename in load_file:
   for cfg in load_file[filename]:
      for loss in ['kld']:
         for policy in ["qumdp"]:
            cfg["policy"] = policy
            cfg['a_loss'] = loss
            cfg['train_deterministic'] = False
            cfg['specification'] = MDPSpec.Rminmax.value
            exp = Experiment("EVADE" + cfg["name"] + policy + loss, cfg, 30)
            try:
               exp.execute(True)
            except Exception as e:
               print("Run failed!", e)
               print(e, file=open(f"./{exp.name}.exception", "w"))
