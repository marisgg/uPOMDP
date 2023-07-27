import json
from experiment import Experiment

filename = 'maze2'
with open('data/input/cfgs/'+filename+'.json') as f:
   load_file = json.load(f)
   cfg = load_file[filename][0]

for cfg in load_file[filename]:
   cfg['a_loss'] = 'mse'
   exp = Experiment(cfg["name"] + "_Large", cfg, 2)
   exp.execute(False)
