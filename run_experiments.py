import json
from experiment import Experiment
from interval_models import MDPSpec
from datetime import datetime

filenames = ['SC_maze_5', 'intercept', 'evade', 'avoid']

GLOBAL_TITLE = "REAL-RUN-" + datetime.now().strftime("%Y%m%d%H%M%S%f") 

for filename in filenames:
    with open('data/input/cfgs/'+filename+'.json') as f:
       load_file = json.load(f)

    for filename in load_file:
        for cfg in load_file[filename]:
            cfg['a_loss'] = 'kld'
            cfg['batch_dim'] = 128
            cfg['train_deterministic'] = False
            cfg['rounds'] = 100

            for setting in [{'dynamic_uncertainty' : False}, {'dynamic_uncertainty' : True}]:
                cfg.update(setting)
                if cfg['dynamic_uncertainty']:
                    for spec in [MDPSpec.Rminmax, MDPSpec.Rminmin]:
                        cfg['specification'] = spec.value
                        exp = Experiment(f'{GLOBAL_TITLE}-{cfg["name"]}-{str(spec.name)}', cfg, 30)
                        exp.execute(True)
                else:
                    spec = MDPSpec.Rminmax
                    cfg['specification'] = spec.value
                    exp = Experiment(f'{GLOBAL_TITLE}-{cfg["name"]}-{str(spec.name)}-BASELINE', cfg, 30)
                    exp.execute(True)
