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
            
            cfg['batch_dim'] = 128
            cfg['rounds'] = 100

            for det_setting in [{'train_deterministic' : False}, {'train_deterministic' : True}]:
                cfg.update(det_setting)

                if cfg['train_deterministic']:
                    cfg['a_loss'] = 'cce'
                else:
                    cfg['a_loss'] = 'kld'

                for setting in [{'dynamic_uncertainty' : True}, {'dynamic_uncertainty' : False}]:
                    cfg.update(setting)
                    if cfg['dynamic_uncertainty']:
                        for spec in [MDPSpec.Rminmax, MDPSpec.Rminmin]:
                            cfg['specification'] = spec.value
                            exp = Experiment(f'{GLOBAL_TITLE}/{"train_deterministic" if cfg["train_deterministic"] else "train_stochastic"}/{cfg["name"]}-{str(spec.name)}', cfg, 30)
                            exp.execute(True)
                    else:
                        spec = MDPSpec.Rminmax
                        cfg['specification'] = spec.value
                        exp = Experiment(f'{GLOBAL_TITLE}/{"train_deterministic" if cfg["train_deterministic"] else "train_stochastic"}/{cfg["name"]}-{str(spec.name)}-BASELINE', cfg, 30)
                        exp.execute(True)
