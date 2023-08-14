import json
from experiment import Experiment
from interval_models import MDPSpec
from datetime import datetime

filenames = ['SC_maze_5'] #, 'intercept', 'evade', 'avoid']

GLOBAL_TITLE = "SECOND-RUN-" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S%f")
GLOBAL_TITLE = "DRY-RUN-14-08-2023_15:05"

NUM_RUNS = 1
MULTITHREAD = False

all_seeds = list(range(100))

run_seeds = all_seeds[:30]

for filename in filenames:
    with open('data/input/cfgs/'+filename+'.json') as f:
       load_file = json.load(f)

    for filename in load_file:
        for cfg in load_file[filename]:
            
            cfg['batch_dim'] = 4
            cfg['rounds'] = 2

            for det_setting in [{'train_deterministic' : False}, {'train_deterministic' : True}]:
                cfg.update(det_setting)

                if cfg['train_deterministic']:
                    continue

                if cfg['train_deterministic']:
                    cfg['a_loss'] = 'cce'
                else:
                    cfg['a_loss'] = 'kld'
                
                for bottle_dim in [{"bottleneck_dim" : 1}, {"bottleneck_dim" : 2}]:
                    cfg.update(bottle_dim)

                    for setting in [{'dynamic_uncertainty' : True}, {'dynamic_uncertainty' : False}]:
                        cfg.update(setting)
                        if cfg['dynamic_uncertainty']:
                            for spec in [MDPSpec.Rminmax, MDPSpec.Rminmin]:
                                cfg['specification'] = spec.value
                                exp = Experiment(f'{GLOBAL_TITLE}/{"train_deterministic" if cfg["train_deterministic"] else "train_stochastic"}/maxk={3**cfg["bottleneck_dim"]}/{cfg["name"]}-{str(spec.name)}', cfg, NUM_RUNS)
                                try:
                                    exp.execute(MULTITHREAD, run_seeds)
                                except Exception as e:
                                    print("Run failed: ", e)
                        else:
                            spec = MDPSpec.Rminmax
                            cfg['specification'] = spec.value
                            exp = Experiment(f'{GLOBAL_TITLE}/{"train_deterministic" if cfg["train_deterministic"] else "train_stochastic"}/maxk={3**cfg["bottleneck_dim"]}/{cfg["name"]}-{str(spec.name)}-BASELINE', cfg, NUM_RUNS)
                            try:
                                exp.execute(MULTITHREAD, run_seeds)
                            except Exception as e:
                                print("Run failed: ", e)
