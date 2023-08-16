import json
from experiment import Experiment
from interval_models import MDPSpec
from datetime import datetime

filenames = ['SC_maze_5', 'intercept', 'evade', 'avoid']

GLOBAL_TITLE = "SECOND-RUN-" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S%f")
GLOBAL_TITLE = "AAAI-RUN-14-08-2023_16h"

DRY_RUN = False

NUM_RUNS = 1 if DRY_RUN else 30
MULTITHREAD = False if DRY_RUN else True

all_seeds = list(range(100))

cfg_standards = {
    "a_memory_dim":3,
    "a_lr":0.001,
    "r_lr": 0.001,
    "length":50,
    "r_batch_size":32,
    "r_epochs":32,
    "a_batch_size":32,
    "a_epochs":16,
    "p_init":{"sl":0.25},
    "p_evals": [{"sl": 0.1}, {"sl": 0.2}, {"sl": 0.3}, {"sl": 0.4}],
    "p_bounds":{"sl":[0.1,0.4]},
    "num_hx_qbns": 8,
    "method": "QBN",
    "memory_dim": 4,
    "a_memory_dim": 3,
    "blow_up": 2,
    "batch_dim" : 128,
    "rounds" : 100
}

run_seeds = all_seeds[:NUM_RUNS] # TODO: FIX THIS IF ADDING RUNS WITH OTHER SEEDS

for bottle_dim in [{"bottleneck_dim" : 1}, {"bottleneck_dim" : 2}]:
    for filename in filenames:
        with open('data/input/cfgs/'+filename+'.json') as f:
            load_file = json.load(f)

        for filename in load_file:
            for cfg in load_file[filename]:
                cfg.update(bottle_dim) # WE SET MAX FSC SIZE HERE

                cfg.update(cfg_standards)

                if DRY_RUN:
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

                        if cfg['bottleneck_dim'] == 1:
                            continue

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
                                        if DRY_RUN:
                                            raise e
                            else:
                                spec = MDPSpec.Rminmax
                                cfg['specification'] = spec.value
                                exp = Experiment(f'{GLOBAL_TITLE}/{"train_deterministic" if cfg["train_deterministic"] else "train_stochastic"}/maxk={3**cfg["bottleneck_dim"]}/{cfg["name"]}-{str(spec.name)}-BASELINE', cfg, NUM_RUNS)
                                try:
                                    exp.execute(MULTITHREAD, run_seeds)
                                except Exception as e:
                                    print("Run failed: ", e)
                                    if DRY_RUN:
                                        raise e
