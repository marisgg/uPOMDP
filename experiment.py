import copy
import multiprocessing
import pickle
import sys
import numpy as np
from copy import deepcopy
import time
from scipy.special import softmax
import stormpy
from scipy.stats import entropy
import tensorflow as tf
import pycarl
from joblib import Parallel, delayed
from mem_top import mem_top
import inspect
from fsc import FiniteMemoryPolicy
from interval_models import IDTMC, IPOMDP, MDPSpec
from models import POMDPWrapper
from datetime import datetime
from net import Net
from instance import Instance
from check import Checker
from in_out import Log, clear_cache
import utils

class Experiment:
    """ Represents a set of cfgs that serve an experiment. """

    def __init__(self, name, cfg, num_runs):
        self.name = name + datetime.now().strftime("%Y%m%d%H%M%S%f")
        self.num_runs = num_runs
        self.cfgs = [cfg]
        self.cfg = cfg

    def add_cfg(self, new_cfg):
        configuration = deepcopy(self.cfg)
        for key in new_cfg:
            configuration[key] = new_cfg[key]
        self.cfgs.append(configuration)


    def execute(self, multi_thread):
        log = Log(self)
        logs = []
        if multi_thread:
            logs = Parallel(n_jobs = min(self.num_runs, multiprocessing.cpu_count()-1))(delayed(self._run)(copy.deepcopy(log), cfg_idx, run_idx) for cfg_idx in range(len(self.cfgs)) for run_idx in range(self.num_runs))
        else:
            for cfg_idx in range(len(self.cfgs)):
                for run_idx in range(self.num_runs):
                    log = self.run(log, cfg_idx, run_idx)
                    logs.append(copy.deepcopy(log))
        utils.inform(f'Finished experiment {self.name}.', indent = 0, itype = 'OKGREEN')
        logs = [log for log in logs if log is not None]
        if len(logs) == 0:
            return
        with open(f"{log.base_output_dir}/logs.pickle", 'wb') as handle:
            pickle.dump(logs, handle)
        log.output_benchmark_table(logs,log.base_output_dir)
        try: log.output_learning_losses(logs,log.base_output_dir)
        except Exception as e: print(e)
        try: log.output_entropy(logs, self.num_runs, log.base_output_dir)
        except Exception as e: print(e)
        try: log.output_memory_usage(logs, self.num_runs, log.base_output_dir)
        except Exception as e: print(e)
    
    def _run(self, log, cfg_idx, run_idx):
        try:
            return self.run(log, cfg_idx, run_idx)
        except Exception as e:
            print(e, file=open(f"{log.base_output_dir}/{cfg_idx}/{run_idx}/exception.log", 'w'))
            return None

    def run(self, log, cfg_idx, run_idx):

        pycarl.clear_pools()
        tf.keras.backend.clear_session()

        utils.inform(f'Starting run {run_idx}.', indent = 0, itype = 'OKBLUE')

        cfg = self.cfgs[cfg_idx]
        instance = Instance(cfg)
        pomdp : POMDPWrapper = instance.build_pomdp()
        ps = cfg['p_init']
        worst_ps = ps
        mdp = instance.build_mdp(ps)
        length = instance.simulation_length()
        checker = Checker(instance, cfg)
        net = Net(instance, cfg)

        dynamic_uncertainty = True # TODO: Make config?
        deterministic_target_policy = cfg['train_deterministic']

        if dynamic_uncertainty:
            T = mdp.T.copy()
        
        assert instance.objective == 'min'
        
        spec = MDPSpec(cfg['specification'])

        utils.inform(f"Max k = {3**cfg['bottleneck_dim']}. Target policy: {cfg['policy']} ({'deterministic' if deterministic_target_policy else 'stochastic'}). Policy loss: {cfg['a_loss']}. Spec: {spec}.", indent = 0, itype = 'OKBLUE')

        mdp_goal_states = [i for i, x in enumerate(mdp.state_labels) if instance.label_to_reach in x]

        ipomdp = IPOMDP(instance, pomdp, cfg['p_bounds'], mdp_goal_states)
        if 'u' in cfg['policy']:
            Q = ipomdp.mdp_action_values(spec)
            utils.inform(f'Synthesized iMDP-policy (min: {Q.min():.2f}, max: {Q.max():.2f}) w/ value = {ipomdp.imdp_V[mdp.initial_state]}')
            print(Q[pomdp.initial_state])
            # assert np.isclose(Q[mdp.initial_state].min(), ipomdp.imdp_V[mdp.initial_state]), (Q[mdp.initial_state].min(), ipomdp.imdp_V[mdp.initial_state].item())

        for round_idx in range(cfg['rounds']):

            timesteps = np.repeat(np.expand_dims(np.arange(length), axis = 0), axis = 0, repeats = cfg['batch_dim'])

            if round_idx == 0 or not dynamic_uncertainty:
                beliefs, states, hs, observations, policies, actions, rewards = net.simulate(pomdp, mdp, greedy = False, length = length)
            else:
                beliefs, states, hs, observations, policies, actions, rewards = net.simulate_with_dynamic_uncertainty(pomdp, T, fsc, greedy = False, length = length)

            num_actions = pomdp.num_choices_per_state[states]

            observation_labels = pomdp.observation_labels[observations]
            empirical_result, _ = utils.evaluate_performance(instance, states, rewards)
            # valid = empirical_result < 4 * mdp.state_values[0]

            until = np.argmax(observation_labels == instance.label_to_reach, axis = -1)
            until[until == 0] = length - 1
            relevant_timesteps = timesteps < np.expand_dims(until, axis = -1)
            num_choices = pomdp.num_choices_per_state[states]
            log_policies = np.log(policies) / np.expand_dims(np.log(num_choices), axis = -1)
            log_policies[np.isinf(log_policies)] = 0
            all_entropies = np.sum(policies * - log_policies, axis = -1)
            relevant_entropies = all_entropies[np.logical_and(relevant_timesteps, num_actions > 1)]
            rnn_entropy = np.mean(relevant_entropies)

            relevant_hs = np.unique(hs[:, 1:][relevant_timesteps[:, 1:]], axis = 0)
            # TRAIN QBN
            r_loss = net.improve_r(relevant_hs)
            utils.inform(f'{run_idx}-{round_idx}\t(RNN)\t\tempir\t%.4f' % empirical_result, indent = 0)
            utils.inform(f'{run_idx}-{round_idx}\t(QBN)\t\trloss \t%.4f' % r_loss[0] + '\t>>>> %3.4f' % r_loss[-1], indent = 0)

            fsc = net.extract_fsc(reshape=True, make_greedy = False)
            utils.inform(f"{run_idx}-{round_idx}\t({fsc.nM_generated}-FSC)\t\tExtracted {fsc.nM_generated}-FSC from RNN", indent = 0)
            idtmc : IDTMC = ipomdp.create_iDTMC(fsc, add_noise=0)
            utils.inform(f"{run_idx}-{round_idx}\t(iMC)\t\tInduced iMC with {pomdp.nS} x {fsc.nM_generated} = {idtmc.T_lower.shape[0]}^2 states ({np.count_nonzero(idtmc.T_lower)} non-zero entries)", indent = 0)

            V = idtmc.check_reward(spec, np.where(np.isin(idtmc.state_labels, np.unique(pomdp.labels_to_states[instance.label_to_reach])))[0])
            utils.inform(f'{run_idx}-{round_idx}\t(iMC)\t\tRVI \t%.4f' % V[0], indent = 0)
            if dynamic_uncertainty:
                iMC_worst_case_T = idtmc.find_transition_model(V, spec)
                T = ipomdp.find_critical_pomdp_transitions(V, instance, iMC_worst_case_T, fsc, add_noise=0, tolerance=1e-6)
            # for n in range(fsc.nM_generated):
                # Assert a valid graph structure in the transition probabilities for all nodes.
                # assert np.array_equal(np.where(np.logical_or(np.isclose(T[n], 0, atol=1e-10), np.isclose(T[n], 1, atol=1e-10)))[0], np.where(np.logical_or(np.isclose(mdp.T, 0, atol=1e-10), np.isclose(mdp.T, 1, atol=1e-10)))[0]), T[n]

            pdtmc = instance.instantiate_pdtmc(fsc, zero = 0)
            fsc_memories, fsc_policies = fsc.simulate(observations)
            log_fsc_policies = np.log(fsc_policies) / np.expand_dims(np.log(num_choices), axis = -1)
            log_fsc_policies[np.logical_not(np.isfinite(log_fsc_policies))] = 0
            all_entropies = np.sum(fsc_policies * - log_fsc_policies, axis = -1)
            relevant_entropies = all_entropies[np.logical_and(relevant_timesteps, num_actions > 1)]
            fsc_entropy = np.mean(relevant_entropies)

            all_cross_entropies = np.sum(fsc_policies * - log_policies, axis = -1)
            relevant_entropies = all_cross_entropies[np.logical_and(relevant_timesteps, num_actions > 1)]
            cross_entropy = np.mean(relevant_entropies) # buggy, can be > 1 for some unknown reason.

            check = checker.check_pdtmc(pdtmc)
            added = instance.add_fsc(check, fsc)
            utils.inform(f'{run_idx}-{round_idx}\t(%i-FSC)' % fsc.nM_generated + '\t\trinit \t%.4f' % check._lb_values[0] + '\t\t%.4f' % check._ub_values[0], indent = 0)

            try:
                error = set(check._lb_values).union(check._ub_values).intersection({np.nan, np.inf})
                assert len(error) == 0, f"pDTMC checking resulting in NaN or infinite value: {error}"
            except AssertionError as ae:
                print(fsc)
                print(pdtmc.model)
                print("is parametric:", pdtmc.is_parametric)
                print(ae)
                exit()

            if cfg['ctrx_gen'] == 'rnd' and pomdp.is_parametric:
                mdp, worst_value = instance.build_mdp(), 0
            elif cfg['ctrx_gen'] == 'crt' and added and pomdp.is_parametric:
                mdp, worst_ps, worst_value = instance.worst_mdp(check, fsc)
            elif cfg['ctrx_gen'] == 'crt_neg' and not added and pomdp.is_parametric:
                mdp, worst_ps, worst_value = instance.worst_mdp(check, fsc)
            elif cfg['ctrx_gen'] == 'rnd_full' and pomdp.is_parametric:
                mdp, worst_value = instance.build_mdp(), 0
            elif cfg['ctrx_gen'] == 'crt_full' and pomdp.is_parametric:
                # Maris: PSO for worst instantiation is done here.
                mdp, worst_ps, worst_value = instance.worst_mdp(check, fsc)
            else:
                worst_value = -1
                worst_ps = {}

            length = instance.simulation_length()
            evalues = check.evaluate(cfg['p_evals'])

            nan_fixer = 10**10 if instance.objective == 'min' else -10**10

            if cfg['policy'].lower() == 'mdp':
                mdp_q_values = mdp.action_values
                q_values = mdp_q_values[states]
            elif cfg['policy'].lower() == 'qmdp':
                mdp_q_values = mdp.action_values
                q = np.nan_to_num(mdp_q_values, nan=nan_fixer)
                assert beliefs.shape[-1] == q.shape[0], "Shape mismatch."
                q_values = np.matmul(beliefs, q)
            elif cfg['policy'].lower() == 'umdp':
                mdp_q_values = ipomdp.imdp_Q
                assert mdp_q_values is not None
                mdp_q_values[mdp.A] = nan_fixer
                assert mdp_q_values.shape == mdp.action_values.shape
                q_values = mdp_q_values[states]
            elif cfg['policy'].lower() == 'qumdp':
                mdp_q_values = ipomdp.imdp_Q
                assert mdp_q_values is not None
                mdp_q_values[mdp.A] = nan_fixer
                assert mdp_q_values.shape == mdp.action_values.shape
                assert beliefs.shape[-1] == mdp_q_values.shape[0], "Shape mismatch."
                q_values = np.matmul(beliefs, mdp_q_values)
                assert q_values.shape == mdp.action_values[states].shape
            else:
                raise ValueError("invalid policy")


            if 'u' in cfg['policy'] and round == 0:
                utils.inform(f'{run_idx}-{round_idx}''\t\iMDP value: \t%.4f' % mdp_q_values[mdp.initial_state].min() if instance.objective == 'min' else q_values[mdp.initial_state].max(), indent = 0)
            
            nanarg = np.nanargmin if instance.objective == 'min' else np.nanargmax

            if deterministic_target_policy:
                a_labels = utils.one_hot_encode(nanarg(q_values, axis = -1), pomdp.nA, dtype ='float32')[1:]
            else:
                def softx(q_values, min=True):
                    x = -q_values if min else q_values
                    return np.exp(x) / np.exp(x).sum()
                q_values_nan_idxs = np.where(np.isnan(q_values))[0]
                q_values = np.nan_to_num(q_values, nan=0)
                soft_min_q = softx(q_values, instance.objective == 'min')
                soft_min_q[q_values_nan_idxs] = 0
                a_labels = utils.normalize(soft_min_q[1:],axis=-1)

            a_inputs = utils.one_hot_encode(observations, pomdp.nO, dtype = 'float32')[1:]
            # TRAIN RNN + Actor
            a_loss = net.improve_a(a_inputs, a_labels)

            label_cross_entropies = np.sum(a_labels * - log_policies, axis = -1)
            relevant_entropies = label_cross_entropies[np.logical_and(relevant_timesteps, num_actions > 1)]
            label_cross_entropy = np.mean(label_cross_entropies)

            log.flush(cfg_idx, run_idx, nM = fsc.nM_generated, lb = check._lb_values[0], ub = check._ub_values[0],
                      ps = ps, mdp_value = mdp.state_values[0], interval_mc_value = V[0],
                      evalues = evalues, worst_ps = worst_ps, added = added, slack = worst_value - check._ub_values[0],
                      empirical_result = empirical_result, front_values = np.array(instance.pareto_values),
                      mdp_policy_q = q_values, a_labels = a_labels, a_loss = np.array(a_loss), r_loss = np.array(r_loss),
                      fsc_entropy = fsc_entropy, rnn_entropy = rnn_entropy, cross_entropy = cross_entropy, label_cross_entropy = label_cross_entropy,
                      bounded_reach_prob = check.bounded_reach_prob)

            utils.inform(f'{run_idx}-{round_idx}\t(RNN)\t\taloss \t%.4f' % a_loss[0] + '\t>>>> %3.4f' % a_loss[-1], indent = 0)
        log.collect(result_at_init=check.result_at_init,duration=time.time()-log.time,cum_rewards=empirical_result,evalues=evalues,k=fsc.nM_generated,interval_mc_value = V[0])
        return log