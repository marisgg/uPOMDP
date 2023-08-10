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
from models import PDTMCModelWrapper, POMDPWrapper
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
        # mdp = instance.build_mdp(ps)
        length = instance.simulation_length()
        checker = Checker(instance, cfg)
        net = Net(instance, cfg)

        dynamic_uncertainty = True # TODO: Make config?
        deterministic_target_policy = cfg['train_deterministic']

        assert instance.objective == 'min'

        spec = MDPSpec(cfg['specification'])

        utils.inform(f"Max k = {3**cfg['bottleneck_dim']}. Target policy: {cfg['policy']} ({'deterministic' if deterministic_target_policy else 'stochastic'}). Policy loss: {cfg['a_loss']}. Spec: {spec}.", indent = 0, itype = 'OKBLUE')

        mdp_goal_states = [i for i, x in enumerate(pomdp.state_labels) if instance.label_to_reach in x]

        ipomdp = IPOMDP(instance, pomdp, cfg['p_bounds'], mdp_goal_states)
        if 'u' in cfg['policy']:
            Q = ipomdp.mdp_action_values(spec)
            utils.inform(f'Synthesized iMDP-policy (min: {np.nanmin(Q):.2f}, max: {np.nanmax(Q):.2f}) w/ value = {ipomdp.imdp_V[pomdp.initial_state]}')
            print(Q[pomdp.initial_state])

        # np.set_printoptions(threshold=sys.maxsize)

        utils.inform(f'{run_idx}\t(empir)\t\t(s,a)-rewards: {ipomdp.state_action_rewards}')

        for round_idx in range(cfg['rounds']):

            timesteps = np.repeat(np.expand_dims(np.arange(length), axis = 0), axis = 0, repeats = cfg['batch_dim'])

            if round_idx == 0:
                beliefs, states, hs, observations, policies, actions, rewards = net.simulate_with_ipomdp(ipomdp, greedy = False, length = length, batch_dim = instance.cfg['batch_dim'])
                assert np.count_nonzero(beliefs[:, 1:]) > 0
                assert np.count_nonzero(rewards) > 0, np.concatenate([states[...,np.newaxis], actions[...,np.newaxis]], axis=-1)
            else:
                beliefs, states, hs, observations, policies, actions, rewards = net.simulate_with_dynamic_uncertainty(ipomdp, pomdp, T, fsc, greedy = False, length = length, batch_dim = instance.cfg['batch_dim'])
                assert np.count_nonzero(beliefs[:, 1:]) > 0
                assert np.count_nonzero(rewards) > 0

            num_actions = pomdp.num_choices_per_state[states]

            observation_labels = pomdp.observation_labels[observations]
            empirical_result, _ = utils.evaluate_performance(instance, states, rewards)


            utils.inform(f'{run_idx}-{round_idx}\t(belief)\t\t{np.count_nonzero(beliefs)} non-zero entries\t', indent = 0)
            utils.inform(f'{run_idx}-{round_idx}\t(RNN)\t\tempir\t%.4f' % empirical_result, indent = 0)
            
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
            utils.inform(f'{run_idx}-{round_idx}\t(QBN)\t\tTraining QBN..', indent = 0)
            r_loss = net.improve_r(relevant_hs)
            utils.inform(f'{run_idx}-{round_idx}\t(QBN)\t\trloss \t%.4f' % r_loss[0] + '\t>>>> %3.4f' % r_loss[-1], indent = 0)

            fsc = net.extract_fsc(reshape=True, make_greedy = False)
            fsc.mask(pomdp.policy_mask)
            utils.inform(f"{run_idx}-{round_idx}\t({fsc.nM_generated}-FSC)\t\tExtracted {fsc.nM_generated}-FSC from RNN", indent = 0)

            pdtmc : PDTMCModelWrapper = ipomdp.instantiate_pdtmc(fsc, zero=0)
            check = checker.check_pdtmc(pdtmc)
            evalues = check.evaluate(cfg['p_evals'])
            utils.inform(f'{run_idx}-{round_idx}\t(pDTMC)\t\tbounds \t%.4f' % check._lb_values[0] + '\t\t%.4f' % check._ub_values[0], indent = 0)
            utils.inform(f'{run_idx}-{round_idx}\t(pDTMC)\t\tevals \t{evalues.ravel()}', indent = 0)

            idtmc : IDTMC = ipomdp.create_iDTMC(fsc, add_noise=0)
            utils.inform(f"{run_idx}-{round_idx}\t(iDTMC)\t\tInduced iMC with {pomdp.nS} x {fsc.nM_generated} = {idtmc.nS} states", indent = 0)
            target_mc_states = idtmc.labels_to_states["goal"]
            V = idtmc.check_reward(spec, target_mc_states)
            utils.inform(f'{run_idx}-{round_idx}\t(iDTMC)\t\tRVI \t%.4f' % V[0], indent = 0)
            if dynamic_uncertainty:
                iMC_worst_case_T = idtmc.find_transition_model(V, spec)
                utils.inform(f'{run_idx}-{round_idx}\t(iDTMC)\t\tFound Markov chain transition model. Computing LP..', indent = 0)
                T = ipomdp.find_critical_pomdp_transitions(V, instance, iMC_worst_case_T, fsc, add_noise=0, tolerance=1e-6)
                utils.inform(f'{run_idx}-{round_idx}\t(LP)\t\tFound worst-case uPOMDP transition model.', indent = 0)
            # for n in range(fsc.nM_generated):
                # Assert a valid graph structure in the transition probabilities for all nodes.
                # assert np.array_equal(np.where(np.logical_or(np.isclose(T[n], 0, atol=1e-10), np.isclose(T[n], 1, atol=1e-10)))[0], np.where(np.logical_or(np.isclose(mdp.T, 0, atol=1e-10), np.isclose(mdp.T, 1, atol=1e-10)))[0]), T[n]
            if round_idx > 0:
                fsc_rewards = fsc.simulate_fsc(ipomdp, pomdp, T, greedy = False, length = length, batch_dim = instance.cfg['batch_dim'])
                fsc_empirical_result, _ = utils.evaluate_performance(instance, states, fsc_rewards)
            else:
                fsc_empirical_result = np.nan
            utils.inform(f'{run_idx}-{round_idx}\t(FSC)\t\tempir\t%.4f' % fsc_empirical_result, indent = 0)

            fsc_memories, fsc_policies = fsc.simulate(observations)
            log_fsc_policies = np.log(fsc_policies) / np.expand_dims(np.log(num_choices), axis = -1)
            log_fsc_policies[np.logical_not(np.isfinite(log_fsc_policies))] = 0
            all_entropies = np.sum(fsc_policies * - log_fsc_policies, axis = -1)
            relevant_entropies = all_entropies[np.logical_and(relevant_timesteps, num_actions > 1)]
            fsc_entropy = np.mean(relevant_entropies)

            all_cross_entropies = np.sum(fsc_policies * - log_policies, axis = -1)
            relevant_entropies = all_cross_entropies[np.logical_and(relevant_timesteps, num_actions > 1)]
            cross_entropy = np.mean(relevant_entropies) # buggy, can be > 1 for some unknown reason.

            if cfg['ctrx_gen'] == 'rnd' and pomdp.is_parametric:
                mdp, worst_value = instance.build_mdp(), 0
            # elif cfg['ctrx_gen'] == 'crt' and added and pomdp.is_parametric:
            #     mdp, worst_ps, worst_value = instance.worst_mdp(check, fsc)
            # elif cfg['ctrx_gen'] == 'crt_neg' and not added and pomdp.is_parametric:
            #     mdp, worst_ps, worst_value = instance.worst_mdp(check, fsc)
            elif cfg['ctrx_gen'] == 'rnd_full' and pomdp.is_parametric:
                mdp, worst_value = instance.build_mdp(), 0
            elif cfg['ctrx_gen'] == 'crt_full' and pomdp.is_parametric and not dynamic_uncertainty:
                # Maris: PSO for worst instantiation is done here.
                mdp, worst_ps, worst_value = instance.worst_mdp(check, fsc)
            else:
                worst_value = -1
                worst_ps = {}

            length = instance.simulation_length()                

            nan_fixer = 10**10 if instance.objective == 'min' else -10**10

            if cfg['policy'].lower() == 'mdp':
                mdp_q_values = mdp.action_values
                q_values = mdp_q_values[states]
            elif cfg['policy'].lower() == 'qmdp':
                mdp_q_values = mdp.action_values
                q_values_nan_idxs = np.where(np.isnan(mdp_q_values))[0]
                mdp_q_values = np.nan_to_num(mdp_q_values, nan=nan_fixer)
                assert beliefs.shape[-1] == mdp_q_values.shape[0], "Shape mismatch."
                q_values = np.matmul(beliefs, mdp_q_values)
            elif cfg['policy'].lower() == 'umdp':
                q_values = Q[states]
            elif cfg['policy'].lower() == 'qumdp':
                mdp_q_values = Q
                q_values_nan_idxs = np.where(np.isnan(mdp_q_values))[0]
                mdp_q_values = np.nan_to_num(mdp_q_values, nan=nan_fixer)
                assert beliefs.shape[-1] == mdp_q_values.shape[0], "Shape mismatch."
                q_values = np.matmul(beliefs, mdp_q_values)
            else:
                raise ValueError("invalid policy")

            if 'u' in cfg['policy'] and round == 0:
                utils.inform(f'{run_idx}-{round_idx}''\t\iMDP value: \t%.4f' % mdp_q_values[pomdp.initial_state].min() if instance.objective == 'min' else q_values[pomdp.initial_state].max(), indent = 0)
            
            nanarg = np.nanargmin if instance.objective == 'min' else np.nanargmax

            if deterministic_target_policy:
                a_labels = utils.one_hot_encode(nanarg(q_values, axis = -1), pomdp.nA, dtype ='float32')[1:]
            else:
                def softx(q_values, min=True):
                    x = -q_values if min else q_values
                    return np.exp(x) / np.exp(x).sum()
                soft_min_q = softx(q_values, instance.objective == 'min')
                soft_min_q[np.isneginf(soft_min_q)] = 0
                soft_min_q[q_values_nan_idxs] = 0
                assert not np.isposinf(soft_min_q).any()
                assert soft_min_q[~np.isfinite(soft_min_q)].size == 0
                do_not_train_idxs = np.where(np.count_nonzero(soft_min_q, axis=-1) == 0)[0]
                train_mask = np.ones(len(soft_min_q), np.bool)
                train_mask[do_not_train_idxs] = 0
                train_mask[pomdp.initial_state] = 0
                a_labels = utils.normalize(soft_min_q[train_mask], axis=-1)

            a_inputs = utils.one_hot_encode(observations, pomdp.nO, dtype = 'float32')[train_mask]
            # TRAIN RNN + Actor
            a_loss = net.improve_a(a_inputs, a_labels)

            label_cross_entropies = np.sum(a_labels * - log_policies[train_mask], axis = -1)
            relevant_entropies = label_cross_entropies[np.logical_and(relevant_timesteps, num_actions > 1)[train_mask]]
            label_cross_entropy = np.mean(label_cross_entropies)

            log.flush(cfg_idx, run_idx, nM = fsc.nM_generated, lb = check._lb_values[0], ub = check._ub_values[0],
                      ps = ps, interval_mc_value = V[0],
                      evalues = evalues, rnn_empirical_result = empirical_result, fsc_empirical_result=fsc_empirical_result, mdp_policy_q = q_values, a_labels = a_labels, a_loss = np.array(a_loss), r_loss = np.array(r_loss),
                      fsc_entropy = fsc_entropy, rnn_entropy = rnn_entropy, cross_entropy = cross_entropy, label_cross_entropy = label_cross_entropy)

            utils.inform(f'{run_idx}-{round_idx}\t(RNN)\t\taloss \t%.4f' % a_loss[0] + '\t>>>> %3.4f' % a_loss[-1], indent = 0)
        log.collect(result_at_init=check.result_at_init,duration=time.time()-log.time,cum_rewards=empirical_result,evalues=evalues,k=fsc.nM_generated,interval_mc_value = V[0])
        return log