from __future__ import annotations
from collections import deque
from datetime import datetime
import math
import multiprocessing
import warnings
from fsc import FiniteMemoryPolicy
import in_out

from instance import Instance

from models import POMDPWrapper

import numpy as np

from utils import value_to_float

from enum import Enum

import mip

USE_PRISM_IMDP = True
USE_PRISM_IDTMC = True

if USE_PRISM_IMDP or USE_PRISM_IDTMC:
    import subprocess, os
    PRISM_PATH = "prism/prism/bin/prism"

class MDPSpec(Enum):
    """
    (PO)MDP objectives, first min/max is the agent's objective, second min/max is how nature resolves the uncertainty
    """
    Rminmax = 1
    Rmaxmin = 2
    Rminmin = 3
    Rmaxmax = 4



class IPOMDP:
    """
    Interval wrapping of a pPOMDP in Stormpy by keeping a lower and upper bound transition matrix induced by the intervals given for the parameters.
    """

    def __init__(self, instance : Instance, pPOMDP : POMDPWrapper, intervals : dict[str, list], target_states : list[int], force_intervals, build_prism_file = True) -> None:
        self.instance = instance
        self.pPOMDP = pPOMDP # the underling pPOMDP wrapper
        self.intervals : dict[str, list] = intervals
        self.T, self.P, self.C, self.D = IPOMDP.parse_transitions(pPOMDP.model, pPOMDP.p_names, intervals)
        self.state_labeling, self.reward_models = pPOMDP.model_reward_and_labeling(True)
        assert len(self.reward_models) == 1, "Only supporting rewards/costs as of yet."
        self.R = np.array(self.reward_models[0])
        assert np.count_nonzero(self.R) > 0
        self.state_action_rewards = len(self.R.shape) > 1
        self.reward_zero_states = set(target_states)
        # print(self.R)
        # if self.state_action_rewards:
            # assert set(np.where(np.isclose(self.R.sum(axis=-1), 0))[0].tolist()) == self.reward_zero_states, set(np.where(np.isclose(self.R.sum(axis=-1), 0))[0].tolist())

        self.nS = pPOMDP.nS
        self.nA = pPOMDP.nA
        # self.reward_zero_states, self.reward_inf_states = self.preprocess(target_states)

        if build_prism_file and USE_PRISM_IMDP:
            self.prism_file = self.to_prism_file(force_intervals)
        
        self.imdp_Q = None
        self.imdp_V = None
    
    def to_prism_file(self, force_intervals=False):
        all_trans_strings = ""
        rewards_strings = ""

        rewards_for_states = set()

        for (s,a), next_s_dict in self.T.items():
            trans_string = f"[a{a}] (s={s}) -> " 
            first = True
            for next_s, interval in next_s_dict.items():
                if interval[0] > 0:
                    if first:
                        first = False
                    else:
                        trans_string += " + "
                    if not force_intervals and np.isclose(interval[0], interval[1]):
                        trans_string += f"{interval[0]} : (s'={next_s})"
                    else:
                        trans_string += f"[{interval[0]}, {interval[1]}] : (s'={next_s})"
            if self.state_action_rewards:
                rewards_strings += f'[a{a}] s={s} : {self.R[s, a]};\n'
            elif s not in rewards_for_states:
                rewards_strings += f's={s} : {self.R[s]};\n'
                rewards_for_states.add(s)
            if first:
                trans_string = ""
            else:
                trans_string += ';\n'
            
            all_trans_strings += trans_string

            
        label_string = 'label "goal" ='
        first = True
        for s in list(self.reward_zero_states):
            if first:
                first = False
            else:
                label_string += " | "
            label_string += f"(s={s})"
        label_string += ';'
        
        file = in_out.cache_pdtmc(in_out._imdp_string(all_trans_strings, label_string, rewards_strings, self.nS), model_str='imdp')
        return file

    def find_critical_pomdp_transitions(self, V : np.ndarray, instance : Instance, MC_T, fsc : FiniteMemoryPolicy, add_noise = 0, tolerance = 1e-6):
        nM = fsc.nM_generated
        if not fsc.is_masked:
            fsc.mask(instance.pomdp.policy_mask, zero = add_noise)
        next_memories = fsc.randomized_next_memories(add = add_noise)
        assert np.allclose(next_memories.sum(axis=-1), 1)

        LP = mip.Model(solver_name=mip.CBC)

        LP.emphasis = mip.SearchEmphasis.FEASIBILITY # We only care about satisfying the constraints

        deterministic_fsc = fsc.is_made_greedy

        uT_sparse_map = {m : {
            (s, a) : {next_s : (None if self.P[(s,a)][next_s] else interval[0]) for next_s, interval in next_s_map.items()} for (s, a), next_s_map in self.T.items()
        } for m in range(nM)}

        for m in range(nM):
            for (s,a) in self.T.keys():
                o = self.pPOMDP.O[s]
                prod_state = s * nM + m
                if prod_state not in MC_T:
                    continue
                for next_s, interval in self.T[(s,a)].items():
                    if self.P[(s,a)][next_s]:
                        if uT_sparse_map[m][(s,a)][next_s] is None:
                            # Add variable and interval constraint
                            uT_sparse_map[m][(s,a)][next_s] = LP.add_var(var_type='C', lb=0, ub=1)
                            LP += float(interval[0]) <= uT_sparse_map[m][(s,a)][next_s]
                            LP += uT_sparse_map[m][(s,a)][next_s] <= float(interval[1])
            for (s,a) in self.T.keys():
                o = self.pPOMDP.O[s]
                prod_state = s * nM + m
                if prod_state not in MC_T:
                    continue  
                for next_s, interval in self.T[(s,a)].items():
                    for next_m in range(nM):
                        prod_next_state = next_s * nM + next_m
                        if prod_next_state not in MC_T[prod_state]:
                            continue
                        elif not self.P[(s,a)][next_s]:
                            continue
                        if MC_T[prod_state][prod_next_state] == {}:
                            continue
                        chain_prob = float(MC_T[prod_state][prod_next_state])
                        varsum = [uT_sparse_map[m][(s,a_sum)][next_s] * float(fsc.action_distributions[m, o, a_sum]) * float(next_memories[m, o, next_m]) for a_sum in range(self.nA) if (s,a_sum) in uT_sparse_map[m] and next_s in self.T[(s,a_sum)]] # and next_s in uT_sparse_map[m][(s,a)]]
                        bound = mip.xsum(varsum) if len(varsum) > 1 else varsum[0]
                        LP += chain_prob >= bound - tolerance
                        LP += chain_prob <= bound + tolerance
                LP += mip.xsum(uT_sparse_map[m][(s,a)][s_] for s_ in range(self.nS) if (s,a) in uT_sparse_map[m] and s_ in self.T[(s,a)]) <= 1 + tolerance
                LP += mip.xsum(uT_sparse_map[m][(s,a)][s_] for s_ in range(self.nS) if (s,a) in uT_sparse_map[m] and s_ in self.T[(s,a)]) >= 1 - tolerance

        LP.verbose = 0 # surpress output
        result = LP.optimize()

        if result not in {mip.OptimizationStatus.FEASIBLE, mip.OptimizationStatus.OPTIMAL}:
            raise Exception(f"LP is infeasible: {result}")
        
        for m, sa_dict in uT_sparse_map.items():
            for (s, a), next_s_map in sa_dict.items():
                for next_s, val in next_s_map.items():
                    if isinstance(val, mip.Var):
                        assert val.x is not None
                        uT_sparse_map[m][(s,a)][next_s] = val.x
        
        return uT_sparse_map

    def create_iDTMC(self, fsc : FiniteMemoryPolicy, add_noise = 0, debug = False) -> tuple[np.ndarray, np.ndarray]:
        nM = fsc.nM_generated
        fsc.mask(self.pPOMDP.policy_mask)

        MC_T = {}
        MC_P = {}

        labels_to_states = {}
        state_labels = []
        memory_labels = []

        rewards = np.full((self.nS * nM), None)

        next_memories = fsc.randomized_next_memories(add = add_noise)
        assert np.allclose(next_memories.sum(axis=-1), 1)

        for (s,a) in self.T.keys():
            o = self.pPOMDP.O[s]
            state_labels.append(s)
            for m in range(nM):
                memory_labels.append(m)
                prod_state = s * nM + m
                for label in self.pPOMDP.states_to_labels[s]:
                    if label in labels_to_states:
                        labels_to_states[label].append(prod_state)
                    else:
                        labels_to_states[label] = [prod_state]
                if not prod_state in MC_T:
                    MC_T[prod_state] = {}
                if not prod_state in MC_P:
                    MC_P[prod_state] = {}
                for next_s, interval in self.T[(s,a)].items():
                    if interval[0] == 0:
                        continue
                    for next_m in range(nM):
                        prod_next_state = next_s * nM + next_m
                        action_prob = fsc.action_distributions[m, o, a]
                        memory_prob = next_memories[m, o, next_m]
                        fsc_prob = action_prob * memory_prob
                        if fsc_prob == 0:
                            continue
                        if prod_next_state in MC_T[prod_state]:
                            MC_T[prod_state][prod_next_state] = (MC_T[prod_state][prod_next_state][0] + interval[0] * fsc_prob, MC_T[prod_state][prod_next_state][1] + interval[1] * fsc_prob)
                        elif interval[0] * fsc_prob > 0:
                            MC_T[prod_state][prod_next_state] = (interval[0] * fsc_prob, interval[1] * fsc_prob)
                            if self.P[(s,a)][next_s]:
                                assert interval[0] != interval[1]
                                MC_P[prod_state][prod_next_state] = True
                            else:
                                if not prod_next_state in MC_P[prod_state]:
                                    MC_P[prod_state][prod_next_state] = False
                                assert interval[0] == interval[1]
                        else:
                            continue
                if self.state_action_rewards:
                    rewards[prod_state] = sum([fsc.action_distributions[m, o, a] * self.R[s, a] for a in range(self.nA)])
                else:
                    rewards[prod_state] = self.R[s]

        next_states = deque([0])
        reachable_states = {0}

        assert np.count_nonzero(rewards) > 0, rewards

        while next_states:
            next_s = next_states.popleft()
            new_reachable_states = MC_T[next_s].keys()
            intervals = MC_T[next_s].values()
            assert all(iter(map(lambda x : not math.isclose(x[0], 0), intervals)))
            unvisited_states = set(list(new_reachable_states)) - reachable_states
            reachable_states = reachable_states.union(set(list(new_reachable_states)))
            next_states.extend(iter(unvisited_states))
        
        unreachable_states = list(set(iter(range(self.nS * nM))) - reachable_states)

        return IDTMC(self.nS * nM, MC_T, MC_P, rewards, state_labels, memory_labels, labels_to_states, unreachable_states)

    def instantiate_pdtmc(self, fsc : FiniteMemoryPolicy, zero = 1e-8, remove_file_from_disk = True):
        """
        Instantiates the (p)DTMC, which is parameterized by values of the policy.

        @param: fsc                 :   a FiniteMemoryPolicy instance.
        @param: zero                :   a pseudo-zero number for graph-preserving the PDTMC.
        @return:                    :   a PDTMCWrapper instance.

        """

        import stormpy

        nM = fsc.nM_generated
        if not fsc.is_masked:
            fsc.mask(self.pPOMDP.policy_mask)
        nS, nA = self.nS, self.nA

        ps = list(self.intervals.keys())
        assert len(ps) == 1
        T = {} # Holds the transition matrix for the Markov chain (nS x nM)
        D = {} # Holds the probabilities for the derived part of the parameter
        C = {} # Holds the probabilities for the constant part of the parameter
        observations_label_set = set.union(*[set(s) for s in self.pPOMDP.observation_labels])
        observation_labels = {observation_label : [] for observation_label in observations_label_set}
        state_labels = []
        memory_labels = []
        rewards_strs = ['' for r_idx in range(1)]
        labels_to_states = {}
        next_memories = fsc.randomized_next_memories(add = zero)

        R = np.full(self.nS * nM, np.nan)

        for (s,a) in self.T.keys():
            o = self.pPOMDP.O[s]
            observation_label = self.pPOMDP.observation_labels[o]
            for m in range(nM):
                prod_state = s * nM + m
                if prod_state not in T:
                    T[prod_state] = {}
                    C[prod_state] = {}
                    D[prod_state] = {}
                for label in self.pPOMDP.states_to_labels[s]:
                    if label in labels_to_states:
                        labels_to_states[label].append(prod_state)
                    else:
                        labels_to_states[label] = [prod_state]
                state_labels.append(s)
                memory_labels.append(m)
                mean_r = 0
                for o_i in observation_label:
                    observation_labels[o_i].append(prod_state)
                for next_s, interval in self.T[(s,a)].items():
                    for next_m in range(nM):
                        prod_next_state = next_s * nM + next_m
                        action_prob = fsc.action_distributions[m, o, a]
                        # memory_prob = fsc._next_memories[m, o] == next_m
                        memory_prob = next_memories[m, o, next_m]
                        fsc_prob = action_prob * memory_prob
                        if self.P[(s, a)][next_s]:
                            derivative = self.D[(s, a)][next_s]
                            constant = self.C[(s, a)][next_s]
                            if derivative == 0 and constant == 0:
                                continue
                            if prod_next_state not in D[prod_state]:
                                D[prod_state][prod_next_state] = derivative * fsc_prob
                            else:
                                D[prod_state][prod_next_state] += derivative * fsc_prob
                            if prod_next_state not in C[prod_state]:
                                C[prod_state][prod_next_state] = constant * fsc_prob
                            else:
                                C[prod_state][prod_next_state] += constant * fsc_prob 
                        else:
                            assert interval[0] == interval[1], (interval[0], interval[1])
                            if prod_next_state not in T[prod_state]:
                                T[prod_state][prod_next_state] = interval[0] * fsc_prob
                            else:
                                T[prod_state][prod_next_state] += interval[0] * fsc_prob
                        has_outgoing = True
                if self.state_action_rewards:
                    # rewards_strs[0] += f'\ts={prod_state} : {sum([fsc.action_distributions[m, o, a] * self.R[s, a] for a in range(self.nA)])};\n'
                    R[prod_state] = sum([fsc.action_distributions[m, o, a] * self.R[s, a] for a in range(self.nA)])
                else:
                    R[prod_state] = self.R[s]
        
        assert np.isfinite(R).all(), R

        p_string = ''
        for idx, p in enumerate(ps):
            p_string += f'const double {p};\n'

        label_strings = ''
        for label in observation_labels:
            if label == 'init' or label == '':
                continue
            states = observation_labels[label]
            label_string = f'label "{label}"='
            for state in states:
                label_string += f's={state}|'
            label_strings += label_string[:-1] + ';\n'

        transitions_strings = ''
        for s, next_dict in T.items():
            transitions_string = f'\t[] s={s} ->'
            has_outgoing = False
            for next_s, trans_prob in next_dict.items():
                if trans_prob > 0:
                    transitions_string += f" {trans_prob} : (s'={next_s}) +"
                    has_outgoing = True
                elif trans_prob == 0:
                    pass
                else:
                    print(T, D, C, sep='\n')
                    raise ValueError("Non-parametric transition that has a negative probability.")
            for next_s in D[s].keys():
                if D[s][next_s] == 0 and C[s][next_s] == 0:
                    continue
                derivative = D[s][next_s]
                constant = C[s][next_s]
                transitions_string += f" ({constant} + {derivative}*{list(self.intervals.keys())[0]}) : (s'={next_s}) +"
                has_outgoing = True

            if has_outgoing:
                transitions_strings += transitions_string[:-2] + ';\n'
            
            rewards_strs[0] += f's={s} : {R[s]};\n'

        rewards_strs = ['true : 0;'] if len(rewards_strs) == 0 else rewards_strs
        contents = in_out.pdtmc_string(p_string, self.nS, nM, transitions_strings, label_strings, rewards_strs[0])
        fn = in_out.cache_pdtmc(contents)
        prism_program = stormpy.parse_prism_program(fn, simplify = False)
        if remove_file_from_disk:
            os.remove(fn)
        if self.pPOMDP.is_parametric:
            model = stormpy.build_sparse_parametric_model(prism_program)
            p_region_dict = {
                x : (stormpy.RationalRF(self.intervals[x.name][0]), 
                    stormpy.RationalRF(self.intervals[x.name][1]))
                for x in model.collect_probability_parameters()
            }
        else:
            assert False, "This is a pDTMC"
            model = stormpy.build_model(prism_program)
            p_region_dict = {}
        
        from models import PDTMCModelWrapper

        pdtmc = PDTMCModelWrapper(model, self.pPOMDP, nM, p_region_dict, state_labels, memory_labels, fn)
        # if pdtmc.nS != np.count_nonzero(hops < np.inf):
            # raise ValueError('Inaccuracies after translating PDTMC to Stormpy model.')
        return pdtmc


    def preprocess(self, target : list[int]):
        reward_zero_states = set(target)
        assert self.R[target].sum() == 0
        reachability_vector = np.zeros(self.nS, dtype=int)
        reachability_vector[target] = 1


        # Backward DFS to find states with 0 probability to reach the target
        queue = deque(target)
        if not queue: raise ValueError("No target states in the DTMC.")

        while queue:
            current_state = queue.popleft()
            next_states = np.where(np.logical_and(self.T_lower[..., current_state].sum(axis=-1) > 0, reachability_vector == 0))[0]
            reachability_vector[next_states] = 1
            queue.extend(next_states)
        # Forward DFS to find states reachable from states with 0 probability to reach the target
        reward_inf_states = set(np.where(reachability_vector == 0)[0].tolist())
        not_reaching_target_states = deque(reward_inf_states)
        while not_reaching_target_states:
            state = not_reaching_target_states.popleft()
            for next_state in range(self.nS):
                for act in range(self.nA):
                    if next_state not in reward_inf_states and self.T_lower[state, act, next_state] > 0:
                        reward_inf_states.add(next_state)
                        not_reaching_target_states.append(next_state)

        return reward_zero_states, reward_inf_states

    def one_step_VI(self, V : np.ndarray, spec : MDPSpec, set_unreachable_to_nan = False):
        assert np.isfinite(V).all(), V

        order = IDTMC.get_direction(spec, np.argsort(V))
        
        if set_unreachable_to_nan:
            Q = np.full((self.nS, self.nA), np.nan)
        else:
            Q = np.zeros((self.nS, self.nA))
        for (s, a), next_states_dict in self.T.items():
            if s in self.reward_zero_states:
                Q[s, a] = 0
            else:
                if not any([self.P[(s,a)][next_s] for next_s in next_states_dict.keys()]):
                    next_s_transition = {next_s : interval[0] for next_s, interval in next_states_dict.items()}
                    # print("Certain transitions:")
                    # print(next_s_transition)
                    assert np.isclose(sum(next_s_transition.values()), 1), (next_s_transition.values(), sum(next_s_transition.values()))
                    Q[s, a] = (self.R[s,a] if self.state_action_rewards else self.R[s]) + sum([V[next_state_idx] * prob for next_state_idx, prob in next_s_transition.items()])
                else:
                    # print("Uncertain transitions:")
                    # print(next_states_dict)
                    next_s_transition = IDTMC.solve_sparse_problem(order, next_states_dict)
                    # print(next_s_transition)
                    assert np.isclose(sum(next_s_transition.values()), 1), (next_s_transition.values(), sum(next_s_transition.values()))
                    Q[s, a] = (self.R[s,a] if self.state_action_rewards else self.R[s]) + sum([V[next_state_idx] * prob for next_state_idx, prob in next_s_transition.items()])
        return Q

    def mdp_action_values(self, spec : MDPSpec, epsilon=1e-6, max_iters=1e4) -> np.ndarray:
        if USE_PRISM_IMDP:
            dt_str = datetime.now().strftime("%Y%m%d%H%M%S%f")
            print(self.reward_zero_states)
            value_file = f"data/cache/Q-{dt_str}.txt"
            property = f"{spec.name}=? [ F \"goal\" ]"
            args = [PRISM_PATH, self.prism_file, "-maxiters", str(int(1e6)), "-zerorewardcheck", "-nocompact", "-pf", property, "-exportvector", value_file]
            # args = [PRISM_PATH, f"{self.instance.prism_path_without_extension}-mdp.prism", "-maxiters", str(int(1e6)), "-zerorewardcheck", "-nocompact", "-pf", property, "-exportvector", value_file, "-const", f'sll={min(self.intervals["sl"])}', "-const", f'slu={max(self.intervals["sl"])}']
            try:
                output = subprocess.run(args, check=True, capture_output=True)
            except Exception as e:
                print(" ".join(args))
                raise e
            try:
                with open(value_file, 'r') as input:
                    V = np.array([float(line.rstrip()) for line in input], dtype=float)
            except Exception as error:
                print("STDOUT:")
                print(output.stdout.decode("utf-8"))
                print("STDERR:")
                print(output.stderr.decode("utf-8"))
                print(output.args)
                raise error
            os.remove(value_file)
            os.remove(self.prism_file)
            assert V.size == self.nS, (V.size, self.nS)
            Q = self.one_step_VI(V, spec, set_unreachable_to_nan=True)

            assert np.allclose(np.nanmin(Q, axis=-1), V), np.concatenate((np.expand_dims(np.nanmin(Q, axis=-1), axis=-1), np.expand_dims(V, axis=-1)), axis=-1)

            self.imdp_Q = Q
            self.imdp_V = V

            return Q
        else:
            return self.__mdp_action_values(spec, epsilon, max_iters)
    
    def __mdp_action_values(self, spec : MDPSpec, epsilon=1e-6, max_iters=1e4) -> np.ndarray:
        """
        Return the Q-values of the robust policy for the underlying interval MDP.
        """
        if spec not in {MDPSpec.Rminmax, MDPSpec.Rminmin}:
            raise NotImplementedError(spec)

        V = np.zeros(self.nS)
        Q = np.zeros((self.nS, self.nA))

        min = spec in {MDPSpec.Rminmax, MDPSpec.Rminmin}

        error = 1.0
        iters = 0
        while error > epsilon and iters < max_iters:
            v_next = np.full(self.nS, np.inf)
            q_next = np.full((self.nS, self.nA), np.inf)

            q_next = self.one_step_VI(V, spec, set_unreachable_to_nan=False)

            v_next = q_next.min(axis=-1) if min else q_next.max(axis=-1)

            error = np.abs(v_next - V).max()
            assert np.isfinite(v_next).all()
            assert np.isfinite(q_next).all()
            V = v_next
            Q = q_next
            iters += 1
        
        self.imdp_Q = Q = self.one_step_VI(V, spec, set_unreachable_to_nan=True)
        self.imdp_V = np.nanmin(Q, axis=-1)

        return self.imdp_Q

    @staticmethod
    def parse_parametric_transition(value, p_names, intervals, lower_is_lower=True, return_constants_and_derivatives=False):
        variables = list(value.gather_variables())
        assert set(v.name for v in variables) == set(p_names), (set([v.name for v in variables]), set(p_names))
        constant = value.constant_part()
        c = float(str(constant.numerator)) / float(str(constant.denominator))
        variable_names = []
        derivatives = []
        constants = []
        for variable in variables:
            variable_names.append(variable.name)
            derivative = value_to_float(value.derive(variable))
            derivatives.append(derivative)
            constants.append(c)
        assert len(constants) == len(derivatives) == len(variable_names)
        bounds = np.array([[c + d * intervals[p][0], c + d * intervals[p][1]] for c, d, p in zip(constants, derivatives, variable_names)])
        if lower_is_lower:
            bounds = np.sort(bounds,axis=-1)
        lower, upper = bounds[:, 0].prod(), bounds[:, 1].prod()
        assert not lower_is_lower or upper >= lower
        if return_constants_and_derivatives:
            assert len(constants) == len(derivatives) == 1
            return lower, upper, constants[0], derivatives[0]
        else:
            return lower, upper


    @staticmethod
    def parse_transitions(model, p_names, intervals, debug=False):
        T = {}
        P = {}

        D = {}
        C = {}

        for state in model.states:
            s = state.id
            for action in state.actions:
                a = action.id
                if (s, a) not in T:
                    T[(s, a)] = {}
                    P[(s, a)] = {}
                    D[(s, a)] = {}
                    C[(s, a)] = {}
                for transition in action.transitions:
                    next_state = transition.column
                    value = transition.value()
                    if value.is_constant():
                        denominator = value.denominator.coefficient
                        numerator = value.numerator.coefficient
                        parsed = float(str(numerator)) / float(str(denominator))
                        lower = upper = parsed
                        P[(s, a)][next_state] = False
                        if debug: print("Value:", value, "parsed:", parsed)
                    else:
                        lower, upper, C[(s, a)][next_state], D[(s, a)][next_state] = IPOMDP.parse_parametric_transition(value, p_names, intervals, return_constants_and_derivatives=True)
                        P[(s, a)][next_state] = True
                        if debug: print(f"Found interval [{lower}, {upper}] for transition {s}, {a}, {next_state} resulting from {value} and {intervals}")
                    T[(s, a)][next_state] = (lower, upper)
        return T, P, C, D
        

class IDTMC:
    """
    Interval model in Numpy format. Instantiating by combining a parametric model with an interval for each of the parameters. In this case a iPOMDP x FSC => iDTMC
    """

    def __init__(self, nS : int, sparse_T : dict, parameter_map : dict, rewards : np.ndarray, state_labels, memory_labels, labels_to_states, unreachable_states) -> None:
        self.labels_to_states = labels_to_states
        self.unreachable_states = unreachable_states
        self.sparse_T = sparse_T
        self.P = parameter_map
        self.state_labels, self.memory_labels = state_labels, memory_labels
        self.nS = nS
        self.R = np.array(rewards, dtype=float)                # 1D: (nS * nM)
        assert self.R.size == self.nS, (self.R.size, self.nS)
        assert self.R.ndim == 1
        assert (self.R >= 0).all() and not np.isinf(self.R).any()

        if USE_PRISM_IDTMC:
            self.materialized_idtmc_filename = self.to_prism_file()
    
    def to_prism_file(self):
        all_trans_strings = ""
        rewards_strings = ""

        for mc_s in self.sparse_T.keys():
            trans_string = f"[] (s={mc_s}) -> " 
            first = True
            for mc_next_s, interval in self.sparse_T[mc_s].items():
                if interval[0] > 0:
                    if first:
                        first = False
                    else:
                        trans_string += " + "
                    if np.isclose(interval[0], interval[1]):
                        trans_string += f"{interval[0]} : (s'={mc_next_s})"
                    else:
                        trans_string += f"[{interval[0]}, {interval[1]}] : (s'={mc_next_s})"
            rewards_strings += f's={mc_s} : {self.R[mc_s]};\n'
            if first:
                trans_string = ""
            else:
                trans_string += ';\n'
            
            all_trans_strings += trans_string

            
        label_string = 'label "goal" ='
        first = True
        for s in self.labels_to_states["goal"]:
            if first:
                first = False
            else:
                label_string += " | "
            label_string += f"(s={s})"
        label_string += ';'
        
        file = in_out.cache_pdtmc(in_out._pdtmc_string("", self.nS, all_trans_strings, label_string, rewards_strings), model_str='idtmc')
        return file

    def preprocess(self, target):
        reward_zero_states = set(target.tolist())
        reachability_vector = np.zeros(self.nS, dtype=int)
        reachability_vector[target] = 1

        queue = deque(target)
        if not queue: raise ValueError("No target states in the DTMC.")

        while queue:
            current_state = queue.popleft()
            for next_state in range(self.nS):
                if reachability_vector[next_state] == 0 and next_state in self.sparse_T[current_state]:
                    reachability_vector[next_state] = 1
                    queue.append(next_state)

        reward_inf_states = set(np.where(reachability_vector == 0)[0].tolist())
        not_reaching_target_states = deque(reward_inf_states)
        while not_reaching_target_states:
            state = not_reaching_target_states.popleft()
            for next_state in range(self.nS):
                if next_state not in reward_inf_states and next_state in self.sparse_T[state]:
                    reward_inf_states.add(next_state)
                    not_reaching_target_states.append(next_state)

        return reward_zero_states, reward_inf_states

    @staticmethod
    def get_direction(spec : MDPSpec, order : np.ndarray):
        """
        Nature's direction given the optimization target of the uMDP.
        """
        if spec in {MDPSpec.Rmaxmax, MDPSpec.Rminmin}:
            return order
        else:
            return order[::-1] # reverse for pessimistic

    @staticmethod
    def solve_sparse_problem(order, next_s_map):
        next_s_resolved_map = {}
        state_ids = list(next_s_map.keys())
        order = sorted(state_ids, key=lambda x : order.tolist().index(x))
        intervals = np.array(list(next_s_map.values()))
        lower_bounds, upper_bounds = intervals[:, 0], intervals[:, 1]
        assert (lower_bounds <= upper_bounds).all()

        i = 0
        t = order[i]
        t_idx = state_ids.index(t)
        limit = lower_bounds.sum()

        while not limit - lower_bounds[t_idx] + upper_bounds[t_idx] >= 1:
            limit = limit - lower_bounds[t_idx] + upper_bounds[t_idx]
            next_s_resolved_map[t] = upper_bounds[t_idx]
            i += 1
            try:
                t = order[i]
            except IndexError as ie:
                print(i, t_idx, limit - lower_bounds[t_idx] + upper_bounds[t_idx], state_ids, order)
                raise ie
            t_idx = state_ids.index(t)
        
        next_s_resolved_map[t] = 1 - (np.round(limit, decimals=6) - lower_bounds[t_idx])

        for k in range(i + 1, len(order)):
            t = order[k]
            t_idx = state_ids.index(t)
            next_s_resolved_map[t] = lower_bounds[t_idx]

        return next_s_resolved_map


    def find_transition_model(self, V : np.ndarray, spec : MDPSpec, sanity_check=True):

        MC_T = {s : {next_s : ({} if self.P[s][next_s] else interval[0]) for next_s, interval in next_s_list.items()} for s, next_s_list in self.sparse_T.items()}

        if sanity_check:
            sanity_V = np.zeros_like(V)

        order = IDTMC.get_direction(spec, np.argsort(V))

        for s, next_s_list in self.sparse_T.items():
            if any([self.P[s][next_s] for next_s in list(next_s_list.keys())]): # if there is an uncertain/interval transition in this sparse map
                MC_T[s] = self.solve_sparse_problem(order, next_s_list)
                if sanity_check:
                    sanity_V[s] = self.R[s] + sum([V[next_state_idx] * prob for next_state_idx, prob in MC_T[s].items()])
            else:
                if sanity_check:
                    sanity_V[s] = self.R[s] + sum([V[next_state_idx] * prob[0] for next_state_idx, prob in next_s_list.items()])
                else:
                    continue
        
        if sanity_check:
            mask = np.ones_like(V, dtype=bool)
            mask[self.unreachable_states] = 0            
            assert np.allclose(V[mask], sanity_V[mask]), np.concatenate((np.expand_dims(V[mask], axis=-1), np.expand_dims(sanity_V[mask], axis=-1)), axis=-1)

        return MC_T

    def check_reward(self, spec : MDPSpec, target : set, epsilon=1e-6, max_iters=1e3):
        reachable_idxs = np.array([x for x in range(self.nS) if x not in self.unreachable_states])
        if USE_PRISM_IDTMC:
            dt_str = datetime.now().strftime("%Y%m%d%H%M%S%f")
            value_file = f"data/cache/V-{dt_str}.txt"
            if spec in {MDPSpec.Rminmax, MDPSpec.Rmaxmax}:
                property = "Rmax=? [ F \"goal\" ]"
            else:
                property = "Rmin=? [ F \"goal\" ]"
            output = subprocess.run([PRISM_PATH, self.materialized_idtmc_filename, "-maxiters", str(int(1e6)), "-zerorewardcheck", "-pf", property, "-exportvector", value_file], check=True, capture_output=True)
            V = np.zeros(self.nS, dtype=float)
            try:
                with open(value_file, 'r') as input:
                    prism_values = np.array([float(line.rstrip()) for line in input], dtype=float)
            except Exception as error:
                print("STDOUT:")
                print(output.stdout.decode("utf-8"))
                print("STDERR:")
                print(output.stderr.decode("utf-8"))
                raise error
            assert prism_values.size + len(self.unreachable_states) == self.nS, (prism_values.size, len(self.unreachable_states), self.nS)
            target_value_state = np.where(np.isclose(prism_values, 0))
            os.remove(self.materialized_idtmc_filename)
            os.remove(value_file)
            V[self.unreachable_states] = np.inf
            V[reachable_idxs] = prism_values
            assert V.size == self.nS, (V.size, self.nS)
            return V
        else:
            return self.__check_reward(spec, target, epsilon=epsilon, max_iters=max_iters)

    def __check_reward(self, spec : MDPSpec, target : set, epsilon=1e-6, max_iters=1e3):
        if spec is not MDPSpec.Rminmax:
            raise NotImplementedError()

        if spec in {MDPSpec.Rminmax, MDPSpec.Rminmin}:
            assert self.R[target].sum() == 0, (self.R[target], target)

        V = np.zeros(self.nS)
        # reward_zero_states, reward_inf_states = self.preprocess(target)

        assert np.count_nonzero(self.R) > 0

        reward_zero_states = set(target)

        error = 1.0
        iters = 0

        while error > epsilon and iters < max_iters:

            order = IDTMC.get_direction(spec, np.argsort(V))
            v_next = np.zeros(self.nS)

            for s, next_s_list in self.sparse_T.items():

                if s in reward_zero_states:
                    v_next[s] = 0
                # elif s in reward_inf_states:
                    # v_next[s] = np.inf
                elif s in self.unreachable_states:
                    v_next[s] = np.inf
                else:
                    if any([self.P[s][next_s] for next_s in list(next_s_list.keys())]): # if there is an uncertain/interval transition in this sparse map
                        T_inner = self.solve_sparse_problem(order, next_s_list)
                    else:
                        T_inner = {next_s : interval[0] for next_s, interval in next_s_list.items()}

                    v_next[s] = self.R[s] + sum([V[next_state_idx] * prob for next_state_idx, prob in T_inner.items()])
            
            assert np.isfinite(v_next).sum() > 0

            error = np.abs(v_next - V).max()
            V = v_next
            iters += 1

        return V


