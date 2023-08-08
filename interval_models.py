from __future__ import annotations
from collections import deque
from datetime import datetime
import math
import multiprocessing
import warnings
import in_out
from fsc import FiniteMemoryPolicy
from instance import Instance

from models import POMDPWrapper

import numpy as np

from utils import value_to_float

from enum import Enum

import mip

USE_PRISM = True

if USE_PRISM:
    import subprocess, os

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

    def __init__(self, instance : Instance, pPOMDP : POMDPWrapper, intervals : dict[str, list], target_states : list[int]) -> None:
        self.instance = instance
        self.pPOMDP = pPOMDP # the underling pPOMDP wrapper
        self.intervals : dict[str, list] = intervals
        self.T, self.P = IPOMDP.parse_transitions(pPOMDP.model, pPOMDP.p_names, intervals)
        self.state_labeling, self.reward_models = pPOMDP.model_reward_and_labeling(True)
        assert len(self.reward_models) == 1, "Only supporting rewards/costs as of yet."
        self.R = np.array(self.reward_models[0])
        self.state_action_rewards = len(self.R.shape) > 1

        self.nS = pPOMDP.nS
        self.nA = pPOMDP.nA
        # self.reward_zero_states, self.reward_inf_states = self.preprocess(target_states)
        self.imdp_Q = None
        self.imdp_V = None

    @staticmethod
    def compute_robust_value(R, V, order, lower, upper):
        return R + IPOMDP.solve_inner_problem(order, lower, upper) @ V

    @staticmethod
    def solve_inner_problem(order, P_low, P_up):
        nS = len(order)
        T_inner = np.zeros(nS)

        i = 0
        t = order[i]
        limit = P_low.sum()

        assert (P_low <= P_up).all(), (P_low, P_up)

        if np.isclose(limit, 0) and np.isclose(np.sum(P_up), 0):
            # Sanity check, transition probability is 0.
            return np.full(T_inner.shape, np.inf)

        while not np.isclose(limit - P_low[t] + P_up[t], 1) and limit - P_low[t] + P_up[t] < 1:
            limit = limit - P_low[t] + P_up[t]
            T_inner[t] = P_up[t]
            i += 1
            t = order[i]

        j = i
        T_inner[t] = 1 - (np.round(limit, decimals=6) - P_low[t])
        assert T_inner[t] >= 0, f"1 - ({limit} - {P_low[t]})"

        for k in range(j + 1, nS):
            t = order[k]
            T_inner[t] = P_low[t]
        assert not (T_inner < 0).any()
        return T_inner

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
            for next_state in range(nS):
                for act in range(self.nA):
                    if next_state not in reward_inf_states and self.T_lower[state, act, next_state] > 0:
                        reward_inf_states.add(next_state)
                        not_reaching_target_states.append(next_state)

        return reward_zero_states, reward_inf_states

    def one_step_VI(self, Q : np.ndarray, V : np.ndarray, spec : MDPSpec):
        order = IDTMC.get_direction(spec, np.argsort(V))
        for (s, a) in self.T.keys():
            next_states_dict = self.T[(s,a)]
            if not self.P[(s,a)][list(next_states_dict.keys())[0]]:
                assert all([not self.P[(s,a)][key] for key in next_states_dict.keys()])
                Q[s, a] = self.R[s,a] if self.state_action_rewards else self.R[s]
            else:
                assert all([self.P[(s,a)][key] for key in next_states_dict.keys()])
                next_s_transition = IDTMC.solve_sparse_problem(order, next_states_dict)
                Q[s, a] = self.R[s,a] if self.state_action_rewards else self.R[s] + sum([V[idx] * prob for idx, prob in next_s_transition.items()])

        return Q

    def mdp_action_values(self, spec : MDPSpec, epsilon=1e-6, max_iters=1e4) -> np.ndarray:
        if USE_PRISM:
            dt_str = datetime.now().strftime("%Y%m%d%H%M%S%f")
            value_file = f"data/cache/Q-{dt_str}.txt"
            property = f"{spec.name}=? [ F \"goal\" ]"
            args = ["prism/prism-4.8/bin/prism", f"{self.instance.prism_path_without_extension}-mdp.prism", "-maxiters", str(int(1e6)), "-zerorewardcheck", "-nocompact", "-pf", property, "-exportvector", value_file, "-const", f'sll={min(self.intervals["sl"])}', "-const", f'slu={max(self.intervals["sl"])}']
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
            assert V.size == self.nS, (V.size, self.nS)
            Q = np.full((V.size, self.nA), np.nan)
            Q = self.one_step_VI(Q, V, spec)
            
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
            order = IDTMC.get_direction(spec, np.argsort(V))

            v_next = np.zeros(self.nS)
            q_next = np.zeros((self.nS, self.nA))
            for s in range(self.nS):
                for a in range(self.nA):
                    if s in self.reward_zero_states:
                        v_next[s] = 0
                        q_next[s,a] = 0
                    elif s in self.reward_inf_states:
                        v_next[s] = np.inf
                        q_next[s, a] = np.inf
                    elif self.pPOMDP.A[s, a]:
                        q_next[s, a] = np.inf
                    else:
                        q_next[s, a] = IPOMDP.compute_robust_value(self.R[s,a] if self.state_action_rewards else self.R[s], V, order, self.T_lower[s, a], self.T_upper[s, a])

                v_next[s] = q_next[s].min() if min else q_next[s].max()

            error = np.abs(v_next - V).max()
            V = v_next
            Q = q_next
            iters += 1

        self.imdp_Q = Q
        self.imdp_V = V
        return self.imdp_Q

    @staticmethod
    def parse_parametric_transition(value, p_names, intervals, lower_is_lower=True):
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
        return lower, upper


    @staticmethod
    def parse_transitions(model, p_names, intervals, debug=False):
        T = {}
        P = {}

        for state in model.states:
            for action in state.actions:
                if (state.id, action.id) not in T:
                    T[(state.id, action.id)] = {}
                if (state.id, action.id) not in P:
                    P[(state.id, action.id)] = {}
                for transition in action.transitions:
                    next_state = transition.column
                    value = transition.value()
                    if value.is_constant():
                        denominator = value.denominator.coefficient
                        numerator = value.numerator.coefficient
                        parsed = float(str(numerator)) / float(str(denominator))
                        lower = upper = parsed
                        P[(state.id, action.id)][next_state] = False
                        if debug: print("Value:", value, "parsed:", parsed)
                    else:
                        lower, upper = IPOMDP.parse_parametric_transition(value, p_names, intervals)
                        P[(state.id, action.id)][next_state] = True
                        if debug: print(f"Found interval [{lower}, {upper}] for transition {state}, {action.id}, {next_state} resulting from {value} and {intervals}")
                    T[(state.id, action.id)][next_state] = (lower, upper)
        return T, P
        

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
        assert self.R.ndim == 1
        assert (self.R >= 0).all() and not np.isinf(self.R).any()

        if USE_PRISM:
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
        
        file = in_out.cache_pdtmc(in_out._pdtmc_string("", self.nS, all_trans_strings, label_string, rewards_strings))
        return file

    def preprocess(self, target):
        reward_zero_states = set(target.tolist())
        num_mc_states = self.T_lower.shape[0]
        reachability_vector = np.zeros(num_mc_states, dtype=int)
        reachability_vector[target] = 1

        queue = deque(target)
        if not queue: raise ValueError("No target states in the DTMC.")

        while queue:
            current_state = queue.popleft()
            # next_states = np.where(np.logical_and(self.T_lower[:, current_state] > 0, reachability_vector == 0))[0]
            # reachability_vector[next_states] = 1
            # queue.extend(next_states)

            # for next_state in np.where(reachability_vector == 0)[0]:
            for next_state in range(num_mc_states):
                if reachability_vector[next_state] == 0 and self.T_lower[next_state, current_state] > 0:
                    reachability_vector[next_state] = 1
                    queue.append(next_state)

        reward_inf_states = set(np.where(reachability_vector == 0)[0].tolist())
        not_reaching_target_states = deque(reward_inf_states)
        while not_reaching_target_states:
            state = not_reaching_target_states.popleft()
            for next_state in range(num_mc_states):
                if next_state not in reward_inf_states and self.T_lower[state, next_state] > 0:
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

        i = 0
        t = order[i]
        t_idx = state_ids.index(t)
        limit = lower_bounds.sum()

        while not limit - lower_bounds[t_idx] + upper_bounds[t_idx] >= 1:
            limit = limit - lower_bounds[t_idx] + upper_bounds[t_idx]
            next_s_resolved_map[t] = upper_bounds[t_idx]
            i += 1
            t = order[i]
            t_idx = state_ids.index(t)
        
        next_s_resolved_map[t] = 1 - (np.round(limit, decimals=6) - lower_bounds[t_idx])

        for k in range(i + 1, len(order)):
            t = order[k]
            t_idx = state_ids.index(t)
            next_s_resolved_map[t] = lower_bounds[t_idx]

        return next_s_resolved_map


    def find_transition_model(self, V : np.ndarray, spec : MDPSpec):

        MC_T = {s : {next_s : ({} if self.P[s][next_s] else interval[0]) for next_s, interval in next_s_list.items()} for s, next_s_list in self.sparse_T.items()}

        order = IDTMC.get_direction(spec, np.argsort(V))

        for s, next_s_list in self.sparse_T.items():
            if self.P[s][list(next_s_list.keys())[0]]:
                assert all([self.P[s][next_s_p] for next_s_p in next_s_list.keys()])
                next_map = self.solve_sparse_problem(order, next_s_list)
                MC_T[s] = next_map
            else:
                continue

        return MC_T

    def check_reward(self, spec : MDPSpec, target : set, epsilon=1e-6, max_iters=1e3):
        reachable_idxs = np.array([x for x in range(self.nS) if x not in self.unreachable_states])
        if USE_PRISM:
            dt_str = datetime.now().strftime("%Y%m%d%H%M%S%f")
            value_file = f"data/cache/V-{dt_str}.txt"
            if spec in {MDPSpec.Rminmax, MDPSpec.Rmaxmax}:
                property = "Rmax=? [ F \"goal\" ]"
            else:
                property = "Rmin=? [ F \"goal\" ]"
            output = subprocess.run(["prism/prism-4.8/bin/prism", self.materialized_idtmc_filename, "-maxiters", str(int(1e6)), "-zerorewardcheck", "-pf", property, "-exportvector", value_file], check=True, capture_output=True)
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
            # os.remove(self.materialized_idtmc_filename)
            os.remove(value_file)
            V[self.unreachable_states] = np.inf
            V[reachable_idxs] = prism_values
            assert V.size == self.nS, (V.size, self.nS)
            return V
        else:
            self.__check_reward(spec, target, epsilon=epsilon, max_iters=max_iters)

    def __check_reward(self, spec : MDPSpec, target : set, epsilon=1e-6, max_iters=1e3):
        if spec is not MDPSpec.Rminmax:
            raise NotImplementedError()

        if spec in {MDPSpec.Rminmax, MDPSpec.Rminmin}:
            assert self.R[target].sum() == 0

        nb_states = self.T_lower.shape[0]
        V = np.zeros(nb_states)
        reward_zero_states, reward_inf_states = self.preprocess(target)

        # self.unreachable_states = set(self.unreachable_states.tolist())

        # if len(reward_zero_states - unreachable_states) != 0:
        #     print(reward_inf_states, unreachable_states)
        #     print("!!!! There are no reachable target states in the Markov chain. !!!!")

        hashmaps = {s : {} for s in range(nb_states) if s not in reward_inf_states.union(reward_zero_states)}

        assert (self.T_lower <= self.T_upper).all()

        error = 1.0
        iters = 0

        while error > epsilon and iters < max_iters:

            order = IDTMC.get_direction(spec, np.argsort(V))
            hashed_order = tuple(order.tolist())
            v_next = np.zeros(nb_states)

            for s in range(nb_states):
                if s in reward_zero_states:
                    v_next[s] = 0
                elif s in reward_inf_states:
                    v_next[s] = np.inf
                # elif s in unreachable_states:
                    # v_next[s] = infinity
                else:
                    if hashed_order in hashmaps[s]:
                        T_inner = hashmaps[s][hashed_order]
                    else:
                        if np.allclose(self.T_lower, self.T_upper, atol=1e-6):
                            T_inner = self.T_lower
                        else:
                            T_inner = IPOMDP.solve_inner_problem(order, self.T_lower[s], self.T_upper[s])
                        hashmaps[s][hashed_order] = T_inner
                    assert not np.isnan(T_inner).any(), T_inner
                    assert not np.isnan(V).any(), V
                    v_next[s] = self.R[s] + T_inner @ V
                    # v_next[s] = IPOMDP.compute_robust_value(self.R[s], V, order, self.T_lower[s], self.T_upper[s])

            error = np.abs(v_next - V).max()
            V = v_next
            iters += 1

        return V


