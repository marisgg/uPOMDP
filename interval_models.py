from __future__ import annotations
from collections import deque
import multiprocessing
import warnings
from fsc import FiniteMemoryPolicy
from instance import Instance

from models import POMDPWrapper

import numpy as np

from utils import value_to_float

from enum import Enum

import mip

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

    def __init__(self, pPOMDP : POMDPWrapper, intervals : dict[str, list], target_states : list[int]) -> None:
        self.pPOMDP = pPOMDP # the underling pPOMDP wrapper
        self.intervals : dict[str, list] = intervals
        self.T_lower, self.T_upper, self.C, self.D, self.P = IPOMDP.parse_transitions(pPOMDP.model, pPOMDP.p_names, intervals)
        self.state_labeling, self.reward_models = pPOMDP.model_reward_and_labeling(True)
        assert len(self.reward_models) == 1, "Only supporting rewards/costs as of yet."
        self.R = np.array(self.reward_models[0])
        assert len(self.T_lower) == len(self.T_lower) == len(self.R) == pPOMDP.nS, (len(self.T_lower), len(self.T_lower), len(self.R), pPOMDP.nS)

        self.nS = pPOMDP.nS
        self.nA = pPOMDP.nA

        assert not (self.T_lower.sum(axis=-1) > 1).any(), self.T_lower[self.T_lower.sum(axis=-1) > 1]
        assert not (self.T_upper[np.nonzero(self.T_upper)].sum(axis=-1) < 1).any(), self.T_upper[self.T_upper.sum(axis=-1) < 1]

        self.reward_zero_states, self.reward_inf_states = self.preprocess(target_states)
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
        limit = np.sum(P_low)

        assert (P_low <= P_up).all(), (P_low, P_up)

        if np.isclose(limit, 0) and np.isclose(np.sum(P_up), 0):
            # Sanity check, transition probability is 0.
            return T_inner

        while not np.isclose(limit - P_low[t] + P_up[t], 1) and limit - P_low[t] + P_up[t] < 1:
            limit = limit - P_low[t] + P_up[t]
            # limit = np.round(limit, decimals=5)
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

        assert MC_T.shape == (self.nS * nM, self.nS * nM)

        LP = mip.Model(solver_name=mip.CBC)

        deterministic_fsc = fsc.is_made_greedy

        T = np.array([[[[LP.add_var(var_type='C', lb=0, ub=1) for _ in range(self.nS)] for _ in range(self.nA)] for _ in range(self.nS)] for _ in range(nM)])

        assert T.shape == (nM, self.nS, self.nA, self.nS)
        assert -1 in instance.pomdp.T

        for n in range(nM):
            for s in range(0, self.nS):
                for a in range(self.nA):
                    if not instance.mdp.A[s, a]: # If there is actually a transition from this (s, a) pair in the model
                        LP += mip.xsum(T[n, s, a, s_] for s_ in range(0, self.nS)) <= 1
                        LP += mip.xsum(T[n, s, a, s_] for s_ in range(0, self.nS)) >= 1 
                    for s_ in range(0, self.nS):
                        if self.P[s, a, s_]: # parametric, set to interval
                            assert self.pPOMDP.T[s, a, s_] == -1
                            LP += float(self.T_lower[s, a, s_]) <= T[n, s, a, s_]
                            LP += T[n, s, a, s_] <= float(self.T_upper[s, a, s_])
                        else: # non-parametric, set to true prob
                            assert self.pPOMDP.T[s, a, s_] != -1
                            LP += T[n, s, a, s_] <= float(instance.pomdp.T[s, a, s_])
                            LP += T[n, s, a, s_] >= float(instance.pomdp.T[s, a, s_])

        for s in range(self.nS):
            o = self.pPOMDP.O[s]
            # a = np.argmin(V[s])
            for m in range(nM):
                prod_state = s * nM + m
                if deterministic_fsc:
                    # for action in self.pPOMDP.model.states[s].actions:
                        # a = action.id
                    for a in range(self.nA):
                        if np.isclose(fsc.action_distributions[m, o, a], 0) or instance.mdp.A[s, a]:
                            continue
                        for next_s in range(self.nS):
                        # for transition in action.transitions:
                            # next_s = transition.column
                            for next_m in range(nM):
                                prod_next_state = next_s * nM + next_m
                                chain_prob = float(MC_T[prod_state, prod_next_state])
                                # a = np.argmax(fsc.action_distributions[m, o])
                                bound = T[m][s][a][next_s] * float(next_memories[m, o, next_m])
                                # bound = T[m][s][a][next_s] * float(fsc.action_distributions[m, o, a]) * float(next_memories[m, o, next_m])
                                LP += chain_prob >= bound
                                LP += chain_prob <= bound
                else:
                    for next_s in range(self.nS):
                        for next_m in range(nM):
                            prod_next_state = next_s * nM + next_m
                            chain_prob = float(MC_T[prod_state, prod_next_state])
                            actions = []
                            # for a in range(self.nA):
                            # for action in self.pPOMDP.model.states[s].actions:
                                # if (a in list(map(lambda x : x.id, self.pPOMDP.model.states[s].actions)) and next_s in list(map(lambda x : x.column, a.transitions))):
                                    # assert not instance.mdp.A[s, a]
                                    # actions.append(a)
                            bound = mip.xsum([T[m][s][a][next_s] * float(fsc.action_distributions[m, o, a]) * float(next_memories[m, o, next_m]) for a in range(self.nA) if not instance.mdp.A[s, a]])
                            LP += chain_prob + tolerance >= bound
                            LP += chain_prob - tolerance <= bound

        LP.verbose = 0 # surpress output
        result = LP.optimize()

        assert result in {mip.OptimizationStatus.FEASIBLE, mip.OptimizationStatus.OPTIMAL}, result

        T = np.abs(np.vectorize(lambda x : x.x)(T))
        # T = np.round(T, decimals=6)

        assert None not in T

        # import sys
        # np.set_printoptions(threshold=sys.maxsize)

        # print(np.round(T, decimals=2), file=open(f'./debug.txt', 'w'))
        for n in range(nM):
            # print(np.concatenate((np.round(T[n], decimals=2)[...,np.newaxis],instance.mdp.T[...,np.newaxis]),axis=-1), file=open(f'./n-debug.txt', 'w'))
            assert (np.logical_and(np.logical_or(self.T_lower < T[n], np.isclose(T[n], self.T_lower, atol=1e-10)), np.logical_or(T[n] < self.T_upper, np.isclose(T[n], self.T_upper, atol=1e-10)))).all(), T[n][np.logical_and(np.logical_or(self.T_lower < T[n], np.isclose(T[n], self.T_lower, atol=1e-10)), np.logical_or(T[n] < self.T_upper, np.isclose(T[n], self.T_upper, atol=1e-10)))]
        return T

    def create_iDTMC(self, fsc : FiniteMemoryPolicy, add_noise = 0, debug = False) -> tuple[np.ndarray, np.ndarray]:
        nM = fsc.nM_generated
        fsc.mask(self.pPOMDP.policy_mask)

        MC_T_lower = np.zeros((self.nS * nM, self.nS * nM), dtype = 'float64') # Holds the transition matrix for the Markov chain (nS x nM)
        MC_T_upper = np.zeros_like(MC_T_lower)

        rewards = np.full((self.nS * nM), None)

        observations_label_set = set.union(*[set(s) for s in self.pPOMDP.observation_labels])
        observation_labels = {observation_label : [] for observation_label in observations_label_set}
        state_labels = []
        memory_labels = []
        labels_to_states = {}

        next_memories = fsc.randomized_next_memories(add = add_noise)
        assert np.allclose(next_memories.sum(axis=-1), 1)

        prod_states = set()
        next_prod_states = set()

        for s in range(self.nS):
            o = self.pPOMDP.O[s]
            observation_label = self.pPOMDP.observation_labels[o]
            for m in range(nM):
                prod_state = s * nM + m
                prod_states.add(prod_state)
                for label in self.pPOMDP.states_to_labels[s]:
                    if label in labels_to_states:
                        labels_to_states[label].append(prod_state)
                    else:
                        labels_to_states[label] = [prod_state]
                state_labels.append(s)
                memory_labels.append(m)
                for o_i in observation_label:
                    observation_labels[o_i].append(prod_state)
                # for a in range(self.nA):
                for action in self.pPOMDP.model.states[s].actions:
                    a = action.id
                    # for next_s in range(self.nS):
                    for transition in action.transitions:
                        next_s = transition.column
                        assert not self.pPOMDP.A[s, a]
                        for next_m in range(nM):
                            prod_next_state = next_s * nM + next_m
                            next_prod_states.add(prod_next_state)
                            assert m < fsc.action_distributions.shape[0], (m, fsc.action_distributions.shape)
                            action_prob = fsc.action_distributions[m, o, a]
                            memory_prob = next_memories[m, o, next_m]
                            fsc_prob = action_prob * memory_prob
                            MC_T_lower[prod_state, prod_next_state] += self.T_lower[s, a, next_s] * fsc_prob
                            MC_T_upper[prod_state, prod_next_state] += self.T_upper[s, a, next_s] * fsc_prob
                            assert MC_T_lower[prod_state, prod_next_state] > 0 or np.isclose(MC_T_lower[prod_state, prod_next_state], MC_T_upper[prod_state, prod_next_state])
                rewards[prod_state] = self.R[s]
    
        assert None not in rewards
        assert (MC_T_lower.sum(axis=-1) <= 1.0 + 1e-6).all(), (MC_T_lower.sum(axis=-1)[MC_T_lower.sum(axis=-1) > 1])
        assert (MC_T_upper.sum(axis=-1) >= 1.0 - 1e-6).all(), MC_T_upper.sum(axis=-1)[MC_T_upper.sum(axis=-1) < 1]

        assert (MC_T_lower <= MC_T_upper).all()

        if debug: print("Lower non-zero/total", np.count_nonzero(MC_T_lower),np.size(MC_T_lower))
        if debug: print("Upper non-zero/total", np.count_nonzero(MC_T_upper),np.size(MC_T_upper))

        # Reachability analysis, delete labels of unreachable states.
        hops = np.full((self.nS * nM), np.inf) # k-hops from init to each state.
        k = 0
        hops[0] = 0
        while np.any(hops < np.inf) and k < len(hops) + 1:
            states, next_states = np.where(np.logical_or(MC_T_lower[hops < np.inf] > 0, MC_T_upper[hops < np.inf] > 0))
            hops[next_states] = np.minimum(k + 1, hops[next_states])
            k += 1

        unreachable_states = np.where(hops == np.inf)[0]

        return IDTMC(MC_T_lower, MC_T_upper, rewards, state_labels, memory_labels, labels_to_states, unreachable_states)

    def preprocess(self, target : list[int]):
        reward_zero_states = set(target)
        assert self.R[target].sum() == 0
        nS = len(self.T_lower)
        reachability_vector = np.zeros(nS, dtype=int)
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

    def mdp_action_values(self, spec : MDPSpec, epsilon=1e-6, max_iters=1e3) -> np.ndarray:
        """
        Return the Q-values of the robust policy for the underlying interval MDP.
        """
        if spec is not MDPSpec.Rminmax:
            raise NotImplementedError()
        
        if self.imdp_Q is None or self.imdp_V is None:

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
                            q_next[s, a] = IPOMDP.compute_robust_value(self.R[s], V, order, self.T_lower[s, a], self.T_upper[s, a])

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
        return lower, upper, np.array(constants).item(), np.array(derivatives).item()


    @staticmethod
    def parse_transitions(model, p_names, intervals, debug=False):

        num_ps = len(p_names)
        nA = max([len(state.actions) for state in model.states])

        T_upper = np.zeros((model.nr_states, nA, model.nr_states)) 
        T_lower = np.zeros_like(T_upper)

        D = np.zeros((model.nr_states, nA, model.nr_states))
        C = np.zeros_like(D)
        P = np.full_like(C, False)

        for state in model.states:
            for action in state.actions:
                for transition in action.transitions:
                    next_state = transition.column
                    assert isinstance(next_state, int)
                    value = transition.value()
                    if value.is_constant():
                        denominator = value.denominator.coefficient
                        numerator = value.numerator.coefficient
                        parsed = float(str(numerator)) / float(str(denominator))
                        lower = upper = parsed
                        if debug: print("Value:", value, "parsed:", parsed)
                    else:
                        P[state.id, action.id, next_state] = True
                        lower, upper, C[state.id, action.id, next_state], D[state.id, action.id, next_state] = IPOMDP.parse_parametric_transition(value, p_names, intervals)
                        # PT[state.id, action.id, next_state] = C[state.id, action.id, next_state] + D[state.id, action.id, next_state]
                        if debug: print(f"Found interval [{lower}, {upper}] for transition {state}, {action.id}, {next_state} resulting from {value} and {intervals}")
                    T_lower[state.id, action.id, next_state] = lower
                    T_upper[state.id, action.id, next_state] = upper
        
        for s in range(model.nr_states):
            for a in range(nA):
                for s_ in range(model.nr_states):
                    assert 0 <= T_lower[s, a, s_] <= T_upper[s, a, s_] <= 1, f"{(s,a,s_)} : {[T_lower[s,a,s_], T_upper[s,a,s_]]}"

        return T_lower, T_upper, C, D, P

class IDTMC:
    """
    Interval model in Numpy format. Instantiating by combining a parametric model with an interval for each of the parameters. In this case a iPOMDP x FSC => iDTMC
    """

    def __init__(self, T_lower : np.ndarray, T_upper : np.ndarray, rewards : np.ndarray, state_labels, memory_labels, labels_to_states, unreachable_states) -> None:
        self.state_labels = state_labels
        self.memory_labels = memory_labels
        self.labels_to_states = labels_to_states
        self.unreachable_states = unreachable_states
        self.T_upper : np.ndarray = T_upper   # 2D: (nS * nM) x (nS * nM)
        self.T_lower : np.ndarray = T_lower # 2D: (nS * nM) x (nS * nM)
        assert (T_lower <= T_upper).all()
        assert self.T_upper.ndim == self.T_lower.ndim == 2
        self.R = np.array(rewards, dtype=float)                # 1D: (nS * nM)
        assert self.R.ndim == 1
        assert (self.R >= 0).all() and not np.isinf(self.R).any()

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

    def find_transition_model(self, V : np.ndarray, spec : MDPSpec):
        nb_states = self.T_lower.shape[0]
        T = np.zeros(self.T_lower.shape)

        order = IDTMC.get_direction(spec, np.argsort(V))

        for s in range(nb_states):
            T[s] = IPOMDP.solve_inner_problem(order, self.T_lower[s], self.T_upper[s])

        return T

    def check_reward(self, spec : MDPSpec, target : set, epsilon=1e-6, max_iters=1e3):
        if spec is not MDPSpec.Rminmax:
            raise NotImplementedError()

        if spec in {MDPSpec.Rminmax, MDPSpec.Rminmin}:
            assert self.R[target].sum() == 0

        nb_states = self.T_lower.shape[0]
        V = np.zeros(nb_states)
        reward_zero_states, reward_inf_states = self.preprocess(target)

        unreachable_states = set(self.unreachable_states.tolist())

        # if len(reward_zero_states - unreachable_states) != 0:
        #     print(reward_inf_states, unreachable_states)
        #     print("!!!! There are no reachable target states in the Markov chain. !!!!")

        assert (self.T_lower <= self.T_upper).all()

        error = 1.0
        iters = 0

        while error > epsilon and iters < max_iters:

            order = IDTMC.get_direction(spec, np.argsort(V))
            v_next = np.zeros(nb_states)

            for s in range(nb_states):
                if s in reward_zero_states:
                    v_next[s] = 0
                elif s in reward_inf_states:
                    v_next[s] = np.inf
                # elif s in unreachable_states:
                    # v_next[s] = infinity
                else:
                    v_next[s] = IPOMDP.compute_robust_value(self.R[s], V, order, self.T_lower[s], self.T_upper[s])

            error = np.abs(v_next - V).max()
            V = v_next
            iters += 1

        return V


