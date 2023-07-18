from __future__ import annotations
from collections import deque
from fsc import FiniteMemoryPolicy

from models import POMDPWrapper

import numpy as np

from utils import value_to_float

from enum import Enum

from queue import Queue


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
        self.T_lower, self.T_upper = IPOMDP.parse_transitions(pPOMDP.model, pPOMDP.p_names, intervals)
        self.state_labeling, self.reward_models = pPOMDP.model_reward_and_labeling(True)
        assert len(self.reward_models) == 1, "Only supporting rewards/costs as of yet."
        self.R = np.array(self.reward_models[0])
        assert len(self.T_lower) == len(self.T_lower) == len(self.R) == pPOMDP.nS, (len(self.T_lower), len(self.T_lower), len(self.R), pPOMDP.nS)

        self.nS = pPOMDP.nS
        self.nA = pPOMDP.nA

        assert not (self.T_lower.sum(axis=-1) > 1).any(), self.T_lower[self.T_lower.sum(axis=-1) > 1]
        assert not (self.T_upper[np.nonzero(self.T_upper)].sum(axis=-1) < 1).any(), self.T_upper[self.T_upper.sum(axis=-1) < 1]

        self.reward_zero_states, self.reward_inf_states = self.preprocess(target_states)


    def create_iDTMC(self, fsc : FiniteMemoryPolicy) -> tuple[np.ndarray, np.ndarray]:
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

        next_memories = fsc.randomized_next_memories(add = 0)

        for s in range(self.nS):
            o = self.pPOMDP.O[s]
            observation_label = self.pPOMDP.observation_labels[o]
            for m in range(nM):
                prod_state = s * nM + m
                for label in self.pPOMDP.states_to_labels[s]:
                    if label in labels_to_states:
                        labels_to_states[label].append(prod_state)
                    else:
                        labels_to_states[label] = [prod_state]
                state_labels.append(s)
                memory_labels.append(m)
                for o_i in observation_label:
                    observation_labels[o_i].append(prod_state)
                for action in self.pPOMDP.model.states[s].actions:
                    a = action.id
                    for transition in action.transitions:
                        next_s = transition.column
                        for next_m in range(nM):
                            prod_next_state = next_s * nM + next_m
                            action_prob = fsc.action_distributions[m, o, a]
                            # memory_prob = fsc._next_memories[m, o] == next_m
                            memory_prob = next_memories[m, o, next_m]
                            MC_T_lower[prod_state, prod_next_state] += self.T_lower[s, a, next_s] * action_prob * memory_prob
                            MC_T_upper[prod_state, prod_next_state] += self.T_upper[s, a, next_s] * action_prob * memory_prob
                rewards[prod_state] = self.R[s]
        
        assert None not in rewards
        assert (MC_T_lower.sum(axis=-1) <= 1).all(), MC_T_lower.sum(axis=-1)
        assert (MC_T_upper.sum(axis=-1) >= 1).all(), MC_T_upper.sum(axis=-1)

        return IDTMC(MC_T_lower, MC_T_upper, rewards, state_labels, memory_labels, labels_to_states)


    def preprocess(self, target : list[int]):
        reward_zero_states = set(target)
        assert self.R[target].sum() == 0
        nS = len(self.T_lower)
        reachability_vector = np.zeros(nS, dtype=int)
        reachability_vector[target] = 1

        queue = deque(target)
        if not queue: raise ValueError("No target states in the DTMC.")

        while queue:
            current_state = queue.popleft()
            next_states = np.where(np.logical_and(self.T_lower[..., current_state].sum(axis=-1) > 0, reachability_vector == 0))[0]
            reachability_vector[next_states] = 1
            queue.extend(next_states)

        reward_inf_states = set(np.where(reachability_vector == 0)[0].tolist())
        not_reaching_target_states = deque(reward_inf_states)
        while not_reaching_target_states:
            state = not_reaching_target_states.popleft()
            for next_state in range(nS):
                for act in range(self.nA):
                    if next_state not in reward_inf_states and self.T_lower[state, act, next_state] > 0:
                        reward_inf_states.add(next_state)
                        not_reaching_target_states.append(next_state)
        
        print(reward_inf_states, not_reaching_target_states, reachability_vector)

        return reward_zero_states, reward_inf_states


    def inner_problem(self, order, s, a):
        T_inner = np.zeros(self.nS)

        i = 0
        t = order[i]
        limit = np.sum(self.T_lower)
        while limit - self.T_lower[s, a, t] + self.T_upper[s, a, t] < 1:
            limit = limit - self.T_lower[s, a, t] + self.T_upper[s, a, t]
            T_inner[t] = self.T_upper[s, a, t]
            i += 1
            t = order[i]

        j = i
        t = order[j]
        T_inner[t] = 1 - (limit - self.T_lower[s, a, t])

        for k in range(j + 1, self.nS):
            t = order[k]
            T_inner[t] = self.T_lower[s, a, t]
        return T_inner

    def mdp_action_values(self, spec : MDPSpec, epsilon=1e-6) -> np.ndarray:
        """
        Return the Q-values of the robust policy for the underlying interval MDP.
        """
        if spec is not MDPSpec.Rminmax:
            raise NotImplementedError()

        V = np.zeros(self.nS)
        Q = np.zeros((self.nS, self.nA))

        error = 1.0
        while error > epsilon:
            error = 0.0
            order = np.argsort(V)

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
                    else:
                        T_inner = self.inner_problem(order, s, a)
                        q_next[s, a] = self.R[s] + np.inner(T_inner, V)

                v_next[s] = q_next[s].max()
            error = np.abs(v_next - V).max()
            V = v_next
            Q = q_next

        # optinal, also return V
        return Q


    @staticmethod
    def parse_parametric_transition(value, p_names, intervals):
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
        bounds = np.sort(bounds,axis=-1)
        lower, upper = bounds[:, 0].prod(), bounds[:, 1].prod()
        assert upper > lower
        return lower, upper


    @staticmethod
    def parse_transitions(model, p_names, intervals, debug=True):

        num_ps = len(p_names)
        nA = max([len(state.actions) for state in model.states])

        T_upper = np.zeros((model.nr_states, nA, model.nr_states)) 
        T_lower = np.zeros_like(T_upper)

        A = np.full((model.nr_states, nA), True) # whether there is NO transition (action) from a state.
        S = np.full((model.nr_states, nA, model.nr_states), False) # whether the action directs back to origin state.

        for state in model.states:
            for action in state.actions:
                A[state.id, action.id] = False
                for transition in action.transitions:
                    next_state = transition.column
                    value = transition.value()
                    if value.is_constant():
                        denominator = value.denominator.coefficient
                        numerator = value.numerator.coefficient
                        parsed = float(str(numerator)) / float(str(denominator))
                        lower = upper = parsed
                        if debug: print("Value:", value, "parsed:", parsed)
                    else:
                        lower, upper = IPOMDP.parse_parametric_transition(value, p_names, intervals)
                        if debug: print(f"Found interval [{lower}, {upper}] for transition {state}, {action.id}, {next_state} resulting from {value} and {intervals}")
                    T_lower[state.id, action.id, next_state] = lower
                    T_upper[state.id, action.id, next_state] = upper
                    S[state.id, action.id, next_state] = state.id == next_state
        
        for s in range(model.nr_states):
            for a in range(nA):
                for s_ in range(model.nr_states):
                    assert 0 <= T_lower[s, a, s_] <= T_upper[s, a, s_] <= 1, f"{(s,a,s_)} : {[T_lower[s,a,s_], T_upper[s,a,s_]]}"

        return T_lower, T_upper

class IDTMC:
    """
    Interval model in Numpy format. Instantiating by combining a parametric model with an interval for each of the parameters. In this case a iPOMDP x FSC => iDTMC
    """

    def __init__(self, T_upper : np.ndarray, T_lower : np.ndarray, rewards : np.ndarray, state_labels, memory_labels, labels_to_states) -> None:
        self.state_labels = state_labels
        self.memory_labels = memory_labels
        self.labels_to_states = labels_to_states
        self.T_upper : np.ndarray = T_upper   # 2D: (nS * nM) x (nS * nM)
        self.T_lower : np.ndarray = T_lower # 2D: (nS * nM) x (nS * nM)
        assert self.T_upper.ndim == self.T_lower.ndim == 2
        self.R = np.array(rewards, dtype=float)                # 1D: (nS * nM)
        assert self.R.ndim == 1
        assert (self.R >= 0).all() and not np.isinf(self.R).any()

    def inner_problem(self, order, s):
        nb_states = self.T_lower.shape[0]
        T_inner = np.zeros(nb_states)

        i = 0
        t = order[i]
        limit = np.sum(self.T_lower)
        while limit - self.T_lower[s, t] + self.T_upper[s, t] < 1:
            limit = limit - self.T_lower[s, t] + self.T_upper[s, t]
            T_inner[t] = self.T_upper[s, t]
            i += 1
            t = order[i]

        j = i
        t = order[j]
        T_inner[t] = 1 - (limit - self.T_lower[s, t])

        for k in range(j + 1, nb_states):
            t = order[k]
            T_inner[t] = self.T_lower[s, t]
        return T_inner

    def preprocess(self, spec : MDPSpec, target):
        reward_zero_states = set(target.tolist())
        nS = len(self.T_lower)
        reachability_vector = np.zeros(nS, dtype=int)
        reachability_vector[target] = 1

        queue = deque(target)
        if not queue: raise ValueError("No target states in the DTMC.")

        while queue:
            current_state = queue.popleft()
            next_states = np.where(np.logical_and(self.T_lower[:, current_state] > 0, reachability_vector == 0))[0]
            reachability_vector[next_states] = 1
            queue.extend(next_states)

            # for next_state in np.where(reachability_vector == 0)[0]:
                # if self.T_lower[next_state, current_state] + self.T_upper[next_state, current_state] > 0:
                    # reachability_vector[next_state] = 1
                    # queue.append(next_state)

        reward_inf_states = set(np.where(reachability_vector == 0)[0].tolist())
        not_reaching_target_states = deque(reward_inf_states)
        while not_reaching_target_states:
            state = not_reaching_target_states.popleft()
            for next_state in range(nS):
                if next_state not in reward_inf_states and self.T_lower[state, next_state] > 0:
                    reward_inf_states.add(next_state)
                    not_reaching_target_states.append(next_state)

        if spec == MDPSpec.Rmaxmax or spec == MDPSpec.Rminmin:
            # optimistic
            nature_direction = 1
        else:
            nature_direction = -1

        return reward_zero_states, reward_inf_states, nature_direction


    def find_transition_model(self, V, nature_direction):
        nb_states = self.T_lower.shape[0]
        T = np.zeros(self.T_lower.shape)
        order = np.argsort(V)
        if nature_direction == 1:
            # reverse for optimistic
            order = order[::-1]

        for s in range(nb_states):
            T[s] = self.inner_problem(order, s)

        return T

    def check_reward(self, spec : MDPSpec, target : set, epsilon=1e-6):
        if spec is not MDPSpec.Rminmax:
            raise NotImplementedError()

        nb_states = self.T_lower.shape[0]
        V = np.zeros(nb_states)
        reward_zero_states, reward_inf_states, nature_direction = self.preprocess(spec, target)

        error = 1.0
        while error > epsilon:
            error = 0.0
            order = np.argsort(V)
            if nature_direction == 1:
                # reverse for optimistic
                order = order[::-1]

            v_next = np.zeros(nb_states)
            for s in range(nb_states):
                if s in reward_zero_states:
                    v_next[s] = 0
                elif s in reward_inf_states:
                    v_next[s] = np.inf
                else:
                    T_inner = self.inner_problem(order, s)
                    v_next[s] = self.R[s] + np.inner(T_inner, V)
            error = np.abs(v_next - V).max()
            V = v_next

        # optional, find transition matrix T
        T = self.find_transition_model(V, nature_direction)
        return V, T


