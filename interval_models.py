from __future__ import annotations
from fsc import FiniteMemoryPolicy

from models import POMDPWrapper

import numpy as np

from utils import value_to_float

from enum import Enum


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

    def __init__(self, pPOMDP : POMDPWrapper, intervals : dict[str, list]) -> None:
        self.pPOMDP = pPOMDP # the underling pPOMDP wrapper
        self.intervals : dict[str, list] = intervals
        self.T_lower, self.T_upper = IPOMDP.parse_transitions(pPOMDP.model, pPOMDP.p_names, intervals)
        self.state_labeling, self.reward_models = pPOMDP.model_reward_and_labeling(True)
        assert len(self.reward_models) == 1, "Only supporting rewards/costs as of yet."
        self.R = self.reward_models[0]

        assert len(self.T_lower) == len(self.T_lower) == len(self.R) == pPOMDP.nS

        self.nS = pPOMDP.nS
        self.nA = pPOMDP.nA

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
        assert np.isclose(MC_T_lower.sum(axis=-1).all(), 1, atol=1e-05)
        assert np.isclose(MC_T_upper.sum(axis=-1).all(), 1, atol=1e-05)

        return IDTMC(MC_T_lower, MC_T_upper, state_labels, rewards)


    def preprocess(self, spec : MDPSpec, target):
        reward_zero_states = set()
        # TODO add state indexes (integers) to reward zero states set
        reward_inf_states = set()
        # TODO add state indexes that cannot reach the target almost surely to reward_inf_states

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

    def mdp_action_values(self, spec : MDPSpec, target : set, epsilon=1e-6) -> np.ndarray:
        """
        Return the Q-values of the robust policy for the underlying interval MDP.
        """
        if spec is not MDPSpec.Rminmax:
            raise NotImplementedError()

        V = np.zeros(self.nS)
        Q = np.zeros((self.nS, self.nA))

        reward_zero_states, reward_inf_states = self.preprocess(spec)

        error = 1.0
        while error > epsilon:
            error = 0.0
            order = np.argsort(V)

            v_next = np.zeros(self.nS)
            q_next = np.zeros((self.nS, self.nA))
            for s in range(self.nS):
                for a in range(self.nA):
                    if s in reward_zero_states:
                        v_next[s] = 0
                        q_next[s,a] = 0
                    elif s in reward_inf_states:
                        v_next[s] = np.inf
                        q_next[s, a] = np.inf
                    else:
                        T_inner = self.inner_problem(order, s, a)
                        q_next[s, a] = self.R[s] + T_inner * V

                v_next[s] = max(q_next[s, :])
            error = max(V - v_next)
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
        bounds = np.array([[c + d * intervals[p][0], c + d * intervals[p][1]] for c, d, p in zip(constants, derivatives, variable_names)])
        lower, upper = bounds[:, 0].prod(), bounds[:, 1].prod()
        return lower, upper


    @staticmethod
    def parse_transitions(model, p_names, intervals, debug=False):

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
                        value = float(str(numerator)) / float(str(denominator))
                        lower = upper = value
                    else:
                        lower, upper = IPOMDP.parse_parametric_transition(value, p_names, intervals)
                        if debug: print(f"Found interval [{lower}, {upper}] for transition {state}, {action.id}, {next_state}")
                    T_lower[state.id, action.id, next_state] = lower
                    T_upper[state.id, action.id, next_state] = upper
                    S[state.id, action.id, next_state] = state.id == next_state
        return T_lower, T_upper

class IDTMC:
    """
    Interval model in Numpy format. Instantiating by combining a parametric model with an interval for each of the parameters. In this case a iPOMDP x FSC => iDTMC
    """

    def __init__(self, T_up : np.ndarray, T_low : np.ndarray, state_labels, rewards : np.ndarray) -> None:
        self.state_labels = state_labels
        self.T_up : np.ndarray = T_up   # 2D: (nS * nM) x (nS * nM)
        self.T_low : np.ndarray = T_low # 2D: (nS * nM) x (nS * nM)
        self.R = rewards                # 1D: (nS * nM)



    def inner_problem(self, order, s):
        nb_states = self.T_low.shape[0]
        T_inner = np.zeros(nb_states)

        i = 0
        t = order[i]
        limit = np.sum(self.T_low)
        while limit - self.T_low[s, t] + self.T_up[s, t] < 1:
            limit = limit - self.T_low[s, t] + self.T_up[s, t]
            T_inner[t] = self.T_up[s, t]
            i += 1
            t = order[i]

        j = i
        t = order[j]
        T_inner[t] = 1 - (limit - self.T_low[s, t])

        for k in range(j + 1, nb_states):
            t = order[k]
            T_inner[t] = self.T_low[s, t]
        return T_inner



    def preprocess(self, spec : MDPSpec, target):
        reward_zero_states = set()
        # TODO add states in target set to reward_zero_states set
        reward_inf_states = set()
        # TODO find states that cannot reach the target with prob. 1 and add them to reward_inf_states


        if spec == MDPSpec.Rmaxmax or spec == MDPSpec.Rminmin:
            # optimistic
            nature_direction = 1
        else:
            nature_direction = -1

        return reward_zero_states, reward_inf_states, nature_direction


    def find_transition_model(self, V, nature_direction):
        nb_states = self.T_low.shape[0]
        T = np.zeros(self.T_low.shape)
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

        nb_states = self.T_low.shape[0]
        V = np.zeros(nb_states)
        reward_zero_states, reward_inf_states, nature_direction = self.preprocess(spec)

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
                    v_next[s] = self.R[s] + T_inner * V
            error = max(V - v_next)
            V = v_next

        # optional, find transition matrix T
        T = self.find_transition_model(V, nature_direction)
        return V, T


