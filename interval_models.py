from __future__ import annotations

from models import POMDPWrapper

import numpy as np

def parse_transitions(model, p_names, check = True):

    num_ps = len(p_names)
    nA = max([len(state.actions) for state in model.states])

    T_upper = np.zeros((model.nr_states, nA, model.nr_states)) 
    T_lower = np.zeros_like(T_upper)
    S = np.full((model.nr_states, nA, model.nr_states), False) # whether the action directs back to origin state.
    for state in model.states:
        for action in state.actions:
            A[state.id, action.id] = False
            for transition in action.transitions:
                next_state = transition.column
                value, variables, constants, derivative_values = parse_transition(transition)
                T[state.id, action.id, next_state] = value
                S[state.id, action.id, next_state] = state.id == next_state
                if variables is not None:
                    if len(variables) != 1:
                        raise NotImplementedError("Only a single parameter is implemented.")
                    for v, variable in enumerate(variables):
                        index_of_variable = p_names.index(variable)
                        P[state.id, action.id, next_state, index_of_variable] = variable
                        C[state.id, action.id, next_state, index_of_variable] = constants[v]
                        D[state.id, action.id, next_state, index_of_variable] = derivative_values[v]
    if check:
        differences = np.sum(T, axis = -1) - 1
        sum_to_1 = np.isclose(differences, 0, atol = 1e-05)
        parameterized = np.any(np.any(P != None, axis = -1), axis = -1)
        no_action = A
        check1 = np.logical_or(np.logical_or(sum_to_1, parameterized), no_action)
        if not np.all(check1):
            raise ValueError(f'Transition distribution does not sum up to 1. \n{np.where(np.logical_not(check1))} \n{differences[np.where(np.logical_not(check1))]}')
        positive = T >= 0
        check2 = np.logical_or(positive, np.expand_dims(parameterized, axis = -1))
        if not np.all(check2):
            raise ValueError('Negative transition probabilities found.')
    return T, C, A, S, P, D

def build_interval_models(pPOMDP : POMDPWrapper, p_bounds : dict[str, list]) -> tuple[IDTMC, IMDP]:


class IMDP:
    """
    Interval model in Numpy format. Instantiating by combining a parametric model with an interval for each of the parameters.
    """

class IDTMC:
    """
    Interval model in Numpy format. Instantiating by combining a parametric model with an interval for each of the parameters.
    """

    def __init__(self, T_up, T_low, state_labels, R) -> None:
        self.T_upper = T_up
        self.T_lower = T_low
        self.state_labels = state_labels
        self.R = R
