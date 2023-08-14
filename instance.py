import stormpy
import stormpy.pomdp
from copy import deepcopy
import numpy as np
from datetime import datetime
from models import POMDPWrapper, MDPWrapper, PDTMCModelWrapper,Wrapper
import math
import os

import utils
from utils import parse_transitions, choice_from_md, inform, parse_transition
import in_out

def parse_formula_str(formula_str):

    formula_str = formula_str.replace(' ', '')
    label_to_reach = None
    label_to_avoid = None

    if 'R' in formula_str:
        kind = 'reward'
        label_to_reach = formula_str.split('F')[-1][1:-2]
    elif 'P' in formula_str:
        kind = 'probability'
        if 'F' in formula_str:
            label_to_reach = formula_str.split('F')[-1][1:-2]
        if 'U' in formula_str:
            label_to_reach = formula_str.split('U')[-1][1:-2]
            label_to_avoid = formula_str.split('U')[0].split('!')[-1][1:-1]
    else:
        raise TypeError(f'Specification could not be parsed from formula string {formula_str}')

    if 'max' in formula_str:
        objective = 'max'
    elif 'min' in formula_str:
        objective = 'min'
    else:
        raise TypeError(f'Objective could not be parsed from formula string {formula_str}')

    return kind, objective, label_to_reach, label_to_avoid


class Instance:

    def __init__(self, cfg):

        inform(f'Initializing {cfg["name"]}.')

        self.cfg = cfg
        self.name = cfg['name']
        self.prism_path_without_extension = f'data/input/envs/prism/{self.name}'
        self.path = f'{self.prism_path_without_extension}.prism'
        self.formula_str = cfg['formula_str']
        self.kind, self.objective, self.label_to_reach, self.label_to_avoid = parse_formula_str(cfg['formula_str'])
        self.properties = stormpy.parse_properties_without_context(cfg['formula_str'])
        self.p_bounds = cfg['p_bounds']
        self.nM = 0
        self.pareto_front = [] # list of Pareto-efficient FScs.
        self.pareto_values = [] # list of Pareto-efficient values.
        self._solutions = [] # list of all FSCs and values.
        self.mdp_include = False # True if cfg.get("mdp_include") else False

    def build_pomdp(self):
        """
        Builds a POMDP from a PRISM file.

        @return:        : a POMDPWrapper instance.

        """

        prism_program = stormpy.parse_prism_program(self.path, simplify = True)
        expression_manager = prism_program.expression_manager
        constants = prism_program.constants
        undefined_constants = []
        for c in constants:
            if not c.defined:
                undefined_constants.append(c.expression_variable)
                if c.name not in self.p_bounds:
                    raise ValueError(f'Parameter {c.name} appears in PRISM program, but no bounds were set.')

        options = stormpy.BuilderOptions([p.raw_formula for p in self.properties])
        options.set_build_choice_labels()
        options.set_build_with_choice_origins()

        if self.kind == 'reward':
            options.set_build_all_labels()
            options.set_build_all_reward_models()
            options.set_build_state_valuations()

        if prism_program.has_undefined_constants:
            model = stormpy.build_sparse_parametric_model_with_options(prism_program, options)
        else:
            assert False
            model = stormpy.build_sparse_model_with_options(prism_program, options)
        
        # for s in model.states:
            # for a in s.actions:
                # for transition in a.transitions:
                    # print("From state {} with action {}, with probability {}, go to state {}".format(model.state_valuations.get_json(s), a, transition.value(), transition.column))

        nr_states = model.nr_states
        # model = stormpy.pomdp.make_canonic(model)
        # added_states = model.nr_states - nr_states
        utils.inform(f'Built POMDP with nS = {model.nr_states} and nO = {model.nr_observations}.')# (Canonical added {added_states} states.)')

        self.pomdp = POMDPWrapper(model, self.properties)

        self.input_dim = self.pomdp.nO
        self.output_dim = self.pomdp.nA

        return self.pomdp

    def build_mdp(self, ps = None):
        """
        Builds an underlying MDP. To do this, the POMDP (if parametric) is first instantiated with
        p equal to ps of args.

        @return:        : an MDPWrapper instance.

        """

        if not hasattr(self, 'pomdp'):
            raise ValueError('You have to build a POMDP before extracting an MDP.')

        p_values = {}
        if self.pomdp.is_parametric:
            for p in self.p_bounds:
                if ps is not None and p in ps:
                    value = ps[p]
                else:
                    lb, ub = self.p_bounds[p]
                    value = np.random.uniform(lb, ub, size = 1)[0]
                p_values[p] = value
            string = ''.join([f'{key} = %.2f ' % p_values[key] for key in p_values])
            # utils.inform('Instantiating POMDP with ' + string)
        elif ps is not None:
            utils.inform('Trying to instantiate an MDP that is not parametric.', itype = 'WARNING')\

        if self.mdp_include:
            raise ValueError("We are not using this.")
            path = in_out.cache_mdp(in_out.read_and_replace(self.name,p_values),self.name)
            prism_program = stormpy.parse_prism_program(path)
            expression_manager = prism_program.expression_manager
            constants = prism_program.constants
            undefined_constants = []
            for c in constants:
                if c in ps:
                    c.defined = True
                    # c.definition = stormpy.storage.Expression()
                if not c.defined:
                    undefined_constants.append(c.expression_variable)
                    if c.name not in self.p_bounds:
                        raise ValueError(f'Parameter {c.name} appears in PRISM program, but no bounds were set.')

            options = stormpy.BuilderOptions([p.raw_formula for p in self.properties])
            options.set_build_choice_labels()
            options.set_build_with_choice_origins()

            if self.kind == 'reward':
                options.set_build_all_labels()
                options.set_build_all_reward_models()
                options.set_build_state_valuations()

            if prism_program.has_undefined_constants:
                model = stormpy.build_sparse_parametric_model_with_options(prism_program, options)
            else:
                model = stormpy.build_sparse_model_with_options(prism_program, options)

        # elif self.pomdp.is_parametric:
        #     instantiator = stormpy.pars.ModelInstantiator(self.pomdp.model)
        #     points = {p : stormpy.RationalRF(value) for p, value in p_values.items()}
        #     model = instantiator.instantiate(points)
        #     components = stormpy.SparseModelComponents(
        #         transition_matrix = model.transition_matrix,
        #         state_labeling = model.labeling,
        #         reward_models = model.reward_models,
        #         rate_transitions = False)
        #     model =  stormpy.storage.SparseMdp(components)
        else:
            transition_matrix, labeling, reward_models = self.pomdp.model_components(p_values)
            components = stormpy.SparseModelComponents(
                transition_matrix = transition_matrix,
                state_labeling = labeling,
                reward_models = reward_models,
                rate_transitions = False)
            model =  stormpy.storage.SparseMdp(components)
        if not hasattr(self, 'old_state_values'):
            self.mdp = MDPWrapper(model, self.pomdp.properties)
            self.old_state_values = np.array(self.mdp.state_values)
        else:
            self.old_state_values = np.array(self.mdp.state_values)
            self.mdp = MDPWrapper(model, self.pomdp.properties)
        if self.cfg['ctrx_gen'] == 'crt_full' or self.cfg['ctrx_gen'] == 'rnd':
            self._remember_labels()
            return self.mdp

    def instantiate_pdtmc(self, fsc, zero = 1e-8, remove_file_from_disk = True):
        """
        Instantiates the (p)DTMC, which is parameterized by values of the policy.

        @param: fsc                 :   a FiniteMemoryPolicy instance.
        @param: zero                :   a pseudo-zero number for graph-preserving the PDTMC.
        @return:                    :   a PDTMCWrapper instance.

        """

        # To do: make faster. Replace for-loops with np.arrays

        nM = fsc.nM_generated
        fsc.mask(self.pomdp.policy_mask)
        nS, nA = self.pomdp.nS, self.pomdp.nA

        ps = list(self.p_bounds.keys())
        assert len(ps) == 1
        T = np.zeros((self.pomdp.nS * nM, self.pomdp.nS * nM), dtype = 'float64') # Holds the transition matrix for the Markov chain (nS x nM)
        D = np.zeros((self.pomdp.nS * nM, self.pomdp.nS * nM, len(ps)), dtype = 'float64') # Holds the probabilities for the derived part of the parameter
        C = np.zeros((self.pomdp.nS * nM, self.pomdp.nS * nM, len(ps)), dtype = 'float64') # Holds the probabilities for the constant part of the parameter
        observations_label_set = set.union(*[set(s) for s in self.pomdp.observation_labels])
        observation_labels = {observation_label : [] for observation_label in observations_label_set}
        state_labels = []
        memory_labels = []
        rewards_strs = ['' for r_idx in range(self.pomdp.num_reward_models)]
        labels_to_states = {}
        next_memories = fsc.randomized_next_memories(add = zero)

        for s in range(self.pomdp.nS):
            o = self.pomdp.O[s]
            observation_label = self.pomdp.observation_labels[o]
            for m in range(nM):
                prod_state = s * nM + m
                for label in self.pomdp.states_to_labels[s]:
                    if label in labels_to_states:
                        labels_to_states[label].append(prod_state)
                    else:
                        labels_to_states[label] = [prod_state]
                state_labels.append(s)
                memory_labels.append(m)
                mean_r = 0
                for o_i in observation_label:
                    observation_labels[o_i].append(prod_state)
                for action in self.pomdp.model.states[s].actions:
                    a = action.id
                    for transition in action.transitions:
                        next_s = transition.column
                        for next_m in range(nM):
                            prod_next_state = next_s * nM + next_m
                            trans_prob = self.pomdp.T[s, a, next_s]
                            action_prob = fsc.action_distributions[m, o, a]
                            # memory_prob = fsc._next_memories[m, o] == next_m
                            memory_prob = next_memories[m, o, next_m]
                            prob = trans_prob * action_prob * memory_prob
                            (p_index, ) = np.where(self.pomdp.P[s, a, next_s]) # only one parameter per transition can exist.
                            if len(p_index) > 0:
                                derivative = self.pomdp.D[s, a, next_s, p_index]
                                constant = self.pomdp.C[s, a, next_s, p_index]
                                D[prod_state, prod_next_state, p_index] += derivative * action_prob * memory_prob
                                C[prod_state, prod_next_state, p_index] += constant * action_prob * memory_prob
                            else:
                                if prob < 0:
                                    print(f"Negative probability? {prob} = {trans_prob} * {action_prob} * {memory_prob}")
                                    raise ValueError()
                                T[prod_state, prod_next_state] += prob
                            has_outgoing = True
                for r_idx in range(self.pomdp.num_reward_models):
                    rewards_strs[r_idx] += f'\ts={prod_state} : {self.pomdp.rewards[s, r_idx]};\n'

        # Reachability analysis, delete labels of unreachable states.
        hops = np.full((self.pomdp.nS * nM), np.inf) # k-hops from init to each state.
        k = 0
        hops[0] = 0
        while np.any(hops < np.inf) and k < len(hops) + 1:
            states, next_states = np.where(np.logical_or(T[hops < np.inf] > 0, np.any(D[hops < np.inf] != 0, axis = -1)))
            hops[next_states] = np.minimum(k + 1, hops[next_states])
            k += 1

        state_labels = list(np.array(state_labels)[hops < np.inf])
        memory_labels = list(np.array(memory_labels)[hops < np.inf])

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
        for s in range(self.pomdp.nS * nM):
            transitions_string = f'\t[] s={s} ->'
            has_outgoing = False
            state_tup = (s,)
            trans_dict_c = dict()
            trans_dict_d = dict()
            for next_s in range(self.pomdp.nS * nM):
                trans_prob = T[s, next_s]
                derivative = D[s, next_s]
                constant = C[s, next_s]
                next_tup = (next_s,)
                if np.any(derivative != 0): # transition is parametric.
                    for p_index, p in enumerate(ps):
                        # trans_dict_d.update({state_tup+next_tup+(p,):derivative[p_index]})
                        # trans_dict_c.update({state_tup+next_tup+(p,):constant[p_index]})
                        d = derivative[p_index]
                        c = constant[p_index]
                        transitions_string += f" ({c} + {d}*{p}) : (s'={next_s}) +"
                    has_outgoing = True
                elif trans_prob > 0:
                    transitions_string += f" {trans_prob} : (s'={next_s}) +"
                    has_outgoing = True
                elif trans_prob == 0:
                    pass
                else:
                    print(T, D, C, sep='\n')
                    raise ValueError("Non-parametric transition that has a negative probability.")
            # for t_d in trans_dict_d:
            #     if trans_dict_c[t_d] == 0:
            #         trans_dict_c[t_d] = 1-sum(trans_dict_c.values())
            #     transitions_string += f" ({trans_dict_c[t_d]} + {trans_dict_d[t_d]}*{t_d[2]}) : (s'={t_d[1]}) +"

            if has_outgoing:
                transitions_strings += transitions_string[:-2] + ';\n'

        rewards_strs = ['true : 0;'] if len(rewards_strs) == 0 else rewards_strs
        contents = in_out.pdtmc_string(p_string, self.pomdp.nS, nM, transitions_strings, label_strings, rewards_strs[0])
        fn = in_out.cache_pdtmc(contents)
        prism_program = stormpy.parse_prism_program(fn, simplify = False)
        if remove_file_from_disk:
            os.remove(fn)
        if self.pomdp.is_parametric:
            model = stormpy.build_sparse_parametric_model(prism_program)
            p_region_dict = {
                x : (stormpy.RationalRF(self.p_bounds[x.name][0]), 
                    stormpy.RationalRF(self.p_bounds[x.name][1]))
                for x in model.collect_probability_parameters()
            }
        else:
            model = stormpy.build_model(prism_program)
            p_region_dict = {}

        pdtmc = PDTMCModelWrapper(model, self.pomdp, nM, p_region_dict, state_labels, memory_labels)
        if pdtmc.nS != np.count_nonzero(hops < np.inf):
            raise ValueError('Inaccuracies after translating PDTMC to Stormpy model.')
        return pdtmc

    def worst_mdp(self, check_result, fsc):
        worst_ps, worst_value = check_result.worst_parameter_value(fsc, self.cfg['p_bounds'])
        if worst_ps is not None:
            utils.inform('(CTRX)' + '\t\t\t%.4f' % worst_value + ' under current FSC.', indent = 0)
            self.build_mdp({p : worst_ps[idx] for idx, p in enumerate(self.p_bounds)})
        return self.mdp, worst_ps, worst_value

    def _remember_labels(self):
        self.mdp.old_state_values = np.array(self.mdp.state_values)
        self.mdp.state_values = np.maximum(self.old_state_values, self.mdp.state_values) # minimization
        transitions = np.logical_not(self.mdp.A)
        action_values = np.full((self.mdp.nS, self.mdp.nA), np.nan)
        if self.mdp.rewards.size > 0:
            rewards = np.sum(self.mdp.T * self.mdp.rewards[:, 0], axis = -1)
        else:
            rewards = np.zeros_like((self.mdp.action_values))
        action_values[transitions] = np.sum(self.mdp.T[transitions] * self.mdp.state_values, axis = -1) + rewards[transitions]
        self.mdp.action_values = action_values
        return

    def add_fsc(self, check_result, fsc):
        """ Adds the FSC that is found to the efficient front, if applicable. """

        new_lb, new_ub = check_result._lb_values[0], check_result._ub_values[0]
        dominates = set()
        dominated = False
        new_pareto_front = list(self.pareto_front)
        new_pareto_values = list(self.pareto_values)
        new_dist = new_ub - new_lb
        for (old_fsc_idx, old_fsc), (old_dist, old_ub) in zip(enumerate(self.pareto_front), self.pareto_values):
            if self.objective == 'min':
                if new_ub > old_ub and new_dist > old_dist:
                    dominated = True
                    break
                elif new_ub < old_ub and new_dist < old_dist:
                    new_pareto_front.remove(old_fsc)
                    new_pareto_values.remove((old_dist, old_ub))
            elif self.objective == 'max':
                raise NotImplementedError('Not implemented.')
        self.pareto_front = new_pareto_front
        self.pareto_values = new_pareto_values
        if not dominated:
            self.pareto_front.append(fsc)
            self.pareto_values.append((new_dist, new_ub))
            return True
        self._solutions.append((fsc, new_lb, new_ub))
        return False

    def simulation_length(self):
        if isinstance(self.cfg['length'], str) and self.cfg['length'].startswith('x'):
            factor = int(self.cfg['length'].split('x')[-1])
            return math.ceil(factor * self.mdp.state_values[0])
        return self.cfg['length']
