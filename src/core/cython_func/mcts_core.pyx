from itertools import count
import math

import numpy as np

import torch

cdef class CyphonNode(object):
    cdef public int visit_count
    cdef public float prior, value_sum, reward
    cdef public dict children
    cdef public object hidden_state, to_play

    def __init__(self, float prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0.
        self.children = {}
        self.hidden_state = None
        self.reward = 0.

    cpdef int expanded(self):
        if len(self.children) > 0:
            return 1
        else:
            return 0

    cpdef float value(self):
        if self.visit_count == 0:
            return 0.
        else:
            return self.value_sum / self.visit_count

    cpdef void expand(self, object to_play, list actions, object network_output):
        cdef int action

        self.to_play = to_play
        self.hidden_state = network_output.hidden_state
        self.reward = network_output.reward
        # softmax over policy logits
        policy_temp = network_output.policy_logits.exp().squeeze()
        policy_temp_sum = (policy_temp * torch.zeros(policy_temp.shape[0]).scatter(0, torch.LongTensor(actions), 1.)).sum(dim=0)
        policy = policy_temp / policy_temp_sum
        for action in actions:
            self.children[action] = CyphonNode(policy[action])

        return

    cpdef void add_exploration_noise(self, float dirichlet_alpha, float exploration_fraction):
        cdef int a
        cdef float n

        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac

        return

cdef class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""
    cdef public float maximum
    cdef public float minimum

    def __init__(self, float min_value_bound=0, float max_value_bound=0):
        self.maximum = min_value_bound if min_value_bound != 0 else -float('inf')
        self.minimum = max_value_bound if max_value_bound != 0 else float('inf')

    cpdef void update(self, float value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

        return

    cpdef float normalize(self, float value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        else:
            return value

cdef class MCTS(object):
    cdef public object config

    def __init__(self, object config):
        self.config = config

    cpdef void run(self, object root, object action_history, object model):
        cdef object min_max_stats = MinMaxStats()
        cdef int i
        cdef int j

        for i in range(self.config.num_simulations):
            history = action_history.clone()
            node = root
            search_path = [node]

            for j in count():
                if node.expanded():
                    action, node = self.select_child(node, min_max_stats)
                    history.add_action(action)
                    search_path.append(node)

            # Inside the search tree we use the dynamics function to obtain the next
            # hidden state given an action and the previous hidden state.
            parent = search_path[-2]
            network_output = model.recurrent_inference(
                parent.hidden_state,
                torch.tensor([[history.last_action()]],device=parent.hidden_state.device)
            )
            node.expand(history.to_play(), history.action_space(), network_output)

            self.backpropagate(search_path, network_output.value.item(), history.to_play(), min_max_stats)

        return

    cpdef list select_child(self, object node, object min_max_stats):
        cdef int i
        cdef int action
        cdef object child
        cdef int action_list[100]
        cdef float ucb_score_list[100]
        cdef list child_list = [_ for _ in range(100)]

        for i, (action, child) in enumerate(node.children.items()):
            ucb_score = self.ucb_score(node, child, min_max_stats)
            ucb_score_list[i] = ucb_score
            action_list[i] = action
            child_list[i] = child

        max_candidate_index = ucb_score_list.index(max(ucb_score_list))
        max_action = action_list[max_candidate_index]
        max_child = child_list[max_candidate_index]

        return max_action, max_child

    cpdef float ucb_score(self, object parent, object child, object min_max_stats):
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = min_max_stats.normalize(child.value())

        return prior_score + value_score

    cpdef void backpropagate(self, list search_path, float value, object to_play, object min_max_stats):
        cdef object node

        for node in search_path:
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
            min_max_stats.update(node.value())

            value = node.reward + self.discount * value

        return