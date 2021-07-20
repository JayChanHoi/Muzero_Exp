include "mcts_minmax.pyx"
include "mcts_node.pyx"

from itertools import count
import math

import numpy as np

import torch

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
        cdef list action_list = [0 for _ in range(20000)]
        cdef list ucb_score_list = [0. for _ in range(20000)]
        cdef list child_list = [CyphonNode(0) for _ in range(20000)]

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