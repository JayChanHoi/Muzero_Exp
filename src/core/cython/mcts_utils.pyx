import numpy as np

import torch

cpdef object node_expand_action_util(list actions,
                                     object node,
                                     object policy,
                                     dict children):

    for action in actions:
        children[action] = node(policy[action])

cpdef node_add_exploration_noise_util(list actions,
                                      object noise,
                                      dict children,
                                      float frac):

    for a, n in zip(actions, noise):
        children[a].prior = children[a].prior * (1 - frac) + n * frac

cdef class CyphonNode():
    cpdef public int visit_count, to_play
    cpdef public float prior, value_sum, reward
    cpdef public dict children
    cpdef public object hidden_state

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

    cpdef void expand(self, int to_play, list actions, object network_output):
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