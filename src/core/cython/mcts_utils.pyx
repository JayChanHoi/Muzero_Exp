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