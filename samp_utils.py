


def true_subgraph(G, nodes_to_keep):
    G_sub = G.copy(as_view=False)
    G_sub.remove_nodes_from([n for n in G if n not in set(nodes_to_keep)])
    return G_sub


