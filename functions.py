import numpy as np
import pandas as pd
from scipy.optimize import bisect
from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm


def line_search(df_link, x, y, x_tram, y_tram, t_tram, cost_function):
    """
    Determine step size by bisection line search

    :param df_link: link data frame
    :param x: current link flows
    :param y: target link flows
    :param cost_function: callable that returns the cost when provided with `df_link` and link flows.
    :return step: optimal step size
    """

    def objective_function(alpha):
        times = cost_function(df_link, x + alpha * (y - x))
        return ((y - x) * times).sum() + ((y_tram - x_tram) * t_tram).sum()

    try:
        return bisect(objective_function, 0, 1, disp=True)
    except ValueError:
        return min(0, 1, key=lambda x: abs(objective_function(x)))


def all_or_nothing_assignment(df_od, df_link, df_link_tram, t, restriction_level = 0):
    """
    All-or-nothing assignment:
    load traffic flows on shortest paths

    :param df_od, df_link: network data frames
    :param t: link cost
    :return x: link flow
    """
    n_nodes = max(df_link.start_node.max(), df_link.end_node.max()) + 1
    x = np.zeros(len(df_link))
    x_tram = np.zeros(len(df_link_tram))

    # Compute time without tram links
    i,j = df_link.start_node, df_link.end_node
    i, j = i.astype(np.int32), j.astype(np.int32)
    A = csr_array((t, (i, j)), shape=(n_nodes, n_nodes))
    
    orgs = df_od.org.unique()
    orgs_index_notram = (df_od.org.values.reshape(-1,1) == orgs.reshape(1, -1)).argmax(1)

    labels_notram, predecessors_notram = dijkstra(csgraph=A, directed=True, return_predecessors=True, indices=orgs)
    
    match restriction_level:
        case 0:
            # Simple case
            for k, (o, d, q) in df_od[["org", "dest", "q"]].iterrows():
                i = int(d)
                while i != int(o):
                    j=i
                    i = predecessors_notram[orgs_index_notram[k], i]
                    if i <0:
                        print("ERROR", o, d, t, sep="\n")
                        break
                    x[df_link[(df_link.start_node == i) & (df_link.end_node == j)].index] += q
        case 1:
            # Compute time on tram links for ODs that can use it
            tram_mask = df_od.dest.isin(df_link_tram.end_node) & df_od.org.isin(df_link_tram.start_node)
            i,j,t = df_link_tram.start_node, df_link_tram.end_node, df_link_tram.cost
            i, j = i.astype(np.int32), j.astype(np.int32)
            A = csr_array((t, (i, j)), shape=(n_nodes, n_nodes))

            orgs = df_od[tram_mask].org.unique()
            orgs_index_tram = (df_od.org.values.reshape(-1,1) == orgs.reshape(1, -1)).argmax(1)

            labels_tram, predecessors_tram = dijkstra(csgraph=A, directed=True, return_predecessors=True, indices=orgs)
            
            # Assign traffic on the best time for it, considering whether this od can use the tram
            for k, (o, d, q) in df_od[["org", "dest", "q"]].iterrows():
                use_tram = tram_mask[k] and (labels_tram[orgs_index_tram[k], int(d)] < labels_notram[orgs_index_notram[k], int(d)])
                predecessors = predecessors_tram if use_tram else predecessors_notram
                orgs_index = orgs_index_tram if use_tram else orgs_index_notram
                i = int(d)
                while i != int(o):
                    j=i
                    i = predecessors[orgs_index[k], i]
                    if i <0:
                        print("ERROR", o, d, t, sep="\n")
                        break
                    if use_tram:
                        x_tram[df_link_tram[(df_link_tram.start_node == i) & (df_link_tram.end_node == j)].index] += q
                    else:
                        x[df_link[(df_link.start_node == i) & (df_link.end_node == j)].index] += q
        case 2:
            # Prepare two level network
            # Road links
            roads_links = df_link[["start_node", "end_node"]].copy()
            roads_links["cost"] = t
            # Connexion links
            onboard_links = df_link_tram[["start_node"]].copy()
            onboard_links["end_node"] = onboard_links["start_node"]
            onboard_links["end_node"] += n_nodes # We offset tram nodes index by n_nodes
            onboard_links["cost"] = 0
            offboard_links = df_link_tram[["end_node"]].copy()
            offboard_links["start_node"] = offboard_links["end_node"]
            offboard_links["start_node"] += n_nodes # We offset tram nodes index by n_nodes
            offboard_links["cost"] = 0
            # Tram links
            tram_links = df_link_tram[["start_node", "end_node", "cost"]].copy()
            tram_links["start_node"] += n_nodes # We offset tram nodes index by n_nodes
            tram_links["end_node"] += n_nodes # We offset tram nodes index by n_nodes
            # => Merge
            df_link_onboard_P_a_R = pd.concat([roads_links, onboard_links, tram_links], ignore_index=True)
            df_link_offboard_P_a_R = pd.concat([roads_links, offboard_links, tram_links], ignore_index=True)

            # Compute the shortest paths
            onboard_mask = df_od.dest.isin(df_link_tram.end_node)
            orgs_onboard = df_od[onboard_mask].org.unique()
            orgs_index_onboard = (df_od.org.values.reshape(-1,1) == orgs_onboard.reshape(1, -1)).argmax(1)
            i,j,t = df_link_onboard_P_a_R.start_node, df_link_onboard_P_a_R.end_node, df_link_onboard_P_a_R.cost
            i, j = i.astype(np.int32), j.astype(np.int32)
            A = csr_array((t, (i, j)), shape=(2*n_nodes, 2*n_nodes))
            labels_onboard, predecessors_onboard = dijkstra(csgraph=A, directed=True, return_predecessors=True, indices=orgs_onboard)

            offboard_mask = df_od.org.isin(df_link_tram.start_node)
            orgs_offboard = df_od[offboard_mask].org.unique() + n_nodes
            orgs_index_offboard = (df_od.org.values.reshape(-1,1)+n_nodes == orgs_offboard.reshape(1, -1)).argmax(1)
            i,j,t = df_link_offboard_P_a_R.start_node, df_link_offboard_P_a_R.end_node, df_link_offboard_P_a_R.cost
            i, j = i.astype(np.int32), j.astype(np.int32)
            A = csr_array((t, (i, j)), shape=(2*n_nodes, 2*n_nodes))
            labels_offboard, predecessors_offboard = dijkstra(csgraph=A, directed=True, return_predecessors=True, indices=orgs_offboard)

            # Do assignment, based on what each OD can do:
            for k, (o, d, q) in df_od[["org", "dest", "q"]].iterrows():
                use_onboard = onboard_mask[k] and (labels_onboard[orgs_index_onboard[k], int(d)+n_nodes] < labels_notram[orgs_index_notram[k], int(d)])
                use_offboard = offboard_mask[k] and (labels_offboard[orgs_index_offboard[k], int(d)] < labels_notram[orgs_index_notram[k], int(d)])
                if use_onboard and use_offboard:
                    if labels_onboard[orgs_index_onboard[k], int(d)+n_nodes] < labels_offboard[orgs_index_offboard[k], int(d)]:
                        use_offboard = False
                    else:
                        use_onboard = False
                predecessors = predecessors_onboard if use_onboard else predecessors_offboard if use_offboard else predecessors_notram
                orgs_index = orgs_index_onboard if use_onboard else orgs_index_offboard if use_offboard else orgs_index_notram
                i = int(d) + (n_nodes if use_onboard else 0)
                while i != int(o) + (n_nodes if use_offboard else 0):
                    j=i
                    i = predecessors[orgs_index[k], i]
                    if i <0:
                        print("ERROR", o, d, use_onboard, use_offboard, sep="\n")
                        break
                    x_tram[df_link_tram[(df_link_tram.start_node + n_nodes == i) & (df_link_tram.end_node + n_nodes == j)].index] += q
                    x[df_link[(df_link.start_node == i) & (df_link.end_node == j)].index] += q
        
        case _:
            raise ValueError(f"Unknown value {restriction_level} of `restriction_level`")

    return x, x_tram


def static_assignment_fw(
    df_od: pd.DataFrame,
    df_link: pd.DataFrame,
    df_link_tram: pd.DataFrame,
    max_iter: int,
    max_gap: float,
    cost_function,
    objective_function,
    restriction_level: int = 0,
):
    """
    Solve static traffic assignment
    using Frank-Wolfe algorithm

    :param df_od, df_link: network data frames
    :param df_link_tram: tram link data frame
    :param max_iter: max number of iterations of main loop
    :param max_gap: gap threshold of main loop
    :param cost_function: callable that returns the cost when provided with `df_link` and link flows.
    :param objective_function: callable that returns the cost when provided with `df_link` and link flows.
    :param restriction_level: level of restriction for the tram links : 0 = no use, 1 = use if o and d on the line, 2 = use if one of o, d on the line
    :return x_star: equilibrium link flow
    :return x_tram: equilibrium tram link flow
    :return gap: gap over iterations
    :return obj: objective value over iterations
    """
    t_tram = df_link_tram.cost # Constant

    # initialize link flow
    x = np.zeros(len(df_link))
    t = cost_function(df_link, x)
    x, x_tram = all_or_nothing_assignment(df_od, df_link, df_link_tram, t, restriction_level)

    # main loop
    gap = np.inf * np.ones(max_iter)
    obj = np.inf * np.ones(max_iter)
    i = 0

    for i in (pbar := tqdm(range(max_iter))):
        pbar.set_description(f"Gap: {gap[i-1]:.4g} - Objective: {obj[i-1]:.4f}")
        pbar.refresh()
        # update link travel time
        t = cost_function(df_link, x)

        # solve target flow
        y, y_tram = all_or_nothing_assignment(df_od, df_link, df_link_tram, t, restriction_level)

        # derive step size by line search
        step = line_search(df_link, x, y, x_tram, y_tram, t_tram, cost_function)

        # compute relative gap
        gap[i] = (np.dot(t, x - y) + np.dot(t_tram, x_tram - y_tram)) / (np.dot(t, x) + np.dot(t_tram, x_tram))

        # update link flow
        x += (y - x) * step
        x_tram += (y_tram - x_tram) * step

        # compute objective value
        obj[i] = objective_function(df_link, x)

        # check convergence
        if gap[i] < max_gap:
            break

    # process outputs
    x_star = x
    gap = gap[: i + 1]
    obj = obj[: i + 1]

    return x_star, x_tram, gap, obj
