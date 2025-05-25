import numpy as np
import pandas as pd
from scipy.optimize import bisect
from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm


def line_search(df_link, x, y, cost_function):
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
        return ((y - x) * times).sum()

    try:
        return bisect(objective_function, 0, 1, disp=True)
    except ValueError:
        return 0


def all_or_nothing_assignment(df_od, df_link, df_link_tram, t, restriction_level = 0):
    """
    All-or-nothing assignment:
    load traffic flows on shortest paths

    :param df_od, df_link: network data frames
    :param t: link cost
    :return x: link flow
    """
    n_nodes = max(df_link.start_node.max(), df_link.end_node.max()) + 1

    # Separate the ODs between those that can use the tram and those that can not
    match restriction_level:
        case 0:
            mask = np.full(len(df_od), False)
        case 1:
            mask = df_od.dest.isin(df_link_tram.end_node) & df_od.org.isin(df_link_tram.start_node)
        case 2:
            mask = df_od.dest.isin(df_link_tram.end_node) | df_od.org.isin(df_link_tram.start_node)
        case _:
            raise ValueError(f"Unknown value {restriction_level} of `restriction_level`")
    od_notram = df_od[~mask].reset_index(drop=True)
    od_tram = df_od[mask].reset_index(drop=True)

    # do assignment without tram links
    i,j = df_link.start_node, df_link.end_node
    i, j = i.astype(np.int32), j.astype(np.int32)
    A = csr_array((t, (i, j)), shape=(n_nodes, n_nodes))
    
    orgs = od_notram.org.unique()
    orgs_index = (od_notram.org.values.reshape(-1,1) == orgs.reshape(1, -1)).argmax(1)

    labels, predecessors = dijkstra(csgraph=A, directed=True, return_predecessors=True, indices=orgs)
    
    x = np.zeros(len(df_link))
    for k, (o, d, q) in od_notram[["org", "dest", "q"]].iterrows():
        i = int(d)
        while i != int(o):
            j=i
            i = predecessors[orgs_index[k], i]
            if i <0:
                print("ERROR", o, d, t, sep="\n")
                break
            x[df_link[(df_link.start_node == i) & (df_link.end_node == j)].link_id] += q

    # Do assignment with tram links
    if len(od_tram) > 0:
        # Road links
        df_link_complete = df_link[["start_node", "end_node"]].copy()
        df_link_complete["cost"] = t
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

        df_link_complete = pd.concat([df_link_complete, onboard_links, offboard_links, tram_links], ignore_index=True)

        # Create the adjacency matrix
        i,j,t = df_link_complete.start_node, df_link_complete.end_node, df_link_complete.cost
        i, j = i.astype(np.int32), j.astype(np.int32)
        A = csr_array((t, (i, j)), shape=(2*n_nodes, 2*n_nodes))

        orgs = od_tram.org.unique()
        orgs_index = (od_tram.org.values.reshape(-1,1) == orgs.reshape(1, -1)).argmax(1)

        labels, predecessors_tram = dijkstra(csgraph=A, directed=True, return_predecessors=True, indices=orgs)

        x_tram = np.zeros(len(df_link_tram))
        for k, (o, d, q) in od_tram[["org", "dest", "q"]].iterrows():
            i = int(d)
            while i != int(o):
                j=i
                i = predecessors_tram[orgs_index[k], i]
                if i <0:
                    print("ERROR", o, d, t, sep="\n")
                    break
                x[df_link[(df_link.start_node == i) & (df_link.end_node == j)].link_id] += q
                x_tram[tram_links[(tram_links.start_node == i) & (tram_links.end_node == j)].index] += q


        return x, x_tram
    return x, 0


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
        step = line_search(df_link, x, y, cost_function)

        # compute relative gap
        gap[i] = np.dot(t, x - y) / np.dot(t, x)

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
