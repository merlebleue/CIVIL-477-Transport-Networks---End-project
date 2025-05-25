import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_network(
    df_node, df_link, df_link_tram=None, df_od=None, link_flows=None, title=None
):
    """
    Plot the network with nodes and links
    :param df_node: node data frame
    :param df_link: link data frame
    :param df_link_tram: tram link data frame
    :param df_od: origin-destination data frame
    :param title: title of the plot
    """

    if title is None:
        if df_od is not None:
            title = "Network with OD pairs"
        else:
            title = "Network representation"

    # Create a figure
    plt.figure(figsize=(7, 7))
    ax = plt.gca()

    normal_handles = []

    # Plot nodes
    node_points = plt.scatter(
        df_node.X,
        df_node.Y,
        c="blue",
        s=50 if df_od is None else df_od.groupby("org").sum().q,
        zorder=20,
    )

    # Add a legend for node sizes
    if df_od is not None:
        handles, labels = node_points.legend_elements(prop="sizes", num=4, c="blue")
        legend = ax.legend(handles, labels, loc="lower right", title="Demand")
        ax.add_artist(legend)

    # Normalize link flows for visualization
    if link_flows is not None and isinstance(link_flows, np.ndarray):
        max_flow = int(np.max(link_flows))+1
        min_flow = 0
        normalized_flows = 5 * (link_flows - min_flow) / (max_flow - min_flow)
    else:
        normalized_flows = np.ones(len(df_link))

    # Plot links
    for k, row in df_link.iterrows():
        plt.plot(
            [df_node.X[row.start_node], df_node.X[row.end_node]],
            [df_node.Y[row.start_node], df_node.Y[row.end_node]],
            color="gray",
            alpha=0.5,
            linewidth=normalized_flows[k],
        )

    # Add a legend for link flows
    if link_flows is not None and isinstance(link_flows, np.ndarray):
        flow_legend_values = [min_flow, (min_flow + max_flow) / 2, max_flow]
        handles = []
        for flow in flow_legend_values:
            handles += plt.plot(
                [],
                [],
                color="gray",
                alpha=0.5,
                linewidth=5 * (flow - min_flow) / (max_flow - min_flow),
                label=f"{flow:.1f}",
            )

        legend1 = ax.legend(handles=handles, loc="best", title="Link Flows")
        ax.add_artist(legend1)
    else:
        normal_handles += plt.plot(
                [],
                [],
                color="gray",
                alpha=0.5,
                linewidth=1,
                label=f"Road links",
            )

    # Plot tram links
    if df_link_tram is not None:
        legend_lines = set()
        for _, row in df_link_tram.iterrows():
            normal_handles += plt.plot(
                [df_node.X[row.start_node], df_node.X[row.end_node]],
                [df_node.Y[row.start_node], df_node.Y[row.end_node]],
                color=['red', 'orange'][int(row.line)-1],
                linewidth=2,
                label=f"Tram Line {row.line}" if row.line not in legend_lines else "",
                zorder = 10
            )
            legend_lines.add(row.line)

    # Plot OD pairs
    if df_od is not None:
        for _, row in df_od.iterrows():
            plt.plot(
                [df_node.X[row.org], df_node.X[row.dest]],
                [df_node.Y[row.org], df_node.Y[row.dest]],
                color="green",
                linewidth=row.q,
                alpha=0.2,
            )

        # Add a legend for the size of the OD pairs
        max_q = int(df_od.q.max())
        min_q = int(df_od.q.min()) + 1
        sizes = [min_q, (min_q + max_q) / 2, max_q]
        for size in sizes:
            plt.plot(
                [], [], color="green", linewidth=size, alpha=0.75, label=f"q = {size}"
            )

    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect("equal")
    plt.legend(
        loc="upper right",
        handles=normal_handles,
    )
    plt.show()
