import numpy as np
import matplotlib.pyplot as plt

const_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]

def transform_indx_into_colors(inds, N):
    colors = np.zeros(N)
    for i, ind in enumerate(inds):
        colors[ind] = i
    final_colors = []
    for c in colors:
        final_colors.append(const_colors[int(c)])
    return final_colors

def transform_weights_into_colors(weights, inds, N):
    colors = np.zeros(N)
    for i, ind in enumerate(inds):
        q = np.ones((len(ind), 1)) * (-1)
        q[weights[i] > 0] = i
        colors[ind] = q.flatten()
    final_colors = []
    for c in colors:
        if c == -1:
            final_colors.append("tab:gray")
        else:
            final_colors.append(const_colors[int(c)])
    return np.array(final_colors)

def plot_coresets(data, sequential_w, full_inds, output, parallel_w, k, name):

    full_inds_i = transform_indx_into_colors(full_inds, len(data))
    output_i = transform_weights_into_colors(output, full_inds, len(data))

    fig, ax = plt.subplots(1, 4, figsize = (20, 8))
    for i, c in enumerate(const_colors):
        ax[0].scatter(data[np.array(full_inds_i) == c, 0], data[np.array(full_inds_i) == c, 1], c = c, label = i + 1)

    ax[1].scatter(data[np.array(output_i) == "tab:gray", 0], data[np.array(output_i) == "tab:gray", 1], c = "tab:gray", label = -1)
    for i, c in enumerate(const_colors):
        ax[1].scatter(data[np.array(output_i) == c, 0], data[np.array(output_i) == c, 1], c = c, label = i + 1)
    
    sc2 = ax[2].scatter(data[:, 0], data[:, 1], c = (parallel_w > 0).astype(int).flatten())
    sc3 = ax[3].scatter(data[:, 0], data[:, 1], c = (sequential_w > 0).astype(int).flatten())

    ax[2].legend(handles = sc2.legend_elements()[0], labels = ["Not in coreset", "Coreset"])
    ax[3].legend(handles = sc3.legend_elements()[0], labels = ["Not in coreset", "Coreset"])
    ax[0].legend()
    ax[1].legend()
    ax[0].set_title("Distribution across processors")
    ax[1].set_title("Coreset on each processor")
    ax[2].set_title("Final parallel coreset")
    ax[3].set_title("Sequential coreset")
    fig.suptitle(f"k = {k}")
    plt.savefig(f"{name}_{k}.png")