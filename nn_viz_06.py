import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")

# Generate a autoencoder neural network visualization
# setting up constant params
FIGURE_WIDTH = 16
FIGURE_HEIGHT = 9
RIGHT_BORDER = 0.7
LEFT_BORDER = 0.7
TOP_BORDER = 0.8
BOTTOM_BORDER = 0.6

N_IMAGE_PIXEL_COLS = 64
N_IMAGE_PIXEL_ROWS = 48
N_NODES_BY_LAYER = [10, 7, 5, 8]

INPUT_IMAGE_BOTTOM = 5
INPUT_IMAGE_HEIGHT = 0.25 * FIGURE_HEIGHT
ERROR_IMAGE_SCALE = 0.7
ERROR_GAP_SCALE = 0.3
BETWEEN_LAYER_SCALE = 0.8
BETWEEN_NODE_SCALE = 0.4


def main():
    # print("All the visualization code goes here")
    # print(f"Node images are {N_IMAGE_PIXEL_ROWS}" + f" by
    # {N_IMAGE_PIXEL_COLS} pixels")
    p = construct_parameters()
    # print("params:")
    # for key, value in p.items():
    #    print(key, ":", value)
    fig = create_background(p)
    save_nn_viz(fig, postfix = "06_empty")


def construct_parameters():
    # Build a dictionary of params that describe the size and location
    # of the elements of visualization.
    aspect_ratio = N_IMAGE_PIXEL_COLS / N_IMAGE_PIXEL_ROWS
    params = {}

    params["figure"] = {"height": FIGURE_HEIGHT, "width": FIGURE_WIDTH}

    params["input"] = {"n_cols": N_IMAGE_PIXEL_COLS, "n_rows": N_IMAGE_PIXEL_ROWS,
                        "aspect_ratio": aspect_ratio,
                        "image": {
                            "bottom": INPUT_IMAGE_BOTTOM,
                            "height": INPUT_IMAGE_HEIGHT,
                            "width": INPUT_IMAGE_HEIGHT * aspect_ratio
                                }
                    }

    params["network"] = {
        "n_nodes": N_NODES_BY_LAYER,
        "n_layers": len(N_NODES_BY_LAYER),
        "max_nodes": np.max(N_NODES_BY_LAYER)
    }

    params["node_image"] = {
        "height": 0,
        "width": 0
    }

    params["error_image"] = {
        "left": 0,
        "bottom": 0,
        "width": params["input"]["image"]["width"] * ERROR_IMAGE_SCALE,
        "height": params["input"]["image"]["height"] * ERROR_IMAGE_SCALE
    }

    params["gap"] = {
        "right_border": RIGHT_BORDER,
        "left_border": LEFT_BORDER,
        "bottom_border": BOTTOM_BORDER,
        "top_border": TOP_BORDER,
        "between_layer": 0,
        "between_layer_scale": BETWEEN_LAYER_SCALE,
        "between_node": 0,
        "between_node_scale": BETWEEN_NODE_SCALE,
        "error_gap_scale": ERROR_GAP_SCALE
    }

    return params


def create_background(p):
    fig = plt.figure(
        figsize=(p["figure"]["width"], p["figure"]["height"])
    )
    return fig


def save_nn_viz(fig, postfix="0"):
    base_name = "nn_viz_"
    filename = base_name + postfix + ".png"
    fig.savefig(filename)


if __name__ == "__main__":
    main()
