"""
Generate a autoencoder neural network visualization
"""


#setting up constant params
FIGURE_WIDTH = 16
FIGURE_HEIGHT = 9
RIGHT_BORDER = 0.7
LEFT_BORDER = 0.7
TOP_BORDER = 0.8
BOTTOM_BORDER = 0.6

N_IMAGE_PIXEL_COLS = 64
N_IMAGE_PIXEL_ROWS = 48
N_NODES_BY_LAYER = [10,7,5,8]

INPUT_IMAGE_BOTTOM = 5
INPUT_IMAGE_HEIGHT = 0.25 * FIGURE_HEIGHT
ERROR_IMAGE_SCALE = 0.7
ERROR_GAP_SCALE = 0.3
BETWEEN_LAYER_SCALE = 0.8 
BETWEEN_NODE_SCALE = 0.4


def main():
    #print("All the visualization code goes here")
    #print(f"Node images are {N_IMAGE_PIXEL_ROWS}" + f" by {N_IMAGE_PIXEL_COLS} pixels")
    p = construct_parameters()
    print("params:")
    for key, value in p.items():
        print(key, ":", value)

def construct_parameters():
    """
    Build a dictionary of params that describe the size and location 
    of the elements of visualization.
    """

    aspect_ratio = N_IMAGE_PIXEL_COLS / N_IMAGE_PIXEL_ROWS
    params = {}
    params["figure"] = {"height":FIGURE_HEIGHT, "width":FIGURE_WIDTH}
    params["input"] = {"n_cols":N_IMAGE_PIXEL_COLS, "n_rows":N_IMAGE_PIXEL_ROWS,
                        "aspect_ratio":aspect_ratio,
                        "image":{
                            "bottom":INPUT_IMAGE_BOTTOM,
                            "height":INPUT_IMAGE_HEIGHT,
                            "width":INPUT_IMAGE_HEIGHT * aspect_ratio
                        }
                        }
    return params


if __name__ == "__main__":
    main()