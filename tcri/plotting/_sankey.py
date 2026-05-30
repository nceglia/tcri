import numpy as np
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors


class SankeyNode(object):
    """Object for sankey node"""

    def __init__(self, x, y, val, dx = 0.2, color = None, **kwargs):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = val
        self.x_gap = 0.05

        self.max_x = self.x + self.dx/2
        self.min_x = self.x - self.dx/2
        self.max_y = self.y + self.dy
        self.min_y = self.y

        self.color = color
        self.patch = mpatches.Rectangle([self.min_x+self.x_gap, self.min_y], self.dx-2*self.x_gap, self.dy, facecolor = self.color, edgecolor = 'None')

    def plot(self, ax):
        ax.add_patch(self.patch)


    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    def plot_node_connection(self, destination_node, ax, **kwargs):
        num_segments = 500
        discretize = np.linspace(0, 1, num_segments)

        # Shape of the connection
        x = self.max_x + (destination_node.min_x - self.max_x) * discretize
        y_shape = 1 / (1 + (10**2 / np.power(10, 4 * discretize)))
        y_shape = (y_shape - y_shape[0]) / (y_shape[-1] - y_shape[0])
        y_top = self.max_y + (destination_node.max_y - self.max_y) * y_shape
        y_bot = self.min_y + (destination_node.min_y - self.min_y) * y_shape

        # Color interpolation
        start_color = np.array(mcolors.to_rgb(self.color))
        end_color = np.array(mcolors.to_rgb(destination_node.color)) if destination_node.color else start_color

        for i in range(num_segments - 1):
            interp_color = (1 - discretize[i]) * start_color + discretize[i] * end_color
            ax.fill_between(x[i:i+2], y_top[i:i+2], y_bot[i:i+2], facecolor=interp_color, edgecolor='none')
