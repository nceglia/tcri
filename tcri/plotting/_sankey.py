"""The phenotype-flow Sankey that ``pl.joint_distribution(view="sankey")`` draws.

:func:`_sankey_layout` places every node and every band end from each clone's phenotype mass at
each covariate value, and :func:`_render_sankey` draws that layout with :class:`SankeyNode`. The
geometry is kept apart from the drawing so a test can pin each position against values computed
by hand, without reading them back off an axes.
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.path import Path

#: Points along a band's outline, and steps of its color ramp.
_SEGMENTS = 500

#: How far each band reaches past its top and bottom, as a fraction of the tallest bar: about
#: half a pixel to a pixel at the usual figure sizes. Two bands that touch then overlap
#: slightly, so the antialiased edge of each does not leave a line of background between them.
_OVERLAP = 0.0015


class SankeyNode:
    """One stacked bar of a Sankey, or one end of a band.

    The rectangle spans ``x - dx/2 + x_gap`` to ``x + dx/2 - x_gap`` with ``x_gap = 0.05``. A
    ``dx`` below ``2 * x_gap`` makes that width negative, and the rectangle is drawn leftward:
    the bar is then ``2 * x_gap - dx`` wide around ``x``, and a band leaving at ``x + dx/2``
    starts inside it.

    Parameters
    ----------
    x
        The center of the bar.
    y
        The bottom of the bar.
    val
        The height of the bar.
    dx
        The nominal width. A band leaves a node at ``x + dx/2`` and arrives at ``x - dx/2``.
    color
        The face color.
    """

    x_gap = 0.05

    def __init__(self, x, y, val, dx=0.2, color=None):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = val
        self.max_x = x + dx / 2
        self.min_x = x - dx / 2
        self.max_y = y + val
        self.min_y = y
        self.color = color
        self.patch = mpatches.Rectangle([self.min_x + self.x_gap, self.min_y],
                                        dx - 2 * self.x_gap, val,
                                        facecolor=color, edgecolor="None")

    def plot(self, ax):
        """Add the bar to ``ax``."""
        ax.add_patch(self.patch)

    def plot_node_connection(self, destination_node, ax, pad=0.0):
        """Draw the band from this node's right edge to ``destination_node``'s left edge.

        The band's top and bottom follow a logistic curve from this node's top and bottom to the
        destination's, and its color runs linearly from this node's color to the destination's.
        The band is one shape: an image of the color ramp, clipped to the band's outline. Drawing
        it as many thin slices instead leaves a faint line of background at every seam, because
        each slice's edge is antialiased on its own. ``pad`` widens the outline by that much
        above and below, in data units.
        """
        t = np.linspace(0, 1, _SEGMENTS)
        x = self.max_x + (destination_node.min_x - self.max_x) * t
        shape = 1 / (1 + (10**2 / np.power(10, 4 * t)))
        shape = (shape - shape[0]) / (shape[-1] - shape[0])
        top = self.max_y + (destination_node.max_y - self.max_y) * shape + pad
        bottom = self.min_y + (destination_node.min_y - self.min_y) * shape - pad
        low, high = min(bottom.min(), top.min()), max(bottom.max(), top.max())
        if high <= low:
            return

        start = np.array(mcolors.to_rgb(self.color))
        end = (np.array(mcolors.to_rgb(destination_node.color))
               if destination_node.color else start)
        ramp = ((1 - t[:, None]) * start + t[:, None] * end)[None, :, :]

        outline = mpatches.PathPatch(
            Path(np.concatenate([np.column_stack([x, top]),
                                 np.column_stack([x[::-1], bottom[::-1]])])),
            facecolor="none", edgecolor="none", transform=ax.transData)
        ax.imshow(ramp, extent=(x[0], x[-1], low, high), origin="lower", aspect="auto",
                  interpolation="bilinear", clip_path=outline)


def _sankey_layout(mass, n_phenotypes):
    """Every node and band end of the Sankey.

    Parameters
    ----------
    mass
        One ``{clone: vector}`` per covariate value, in drawing order. A vector holds the
        clone's mass on each phenotype, in category order.
    n_phenotypes
        The length of every vector.

    Returns
    -------
    dict
        ``nodes``: one ``(n_phenotypes, 2)`` array of (bottom, height) per covariate value, the
        phenotype totals over clones stacked from 0 in category order.

        ``bands``: one ``(origin, destination)`` pair per adjacent pair of values, each an
        ``(n_phenotypes, n_phenotypes, 2)`` array of (bottom, height) indexed by origin then
        destination phenotype.

    Notes
    -----
    Only a clone with mass at both values carries flow. With ``o`` and ``d`` its vectors, the
    origin end of the band from phenotype ``j`` to ``k`` grows by ``o[j] * d[k] / sum(d)`` and
    the destination end by ``o[j] / sum(o) * d[k]``, so each end carries the clone's own mass
    at its value. Origin ends stack up from the bottom of their node, ``k`` within ``j``.
    Destination ends stack in the same order but end at the top of their node: the mass of
    clones absent from the origin value sits beneath them.
    """
    p = n_phenotypes
    totals = [np.sum(list(vectors.values()), axis=0) if vectors else np.zeros(p)
              for vectors in mass]
    nodes = [np.column_stack([np.concatenate([[0.0], np.cumsum(total)[:-1]]), total])
             for total in totals]

    bands = []
    for i in range(len(mass) - 1):
        origin, destination = mass[i], mass[i + 1]
        flow_o = np.zeros((p, p))
        flow_d = np.zeros((p, p))
        for clone in origin:
            if clone not in destination:
                continue
            o, d = origin[clone], destination[clone]
            if o.sum() > 0 and d.sum() > 0:
                flow_o += np.outer(o, d / d.sum())
                flow_d += np.outer(o / o.sum(), d)

        run_o = nodes[i][:, 0].copy()
        run_d = np.cumsum(totals[i + 1]) - flow_d.sum(axis=0)
        ends_o = np.zeros((p, p, 2))
        ends_d = np.zeros((p, p, 2))
        for j in range(p):
            for k in range(p):
                ends_o[j, k] = (run_o[j], flow_o[j, k])
                run_o[j] += flow_o[j, k]
                ends_d[j, k] = (run_d[k], flow_d[j, k])
                run_d[k] += flow_d[j, k]
        bands.append((ends_o, ends_d))
    return {"nodes": nodes, "bands": bands}


def _render_sankey(ax, layout, *, phenotypes, colors, dx):
    """Draw ``layout`` on ``ax``, with covariate value ``i`` at x = ``i``.

    Each value's bars are drawn, then the bands leaving it, so the next value's bars sit on top
    of the bands arriving at them. A band with no flow is not drawn.
    """
    n = len(layout["nodes"])
    pad = _OVERLAP * max(float((nodes[:, 0] + nodes[:, 1]).max()) for nodes in layout["nodes"])
    for i in range(n):
        for j, phenotype in enumerate(phenotypes):
            bottom, height = layout["nodes"][i][j]
            SankeyNode(i, bottom, height, dx=dx, color=colors[phenotype]).plot(ax)
        if i == n - 1:
            break
        ends_o, ends_d = layout["bands"][i]
        for j, origin in enumerate(phenotypes):
            for k, destination in enumerate(phenotypes):
                if ends_o[j, k, 1] == 0 and ends_d[j, k, 1] == 0:
                    continue
                start = SankeyNode(i, *ends_o[j, k], dx=dx, color=colors[origin])
                end = SankeyNode(i + 1, *ends_d[j, k], dx=dx, color=colors[destination])
                start.plot_node_connection(end, ax, pad=pad)
