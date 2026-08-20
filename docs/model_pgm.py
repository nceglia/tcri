"""Plate diagram for the TCRi generative model, annotated with the knobs that tune it.

Run this to regenerate ``docs/images/model_pgm.png``::

    .venv-docs/bin/python docs/model_pgm.py

The output is **committed** rather than built by Sphinx: the docs build installs only
``docs/requirements.txt`` (mirroring Read the Docs), and a figure that must be generated at
build time is a dependency the published build would otherwise have to carry.

This replaces an earlier ``daft``-based version that had been dead for some time — it imported
``daft``, which now resolves to a distributed dataframe library rather than the PGM package, so
the file raised ``AttributeError`` on import and nothing noticed. Plain matplotlib has no such
ambiguity, and it gives the control needed to place the annotation column.

The point of the annotations: the plate diagram alone says what depends on what, but not which
*argument* changes which piece. Someone reading it wants to know that ``local_scale`` is the
concentration on the covariate-level Dirichlet, so raising it pins each covariate closer to its
clone's overall distribution. That is the mapping this figure makes explicit.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# tcri green, used for the argument names so they read as one family
ACCENT = "#1f9e16"
OBSERVED = "#c9c9c9"
INK = "#1a1a1a"

NODE_R = 0.42


def _plate(ax, x, y, w, h, label):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.3, edgecolor=INK, facecolor="none", zorder=1,
    ))
    ax.text(x + w - 0.18, y + 0.22, label, ha="right", va="bottom",
            fontsize=11, color=INK, zorder=2)


def _node(ax, x, y, label, observed=False):
    ax.add_patch(Circle((x, y), NODE_R, linewidth=1.4, edgecolor=INK,
                        facecolor=OBSERVED if observed else "white", zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=15, color=INK, zorder=4)


def _edge(ax, p0, p1):
    """Arrow between node centres, trimmed to the circle boundary at both ends."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    d = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / d, dy / d
    ax.add_patch(FancyArrowPatch(
        (x0 + ux * NODE_R, y0 + uy * NODE_R),
        (x1 - ux * (NODE_R + 0.06), y1 - uy * (NODE_R + 0.06)),
        arrowstyle="-|>", mutation_scale=15, linewidth=1.4,
        color=INK, shrinkA=0, shrinkB=0, zorder=2,
    ))


def _annotate(ax, y, title, body, arg=None):
    """One row of the right-hand column: the distribution, then the knob that tunes it.

    Rows are spaced EVENLY rather than aligned to their node's height. Aligning them was the
    obvious first idea and it collided: ``z_i`` and ``x_i`` sit close together in the graph,
    so their annotation blocks overlapped and the text became unreadable. The column reads
    top-to-bottom in the same order as the graph, which is what actually matters.
    """
    x = 7.5
    ax.text(x, y + 0.34, title, ha="left", va="center", fontsize=12.5, color=INK)
    ax.text(x, y - 0.06, body, ha="left", va="center", fontsize=10.5, color="#444")
    if arg:
        ax.text(x, y - 0.48, arg, ha="left", va="center", fontsize=10.5,
                color=ACCENT, family="DejaVu Sans Mono")


def build_pgm():
    fig, ax = plt.subplots(figsize=(13.0, 8.2))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 8.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Nested plates: a clone, its covariate levels, the cells at each level.
    #
    # The vertical gap between nested bottoms is 0.62, not the 0.40 that reads as "enough".
    # Each plate's label sits inside its own bottom-right corner, so at 0.40 the label for an
    # outer plate collided with the border of the plate nested inside it. The gap has to clear
    # the label's height, not merely separate the lines.
    _plate(ax, 0.45, 0.35, 6.30, 7.95, "Clonotype $c$")
    _plate(ax, 0.85, 0.97, 5.50, 6.55, "Covariate $ct$")
    _plate(ax, 1.25, 1.59, 4.70, 4.35, "Cell $i$")

    p_c = (4.75, 7.35)
    p_ct = (4.75, 5.95)
    z_f = (4.75, 4.35)
    z_i = (2.55, 4.35)
    x_i = (2.55, 2.55)

    _node(ax, *p_c, r"$p_c$")
    _node(ax, *p_ct, r"$p_{ct}$")
    _node(ax, *z_f, r"$z_i^{\phi}$")
    _node(ax, *z_i, r"$z_i$")
    _node(ax, *x_i, r"$x_i$", observed=True)

    _edge(ax, p_c, p_ct)
    _edge(ax, p_ct, z_f)
    _edge(ax, z_i, z_f)
    _edge(ax, z_i, x_i)

    # the annotation column, evenly spaced top-to-bottom in graph order
    rows = [
        (r"$p_c \sim \mathrm{MixtureDirichlet}(\alpha\,\psi_b)$",
         "the clone's phenotype distribution overall",
         "global_scale = α   (default 5.0)"),
        (r"$p_{ct} \sim \mathrm{Dirichlet}(\beta\, p_c)$",
         "its distribution at ONE covariate level — raise β to pin it to the clone",
         "local_scale = β   (default 3.0)"),
        (r"$z_i^{\phi} \sim \mathrm{Cat}(\mathrm{softmax}\,\ell_i)$",
         r"$\ell_i = \pi\, f_{\mathrm{cls}}(z_i) + (1-\pi)\log p_{ct}$   —   π=1 pure classifier, π=0 pure prior",
         "gate_prob = π   (default 0.5)"),
        (r"$z_i \sim \mathrm{VampPrior}$",
         "latent expression state, encoded from counts",
         "n_latent, n_hidden, n_layers"),
        (r"$x_i \sim \mathrm{ZINB}(\mathrm{dec}(z_i))$",
         "observed counts — shaded, the only observed node",
         "reconstruction_loss_scale"),
    ]
    top, step = 7.55, 1.34
    for i, (title, body, arg) in enumerate(rows):
        _annotate(ax, top - i * step, title, body, arg)

    # the one relationship the diagram cannot show, because it is a training term rather
    # than an edge in the generative graph
    ax.plot([7.5, 13.9], [1.30, 1.30], color="#cccccc", linewidth=1.0, zorder=1)
    ax.text(7.5, 0.92,
            "The classifier $f_{\\mathrm{cls}}$ is trained ONLY by an alignment penalty pulling\n"
            "softmax$(\\ell_i)$ toward $p_{ct}$. Without it the logits never enter the\n"
            "objective at all and phenotype recovery sits at chance.",
            ha="left", va="center", fontsize=10, color="#444", linespacing=1.5)
    ax.text(7.5, 0.20, "phenotype_kl_weight = γ   (default 1.0)",
            ha="left", va="center", fontsize=10.5, color=ACCENT,
            family="DejaVu Sans Mono")

    fig.tight_layout(pad=0.4)
    return fig


if __name__ == "__main__":
    import pathlib

    out = pathlib.Path(__file__).parent / "images" / "model_pgm.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = build_pgm()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
