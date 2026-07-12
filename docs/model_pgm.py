"""Model plate-diagram (PGM) generators — moved OUT of the tcri package (PR9, §9).
Requires `daft` (a docs-only extra, NOT a tcri runtime dependency)."""
import daft
import matplotlib.pyplot as plt
import numpy as np


def build_nested_tcri_pgm():
    """
    A fully explicit TCRI PGM matching the implementation in _model.py
    with improved layout to minimize edge crossings
    """
    # Define colors
    red, yellow, green, gray, blue = "#cd442a", "#f0bd00", "#7e9437", "#eee", "#009de1"

    # Create a PGM canvas
    pgm = daft.PGM(
        shape=[8, 8],  # width x height
        origin=[0, 0],
        grid_unit=1.6,
        node_unit=1.5
    )

    # ------------------------------------------------------------------
    # 1) Global hyperparameters for Dirichlet priors
    # ------------------------------------------------------------------
    pgm.add_node(
        "global_scale",
        r"$\mathrm{global\_scale}$",
        5.1,  # x
        6.6,  # y
        fixed=True,
        plot_params={"fc": "#DDD"}
    )

    # ------------------------------------------------------------------
    # 2) Plate: batch (b) - outermost plate
    # ------------------------------------------------------------------

    # Batch-level variables - aligned vertically
    # ------------------------------------------------------------------
    # 3) Plate: clonotypes (c) - middle plate
    # ------------------------------------------------------------------
    pgm.add_plate(
        [1.0, 1.0, 6.0, 6.3],  # [x, y, width, height]
        label=r"clonotypes $(c)$",
        shift=-0.1
    )

    # p_c (Dirichlet)
    pgm.add_node(
        "p_c",
        r"$p_c$",
        3.8,  # x
        6.6,  # y
        observed=False,
        plot_params={"fc": blue}
    )

    # Edge: global_scale -> p_c
    pgm.add_edge("global_scale", "p_c")

    # ------------------------------------------------------------------
    # 4) Plate: clone-covariate (ct) - inner plate
    # ------------------------------------------------------------------
    pgm.add_plate(
        [1.5, 1.5, 5.0, 4.5],  # [x, y, width, height]
        label=r"clone-covariate $(ct)$",
        shift=-0.1
    )

    # local_scale moved inside clone-covariate plate
    pgm.add_node(
        "local_scale",
        r"$\mathrm{local\_scale}$",
        5.2,  # x
        5.3,  # y
        fixed=True,
        plot_params={"fc": "#DDD"}
    )

    # p_ct (Dirichlet)
    pgm.add_node(
        "p_ct",
        r"$p_{ct}$",
        3.8,  # x
        5.3,  # y
        observed=False,
        plot_params={"fc": yellow}
    )

    # Edges: p_c -> p_ct, local_scale -> p_ct
    pgm.add_edge("p_c", "p_ct")
    pgm.add_edge("local_scale", "p_ct")

    # ------------------------------------------------------------------
    # 5) Plate: data (i) - innermost plate
    # ------------------------------------------------------------------
    pgm.add_plate(
        [1.9, 2., 4.0, 2.7],  # [x, y, width, height]
        label=r"data $(i)$",
        shift=-0.1
    )

    # Grid layout for data-level variables - aligned vertically
    # Column 1: Observed variables
    pgm.add_node(
        "obs",
        r"$X_{i}$",
        5,  # x
        4.0,  # y
        observed=True,
        plot_params={"fc": gray}
    )

    pgm.add_node(
        "obs_label",
        r"$Pheno_{i}$",
        2.5,  # x
        4.0,  # y
        observed=True,
        plot_params={"fc": gray}
    )

    # Column 2: Latent variables
    pgm.add_node(
        "latent",
        r"$z_i$",
        3.8,  # x
        2.8,  # y
        observed=False,
        plot_params={"fc": green}
    )

    pgm.add_node(
        "z_i_phen",
        r"$z_{i,\mathrm{phen}}$",
        3.8,  # x
        4.0,  # y
        observed=False,
        plot_params={"fc": red}
    )

    # Column 3: Decoder inputs
    pgm.add_node(
        "px_r",
        r"$ZINB(X_{i})$",
        5,  # x
        2.8,  # y
        observed=False,
        plot_params={"fc": "#DDD"}
    )

    # Edges - now mostly vertical and horizontal
    # Data-level edges
    pgm.add_edge("p_ct", "z_i_phen")
    pgm.add_edge("latent", "z_i_phen")
    pgm.add_edge("z_i_phen", "obs_label")

    # Direct connections to z_i (previously through decoder)
    pgm.add_edge("latent", "obs")
    pgm.add_edge("px_r", "obs")

    # ------------------------------------------------------------------
    # Text / Title
    # ------------------------------------------------------------------
    pgm.add_text(3.1,7.5, "TCRi Model", fontsize=14)

    return pgm


def draw_tcri_pgm_nested():
    pgm = build_nested_tcri_pgm()
    pgm.render()
    pgm.figure.savefig("tcri_model_fully_explicit.pdf", dpi=300)
    plt.show()
