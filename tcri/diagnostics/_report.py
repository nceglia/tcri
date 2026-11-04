"""What a diag check prints, and what it returns.

Every check computes, then draws (``plot``), prints (``print``) and returns (``return_dict``)
what it computed. Nothing is cached, so drawing or printing again means calling again. A call
with all three off would compute and show nothing; it raises instead.
"""
from __future__ import annotations

from .. import _output


def check_outputs(plot: bool, print: bool, return_dict: bool) -> None:
    """Raise ``ValueError`` when ``plot``, ``print`` and ``return_dict`` are all off."""
    if not (plot or print or return_dict):
        raise ValueError(
            "plot=False, print=False and return_dict=False leave the check nothing to do: it "
            "would compute, then draw, print and return nothing. Turn one of them on."
        )


def _where(keys: str, returned: bool) -> str:
    """Where a cut table's rows are: the returned dict, or a call with ``return_dict=True``."""
    return (f"the returned {keys}" if returned
            else f"call again with return_dict=True for {keys}")


def _ppc_table(result, returned):
    calibration = result["calibration"]
    frame = (calibration.groupby(["covariate", "size_bin"], sort=False)
             .agg(n_clones=("n_clones", "first"), median_ks=("ks", "median"))
             .reset_index())
    return _output.frame_table(
        frame, title="joint_distribution_ppc · KS distance of p from uniform, median over draws",
        rest=_where("'calibration' (per draw), 'summary' and 'draws'", returned))


def _calibration_table(result, returned):
    summary = result["summary"].iloc[0]
    return _output.frame_table(
        result["bins"], title="phenotype_calibration",
        notes=[f"ECE {_output.cell(summary['ece'])} over {int(summary['n_cells']):,} cells"],
        rest=_where("'bins'", returned))


def _summary_table(name):
    def table(result, returned):
        return _output.frame_table(result["summary"], title=name,
                                   rest=_where("'summary'", returned))
    return table


_TABLES = {
    "joint_distribution_ppc": _ppc_table,
    "phenotype_calibration": _calibration_table,
    "reconstruction_ppc": _summary_table("reconstruction_ppc"),
    "permutation_null": _summary_table("permutation_null"),
}


def finish(name: str, result: dict, *, print: bool, return_dict: bool):
    """Print check ``name``'s table if asked, and return ``result`` or ``None``."""
    if print:
        _output.print_out(_TABLES[name](result, return_dict))
    return result if return_dict else None
