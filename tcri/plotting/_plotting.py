import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram, linkage
import scanpy as sc
from gseapy import dotplot
import numpy as np, pandas as pd, torch, umap
from tqdm.auto import tqdm
from scvi import REGISTRY_KEYS

import collections
import operator
import itertools

from ..utils._utils import Phenotypes, CellRepertoire, Tcell, plot_pheno_sankey, plot_pheno_ternary_change_plots, draw_clone_bars, probabilities, set_ternary_corner_label, ternary_plot_projection
from ..preprocessing._preprocessing import clone_size, joint_distribution, joint_distribution_posterior
from ..metrics._metrics import clonotypic_entropy as centropy
from ..metrics._metrics import phenotypic_entropy as pentropy
from ..metrics._metrics import clonality as clonality_tl
from ..metrics._metrics import flux as flux_tl
from ..metrics._metrics import mutual_information as mutual_information_tl
from ..metrics._metrics import clone_fraction as clone_fraction_tl


import warnings
warnings.filterwarnings('ignore')

sc._settings.settings._vector_friendly=True

# ╭─ colour / pretty-print helpers ─────────────────────────────────────────╮
RESET  = "\x1b[0m";  BOLD  = "\x1b[1m";  DIM  = "\x1b[2m"
GRN = "\x1b[32m";  CYN = "\x1b[36m";  MAG = "\x1b[35m";  YLW = "\x1b[33m"; RED = "\x1b[31m"

def _ok(msg:str, quiet=False):    # success mark
    if not quiet: print(f"{GRN}✅ {msg}{RESET}")

def _info(key:str, txt:str, quiet=False):       # key-value info line
    if not quiet: print(f"   {CYN}🎯 {key:<22}{DIM}{txt}{RESET}")

def _warn(msg:str, quiet=False):   # warning line
    if not quiet: print(f"{YLW}⚠️  {msg}{RESET}")

def _fin(quiet=False):             # final flourish
    if not quiet: print(f"{MAG}✨  Done!{RESET}")
# ╰──────────────────────────────────────────────────────────────────────────╯

red = "#cd442a"
yellow = "#f0bd00"
green = "#7e9437"

tcri_colors = [
    red,
    yellow,
    green,
    "#004d47",  # Darker Teal
    "#AE81FF",  # Purple
    "#FD971F",  # Orange
    "#E6DB74",  # Yellow
    "#A6E22E",  # Green
    "#F28E7F",  # Salmon
    "#75715E",  # Brown
    "#F92659",  # Pink
    "#D65F0E",  # Abricos
    "#66D9EF",  # Blue
    "#F92672",  # Red
    "#1E1E1E",   # Black
    "#272822",  # Background
    "#D291BC",  # Soft Pink
    "#3A506B",  # Dark Slate Blue
    "#5D8A5E",  # Sage Green
    "#A6A1E2",  # Dull Lavender
    "#E97451",  # Burnt Sienna
    "#6C8D67",  # Muted Lime Green
    "#832232",  # Dim Maroon
    "#669999",  # Desaturated Cyan
    "#C08497",  # Dusty Rose
    "#587B7F",  # Ocean Blue
    "#9A8C98",  # Muted Purple
    "#F3B61F",  # Goldenrod
    "#FFD8B1",  # Light Peach
    "#88AB75",  # Moss Green
    "#C38D94",  # Muted Rose
    "#6D6A75",  # Purple Gray
]

sns.set_palette(sns.color_palette(tcri_colors))

def compare_phenotypes(adata, variable1, variable2):
    df = adata.obs[[variable1,variable2]]
    df=pd.crosstab(df[variable1],df[variable2],normalize='index')
    return sns.heatmap(df)

def compare_joint_distribution(adata, temperature=1):
    # -----------------------------
    # 1. Get Model-Inferred Distributions
    # -----------------------------
    # Create a dictionary mapping each tissue (treatment group) to its clone phenotype DataFrame
    covariate_col = adata.uns["tcri_metadata"]["covariate_col"]
    model_dists = dict()
    for tissue in set(adata.obs[covariate_col]):
        # Use your function to get the inferred p_ct distribution (with temperature scaling)
        df_tissue = joint_distribution(adata,tissue, temperature=temperature)
        df_tissue[covariate_col] = tissue
        # Set clonotype_id as index for easier merging/comparison later
        #df_tissue.set_index("clonotype_id", inplace=True)
        model_dists[tissue] = df_tissue
    
    # Concatenate the inferred distributions from all tissues into a single DataFrame
    df_model = pd.concat(model_dists.values(), axis=0)
    
    empirical_dists = dict()
    clonotype_col = model.adata_manager.registry["clonotype_col"]
    phenotype_col = model.adata_manager.registry["phenotype_col"]
    
    for tissue in set(adata.obs[covariate_col]):
        # Filter for cells in the given tissue
        adata_tissue = adata[adata.obs[covariate_col] == tissue].copy()
        # Group by clonotype and compute normalized counts of each phenotype
        emp = (
            adata_tissue.obs.groupby(clonotype_col)[phenotype_col]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )
        # Ensure that the DataFrame uses the actual phenotype category names as columns.
        # If some phenotype categories are missing in a tissue, add them with 0.
        phenotype_categories = list(adata.obs[phenotype_col].astype("category").cat.categories)
        for ph in phenotype_categories:
            if ph not in emp.columns:
                emp[ph] = 0.0
        # Reorder columns
        emp = emp[phenotype_categories]
        emp[covariate_col] = tissue
        # Use clonotype ID (from the index) as a column if needed
        emp.index.name = "clonotype_id"
        empirical_dists[tissue] = emp
    
    # Concatenate the empirical distributions from all tissues into one DataFrame.
    df_empirical = pd.concat(empirical_dists.values(), axis=0)
    
    # -----------------------------
    # 3. Compare Distributions: Plotting Side-by-Side
    # -----------------------------
    # We'll loop over the unique tissues and for each, plot the inferred (model) and empirical distributions.
    unique_tissues = df_model[covariate_col].unique()
    n_tissues = len(unique_tissues)
    
    fig, axes = plt.subplots(n_tissues, 4, figsize=(20, 4 * n_tissues),
                             gridspec_kw={'width_ratios': [1, 4, 1, 4]})
    
    # In case there's only one tissue, ensure axes is 2D.
    if n_tissues == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for i, tissue in enumerate(unique_tissues):
        # Select the rows for the current tissue for both distributions
        model_data = df_model[df_model[covariate_col] == tissue]
        empirical_data = df_empirical[df_empirical[covariate_col] == tissue]
        
        # Determine the phenotype columns (assumed to be common to both)
        phenotype_cols = [col for col in model_data.columns if col not in ["clonotype_index", covariate_col]]
        
        # --- Model-Inferred Distribution ---
        # Compute hierarchical clustering for the model distribution.
        Z_model = linkage(model_data[phenotype_cols], method='average')
        dendro_model = dendrogram(Z_model, orientation='left', ax=axes[i, 0], no_labels=True)
        # Order the data
        ordered_model = model_data.iloc[dendro_model['leaves']]
        sns.heatmap(ordered_model[phenotype_cols], ax=axes[i, 1], cmap="viridis", cbar=True)
        axes[i, 1].set_title(f"Model-Inferred: {tissue}")
        axes[i, 0].set_title("Dendrogram")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 1].set_yticklabels([])
        
        # --- Empirical Distribution ---
        # Compute hierarchical clustering for the empirical distribution.
        Z_emp = linkage(empirical_data[phenotype_cols], method='average')
        dendro_emp = dendrogram(Z_emp, orientation='left', ax=axes[i, 2], no_labels=True)
        ordered_emp = empirical_data.iloc[dendro_emp['leaves']]
        sns.heatmap(ordered_emp[phenotype_cols], ax=axes[i, 3], cmap="viridis", cbar=True)
        axes[i, 3].set_title(f"Empirical: {tissue}")
        axes[i, 2].set_title("Dendrogram")
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        axes[i, 3].set_yticklabels([])
    
    plt.tight_layout()
    plt.show()

def phenotypic_flux(adata, splitby, order, clones=None, normalize=False, n_samples=50, phenotype_colors=None, save=None, figsize=(6,3), show_legend=True, temperature=1):
    nt = False
    phenotypes = Phenotypes(adata.uns["tcri_phenotype_categories"])
    cell_probabilities = collections.defaultdict(dict)
    for s in order:
        jds = []
        print("Sampling joint distributions for {}".format(s))
        for _ in tqdm.tqdm(range(n_samples)):
            jd = joint_distribution_posterior(
                    adata,
                    covariate_label     = s,
                    temperature         = temperature,
                    weighted            = False,
                    combine_with_logits = True,
                    silent              = True)
            jds.append(jd)
        jd = (
            pd.concat(jds)             # stack them one on top of another
            .groupby(level=0)        # regroup by the original row index
            .mean()                  # take the mean across stacks
        )
        for x in jd.T:
            dist = jd.T[x] / jd.T[x].sum()
            cell_probabilities[s][x] = dist.to_dict()
    repertoires = dict()
    times = list(range(len(order)))
    if nt:
        chains_to_use = "ntseq"
    else:
        chains_to_use = "aaseq"
    
    for s in order:
        print("Creating cell repertoires for {}".format(s))
        repertoires[s] = CellRepertoire(clones_and_phenos = {}, 
                                        phenotypes = phenotypes, 
                                        use_genes = False, 
                                        use_chain = False,
                                        seq_type = chains_to_use,
                                        chains_to_use = ['TRB'],
                                        name = s)
    print("Adding cells to repertoire")
    for bc, condition, seq, phenotype in tqdm.tqdm(list(zip(adata.obs.index,
                                         adata.obs[splitby],
                                         adata.obs[adata.uns["tcri_clone_key"]],
                                         adata.obs[adata.uns["tcri_phenotype_key"]]))):
        if str(seq) != "nan" and condition in repertoires and seq in cell_probabilities[condition]:
            phenotypes_and_counts = cell_probabilities[condition][seq]
            if nt:
                t = Tcell(phenotypes = phenotypes, phenotypes_and_counts = phenotypes_and_counts, 
                                                          TRB = dict(ntseq = seq), 
                                                          use_genes = False)
            else:
                t = Tcell(phenotypes = phenotypes, phenotypes_and_counts = phenotypes_and_counts, 
                                                          TRB = dict(aaseq = seq), 
                                                          use_genes = False)    
            repertoires[condition].cell_list.append(t)
    for condition, rep in repertoires.items():
        rep._set_consistency()
    if phenotype_colors==None:
        phenotype_colors = dict(zip(set(adata.obs[adata.uns["tcri_phenotype_key"]]), tcri_colors))
    print("Generating Sankey plot...")
    fig, ax = plot_pheno_sankey(phenotypes = phenotypes, 
                                cell_repertoires = [repertoires[condition] for condition in order], 
                                clones = clones,
                                times = times,
                                xlim = [min(times), max(times)],
                                time_rescale = 1,
                                normalize=normalize,
                                xlabel = splitby,
                                return_axes = True, 
                                show_legend = show_legend,
                                figsize = figsize,
                                phenotype_colors=phenotype_colors)
    ax.set_xticks(times)
    ax.set_xticklabels(order)
    if save != None:
        fig.savefig(save)

def freq_to_size_scaling(freq):
    return 10*(freq**(1/2))

def freq_to_size_legend(ax, min_freq = 1e-6, max_freq = 1, loc = [0.85, 0.92], size = 0.25, x_offset = 0.1):
    freq_to_y_pos = lambda f: loc[1]-np.log(max_freq/f)*size/np.log(max_freq/min_freq)
    legend_pnts = np.exp(np.arange(np.log(min_freq), np.log(max_freq), 0.001))
    ax.scatter(loc[0]*np.ones(len(legend_pnts)), [freq_to_y_pos(y) for y in legend_pnts], s = [freq_to_size_scaling(x)**2 for x in legend_pnts], marker = '_', c = 'k')
    ax.plot(loc[0]*np.ones(2)+ x_offset, [freq_to_y_pos(min_freq), freq_to_y_pos(max_freq)], 'k', lw = 1)
    major_ticks = [10**x for x in  range(int(np.ceil(np.log10(min_freq))), int(np.floor(np.log10(max_freq))+1))]
    minor_ticks = sum([[x*major_tick for x in range(1, 10)] for major_tick in major_ticks[:-1]], [])
    for minor_tick in minor_ticks:
        ax.plot([loc[0]*np.ones(2)+ x_offset, loc[0]*np.ones(2)+ 1.2*x_offset], [freq_to_y_pos(minor_tick), freq_to_y_pos(minor_tick)], 'k', lw = 0.5)
    for major_tick in major_ticks:
        ax.plot([loc[0]*np.ones(2)+ x_offset, loc[0]*np.ones(2)+ 1.5*x_offset], [freq_to_y_pos(major_tick), freq_to_y_pos(major_tick)], 'k', lw = 0.75)
        if major_tick < 0.1:
            ax.text(loc[0]+ 1.7*x_offset, freq_to_y_pos(major_tick), '%.0e'%(major_tick), ha = 'left', va = 'center', rotation = 0, fontsize = 8)
        else:
            ax.text(loc[0]+ 1.7*x_offset, freq_to_y_pos(major_tick), str(major_tick), ha = 'left', va = 'center', rotation = 0, fontsize = 8)

def setup_ternary_plot():
    """Set up a ternary plot with proper axes and labels."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_axis_off()
    return fig, ax

def probability_ternary(adata, phenotype_names, covariate, splitby, conditions,n_samples=1,
                        temperature=1,top_n=None, scale_function=None,color="k",save=None):
    if scale_function == None:
        scale_function = freq_to_size_scaling
    phenotypes = Phenotypes(adata.uns["tcri_phenotype_categories"])

    cell_probabilities = collections.defaultdict(dict)
    for s in set(adata.obs[splitby]):
        sdata = adata[adata.obs[splitby] == s].copy()
        clones = list(set(sdata.obs[adata.uns["tcri_clone_key"]]))
        rtab = collections.defaultdict(lambda : collections.defaultdict(list))
        for _ in range(n_samples):
            jd = joint_distribution_posterior(
                    adata,
                    covariate_label     = covariate,
                    temperature         = temperature,
                    clones              = clones,
                    weighted            = False,
                    combine_with_logits = True,
                    silent              = True)
            for x in jd.T:
                for p,v in jd.T[x].to_dict().items():
                    rtab[x][p].append(v)
        for x,v in rtab.items():
            phs = []
            pbs = []
            for p, vals in v.items():
                pbs.append(np.sum(vals))
                phs.append(p)
            cell_probabilities[s][x] = dict(zip(phs,pbs))
    
    repertoires = dict()
    chains_to_use = "aaseq"
    
    for s in set(adata.obs[splitby]):
        repertoires[s] = CellRepertoire(clones_and_phenos = {}, 
                                        phenotypes = phenotypes, 
                                        use_genes = False, 
                                        use_chain = False,
                                        seq_type = chains_to_use,
                                        chains_to_use = ['TRB'],
                                        name = s)
    
    for bc, condition, seq, phenotype in zip(adata.obs.index,
                                             adata.obs[splitby],
                                             adata.obs[adata.uns["tcri_clone_key"]],
                                             adata.obs[adata.uns["tcri_phenotype_key"]]):
    
        if str(seq) != "nan" and condition in repertoires and seq in cell_probabilities[condition]:
            phenotypes_and_counts = cell_probabilities[condition][seq]
            t = Tcell(phenotypes = phenotypes, phenotypes_and_counts = phenotypes_and_counts, 
                                                      TRB = dict(aaseq = seq), 
                                                      use_genes = False)
            repertoires[condition].cell_list.append(t)
    
    for condition, rep in repertoires.items():
        rep._set_consistency()

    phenotype_names_dict = {p: p for p in phenotype_names}
    if type(conditions) == list:
        if len(conditions) == 1:
            start_clones_and_phenos = repertoires[conditions[0]]
            end_clones_and_phenos = None
        elif len(conditions) == 2:
            start_clones_and_phenos = repertoires[conditions[0]]
            end_clones_and_phenos = repertoires[conditions[1]]
        else:
            raise ValueError("Only two conditions supported.")
    else:
        print(repertoires.keys())
        start_clones_and_phenos = repertoires[conditions]
        end_clones_and_phenos = None
    s_dict = {c: scale_function(sum(start_clones_and_phenos[c].values())/start_clones_and_phenos.norm) for c in start_clones_and_phenos.clones}
    if top_n == None:
        top_n = len(set(adata.obs[adata.uns["tcri_clone_key"]]))
    c_clones = sorted(start_clones_and_phenos.clones, key = s_dict.get, reverse = True)[:top_n]
    fig, ax = plot_pheno_ternary_change_plots(start_clones_and_phenotypes = start_clones_and_phenos,
                                            end_clones_and_phenotypes = end_clones_and_phenos,
                                            phenotypes = phenotype_names, 
                                            phenotype_names = phenotype_names_dict,
                                            clones = c_clones,
                                            line_type = 'arrows', 
                                            kwargs_for_plots={"color":color,'alpha':0.8},
                                            s_dict = s_dict,
                                            return_axes  = True)
    freq_to_size_legend(ax)
    if save != None:
        fig.savefig(save)

def probability_distribution(adata, phenotype_order=None, color="#000000", rotation=90, splitby=None, order=None, figsize=(7,5), save=None):
    columns = []
    if splitby != None:
        ncols = len(set(adata.obs[splitby]))
    else:
        ncols = 1
    fig, ax = plt.subplots(1,ncols,figsize=figsize)

    if order == None:
        order = list(sorted(adata.obs[splitby]))
    for i, o in enumerate(order):
        zdata = adata[adata.obs[splitby] == o]
        pdist = probability_distribution(zdata)
        sns.barplot(data=pdist,ax=ax[i],order=phenotype_order,color=color)
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=rotation)
        ax[i].set_title(o)

    fig.tight_layout()
    if save != None:
        fig.savefig(save)


def top_clone_umap(adata, reduction="umap", top_n=10, fg_alpha=0.9, fg_size=25, bg_size=0.1, bg_alpha=0.6, figsize=(12,5), return_df=False,save=None):
    df = adata.obs
    seq_column = adata.uns["tcri_clone_key"]
    plt.figure(figsize = figsize)
    clonotype_counts = collections.defaultdict(int)
    for clonotype in df[seq_column]:
        clonotype_counts[clonotype] += 1
    top_clonotypes = sorted(clonotype_counts.items(), key=operator.itemgetter(1),reverse=True)
    top_clonotypes = [x[0] for x in top_clonotypes[:top_n]]
    ax1 = plt.subplot(1,1,1)
    x = [x[0] for x in adata.obsm["X_{}".format(reduction)]]
    y = [x[1] for x in adata.obsm["X_{}".format(reduction)]]
    sns.scatterplot(x=x,y=y, color="#75715E",  alpha=bg_alpha, ax=ax1, s=bg_size, linewidth=0.0)
    xonly = []
    yonly = []
    clonotype_labels = []
    size = []
    for clonotype,x1,y1 in zip(df[seq_column],x,y):
        clonotype = str(clonotype)
        if clonotype not in top_clonotypes or clonotype == "None" or clonotype == "nan":
            continue
        else:
            xonly.append(x1)
            yonly.append(y1)
            size.append(1/clonotype_counts[clonotype])
            clonotype_labels.append(str(clonotype) + " {}".format(clonotype_counts[clonotype]))
    dftop = pd.DataFrame.from_dict({"TCR Sequence":clonotype_labels,"Cells":size, "UMAP1":xonly,"UMAP2":yonly})
    colors = tcri_colors + tcri_colors + tcri_colors
    order = []
    for c in set(clonotype_labels):
        if c != "_Other":
            order.append(c)
    colors = colors[:len(set(clonotype_labels))]
    sns.scatterplot(data=dftop, x="UMAP1", y="UMAP2", hue="TCR Sequence", hue_order=order, ax=ax1, alpha=fg_alpha,s=fg_size, linewidth=0.0,palette=colors)
    ax1.set_xlabel('UMAP-1')
    ax1.set_ylabel('UMAP-2')
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    ax1.xaxis.set_ticks([])
    ax1.yaxis.set_ticks([])
    ax1.set_title("Top 10 TCR Clone by Size")
    h,l = ax1.get_legend_handles_labels()
    ax1.legend(h[:top_n-1], l[:top_n-1], borderaxespad=2.,fontsize='9',bbox_to_anchor=(0, 1), loc='best')
    plt.tight_layout()
    if return_df:
        return dftop
    elif save != None:
        plt.savefig(save)


def tcri_boxplot(adata, function, groupby=None,ylabel="", splitby=None,figsize=(8,4),s=20,order=None, palette=None):
    if palette == None:
        palette = tcri_colors
    if groupby == None and splitby == None:
        data = function(adata)
        df = pd.DataFrame(list(data.items()), columns=['Phenotype', 'Clonotypic Entropy'])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        fig,ax=plt.subplots(1,1,figsize=figsize)
        sns.stripplot(data=df,x="Phenotype",y=ylabel,s=s,ax=ax, palette=palette)
        ax.set_ylim(0,max(df[ylabel] + 0.1))
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        fig.tight_layout()
    elif groupby != None and splitby == None:
        groups = adata.obs[groupby].unique()
        dfs = []
        for group in groups:
            data = function(adata[adata.obs[groupby]==group])
            df = pd.DataFrame(list(data.items()), columns=['Phenotype',ylabel])
            df[groupby] = group
            dfs.append(df)
        df = pd.concat(dfs)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        if order == None:
            order = df.groupby(["Phenotype"]).median(ylabel).sort_values(ylabel).index.tolist()
        fig,ax=plt.subplots(1,1,figsize=figsize)
        sns.stripplot(data=df,x="Phenotype",y=ylabel,s=s,hue=groupby,ax=ax,order=order, palette=palette)
        sns.boxplot(data=df,x="Phenotype",y=ylabel,ax=ax, color="#999999",order=order)
        ax.set_ylim(0,max(df[ylabel] + 0.1))
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        fig.tight_layout()
    elif groupby != None and splitby != None:
        groups = adata.obs[groupby].unique()
        dfs = []
        for group in groups:
            sub = adata[adata.obs[groupby]==group]
            splits = sub.obs[splitby].unique()
            for split in splits:
                data = function(sub[sub.obs[splitby]==split])
                df = pd.DataFrame(list(data.items()), columns=['Phenotype', ylabel])
                df[groupby] = group
                df[splitby] = split
                dfs.append(df)
        df = pd.concat(dfs)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        fig,ax=plt.subplots(1,1,figsize=figsize)
        if order == None:
            order = df.groupby(["Phenotype"]).median(ylabel).sort_values(ylabel).index.tolist()
        sns.boxplot(data=df,x="Phenotype",y=ylabel,ax=ax, hue=splitby,order=order,palette=palette)
        ax.set_ylim(0,max(df[ylabel] + 0.1))
        ax.set_title(ylabel)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        ax.set_ylabel(ylabel)
        fig.tight_layout()
    else:
        raise ValueError("'groupby' must be set to use 'splitby'.")
    return ax

def clonality(adata, groupby = None, splitby=None, s=10, order=None, figsize=(12,5), palette=None):
    return tcri_boxplot(adata,clonality_tl, ylabel="Clonality", groupby=groupby, splitby=splitby, s=s, figsize=figsize, order=order, palette=palette)

def clonotypic_entropy_by_phenotype(
    adata,
    *,
    temperature       = 1.0,
    n_samples         = 0,
    weighted          = False,
    normalised        = True,
    posterior         = True,
    combine_with_logits = True,
    bayesian          = True,
    bayes_samples     = 1_000,
    palette           = None,
    group_colors      = None,        # {covariate: colour}
    hue_order         = None,
    legend_fontsize   = 6,
    bbox_to_anchor    = (1.15, 1.),
    figsize           = (6, 3),
    rotation          = 90,
    save              = None,
    return_df         = False,
    progress          = True,
):
    """Box-and-dot plot of clonotypic entropy per phenotype / covariate."""

    # ---- meta columns --------------------------------------------- #
    meta        = adata.uns["tcri_metadata"]
    cov_col     = meta["covariate_col"]
    clone_col   = meta["clone_col"]
    phen_col    = meta["phenotype_col"]
    batch_col   = meta["batch_col"]

    covariates  = adata.obs[cov_col].astype("category").cat.categories.tolist()
    phenotypes  = adata.obs[phen_col].astype("category").cat.categories.tolist()
    batches     = adata.obs[batch_col].astype("category").cat.categories.tolist()

    if hue_order is None:
        hue_order = covariates
    # # ---- colours -------------------------------------------------- #
    # if group_colors is not None:
    #     palette = tcri.pl.tcri_colors#[group_colors[c] for c in hue_order]
    # elif palette is None:
    #     palette = sns.color_palette("Set2", len(hue_order))
    palette=tcri_colors
    cov2col = dict(zip(hue_order, palette))

    # ---- compute entropy values ----------------------------------- #
    records = []
    iterator = itertools.product(batches, hue_order, phenotypes)
    if progress:
        iterator = tqdm.tqdm(list(iterator), desc="clonotypic entropy")

    for patient, cov_value, phen in iterator:
        mask = (
            (adata.obs[batch_col] == patient) &
            (adata.obs[cov_col]   == cov_value) &
            (adata.obs[phen_col]  == phen)
        )
        sub = adata[mask]
        if sub.n_obs == 0:
            continue
        clones = list(set(sub.obs["trb_unique"]))
        ent = centropy(
            adata,
            covariate          = cov_value,
            phenotype          = phen,
            temperature        = temperature,
            n_samples          = n_samples,
            weighted           = weighted,
            clones             = clones,
            normalised         = normalised,
            posterior          = posterior,
            combine_with_logits= combine_with_logits,
            verbose            = False,
        )
        ent_scalar = float(ent)  # always scalar now
        records.append(
            {cov_col: cov_value, batch_col: patient,
             phen_col: phen, "entropy": ent_scalar}
        )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        _warn("no data – nothing to plot")
        return None

    # ---- x-position jitter per hue -------------------------------- #
    x_levels = df[phen_col].unique().tolist()
    n_hue    = len(hue_order)
    w_tot    = .8
    step     = w_tot / n_hue
    offsets  = np.linspace(-w_tot/2 + step/2, w_tot/2 - step/2, n_hue)
    cov2off  = dict(zip(hue_order, offsets))
    df["_x"] = df[phen_col].map(lambda x: x_levels.index(x)) + df[cov_col].map(cov2off)

    # ---- plotting ------------------------------------------------- #
    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data      = df,
        x         = phen_col,
        y         = "entropy",
        hue       = cov_col,
        palette   = palette,
        hue_order = hue_order,
        width     = w_tot,
        fliersize = 0,
        ax        = ax,
        zorder    = 1,
    )
    ax.legend_.remove()

    # ---- Bayesian / MW stats inside each phenotype ---------------- #
    y_max = df["entropy"].max();  y_min = df["entropy"].min()
    h_bar = (y_max - y_min) * .05
    v_gap = h_bar * 1.25

    for i, ph in enumerate(x_levels):
        for j, (g1, g2) in enumerate(itertools.combinations(hue_order, 2)):
            d1 = df[(df[phen_col]==ph)&(df[cov_col]==g1)]["entropy"].values
            d2 = df[(df[phen_col]==ph)&(df[cov_col]==g2)]["entropy"].values
            if len(d1)==0 or len(d2)==0:
                continue

            if bayesian:
                samp1 = np.random.choice(d1, (bayes_samples, len(d1)), replace=True).mean(1)
                samp2 = np.random.choice(d2, (bayes_samples, len(d2)), replace=True).mean(1)
                delta = samp2 - samp1
                lbl   = f"Δ={delta.mean():.2g}"
            else:
                from scipy.stats import mannwhitneyu
                _, p = mannwhitneyu(d1,d2,alternative="two-sided")
                lbl  = "ns" if p>=.05 else ("*" if p<.05 else "**" if p<.01 else "***" if p<.001 else "****")

            x1 = i+cov2off[g1];  x2 = i+cov2off[g2];  y = y_max + j*v_gap
            ax.plot([x1,x1,x2,x2], [y,y+h_bar,y+h_bar,y], color="k", lw=1.4)
            ax.text((x1+x2)/2, y+h_bar, lbl, ha="center", va="bottom", fontsize=legend_fontsize)

    # patient dots
    pt_pal = sns.color_palette("tab20b", len(batches))
    for pt,c in zip(batches, pt_pal):
        sub = df[df[batch_col]==pt]
        ax.scatter(sub["_x"], sub["entropy"], color=c, s=80, alpha=.85, zorder=3, label=pt)
    ax.legend(title="Patient", fontsize=legend_fontsize, loc="upper right", bbox_to_anchor=bbox_to_anchor)

    ax.set_xticklabels(x_levels, rotation=rotation)
    ax.set_xlabel("Phenotype"); ax.set_ylabel("Clonotypic entropy")
    ax.set_title("Clonotypic entropy per phenotype / covariate")
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150);  _ok(f"figure saved → {save}")
    else:
        plt.show()

    if return_df:
        return df

def plot_phenotype_probabilities(adata, phenotype_prob_slot="X_tcri_phenotypes", add_outline=False, save=None,ncols=2,cmap="magma"):
    phenotypes = adata.uns["tcri_phenotype_categories"]
    prob_labels = []
    adata = adata.copy()
    for y,x in zip(phenotypes, adata.obsm[phenotype_prob_slot].T):
        adata.obs['{}_probability'.format(y)] = x
        prob_labels.append('{}_probability'.format(y))
    sc.pl.umap(adata,color=prob_labels, cmap=cmap, s=30, ncols=ncols, show=False, add_outline=add_outline)
    if save != None:
        plt.savefig(save)

def clone_size_umap(adata, reduction="umap",figsize=(10,8),size=1,alpha=0.7,palette="coolwarm",save=None):
    clone_size(adata)
    df = adata.obs
    reduction="umap"
    sizes = np.log10(adata.obs["clone_size"].to_numpy())
    df["UMAP1"] = [x[0] for x in adata.obsm["X_{}".format(reduction)]]
    df["UMAP2"] = [x[1] for x in adata.obsm["X_{}".format(reduction)]]
    df["log(Clone Size)"] = sizes
    fig,ax=plt.subplots(1,1,figsize=figsize)
    sns.scatterplot(data=df,x="UMAP1", y="UMAP2", hue="log(Clone Size)",s=size,palette=palette, ax=ax, alpha=alpha,linewidth=0.)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    fig.tight_layout()
    if save != None:
        fig.savefig(save)
    return ax


def ridge_delta_entropy(
    df_delta      : pd.DataFrame,
    *,
    splitby       : str   = "complete_response",
    order_group   : list  = None,
    order_phen    : list  = None,
    palette       : dict  = None,
    bw_adjust     : float = 0.8,
    jitter        : float = .15,
    density_scale : float = .9,
    significance  : bool  = True,
    sig_test      : str   = "mannwhitney",     # or "bayesian"
    bayes_iters   : int   = 5_000,
    bracket_pad   : float = 0.15,              # ↑ lift for bracket
    star_size     : int   = 16,                # ★ marker size
    figsize       : tuple = (10, 6),
    ax            = None
):
    """
    Ridge plot of Δ-entropy posteriors per phenotype.

    For each phenotype the *first two* groups in `order_group`
    are compared and annotated:

        ┌──────────┐
    CR  │          │ NR
        ★ p-value / stars

    Bracket anchors are placed at the group means.
    """
    # ───── tidy → long Δ samples ─────────────────────────────────
    long = (df_delta
            .explode("delta_samples")
            .rename(columns={"delta_samples": "delta"}))

    if order_group is None:
        order_group = sorted(long[splitby].unique())
    if order_phen is None:
        order_phen  = sorted(long["phenotype"].unique())

    # ───── palette ───────────────────────────────────────────────
    if palette is None:
        tab = cm.tab10.colors
        palette = {g: tab[i % 10] for i, g in enumerate(order_group)}

    # ───── basic geometry ───────────────────────────────────────
    phen2y   = {p: i for i, p in enumerate(order_phen)}
    n_groups = len(order_group)

    x_all        = long["delta"].astype(float).to_numpy()
    x_min,x_max  = np.percentile(x_all, [0.5, 99.5])
    pad          = 0.06*(x_max-x_min)
    xs           = np.linspace(x_min-pad, x_max+pad, 500)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # ───── draw ridges ───────────────────────────────────────────
    for ph in order_phen:
        base_y = phen2y[ph]

        # plot each group’s ridge
        means = {}                               # keep per-group mean (for bracket)
        for g_idx, g in enumerate(order_group):
            data = long[(long["phenotype"]==ph) & (long[splitby]==g)]["delta"].astype(float)
            if data.empty:
                continue
            kde = st.gaussian_kde(data, bw_method=bw_adjust)
            ys  = kde(xs)
            ys  = ys/ys.max()*density_scale

            shift   = (g_idx - (n_groups-1)/2)*2*jitter
            y_line  = base_y + shift

            ax.fill_between(xs, y_line, y_line+ys,
                            color=palette[g], alpha=.85, lw=0)
            ax.plot(xs, y_line+ys, color=palette[g], lw=.8)

            means[g] = data.mean()

        # ── bracket + significance for first two groups ─────────
        if significance and len(order_group) >= 2 and all(k in means for k in order_group[:2]):
            g1, g2   = order_group[:2]
            d1 = long[(long["phenotype"]==ph)&(long[splitby]==g1)]["delta"].to_numpy(float)
            d2 = long[(long["phenotype"]==ph)&(long[splitby]==g2)]["delta"].to_numpy(float)

            # statistical label
            if sig_test == "mannwhitney":
                _, pval = st.mannwhitneyu(d1,d2,alternative="two-sided")
                label = ("ns" if pval>=.05 else
                         "*"  if pval<.05   else
                         "**" if pval<.01  else
                         "***" if pval<.001 else "****")
            else:                                 # Bayesian Δ > 0
                idx  = np.random.randint(0, min(len(d1),len(d2)), size=bayes_iters)
                p_gt = ((d2[idx]-d1[idx]) > 0).mean()
                label = f"P={p_gt:.2f}"

            # bracket coordinates
            x1, x2  = means[g1], means[g2]
            y_brk   = base_y + density_scale + bracket_pad
            ax.plot([x1, x1, x2, x2], [y_brk, y_brk+bracket_pad,
                                        y_brk+bracket_pad, y_brk],
                    color="k", lw=1.2)

            # star / label on top
            x_star  = (x1+x2)/2
            ax.text(x_star, y_brk+bracket_pad*1.05, label,
                    ha="center", va="bottom", fontsize=star_size,
                    color="k")

    # ───── axis cosmetics ───────────────────────────────────────
    ax.set_yticks(list(phen2y.values()), list(order_phen))
    ax.axvline(0, color="k", ls="--", lw=.8)
    ax.set_xlabel("Δ clonotypic entropy  (post – pre)")
    ax.set_xlim(x_min-pad, x_max+pad)
    ax.set_title("Δ Clonotypic Entropy")

    # legend
    handles = [plt.Line2D([0],[0],lw=8,color=palette[g],label=g) for g in order_group]
    ax.legend(handles=handles, title=splitby, frameon=False)

    plt.tight_layout()
    return fig, ax


    
def phenotypic_entropy(adata, splitby=None, temperature=1, n_samples=0, normalized=True, palette=None, save=None, legend_fontsize=6, bbox_to_anchor=(1.15,1.), figsize=(8,4), rotation=90):
    if palette == None:
        palette=tcri_colors
    cov_col = adata.uns["tcri_metadata"]["covariate_col"]
    clone_col = adata.uns["tcri_metadata"]["clone_col"]
    phenotype_col = adata.uns["tcri_metadata"]["phenotype_col"]
    batch_col = adata.uns["tcri_metadata"]["batch_col"]

    covs = adata.obs[cov_col].astype("category").cat.categories.tolist()
    clones = adata.obs[clone_col].astype("category").cat.categories.tolist()
    phenotypes = adata.obs[phenotype_col].astype("category").cat.categories.tolist()
    batches = adata.obs[batch_col].astype("category").cat.categories.tolist()

    mi = []
    ps = []
    ts = []
    rs = []
    cl = []
    phs = []
    for p in tqdm.tqdm(batches):
        sub = adata[adata.obs[batch_col] == p].copy()
        for t in covs:
            subt = sub[sub.obs[cov_col] == t]
            vclones = list(set(subt.obs[clone_col]))
            if len(vclones) == 0: continue
            for ph in phenotypes:
                if splitby == None:
                    vclones = list(set(subt.obs[clone_col]))
                    mi.append(pentropy(subt,t,ph, temperature=temperature, clones=vclones,n_samples=n_samples, normalized=normalized))
                    ps.append(p)
                    ts.append(t)
                    phs.append(ph)
                else:
                    for s in set(subt.obs[splitby]):
                        subts = subt[subt.obs[cov_col] == t]
                        vclones = list(set(subts.obs[clone_col]))
                        mi.append(pentropy(subts,t,ph, temperature=temperature, clones=vclones,n_samples=n_samples, normalized=normalized))
                        ps.append(p)
                        ts.append(t)
                        cl.append(s)
                        phs.append(ph)
    fig,ax = plt.subplots(1,1,figsize=figsize)
    if splitby != None:
        df = pd.DataFrame.from_dict({cov_col: ts, batch_col:ps, "Phenotypic Entropy":mi, splitby:cl, clone_col:phs})
        sns.boxplot(data=df,x=splitby,y="Phenotypic Entropy",hue=cov_col,palette=palette)
        palette_black = {level: "black" for level in df[cov_col].unique()}
        sns.stripplot(data=df,x=splitby,y="Phenotypic Entropy",hue=cov_col, dodge=True, palette=palette_black)
    else:
        df = pd.DataFrame.from_dict({cov_col: ts, batch_col:ps, "Phenotypic Entropy":mi,clone_col:phs})
        sns.boxplot(data=df,x=cov_col,y="Phenotypic Entropy",color="#999999")
        sns.stripplot(data=df,x=cov_col,y="Phenotypic Entropy",palette=palette,dodge=False)
    plt.xticks(rotation=rotation)
    leg = ax.legend(loc='upper right', bbox_to_anchor=bbox_to_anchor, fontsize=legend_fontsize)
    fig.tight_layout()
    if save:
        fig.savefig(save)

def set_color_palette(adata, columns):
    i = 0
    main_color_map = dict()
    adata = adata.copy()
    colors = tcri_colors.copy() + tcri_colors.copy() + tcri_colors.copy()
    for x in columns:
        ct = []
        for i, val in enumerate(set(adata.obs[x].tolist())):
            c = colors.pop(i)
            ct.append(c)
            main_color_map[val] = c
        adata.uns["{}_colors".format(x)] = ct
    return main_color_map

def clone_fraction(adata, groupby):
    fractions = clone_fraction_tl(adata,groupby=groupby)
    draw_clone_bars(fractions, title=groupby)

def flux(adata, key, order, groupby, paint_dict=None, method="probabilistic", paint=None, distance_metric="l1", figsize=(12,5), paint_order=None, palette=None):
    dfs = []
    if paint != None:
        palette = []
        legend_handles = [] 
        paint_categories = adata.obs[paint].unique()
        if paint_dict != None:
            pcolors = paint_dict
        else:
            pcolors = dict(zip(paint_categories, tcri_colors))
        for category in paint_categories:
            handle = mpatches.Patch(color=pcolors[category], label=category)
            legend_handles.append(handle)
    else:
        if palette == None:
            if "{}_colors".format(paint) in adata.uns:
                palette = adata.uns["{}_colors".format(paint)]
            else:
                palette = tcri_colors
    for x in tqdm.tqdm(list(set(adata.obs[groupby]))):
        sdata = adata[adata.obs[groupby]==x]
        hue_order = []
        for i in range(len(order)-1):
            l1_distances = flux_tl(sdata,key=key,from_this=order[i],to_that=order[i+1],distance_metric=distance_metric)
            df = pd.DataFrame(list(l1_distances.items()), columns=['Clone', distance_metric])
            df[groupby] = x
            if paint!=None:
                pcat = sdata.obs[paint].unique().tolist()[0]
                palette.append(pcolors[pcat])
            df["Comparison"] = pcat
            dfs.append(df)
    print(palette)
    df = pd.concat(dfs)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    order = df.groupby(groupby).median(distance_metric).sort_values(distance_metric).index.tolist()
    fig,ax=plt.subplots(1,1,figsize=figsize)
    sns.boxplot(data=df,x=groupby,y=distance_metric,hue="Comparison",order=order,palette=pcolors,ax=ax)
    fig.tight_layout()
    return ax

def mutual_information(adata, splitby=None, temperature=1.0, n_samples=0, normalized=True, palette=None, save=None, legend_fontsize=6, bbox_to_anchor=(1.15,1.), figsize=(8,4), rotation=90, weighted=True, return_plot=True):
    """
    Compute and plot mutual information between clonotypes and phenotypes.
    
    This function calculates mutual information between TCR clonotypes and cell phenotypes,
    which quantifies how much information one variable provides about the other. The function
    can optionally split the calculation by a specified categorical variable (e.g., timepoint,
    condition). Results are displayed as a box plot with individual data points.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing the data with TCR and phenotype information
    splitby : str, optional
        Column name to split the data by. If None, uses the covariate column stored in adata.uns["tcri_metadata"]
    temperature : float, default=1.0
        Temperature parameter for softening/sharpening distributions in the joint distribution calculation
    n_samples : int, default=0
        Number of samples to use for Monte Carlo estimation of mutual information
    normalized : bool, default=True
        Whether to normalize the mutual information values to [0,1] range
    palette : list, optional
        Color palette for the plot. If None, uses tcri_colors
    save : str, optional
        Path to save the plot figure
    legend_fontsize : int, default=6
        Font size for the plot legend
    bbox_to_anchor : tuple, default=(1.15,1.)
        Position of the legend box
    figsize : tuple, default=(8,4)
        Size of the figure in inches (width, height)
    rotation : int, default=90
        Rotation angle for x-axis labels
    weighted : bool, default=True
        Whether to weight the mutual information by clone size
    return_plot : bool, default=True
        Whether to return the plot axis. If False, returns only the DataFrame with MI values
        
    Returns
    -------
    Union[matplotlib.axes.Axes, pd.DataFrame]
        If return_plot is True, returns the plot axis. Otherwise returns a DataFrame with MI values.
        
    Examples
    --------
    >>> import tcri
    >>> # Calculate and plot mutual information
    >>> ax = tcri.pl.mutual_information(adata, splitby="timepoint", temperature=1.0)
    >>> 
    >>> # Get the mutual information values as a DataFrame without plotting
    >>> mi_df = tcri.pl.mutual_information(adata, splitby="condition", return_plot=False)
    """
    if palette is None:
        palette = tcri_colors

    # Retrieve metadata from adata
    cov_col = adata.uns["tcri_metadata"]["covariate_col"]
    clone_col = adata.uns["tcri_metadata"]["clone_col"]
    phenotype_col = adata.uns["tcri_metadata"]["phenotype_col"]
    batch_col = adata.uns["tcri_metadata"]["batch_col"]
    
    covs = adata.obs[cov_col].astype("category").cat.categories.tolist()
    batches = adata.obs[batch_col].astype("category").cat.categories.tolist()

    mi_vals = []
    ps = []
    ts = []
    cl = []

    for p in tqdm.tqdm(batches, desc="Computing mutual information"):
        sub = adata[adata.obs[batch_col] == p].copy()
        for t in covs:
            subt = sub[sub.obs[cov_col] == t]
            # figure out which clones are actually present
            vclones = list(set(subt.obs[clone_col]))
            
            if splitby is None:
                # Compute MI for all clones in cov t, batch p
                val = mutual_information_tl(
                    subt, t, 
                    temperature=temperature, 
                    clones=vclones,
                    n_samples=n_samples,
                    weighted=weighted
                )
                mi_vals.append(val)
                ps.append(p)
                ts.append(t)

            else:
                # If you want to split further by some obs column
                for s in sorted(subt.obs[splitby].unique()):
                    subts = subt[subt.obs[splitby] == s]
                    vclones2 = list(set(subts.obs[clone_col]))
                    val = mutual_information_tl(
                        subts, t,
                        temperature=temperature,
                        clones=vclones2,
                        n_samples=n_samples,
                        weighted=weighted
                    )
                    mi_vals.append(val)
                    ps.append(p)
                    ts.append(t)
                    cl.append(s)

    # Build a DataFrame for plotting
    if splitby is None:
        df = pd.DataFrame({
            cov_col: ts,
            batch_col: ps,
            "Mutual Information": mi_vals
        })
    else:
        df = pd.DataFrame({
            cov_col: ts,
            batch_col: ps,
            "Mutual Information": mi_vals,
            splitby: cl
        })

    if not return_plot:
        return df

    # Create the plot
    fig, ax = plt.subplots(1,1, figsize=figsize)
    if splitby is None:
        sns.boxplot(data=df, x=cov_col, y="Mutual Information", color="#999999", ax=ax)
        sns.stripplot(data=df, x=cov_col, y="Mutual Information", palette=palette, dodge=False, ax=ax)
    else:
        sns.boxplot(data=df, x=splitby, y="Mutual Information", hue=cov_col, color="#999999", ax=ax)
        sns.stripplot(data=df, x=splitby, y="Mutual Information", hue=cov_col, palette=palette, dodge=True, ax=ax)
        ax.legend(loc='upper right', bbox_to_anchor=bbox_to_anchor, fontsize=legend_fontsize)

    plt.xticks(rotation=rotation)
    plt.title("Mutual Information" + (" (Weighted)" if weighted else ""))
    fig.tight_layout()

    if save:
        plt.savefig(save, dpi=150)
    plt.show()

    return ax


def bayesian_mutual_information(
    adata,
    *,
    group1,
    group2,
    splitby,
    n_samples       = 200,
    temperature     = 1.0,
    normalised      = True,
    normalise_mode  = "average",
    weighted        = False,
    posterior       = True,
    combine_with_logits=True,
    seed            = 42,
    palette         = None,
):
    np.random.seed(seed)

    meta       = adata.uns["tcri_metadata"]
    cov_col    = meta["covariate_col"]
    clone_col  = meta["clone_col"]

    groups = sorted(adata.obs[splitby].dropna().unique().tolist())
    if palette == None:
        palette = dict()
        for i, g in enumerate(groups):
            palette[g] = tcri_colors[i]
    print(f"{BOLD}{MAG}──────── MI summary ({group1} → {group2}) ────────{RESET}")
    _info("split column",      splitby)
    _info("# groups",          len(groups))
    _info("Δ samples / group", n_samples)
    _info("weighted",          weighted)
    print(f"{MAG}────────────────────────────────────────────────────{RESET}")

    results = {}

    # ---------- iterate over strata --------------------------------
    for g in groups:
        mask_g  = adata.obs[splitby] == g
        clones  = adata.obs.loc[mask_g, clone_col].unique().tolist()

        mi_pre  = []; mi_post = []
        bar = tqdm(range(n_samples), desc=f"Δ-MI samples  ({g})")
        for _ in bar:
            mi_pre.append( mutual_information_tl(
                adata, group1, temperature=temperature, n_samples=1,
                clones=clones, weighted=weighted, normalised=normalised,
                normalise_mode=normalise_mode, posterior=posterior,
                combine_with_logits=combine_with_logits, verbose=False))
            mi_post.append( mutual_information_tl(
                adata, group2, temperature=temperature, n_samples=1,
                clones=clones, weighted=weighted, normalised=normalised,
                normalise_mode=normalise_mode, posterior=posterior,
                combine_with_logits=combine_with_logits, verbose=False))

        mi_pre  = np.array(mi_pre)
        mi_post = np.array(mi_post)
        delta   = mi_post - mi_pre

        d_mean, d_std = delta.mean(), delta.std()
        hdi_low, hdi_hi = np.percentile(delta, [2.5,97.5])
        p_gt = (delta>0).mean(); p_lt = 1-p_gt
        cohens_d = d_mean/d_std if d_std>0 else 0.0

        print(f"\n{BOLD}{g}{RESET}  ΔMI = {d_mean:.4f} ± {d_std:.4f}   "
              f"95 % HDI [{hdi_low:.4f}, {hdi_hi:.4f}]   "
              f"P(>0)={p_gt:.3f}")

        results[g] = dict(delta_samples=delta, mi_pre_samples=mi_pre,
                          mi_post_samples=mi_post, delta_mean=d_mean,
                          delta_std=d_std, cohens_d=cohens_d,
                          p_greater=p_gt, p_less=p_lt,
                          hdi=(hdi_low, hdi_hi))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4),
                           gridspec_kw=dict(width_ratios=[2, 2, 1]),
                           constrained_layout=True)
    
    # A ─── Δ-MI KDEs
    for i, g in enumerate(groups):
        sns.kdeplot(results[g]["delta_samples"],
                    fill=True, ax=ax[0],
                    palette=[palette[g]],
                    alpha=.9, linewidth=1.2, label=g)
    ax[0].axvline(0, color="k", ls="--")
    ax[0].set(title="Δ MI (post – pre)",
              xlabel="Δ normalised MI", ylabel="density")
    ax[0].legend(title=splitby)
    
    # B ─── pre / post KDEs
    for i, g in enumerate(groups):
        sns.kdeplot(results[g]["mi_pre_samples"],
                    ax=ax[1],  palette=[palette[g]],
                    ls="-",  label=f"{g} – {group1}")
        sns.kdeplot(results[g]["mi_post_samples"],
                    ax=ax[1],  palette=[palette[g]],
                    ls="--", label=f"{g} – {group2}")
    ax[1].set(title="MI posterior per condition",
              xlabel="normalised MI")
    ax[1].legend()
    
    # C ─── bar summary of Δ
    means = [results[g]["delta_mean"] for g in groups]
    errs  = [results[g]["delta_std"]  for g in groups]
    ax[2].bar(groups, means, yerr=errs, capsize=5,
              color=[palette[g] for g in groups])
    ax[2].axhline(0, color="k", ls="--")
    ax[2].set(title="Δ MI summary", ylabel="Δ normalised MI")
    
    fig.suptitle("Bayesian MI Analysis of clonotype ⇄ phenotype coupling",
                 fontsize=14, weight="bold")
    return results



def polar_plot(adata, phenotypes=None, statistic="distribution", method="joint_distribution", splitby=None, color_dict=None, temperature=1.0):
    """
    Create a polar plot showing phenotype distributions or entropies.
    
    This function creates a radar/polar chart that visualizes either the distribution of phenotypes
    or entropy values across different conditions. It's useful for comparing phenotype proportions
    or entropy patterns across experimental groups.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing the data with TCR and phenotype information
    phenotypes : list, optional
        List of phenotype names to include in the plot. If None, uses all phenotypes
        defined in adata.uns["tcri_metadata"]["phenotype_col"]
    statistic : str, default="distribution"
        Type of statistic to plot, one of "distribution" or "entropy"
    method : str, default="joint_distribution"
        Method to compute phenotype distributions, either "joint_distribution" (model-based)
        or "empirical" (raw cell counts)
    splitby : str, optional
        Column name to split the data by. If None, uses the covariate column stored
        in adata.uns["tcri_metadata"]["covariate_col"]
    color_dict : dict, optional
        Dictionary mapping split categories to colors
    temperature : float, default=1.0
        Temperature parameter for softening/sharpening distributions
        
    Returns
    -------
    matplotlib.axes.Axes
        The polar plot axis
        
    Examples
    --------
    >>> import tcri
    >>> # Basic phenotype distribution polar plot
    >>> ax = tcri.pl.polar_plot(adata, statistic="distribution")
    >>> 
    >>> # Entropy polar plot with custom colors
    >>> color_dict = {"Day0": "#FF5733", "Day7": "#33FF57", "Day14": "#3357FF"}
    >>> ax = tcri.pl.polar_plot(adata, statistic="entropy", color_dict=color_dict)
    """
    if phenotypes is None:
        phenotypes = adata.uns["tcri_metadata"]["phenotype_col"]
    
    if splitby is None:
        splitby = adata.uns["tcri_metadata"]["covariate_col"]
    
    # Get unique splits
    splits = adata.obs[splitby].unique()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    # Calculate angles for each phenotype
    angles = np.linspace(0, 2*np.pi, len(phenotypes), endpoint=False)
    
    # Plot for each split
    for i, split in enumerate(splits):
        if statistic == "distribution":
            if method == "joint_distribution":
                jd = joint_distribution(adata, split, temperature=temperature)
                values = jd.mean().values
            else:
                subset = adata[adata.obs[splitby] == split]
                values = np.zeros(len(phenotypes))
                for j, pheno in enumerate(phenotypes):
                    mask = subset.obs[adata.uns["tcri_metadata"]["phenotype_col"]] == pheno
                    values[j] = np.sum(mask) / len(subset)
        else:  # entropy
            values = np.zeros(len(phenotypes))
            for j, pheno in enumerate(phenotypes):
                values[j] = clonotypic_entropy(adata, split, pheno, temperature=temperature)
        
        # Normalize values
        values = values / np.sum(values)
        
        # Plot values
        color = color_dict[split] if color_dict else None
        ax.plot(angles, values, 'o-', linewidth=2, label=split, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    # Set the labels
    ax.set_xticks(angles)
    ax.set_xticklabels(phenotypes)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    return ax