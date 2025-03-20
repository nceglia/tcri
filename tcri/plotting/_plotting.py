import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram, linkage

from gseapy import dotplot
import tqdm

import collections
import operator
import itertools

from ..utils._utils import Phenotypes, CellRepertoire, Tcell, plot_pheno_sankey, plot_pheno_ternary_change_plots, draw_clone_bars, probabilities, set_ternary_corner_label, ternary_plot_projection
from ..preprocessing._preprocessing import clone_size, joint_distribution
from ..metrics._metrics import clonotypic_entropy as centropy
from ..metrics._metrics import phenotypic_entropy as pentropy
from ..metrics._metrics import clonality as clonality_tl
from ..metrics._metrics import flux as flux_tl
from ..metrics._metrics import mutual_information as mutual_information_tl
from ..metrics._metrics import clone_fraction as clone_fraction_tl


import warnings
warnings.filterwarnings('ignore')

tcri_colors = [
    "#F28E7F",  # Salmon
    "#6A6E75",  # Iron Gray
    "#AE81FF",  # Purple
    "#FD971F",  # Orange
    "#E6DB74",  # Yellow
    "#A6E22E",  # Green
    "#66D9EF",  # Blue
    "#75715E",  # Brown
    "#F92659",  # Pink
    "#D65F0E",  # Abricos
    "#F92672",  # Red
    "#1E1E1E",   # Black
    "#004d47",  # Darker Teal
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

def phenotypic_flux(adata, splitby, order, clones=None, normalize=True, nt=False, n_samples=0, phenotype_colors=None, save=None, figsize=(6,3), show_legend=True, temperature=1):
    phenotypes = Phenotypes(adata.uns["tcri_phenotype_categories"])
    cell_probabilities = collections.defaultdict(dict)
    for s in order:
        jd = joint_distribution(adata,s,n_samples=n_samples,temperature=temperature)
        if n_samples > 0:
            jd["clonotype_id"] = ["_".join(x.split("_")[:-1]) for x in jd.index]
            jd = jd.groupby("clonotype_id").mean()
        for x in jd.T:
            cell_probabilities[s][x] = jd.T[x].to_dict()
    repertoires = dict()
    times = list(range(len(order)))
    if nt:
        chains_to_use = "ntseq"
    else:
        chains_to_use = "aaseq"
    for s in order:
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

def probability_ternary(adata, phenotype_names, splitby=None, conditions=None, top_n=None):
    """Create a ternary plot showing phenotype probabilities."""
    if splitby is None:
        splitby = adata.uns["tcri_metadata"]["covariate_col"]
    
    if conditions is None:
        conditions = adata.obs[splitby].unique()
    
    # Get joint distribution for each condition
    jds = {}
    for cond in conditions:
        jd = joint_distribution(adata, cond, temperature=1.0)
        if top_n is not None:
            jd = jd.nlargest(top_n, phenotype_names[0])
        jds[cond] = jd
    
    # Create the plot
    fig, ax = setup_ternary_plot()
    
    # Plot points for each condition
    for i, (cond, jd) in enumerate(jds.items()):
        # Convert to ternary coordinates
        x = jd[phenotype_names[0]].values
        y = jd[phenotype_names[1]].values
        z = jd[phenotype_names[2]].values
        
        # Plot points
        ax.scatter(x, y, z, label=cond, alpha=0.6)
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    return fig, ax

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

    
def clonotypic_entropy(adata, splitby=None, temperature=1, n_samples=0, normalized=True, palette=None, save=None, legend_fontsize=6, bbox_to_anchor=(1.15,1.), figsize=(8,4), rotation=90):
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
            for ph in phenotypes:
                if splitby == None:
                    mi.append(centropy(subt,t,ph, temperature=temperature, clones=vclones,n_samples=n_samples, normalized=normalized))
                    ps.append(p)
                    ts.append(t)
                    phs.append(ph)
                else:
                    for s in set(subt.obs[splitby]):
                        subts = subt[subt.obs[cov_col] == t]
                        vclones = list(set(subts.obs[clone_col]))
                        mi.append(centropy(subt,t,ph, temperature=temperature, clones=vclones,n_samples=n_samples, normalized=normalized))
                        ps.append(p)
                        ts.append(t)
                        cl.append(s)
                        phs.append(ph)
    fig,ax = plt.subplots(1,1,figsize=figsize)
    if splitby != None:
        df = pd.DataFrame.from_dict({cov_col: ts, batch_col:ps, "Clonotypic Entropy":mi, splitby:cl, phenotype_col:phs})
        sns.boxplot(data=df,x=splitby,y="Clonotypic Entropy",hue=cov_col,palette=palette)
        palette_black = {level: "black" for level in df[cov_col].unique()}
        sns.stripplot(data=df,x=splitby,y="Clonotypic Entropy",hue=cov_col, dodge=True, palette=palette_black)
    else:
        df = pd.DataFrame.from_dict({cov_col: ts, batch_col:ps, "Clonotypic Entropy":mi,phenotype_col:phs})
        sns.boxplot(data=df,x=phenotype_col,hue=cov_col,y="Clonotypic Entropy",color="#999999")
        sns.stripplot(data=df,x=phenotype_col,hue=cov_col,y="Clonotypic Entropy",palette=palette,dodge=True)
    plt.xticks(rotation=rotation)
    leg = ax.legend(loc='upper right', bbox_to_anchor=bbox_to_anchor, fontsize=legend_fontsize)
    fig.tight_layout()
    if save:
        fig.savefig(save)

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
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing the data
    splitby : str, optional
        Column name to split the data by. If None, uses the covariate column.
    temperature : float, optional
        Temperature parameter for the joint distribution calculation
    n_samples : int, optional
        Number of samples to use for Monte Carlo estimation
    normalized : bool, optional
        Whether to normalize the mutual information values
    palette : list, optional
        Color palette for the plot
    save : str, optional
        Path to save the plot
    legend_fontsize : int, optional
        Font size for the legend
    bbox_to_anchor : tuple, optional
        Position of the legend box
    figsize : tuple, optional
        Figure size
    rotation : int, optional
        Rotation angle for x-axis labels
    weighted : bool, optional
        Whether to weight the mutual information by clone size
    return_plot : bool, optional
        Whether to return the plot axis. If False, returns only the DataFrame.
        
    Returns
    -------
    Union[matplotlib.axes.Axes, pd.DataFrame]
        If return_plot is True, returns the plot axis. Otherwise returns a DataFrame with MI values.
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

def mutual_information_plot(adata, splitby=None, temperature=1.0):
    """Create a plot showing mutual information between clonotypes and phenotypes."""
    if splitby is None:
        splitby = adata.uns["tcri_metadata"]["covariate_col"]
    
    # Calculate mutual information for each split
    splits = adata.obs[splitby].unique()
    mi_values = []
    
    for split in splits:
        subset = adata[adata.obs[splitby] == split]
        mi = mutual_information_tl(subset, split, temperature=temperature)
        mi_values.append(mi)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        splitby: splits,
        "Mutual Information": mi_values
    })
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df, x=splitby, y="Mutual Information", ax=ax)
    
    plt.tight_layout()
    return ax

def polar_plot(adata, phenotypes=None, statistic="distribution", method="joint_distribution", splitby=None, color_dict=None, temperature=1.0):
    """Create a polar plot showing phenotype distributions or entropies."""
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