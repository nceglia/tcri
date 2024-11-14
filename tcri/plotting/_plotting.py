import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation
import matplotlib.patches as mpatches

from gseapy import dotplot
import tqdm

import collections
import operator
import itertools

from ..utils._utils import Phenotypes, CellRepertoire, Tcell, plot_pheno_sankey, plot_pheno_ternary_change_plots, draw_clone_bars, probabilities
from ..preprocessing._preprocessing import clone_size, joint_distribution
from ..metrics._metrics import clonotypic_entropies as centropies
from ..metrics._metrics import phenotypic_entropies as pentropies
from ..metrics._metrics import clonality as clonality_tl
from ..metrics._metrics import flux as flux_tl
from ..metrics._metrics import probability_distribution as pdistribution
from ..metrics._metrics import mutual_information as mutual_information_tl
from ..metrics._metrics import phenotypic_entropy_delta as phenotypic_entropy_delta_tl
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


def phenotypic_flux(adata, splitby, order, clones=None, normalize=True, nt=True, method="probabilistic", phenotype_colors=None, save=None, figsize=(6,3), show_legend=True):
    joint_distribution(adata, method=method)
    phenotypes = Phenotypes(adata.obs[adata.uns["tcri_phenotype_key"]].unique())
    if method == "probabilistic":
        cell_probabilities = probabilities(adata)
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
        if str(seq) != "nan" and condition in repertoires:
            if method == "probabilistic":
                phenotypes_and_counts = cell_probabilities[bc]
            elif method == "empirical":
                phenotypes_and_counts = {phenotype: 1}
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

def probability_ternary(adata, phenotype_names, splitby, conditions, method="probabilistic", nt=False, top_n=None, scale_function=None,color="k",save=None):
    if scale_function == None:
        scale_function = freq_to_size_scaling
    phenotypes = Phenotypes(phenotype_names)
    cell_probabilities = probabilities(adata)
    repertoires = dict()
    if nt:
        chains_to_use = "ntseq"
    else:
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
        if str(seq) != "nan" and condition in repertoires:
            if method == "probabilistic":
                phenotypes_and_counts = cell_probabilities[bc]
            elif method == "empirical":
                phenotypes_and_counts = {phenotype: 1}
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
                                            kwargs_for_plots={"color":color},
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

def expression_ternary(adata, gene_symbols, splitby, conditions, temperature=0.001, nt=False, top_n=None, color="k",save=None, scale_function=None):
    def get_expression(gene):
        return adata.X[:,adata.var.index.tolist().index(gene)].T.todense().tolist()[0]
    assert len(gene_symbols) == 3, "Must select three genes."
    def normalized_exponential(values):
        assert temperature > 0, "Temperature must be positive"
        exps = np.exp(values / temperature)
        dist = exps / np.sum(exps)
        return np.nan_to_num(dist,nan=1)
    expression_matrix = []
    for gene in gene_symbols:
        expression = get_expression(gene)
        expression_matrix.append(expression)
    matrix = np.array(expression_matrix).T
    cell_probabilities = dict()
    for bc, exp in zip(adata.obs.index.tolist(), matrix):
        probs = normalized_exponential(exp)
        cell_probabilities[bc] = dict(zip(gene_symbols,probs))
    if scale_function == None:
        scale_function = freq_to_size_scaling
    phenotypes = Phenotypes(gene_symbols)
    repertoires = dict()
    if nt:
        chains_to_use = "ntseq"
    else:
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
        if str(seq) != "nan" and condition in repertoires:
            phenotypes_and_counts = cell_probabilities[bc]
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
    phenotype_names_dict = {p: p for p in gene_symbols}
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
        start_clones_and_phenos = repertoires[conditions]
        end_clones_and_phenos = None
    s_dict = {c: scale_function(sum(start_clones_and_phenos[c].values())/start_clones_and_phenos.norm) for c in start_clones_and_phenos.clones}
    if top_n == None:
        top_n = len(set(adata.obs[adata.uns["tcri_clone_key"]]))
    c_clones = sorted(start_clones_and_phenos.clones, key = s_dict.get, reverse = True)[:top_n]
    fig, ax = plot_pheno_ternary_change_plots(start_clones_and_phenotypes = start_clones_and_phenos,
                                            end_clones_and_phenotypes = end_clones_and_phenos,
                                            phenotypes = gene_symbols, 
                                            phenotype_names = phenotype_names_dict,
                                            clones = c_clones,
                                            line_type = 'arrows', 
                                            kwargs_for_plots={"color":color},
                                            s_dict = s_dict,
                                            return_axes  = True)
    freq_to_size_legend(ax)
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

def phenotypic_entropy_delta(adata, groupby, key, from_this, to_that, palette=None, figsize=(7,5),save=None):
    df = phenotypic_entropy_delta_tl(adata, groupby, key, from_this, to_that)
    if palette==None:
        print("hit")
        palette=tcri_colors
    fig, ax = plt.subplots(1,1,figsize=figsize)
    sns.boxplot(data=df, y='Delta Phenotypic Entropy', x=groupby, palette=palette,ax=ax)
    if save != None:
        fig.savefig(save)

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

def clonotypic_entropy(adata, method="probabilistic", normalized=True, groupby=None, splitby=None, s=10, figsize=(12,5), order=None, palette=None):
    func = lambda x : centropies(x, normalized=normalized, method=method)
    return tcri_boxplot(adata, func, groupby=groupby, ylabel="Clonotypic Entropy", splitby=splitby, s=s, figsize=figsize, order=order, palette=palette)

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

def phenotypic_entropy(adata, groupby, splitby, method="probabilistic", return_df=False, normalized=True, decimals=5, figsize=(5,4), save=None, order=None, rotation=0, minimum_clone_size=1, palette=None):
    ps = []
    rs = []
    r2 = []
    ts = []
    for r in set(adata.obs[groupby]):
        rdata = adata[adata.obs[groupby] == r]
        clone_size(rdata)
        rdata = rdata[rdata.obs["clone_size"] >= minimum_clone_size]
        for p in set(rdata.obs[splitby]):
            pdata = rdata[rdata.obs[splitby] == p]
            for clone, ent in pentropies(pdata,method=method,normalized=normalized,decimals=decimals).items():
                rs.append(p)
                r2.append(ent)
                ts.append(clone)
                ps.append(r)
    df = pd.DataFrame.from_dict({groupby:ps,splitby:rs,"Phenotypic Entropy":r2,"Clone":ts})
    fig, ax = plt.subplots(1,1,figsize=figsize)
    if order == None:
        order = list(set(rs))
    if palette == None:
        palette = tcri_colors
    sns.boxplot(data=df, x=splitby,y="Phenotypic Entropy",ax=ax,order=order,palette=palette)
    plt.xticks(rotation=rotation)
    if save!=None:
        fig.savefig(save)
    if return_df:
        return df

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

def probability_distribution_bar(adata, phenotypes=None, method="probabilistic", save=None, figsize=(6,2)):
    pdist = pdistribution(adata, method=method)
    if phenotypes == None:
        phenotypes = adata.uns["joint_distribution"].index
    fig, ax = plt.subplots(1,1,figsize=figsize)
    sns.barplot(data=pdist,ax=ax,order=phenotypes,color="#000000")
    ax.set_xticklabels(ax[1].get_xticklabels(), rotation=45)
    ax.set_title("BCC CD8 Post")
    fig.tight_layout()
    if save:
        fig.savefig(save)

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

def mutual_information(adata, groupby, splitby=None, method="probabilistic", box_color="#999999", size=10, figsize=(6,5), colors=None, minimum_clone_size=1, rotation=90,return_df=False,bbox_to_anchor=(1.15, 1.), order=None):
    mis = []
    groups = []
    splits = []
    for group in set(adata.obs[groupby]):
        gdata = adata[adata.obs[groupby] == group]
        clone_size(gdata)
        gdata = gdata[gdata.obs["clone_size"] >= minimum_clone_size]
        if splitby != None:
            for split in set(gdata.obs[splitby]):
                sdata = gdata[gdata.obs[splitby] == split]
                mi = mutual_information_tl(sdata, method=method)
                mis.append(mi)
                groups.append(group)
                splits.append(split)
        else:
            joint_distribution(gdata)
            mi = mutual_information_tl(gdata,method=method)
            mis.append(mi)
            groups.append(group)
    df = pd.DataFrame.from_dict({"MI":mis, groupby: groups})
    if splitby != None:
        df[splitby] = splits
    if order == None:
        order = list(set(adata.obs[splitby]))
    if colors == None:
        colors = tcri_colors
    fig, ax = plt.subplots(1,1,figsize=figsize)
    sns.boxplot(data=df,x=splitby,y="MI",ax=ax,order=order, color=box_color)
    sns.swarmplot(data=df,x=splitby,y="MI",order=order ,s=size, hue=groupby, palette=colors)
    fig.tight_layout()
    plt.xticks(rotation=rotation)
    _ = ax.legend(loc='upper right', bbox_to_anchor=bbox_to_anchor)
    if return_df:
        return df
    else:
        return ax

def polar_plot(adata, phenotypes=None, statistic="entropy", method="probabilistic", save=None, figsize=(6,6), title=None, alpha=0.6, fontsize=15, splitby=None, bbox_to_anchor=(1.15,1.), linewidth=5., legend_fontsize=15, color_dict=None):
    joint_distribution(adata,method=method )
    plt.figure(figsize=figsize)
    ax = plt.subplot(111, projection='polar')
    if splitby is None:
        splits = ['All']
    else:
        splits = list(set(adata.obs[splitby]))
    if phenotypes is None:
        phenotypes = adata.uns["joint_distribution"].index
    N = len(phenotypes)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    plot_theta = np.append(theta, theta[0])
    subset = adata[adata.obs[adata.uns["tcri_phenotype_key"]].isin(phenotypes)]
    for i, split in enumerate(splits): 
        if color_dict == None:
            colorx = tcri_colors[i]
        else:
            colorx = color_dict[split]
        psubset = adata[adata.obs[splitby] == split]
        if statistic == "entropy":
            pdist = pd.Series(centropies(psubset))
        else:    
            pdist = pdistribution(psubset, method=method)
        pdist = pdist.tolist()
        pdist.append(pdist[0])
        ax.plot(plot_theta, pdist, color=colorx, alpha=alpha, label=split, linewidth=linewidth)
        ax.fill_between(plot_theta, 0, pdist, color=colorx, alpha=alpha)
    ax.set_xticks(theta)
    ax.set_xticklabels(phenotypes, fontsize=fontsize)
    ax.grid(True)
    leg = ax.legend(loc='upper right', bbox_to_anchor=bbox_to_anchor, fontsize=legend_fontsize)
    for line in leg.get_lines():
        line.set_linewidth(8.0)  # Set the line width
    if title:
        plt.title(title, va='bottom', fontsize=fontsize, fontweight="bold")
    if save:
        plt.savefig(save)