import scanpy as sc
import numpy as np
import pandas
import sys
from tqdm import tqdm
import numpy
import collections
import multiprocessing

def dicts2collated_arrays(dictsA, dictsB = []):
    if type(dictsA) is dict: dictsA = [dictsA]
    if type(dictsB) is dict: dictsB = [dictsB]
    all_clonotypes = list(set(sum([list(d.keys()) for d in dictsA], []) + sum([list(d.keys()) for d in dictsB], [])))
    n_phenotypes = len(list(dictsA[0].values())[0])
    arrayA = numpy.array([list(sum([d.get(clonotype, numpy.zeros(n_phenotypes)) for d in dictsA], numpy.zeros(n_phenotypes))) for clonotype in all_clonotypes])

    if len(dictsB) > 0:
        arrayB = numpy.array([list(sum([d.get(clonotype, numpy.zeros(n_phenotypes)) for d in dictsB], numpy.zeros(n_phenotypes))) for clonotype in all_clonotypes])
        return arrayA, arrayB, all_clonotypes
    else:
        return arrayA, all_clonotypes

def initializer(clonotype_column, phenotype_column, min_clone_size, phenotypes):
    global g_clonotype_column, g_phenotype_column, g_min_clone_size, g_phenotypes
    g_clonotype_column = clonotype_column
    g_phenotype_column = phenotype_column
    g_min_clone_size = min_clone_size
    g_phenotypes = phenotypes

def process_parallel(ta):
    global g_clonotype_column, g_phenotype_column, g_min_clone_size, g_phenotypes
    sample, tadata = ta
    return process(sample, tadata, g_clonotype_column, g_phenotype_column, g_min_clone_size, g_phenotypes)

def process(sample, tadata, clonotype_column, phenotype_column, min_clone_size, phenotypes):
    num_cells = 0
    joint_d = []
    clones = []
    for clonotype in tqdm(list(set(tadata[clonotype_column].tolist())), ncols=50):
        sub = tadata[tadata[clonotype_column]==clonotype]
        if len(sub.index) < min_clone_size: continue
        dist = []
        for phenotype in phenotypes:
            ph = phenotype+" Pseudo-probability"
            if ph in sub.columns:
                count = sum(sub[ph].tolist())
            else:
                count = np.count_nonzero(sub[phenotype_column] == phenotype)
            dist.append(count)
            num_cells += count
        joint_d.append(dist)
        clones.append(clonotype)
    joint_d = numpy.array(joint_d)# / num_cells
    _joint_d = {clone: dist for clone, dist in zip(clones, joint_d)}
    return sample, _joint_d

def joint_distribution(adata, condition_column="condition", sample_column="sample",
                       clonotype_column="IR_VDJ_1_junction_aa", phenotype_column="phenotype", min_clone_size=1, cores=1):
    if clonotype_column not in adata.obs:
        raise ValueError("{} not found in adata.obs.keys".format(clonotype_column))
    if phenotype_column not in adata.obs:
        raise ValueError("{} not found in adata.obs.keys".format(phenotype))
    if condition_column not in adata.obs:
        raise ValueError("{} not found in adata.obs.keys".format(condition_column))
    if sample_column not in adata.obs:
        raise ValueError("{} not found in adata.obs.keys".format(sample_column))
    phenotypes = list(set(adata.obs[phenotype_column]))
    clonotypes = list(set(adata.obs[clonotype_column]))
    samples = list(set(adata.obs[sample_column]))
    conditions = list(set(adata.obs[condition_column]))

    adata.uns["phenotype_order"] = phenotypes
    adata.uns["clonotype_order"] = clonotypes
    adata.uns["sample_order"] = samples
    adata.uns["condition_order"] = conditions

    adata.uns["phenotype_column"] = phenotype_column
    adata.uns["clonotype_column"] = clonotype_column
    adata.uns["condition_column"] = condition_column
    adata.uns["sample_column"] = sample_column

    joint_ds = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(dict)))
    cadatas = adata.obs.groupby(condition_column)
    for condition, cadata in cadatas:
        tadatas = cadata.groupby(sample_column)
        
        if cores > 1:
            with multiprocessing.Pool(
                processes=cores, initializer=initializer, initargs=(clonotype_column, phenotype_column, min_clone_size, phenotypes)
            ) as pool:
                results = list(tqdm(pool.imap(process_parallel, tadatas), total=tadatas.ngroups))
            joint_ds[condition] = {sample: _joint_d for sample, _joint_d in results}

        else:
            joint_ds[condition] = {sample : process(sample, tadata, clonotype_column, phenotype_column, min_clone_size, phenotypes)[1] for sample, tadata in tadatas}
    adata.uns["joint_probability_distribution"] = joint_ds
    return joint_ds
