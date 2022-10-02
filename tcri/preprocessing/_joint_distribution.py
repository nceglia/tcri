import scanpy as sc
import numpy as np
import pandas
import sys
import tqdm
import numpy
import collections

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

def joint_distribution(adata, condition_column="condition", sample_column="sample",
                       ir_column="IR_VDJ_1_junction_aa", phenotype_column="phenotype", min_clone_size=1):
    if ir_column not in adata.obs:
        raise ValueError("{} not found in adata.obs.keys".format(ir_column))
    if phenotype_column not in adata.obs:
        raise ValueError("{} not found in adata.obs.keys".format(phenotype))
    if condition_column not in adata.obs:
        raise ValueError("{} not found in adata.obs.keys".format(condition_column))
    if sample_column not in adata.obs:
        raise ValueError("{} not found in adata.obs.keys".format(sample_column))
    phenotypes = list(set(adata.obs[phenotype_column]))
    clonotypes = list(set(adata.obs[ir_column]))
    samples = list(set(adata.obs[sample_column]))
    conditions = list(set(adata.obs[condition_column]))

    adata.uns["phenotype_order"] = phenotypes
    adata.uns["clonotype_order"] = clonotypes
    adata.uns["sample_order"] = samples
    adata.uns["condition_order"] = conditions


    adata.uns["phenotype_column"] = phenotype_column
    adata.uns["clonotype_column"] = ir_column
    adata.uns["condition_column"] = condition_column
    adata.uns["sample_column"] = sample_column

    joint_ds = collections.defaultdict(lambda : collections.defaultdict(lambda : collections.defaultdict(dict)))
    for condition in conditions:
        cadata = adata[adata.obs[condition_column]==condition]
        for sample in list(set(cadata.obs[sample_column])):
            tadata = cadata[cadata.obs[sample_column]==sample]
            num_cells = 0
            joint_d = []
            clones = []
            for clonotype in tqdm.tqdm(list(set(tadata.obs[ir_column].tolist())), ncols=50):
                sub = tadata[tadata.obs[ir_column]==clonotype]
                if len(sub.obs.index.tolist()) < min_clone_size: continue
                dist = []
                for phenotype in adata.uns["phenotype_order"]:
                    count = sum(sub.obs[phenotype+" Pseudo-probability"].tolist())
                    dist.append(count)
                    num_cells += count
                joint_d.append(dist)
                clones.append(clonotype)
            joint_d = numpy.array(joint_d)# / num_cells
            _joint_d = dict()
            for clone, dist in zip(clones,joint_d):
                _joint_d[clone] = dist
            joint_ds[condition][sample] = _joint_d
    adata.uns["joint_probability_distribution"] = joint_ds