import numpy
import pandas 
import collections
import tqdm

def calc_entropy(P, log_units = 2):
    P = P[P>0].flatten()
    return numpy.dot(P, -numpy.log(P))/numpy.log(log_units)

def clonotypic_entropy(adata, phenotype):
    clonotype_dist = collections.defaultdict(list)
    for ct, clonotype in zip(adata.obs[adata.uns["phenotype_column"]], adata.obs[adata.uns["clonotype_column"]]):
        if ct == phenotype:
            prob = 1.0
        else:
            prob = 0.0
        clonotype_dist[clonotype].append(prob)
    clonotype_expected_cells = dict()
    total_sum = 0
    for clone, ps in clonotype_dist.items():
        clonotype_expected_cells[clone] = sum(ps)
        total_sum += sum(ps)
    phenotype_entropy = []
    for clone, psum in clonotype_expected_cells.items():
        phenotype_entropy.append(psum/total_sum)
    return calc_entropy(numpy.array(phenotype_entropy))  / numpy.log2(total_sum)

def phenotypic_entropy(adata, min_clone_size=3):
    entropies = dict()
    clonotypes = adata.obs[adata.uns["clonotype_column"]].tolist()
    for target_clone in set(clonotypes):
        if clonotypes.count(target_clone) < min_clone_size: continue
        phenotype_dist = collections.defaultdict(list)
        for ct, clonotype in zip(adata.obs[adata.uns["phenotype_column"]], adata.obs[adata.uns["clonotype_column"]]):
            if clonotype == target_clone:
                prob = 1.0
            else:
                prob = 0.0
            phenotype_dist[ct].append(prob)
        expected_cells = dict()
        total_sum = 0
        for clone, ps in phenotype_dist.items():
            expected_cells[clone] = sum(ps)
            total_sum += sum(ps)
        phenotype_entropy = []
        for clone, psum in expected_cells.items():
            if psum == 0.0:
                continue
            else:
                phenotype_entropy.append(psum/total_sum)
        entropies[target_clone] = calc_entropy(numpy.array(phenotype_entropy)) / numpy.log2(total_sum) 
    return entropies


def phenotypic_flux(adata, from_this="Pre", to_that="Post", min_clone_size=3):
    precounts = collections.defaultdict(lambda : collections.defaultdict(int))
    postcounts = collections.defaultdict(lambda : collections.defaultdict(int))
    phenotype_column = adata.uns["phenotype_column"]
    clonotypes = adata.obs.groupby(adata.obs[adata.uns["clonotype_column"]])
    for clone, sub in tqdm.tqdm(clonotypes):
        if str(clone) == "nan": continue
        if len(sub.index) < min_clone_size: continue
        pre = sub[sub[adata.uns["condition_column"]] == from_this].value_counts(phenotype_column)
        post = sub[sub[adata.uns["condition_column"]] == to_that].value_counts(phenotype_column)
        for ph_pre, precount in pre.items():
            for ph_post, postcount in post.items():
                precounts[ph_pre][ph_post] += precount
                postcounts[ph_pre][ph_post] += postcount

    table = dict()
    table["Pre"] = []
    table["Post"] = []
    table["Phenotype A"] = []
    table["Phenotype B"] = []
    for ph_pre, posts in precounts.items():
        for ph_post, values in posts.items():
            table["Pre"].append(values)
            table["Post"].append(postcounts[ph_pre][ph_post])
            table["Phenotype A"].append(ph_pre)
            table["Phenotype B"].append(ph_post)
    table = pandas.DataFrame.from_dict(table)
    table["Pre"] = table["Pre"] / table["Pre"].sum()
    table["Post"] = table["Post"] / table["Post"].sum()
    return table