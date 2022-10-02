import numpy
import pandas 
import collections

def calc_entropy(P, log_units = 2):
    P = P[P>0].flatten()
    return numpy.dot(P, -numpy.log(P))/numpy.log(log_units)

def clonotypic_entroy(adata, phenotype):
    clonotype_dist = collections.defaultdict(list)
    for ct, clonotype in zip(adata.obs[adata.uns["phenotype_column"]], adata.obs["IR_VDJ_1_junction_aa"]):
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

def phenotypic_entropy():
    pass


def phenotypic_flux():
    pass