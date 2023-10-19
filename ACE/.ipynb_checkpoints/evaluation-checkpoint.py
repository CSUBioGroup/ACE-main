import numpy as np
import pandas as pd
import scanpy as sc
from src.nmiari import nmi, ari
from src.clustering import opt_louvain
from src.matching import eval_FOSCTTM, eval_ACC, eval_matching_score
from src.matching import eval_FOSCTTM_above2, eval_matching_score_above2
from datetime import datetime
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.cluster import silhouette_samples, silhouette_score
from scib.metrics import lisi

def print_results(results, exclude_keys=['df_lisi']):
    for k, v in results.items():
        if k not in exclude_keys:
            print(f'{k}={v:.5f}')

def eval_clustering(
        adata, 
        label_key='cell_type', 
        cluster_key='cluster', 
        resolutions=None,
        use_rep='X_emb', use_neighbors=False,
        use='ari',
        nmi_method='arithmetic',
        nmi_dir=None,
        verbose=False
    ):
    adata.obs[label_key] = adata.obs[label_key].astype('category')

    res_max, nmi_max, nmi_all = opt_louvain(
            adata,
            label_key=label_key,
            cluster_key=cluster_key,
            use_rep=use_rep, use_neighbors=use_neighbors,
            function=ari if use=='ari' else nmi,
            resolutions=resolutions,
            plot=False,
            verbose=verbose,
            inplace=True,
            force=True
        )
    nmi_score = nmi(
            adata,
            group1=cluster_key,
            group2=label_key,
            method=nmi_method,
            nmi_dir=nmi_dir
        )
    ari_score = ari(
            adata,
            group1=cluster_key,
            group2=label_key
        )
    return nmi_score, ari_score

def eval_GC(adata, label_key, use_rep='X_emb'):
    """"
    Quantify how connected the subgraph corresponding to each batch cluster is.
    Calculate per label: #cells_in_largest_connected_component/#all_cells
    Final score: Average over labels

    :param adata: adata with computed neighborhood graph
    :param label_key: name in adata.obs containing the cell identity labels
    """
    adata.obs[label_key] = adata.obs[label_key].astype('category')

    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=15, use_rep=use_rep)
        print(f'Compute neighbors using {use_rep}')

    clust_res = []
    for label in adata.obs[label_key].cat.categories:
        adata_sub = adata[adata.obs[label_key].isin([label])]
        _, labels = connected_components(
            adata_sub.obsp['connectivities'],
            connection='strong'
        )
        tab = pd.value_counts(labels)
        clust_res.append(tab.max() / sum(tab))

    return np.mean(clust_res)


def eval_lisi(
        adata,
        batch_keys=['domain', 'batch'],
        use_rep='X_emb', use_neighbors=False,
    ):
    res = {}
    for key in batch_keys:
        adata.obs[key] = adata.obs[key].astype('category')

        _lisi = lisi.ilisi_graph(
            adata,
            key,
            'embed' if not use_neighbors else 'knn',
            use_rep=use_rep,
            k0=90,
            subsample=None,
            scale=True,
            n_cores=1,
            verbose=False,
        )
        res[key] = _lisi
    df = pd.DataFrame.from_dict(res, orient='index').T
    df.columns = [_+'_LISI' for _ in df.columns]
    return df

def eval_bridge(
        adata1, adata2,
        label_key='cell_type',
        batch_key='batch',
        use_rep='X_emb',
        use_fosc=True, use_acc=False, use_score=True,
    ):
    # foscttm
    fosc, _dist = eval_FOSCTTM(adata1, adata2, use_rep=use_rep, return_dist=True)
    results = {'FOSCTTM': fosc}

    if use_acc:
        # acc
        acc1 = eval_ACC(
            scores=-_dist, 
            label_x=adata1.obs[label_key].to_numpy(), 
            label_y=adata2.obs[label_key].to_numpy(),
            K=1
        )
        acc5, knn_acc_x, knn_acc_y = eval_ACC(
            scores=-_dist, 
            label_x=adata1.obs[label_key].to_numpy(), 
            label_y=adata2.obs[label_key].to_numpy(),
            K=5
        )
        results.update({'ACC_top1':acc1, 'ACC_top5':acc5, 'knn_acc_x':knn_acc_x, 'knn_acc_y':knn_acc_y})

    if use_score:
        score = eval_matching_score(adata1, adata2, split_by=batch_key, k=1, use_rep=use_rep)
        results.update({'Match_score':score})

    print_results(results)
    return results

def eval_bridge_above2(
        adatas,
        label_key='cell_type',
        batch_key='batch',
        mod_key='mod',
        use_rep='X_emb',
        use_fosc=True, use_acc=False, use_score=True,
    ):
    # foscttm
    fosc_dict = eval_FOSCTTM_above2(adatas, use_rep=use_rep, mod_key=mod_key, return_dist=False)
    fosc_dict_vs = [v for k,v in fosc_dict.items()]
    results = fosc_dict
    results.update({'FOSCTTM':np.mean(fosc_dict_vs)})

    if use_score:
        score_dict = eval_matching_score_above2(adatas, split_by=batch_key, mod_key=mod_key, k=1, use_rep=use_rep)
        score_dict_vs = [v for k,v in score_dict.items()]
        results.update(score_dict)
        results.update({'Match_score':np.mean(score_dict_vs)})

    print_results(results)
    return results
