#!/usr/bin/env python3
"""Single-cell anchoring on GSE167363 (10x scRNA-seq), patient level.

Two sepsis patients were sampled at T0 and T6; to avoid pseudoreplication
we first average T0/T6 within patient, so each patient contributes exactly
one data point. With n=2 healthy vs n=5 sepsis patients we report
descriptive statistics only -- no inferential testing at this sample size.

Input: GSE167363_RAW.tar unpacked under --raw/sc/ (per-sample
*_matrix.mtx.gz / *_features.tsv.gz / *_barcodes.tsv.gz triples).

The clustering/UMAP part is memory-hungry; a subsample cap per sample
keeps it runnable on a 16 GB machine.
"""
import os
import glob
import argparse

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import sparse

HC_SAMPLES = ('GSM5102900', 'GSM5102901')   # the two healthy controls

LYM13 = ['IL7R', 'CD3D', 'CD3E', 'CD247', 'LCK', 'ZAP70', 'GIMAP1',
         'GIMAP4', 'CCR7', 'TCF7', 'CD28', 'CD2', 'ITK']


def sample_group(sid):
    return 'HC' if sid.startswith(HC_SAMPLES) else 'Sepsis'


def sample_patient(sid):
    """GSMxxxx_<patient>_T0/T6 -> (patient, timepoint)."""
    parts = sid.split('_')
    tp = parts[-1] if parts[-1] in ('T0', 'T6') else 'T0'
    patient = '_'.join(parts[:-1]) if parts[-1] in ('T0', 'T6') else sid
    return patient, tp


def load_sample(f):
    base = f.replace('_matrix.mtx.gz', '')
    sid = os.path.basename(base)
    mtx = sio.mmread(f).tocsr().T.astype(np.float32)
    feats = pd.read_csv(base + '_features.tsv.gz', sep='\t', header=None,
                        compression='gzip')
    return sid, mtx, feats[1].astype(str).values


def qc_normalize(mtx):
    """Cell QC (200<=nGene<5000, MT<15%), then log-normalise per cell."""
    ncount = np.asarray(mtx.sum(1)).ravel()
    ngene = np.asarray((mtx > 0).sum(1)).ravel()
    keep = (ngene >= 200) & (ngene < 5000) & (ncount > 0)
    m = mtx[keep]
    lib = np.asarray(m.sum(1)).ravel()
    lib[lib == 0] = 1
    norm = sparse.diags(1e4 / lib) @ m
    norm.data = np.log1p(norm.data)
    return norm


def marker_summary(raw, out):
    """Per-sample marker metrics -> averaged to per-patient rows."""
    rows = []
    for f in sorted(glob.glob(os.path.join(raw, 'sc', '*_matrix.mtx.gz'))):
        sid, mtx, genes = load_sample(f)
        norm = qc_normalize(mtx)
        gidx = {g: i for i, g in enumerate(genes)}

        def expr(g):
            if g not in gidx:
                return np.zeros(norm.shape[0])
            return np.asarray(norm[:, gidx[g]].todense()).ravel()

        cd3d, cd14, fcgr3a = expr('CD3D'), expr('CD14'), expr('FCGR3A')
        il7r, hladra = expr('IL7R'), expr('HLA-DRA')
        lym_mat = np.vstack([expr(g) for g in LYM13])
        tmask = cd3d > 0
        mmask = (cd14 > 0) | (fcgr3a > 0)
        patient, tp = sample_patient(sid)
        rows.append({
            'sample': sid, 'patient': patient, 'tp': tp,
            'group': sample_group(sid), 'n_cells': norm.shape[0],
            'T_frac': tmask.mean(),
            'T_IL7R_mean': il7r[tmask].mean() if tmask.sum() else np.nan,
            'T_LymphoScore': lym_mat[:, tmask].mean() if tmask.sum() else np.nan,
            'Mono_HLADRA': hladra[mmask].mean() if mmask.sum() else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, 'per_sample_marker_summary.csv'), index=False)

    # average T0/T6 within patient -> one row per patient
    pat = (df.groupby(['patient', 'group'], as_index=False)
             [['T_frac', 'T_IL7R_mean', 'T_LymphoScore', 'Mono_HLADRA']].mean())
    pat.to_csv(os.path.join(out, 'per_patient_marker_summary.csv'), index=False)
    print(pat.groupby('group')[['T_frac', 'T_IL7R_mean',
                                'T_LymphoScore', 'Mono_HLADRA']].agg(['mean', 'std', 'count']))
    # descriptive only; no group test (n=2 vs n=5 patients)


def cluster_umap(raw, out, cap=1500):
    """Optional: QC -> HVG -> ComBat -> PCA -> KMeans clusters -> UMAP.
    Used only for cell-type annotation sanity checks."""
    import scanpy as sc
    import anndata
    from sklearn.cluster import KMeans

    ads = []
    for f in sorted(glob.glob(os.path.join(raw, 'sc', '*_matrix.mtx.gz'))):
        sid, mtx, genes = load_sample(f)
        barc = pd.read_csv(f.replace('_matrix.mtx.gz', '_barcodes.tsv.gz'),
                           sep='\t', header=None, compression='gzip')[0]
        a = sc.AnnData(X=mtx,
                       obs=pd.DataFrame(index=[sid + '|' + b for b in barc]),
                       var=pd.DataFrame(index=genes))
        a.var_names_make_unique()
        a.obs['sample'] = sid
        sc.pp.filter_cells(a, min_genes=200)
        a = a[a.obs.n_genes < 5000].copy()
        a.var['mt'] = a.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(a, qc_vars=['mt'], inplace=True)
        a = a[a.obs.pct_counts_mt < 15].copy()
        if a.n_obs > cap:
            idx = np.random.default_rng(1).choice(a.n_obs, cap, replace=False)
            a = a[idx].copy()
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        ads.append(a)

    ad = anndata.concat(ads, join='inner')
    del ads
    ad.obs['group'] = np.where(ad.obs['sample'].str.startswith(HC_SAMPLES),
                               'HC', 'Sepsis')
    sc.pp.filter_genes(ad, min_cells=10)
    sc.pp.highly_variable_genes(ad, n_top_genes=1500, batch_key='sample')
    ad = ad[:, ad.var.highly_variable].copy()
    ad.X = ad.X.toarray() if hasattr(ad.X, 'toarray') else ad.X
    sc.pp.scale(ad, max_value=10)
    sc.pp.combat(ad, key='sample')
    sc.tl.pca(ad)
    sc.pp.neighbors(ad, n_pcs=25)
    ad.obs['cluster'] = KMeans(n_clusters=14, random_state=42,
                               n_init=10).fit_predict(
        ad.obsm['X_pca'][:, :25]).astype(str)
    sc.tl.umap(ad)

    markers = {'CD14 mono': ['CD14', 'LST1', 'S100A8'],
               'FCGR3A mono': ['FCGR3A', 'MS4A7'],
               'T cell': ['CD3D', 'CD3E', 'IL7R'],
               'NK': ['NKG7', 'GNLY'],
               'B cell': ['CD79A', 'MS4A1'],
               'Platelet': ['PPBP'],
               'DC': ['FCER1A', 'CST3']}
    scored = []
    for name, gs in markers.items():
        gg = [g for g in gs if g in ad.var_names]
        if gg:
            sc.tl.score_genes(ad, gg, score_name='sc_' + name)
            scored.append('sc_' + name)
    scdf = ad.obs.groupby('cluster', observed=True)[scored].mean()
    lab = {cl: scdf.loc[cl].idxmax().replace('sc_', '') for cl in scdf.index}
    ad.obs['celltype'] = ad.obs['cluster'].map(lab)

    um = pd.DataFrame(ad.obsm['X_umap'], columns=['UMAP1', 'UMAP2'],
                      index=ad.obs.index)
    um['celltype'] = ad.obs['celltype'].values
    um['group'] = ad.obs['group'].values
    um['sample'] = ad.obs['sample'].values
    um.to_csv(os.path.join(out, 'umap.csv'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--out', default='./output')
    ap.add_argument('--skip-umap', action='store_true',
                    help='only compute per-patient marker summary')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    marker_summary(args.raw, args.out)
    if not args.skip_umap:
        cluster_umap(args.raw, args.out)


if __name__ == '__main__':
    main()
