#!/usr/bin/env python3
"""External-cohort assessment: GSE154918 (severity gradient) and
GSE54514 (longitudinal trajectories).

These cohorts do not carry an ICU-AI endpoint, so they are used for
endpoint-hierarchy assessment only, not as independent ICU-AI validation.

Raw files expected under --raw:
  GSE154918_series_matrix.txt.gz, GSE154918_suppl.txt.gz
  GSE54514_series_matrix.txt.gz, illuminaHumanv3.db*/**/*.sqlite
"""
import os
import glob
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from geo_utils import parse_series_matrix, probe2gene, LYM_GENES


def severity_gradient(raw, gene_info, out):
    sup = pd.read_csv(f'{raw}/GSE154918_suppl.txt.gz', sep=' ',
                      compression='gzip').drop_duplicates(
        subset='gene_symbol').set_index('gene_symbol')
    _, pheno = parse_series_matrix(f'{raw}/GSE154918_series_matrix.txt.gz')
    expr = sup[[c for c in sup.columns
                if c.startswith(('INFG_', 'WIMR_'))]].astype(float)
    Z = expr.sub(expr.mean(1), axis=0).div(expr.std(1), axis=0)
    lym = [g for g in LYM_GENES if g in Z.index]
    score = Z.loc[lym].mean(0)

    # severity order from sample titles: Healthy / Infection / Sepsis / Shock
    order = {'Healthy': 0, 'Infection': 1, 'Sepsis': 2, 'Septic shock': 3}
    grp = pheno.set_index('gsm')['group'].map(
        lambda s: next((k for k in order if k.lower() in str(s).lower()), None))
    df = pd.DataFrame({'LymphoScore': score, 'group': grp}).dropna()
    df['ord'] = df.group.map(order)
    rho, p = stats.spearmanr(df['ord'], df['LymphoScore'])
    print('severity gradient: Spearman rho=%.3f, P=%.2e' % (rho, p))
    print(df.groupby('group')['LymphoScore'].mean().reindex(order.keys()))
    df.to_csv(f'{out}/gse154918_lymphoscore.csv', index=False)


def longitudinal(raw, gene_info, out):
    expr_raw, pheno = parse_series_matrix(f'{raw}/GSE54514_series_matrix.txt.gz')
    sqlite_path = glob.glob(f'{raw}/illuminaHumanv3*/**/*.sqlite',
                            recursive=True)[0]
    X = probe2gene(expr_raw, sqlite_path, gene_info)
    Z = X.sub(X.mean(1), axis=0).div(X.std(1), axis=0)

    pheno['grp'] = pheno['group_day'].str.split('_').str[0]      # S / NS / HC
    pheno['dayn'] = pheno['group_day'].str.split('_').str[1].str[1].astype(int)
    lym = [g for g in LYM_GENES if g in Z.index]
    pheno['LymphoScore'] = pheno['gsm'].map(Z.loc[lym].mean(0))

    # per-patient slope over time; require >=3 time points
    rows = []
    for pid, d in pheno.groupby('patient'):
        d = d.dropna(subset=['LymphoScore']).sort_values('dayn')
        if len(d) < 3 or d['dayn'].nunique() < 3:
            continue
        slope = np.polyfit(d['dayn'], d['LymphoScore'], 1)[0]
        rows.append({'patient': pid, 'group': d['grp'].iloc[0],
                     'slope': slope, 'n_tp': len(d)})
    slopes = pd.DataFrame(rows)
    surv = slopes[slopes.group == 'S']['slope']
    nons = slopes[slopes.group == 'NS']['slope']
    p = stats.mannwhitneyu(surv, nons).pvalue
    print('slopes/day: survivors %.3f (n=%d) vs non-survivors %.3f (n=%d), MWU P=%.3f'
          % (surv.mean(), len(surv), nons.mean(), len(nons), p))
    slopes.to_csv(f'{out}/gse54514_slopes.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--out', default='./output')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    gene_info = pd.read_csv(f'{args.raw}/Homo_sapiens.gene_info.gz',
                            sep='\t', dtype=str)
    severity_gradient(args.raw, gene_info, args.out)
    longitudinal(args.raw, gene_info, args.out)


if __name__ == '__main__':
    main()
