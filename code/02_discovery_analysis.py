#!/usr/bin/env python3
"""Discovery-cohort analysis (GSE65682 / MARS).

Steps:
  1. parse series matrix, map probes to gene symbols
  2. build ICU-AI labels and the 17-gene LymphoScore
  3. whole-cohort differential expression (Welch t + BH FDR)
  4. whole-cohort 73-gene elastic-net panel with nested CV (negative control)
  5. CAP/HAP subgroup: candidate-gene tests, logistic models, Firth
     correction, bootstrap optimism correction, LOO, Cook's distance,
     Hosmer-Lemeshow, LymphoScore x pneumonia-type interaction
  6. secondary outcome: 28-day mortality

Raw data layout expected under --raw (see 01_download_geo.py):
  GSE65682_series_matrix.txt.gz
  Homo_sapiens.gene_info.gz
  hgu219.db*/**/*.sqlite
"""
import os
import glob
import argparse

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from geo_utils import parse_series_matrix, probe2gene, LYM_GENES

# prior panels for the whole-cohort elastic-net screen
PANEL = {
    'AgPresentation': ['HLA-DRA', 'HLA-DRB1', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1',
                       'HLA-DQB1', 'HLA-DMA', 'HLA-DMB', 'CD74', 'RFX5', 'RFXANK'],
    'Tcell': ['CD3D', 'CD3E', 'CD28', 'CD2', 'LCK', 'ZAP70', 'GIMAP1', 'GIMAP4',
              'IL7R', 'CCR7', 'TCF7', 'LEF1', 'ITK', 'CD247', 'CD8A', 'CD4',
              'GZMK', 'PRF1', 'GZMB', 'NKG7', 'GNLY', 'KLRB1'],
    'MDSC': ['S100A8', 'S100A12', 'MMP8', 'MMP9', 'RETN', 'ARG1', 'CEACAM8',
             'LTF', 'LCN2', 'DEFA4', 'ELANE', 'OLFM4', 'CAMP', 'CRISP3',
             'MCEMP1', 'ANXA3', 'VNN2', 'IL1R2', 'CD163', 'MSR1'],
    'Checkpoint': ['CD274', 'PDCD1LG2', 'IL10', 'IL10RA', 'TGFB1', 'LGALS9',
                   'VSIR', 'HAVCR2', 'LAG3', 'CTLA4', 'PDCD1', 'TIGIT', 'IDO1'],
    'Monocyte': ['CD14', 'ITGAM', 'FCN1', 'VCAN', 'MS4A6A', 'CSF1R', 'FCGR3A'],
}


# ---------------------------------------------------------------- Firth logit
def _ploglik(X, y, beta):
    """Penalised (Jeffreys) log-likelihood."""
    eta = X @ beta
    p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    W = p * (1 - p)
    sign, logdet = np.linalg.slogdet(X.T @ (X * W[:, None]))
    if sign <= 0:
        return -np.inf
    return ll + 0.5 * logdet


def firth_logit(X, y, tol=1e-8, max_iter=200):
    """Firth logistic regression via modified IRLS.

    Score equation  U* = X'(y - p + h*(0.5 - p))  with h the diagonal of
    the hat matrix  H = W^1/2 X (X'WX)^-1 X' W^1/2.
    X must include the intercept. Returns beta, SEs and penalised ll.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta
        p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1 - 1e-12)
        W = p * (1 - p)
        XtWX = X.T @ (X * W[:, None])
        XtWX_inv = np.linalg.inv(XtWX)
        Wh = np.sqrt(W)
        H = (X * Wh[:, None]) @ XtWX_inv @ (X * Wh[:, None]).T
        h = np.diag(H)
        Ustar = X.T @ (y - p + h * (0.5 - p))
        step = XtWX_inv @ Ustar
        beta_new = beta + step
        # step-halving if penalised ll does not improve
        while _ploglik(X, y, beta_new) < _ploglik(X, y, beta):
            step *= 0.5
            beta_new = beta + step
            if np.max(np.abs(step)) < 1e-10:
                break
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    W = p * (1 - p)
    cov = np.linalg.inv(X.T @ (X * W[:, None]))
    se = np.sqrt(np.diag(cov))
    return beta, se, _ploglik(X, y, beta)


def firth_profile_ci(X, y, beta, j, alpha=0.05):
    """Profile penalised-likelihood CI for coefficient j."""
    from scipy.stats import chi2
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    crit = _ploglik(X, y, beta) - chi2.ppf(1 - alpha, 1) / 2.0
    others = [k for k in range(X.shape[1]) if k != j]

    def pll_at(bj):
        # maximise over the other coefficients with beta[j] fixed:
        # IRLS on the sub-design with offset bj * X[:, j]
        off = X[:, j] * bj
        Xs = X[:, others]
        bs = beta[others].copy()
        for _ in range(200):
            eta = off + Xs @ bs
            p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1 - 1e-12)
            W = p * (1 - p)
            A = Xs.T @ (Xs * W[:, None])
            Ainv = np.linalg.pinv(A)
            Wh = np.sqrt(W)
            H = (Xs * Wh[:, None]) @ Ainv @ (Xs * Wh[:, None]).T
            h = np.diag(H)
            Ustar = Xs.T @ (y - p + h * (0.5 - p))
            bs_new = bs + Ainv @ Ustar
            if np.max(np.abs(bs_new - bs)) < 1e-8:
                bs = bs_new
                break
            bs = bs_new
        b = np.empty(X.shape[1])
        b[j] = bj
        b[others] = bs
        return _ploglik(X, y, b)

    def find(direction):
        lo, hi = beta[j], beta[j] + direction * 0.05
        for _ in range(100):
            if pll_at(hi) < crit:
                break
            hi += direction * 0.05
        else:
            return np.nan
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if pll_at(mid) < crit:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    return find(-1), find(1)


def logit_table(df, ycol, xcols, firth=True):
    """Ordinary MLE logit OR + CI, and Firth OR + profile CI."""
    d = df[[ycol] + xcols].dropna()
    X = sm.add_constant(d[xcols].astype(float).values)
    y = d[ycol].values.astype(float)
    m = sm.Logit(y, X).fit(disp=0)
    out = {}
    for i, name in enumerate(['const'] + xcols):
        out[name] = {'OR': np.exp(m.params[i]),
                     'lo': np.exp(m.conf_int()[i, 0]),
                     'hi': np.exp(m.conf_int()[i, 1]),
                     'P': m.pvalues[i]}
    if firth:
        beta, se, pll = firth_logit(X, y)
        for i, name in enumerate(['const'] + xcols):
            lo, hi = firth_profile_ci(X, y, beta, i)
            z = beta[i] / se[i]
            out[name].update({'firth_OR': np.exp(beta[i]),
                              'firth_lo': np.exp(lo),
                              'firth_hi': np.exp(hi),
                              'firth_P': 2 * stats.norm.sf(abs(z))})
    return out


# ------------------------------------------------------- bootstrap optimism
def boot_auc(df, xcols, B=500, seed=42):
    """Apparent AUC, bootstrap percentile CI, optimism-corrected AUC."""
    d = df[['ICUA'] + xcols].dropna()
    X = StandardScaler().fit_transform(d[xcols].values)
    y = d['ICUA'].values.astype(int)
    rng = np.random.default_rng(seed)
    full = LogisticRegression(C=1.0, max_iter=5000).fit(X, y)
    app = roc_auc_score(y, full.predict_proba(X)[:, 1])
    opts, boots = [], []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        mb = LogisticRegression(C=1.0, max_iter=5000).fit(X[idx], y[idx])
        pa = roc_auc_score(y[idx], mb.predict_proba(X[idx])[:, 1])
        pt = roc_auc_score(y, mb.predict_proba(X)[:, 1])
        opts.append(pa - pt)
        boots.append(pa)
    ci = np.percentile(boots, [2.5, 97.5])
    return {'apparent': app, 'ci_lo': ci[0], 'ci_hi': ci[1],
            'optimism': np.mean(opts), 'corrected': app - np.mean(opts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='folder with downloaded GEO files')
    ap.add_argument('--out', default='./output')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    RAW = args.raw

    gene_info = pd.read_csv(f'{RAW}/Homo_sapiens.gene_info.gz', sep='\t', dtype=str)

    # ---- 1-2. expression matrix, labels, LymphoScore ----
    expr_raw, pheno = parse_series_matrix(f'{RAW}/GSE65682_series_matrix.txt.gz')
    sqlite_path = glob.glob(f'{RAW}/hgu219*/**/*.sqlite', recursive=True)[0]
    X65 = probe2gene(expr_raw, sqlite_path, gene_info)

    pheno['ICUA'] = pheno['icu_acquired_infection'].map({'ICUA': 1, 'No_ICUA': 0})
    pheno['death28'] = pd.to_numeric(
        pheno['mortality_event_28days'].replace('NA', np.nan), errors='coerce')
    sub = pheno[pheno['ICUA'].notna()].copy()          # n=346 (50 ICU-AI)
    sub['pna'] = sub['pneumonia diagnoses']            # cap / hap
    samples = sub['gsm'].tolist()
    Z65 = X65[samples].sub(X65[samples].mean(1), axis=0).div(X65[samples].std(1), axis=0)

    lym = [g for g in LYM_GENES if g in X65.index]
    sub['LymphoScore'] = Z65.loc[lym, sub['gsm']].mean(0).values

    # ---- 3. whole-cohort differential expression ----
    g1 = sub[sub.ICUA == 1]['gsm'].tolist()
    g0 = sub[sub.ICUA == 0]['gsm'].tolist()
    t, p = stats.ttest_ind(X65[g1].values, X65[g0].values, axis=1, equal_var=False)
    de = pd.DataFrame({'gene': X65.index,
                       'log2FC': X65[g1].mean(1) - X65[g0].mean(1),
                       't': t, 'P': p})
    de['FDR'] = multipletests(de.P.fillna(1), method='fdr_bh')[1]
    de.to_csv(f'{args.out}/whole_cohort_DE.csv', index=False)
    # -> no gene passes FDR<0.05, consistent with the original MARS report

    # ---- 4. whole-cohort elastic-net panel, nested CV ----
    allgenes = [g for v in PANEL.values() for g in v if g in X65.index]
    Xm = Z65.loc[allgenes, samples].T.values
    y = sub.set_index('gsm').loc[samples, 'ICUA'].values.astype(int)
    outer = StratifiedKFold(5, shuffle=True, random_state=42)
    aucs = []
    for tr, te in outer.split(Xm, y):
        mdl = LogisticRegressionCV(Cs=20, penalty='elasticnet', solver='saga',
                                   l1_ratios=[0.5, 0.9, 1.0], cv=5,
                                   scoring='roc_auc', max_iter=20000,
                                   random_state=42)
        mdl.fit(Xm[tr], y[tr])
        aucs.append(roc_auc_score(y[te], mdl.predict_proba(Xm[te])[:, 1]))
    print('nested-CV AUC: %.3f +/- %.3f' % (np.mean(aucs), np.std(aucs)))
    # -> ~0.47, baseline whole-cohort transcriptome does not predict ICU-AI

    # ---- 5. CAP/HAP subgroup ----
    pna = sub[sub['pna'].isin(['cap', 'hap'])].copy()
    cap = pna[pna.pna == 'cap'].copy()
    cap['age_n'] = pd.to_numeric(cap['age'], errors='coerce')
    cap['male'] = (cap['gender'] == 'male').astype(int)
    cap['dm'] = (cap['diabetes_mellitus'] == 'DM').astype(int)
    cap = cap.dropna(subset=['age_n'])

    # candidate genes (primary: IL7R, CD3D, PRKACB; exploratory: CFLAR, TNFAIP6)
    cand = []
    for g in ['IL7R', 'CD3D', 'PRKACB', 'CFLAR', 'TNFAIP6']:
        a = X65.loc[g, cap[cap.ICUA == 1]['gsm']]
        b = X65.loc[g, cap[cap.ICUA == 0]['gsm']]
        cand.append({'gene': g, 'log2FC': a.mean() - b.mean(),
                     'P': stats.mannwhitneyu(a, b).pvalue})
    pd.DataFrame(cand).to_csv(f'{args.out}/cap_candidate_genes.csv', index=False)

    # logistic models: univariable / parsimonious (+age) / full (+sex+diabetes)
    models = {'univariable': ['LymphoScore'],
              'parsimonious': ['LymphoScore', 'age_n'],
              'full': ['LymphoScore', 'age_n', 'male', 'dm']}
    rows = []
    for name, xcols in models.items():
        tb = logit_table(cap, 'ICUA', xcols)['LymphoScore']
        bs = boot_auc(cap, xcols)
        rows.append({'model': name, 'EPV': 11 / len(xcols),
                     'OR': tb['OR'], 'OR_lo': tb['lo'], 'OR_hi': tb['hi'],
                     'P': tb['P'],
                     'firth_OR': tb['firth_OR'], 'firth_lo': tb['firth_lo'],
                     'firth_hi': tb['firth_hi'],
                     **bs})
    pd.DataFrame(rows).to_csv(f'{args.out}/cap_models.csv', index=False)

    # leave-one-out
    loo = []
    for i in cap.index:
        m = sm.Logit(cap.drop(i)['ICUA'],
                     sm.add_constant(cap.drop(i)[['LymphoScore', 'age_n']]
                                     .astype(float))).fit(disp=0)
        loo.append(np.exp(m.params['LymphoScore']))
    print('LOO OR range: %.3f - %.3f' % (min(loo), max(loo)))

    # Cook's distance sensitivity
    Xd = sm.add_constant(cap[['LymphoScore', 'age_n']].astype(float))
    mf = sm.Logit(cap['ICUA'], Xd).fit(disp=0)
    infl = mf.get_influence()
    cooks = infl.cooks_distance[0]
    thr = 4.0 / len(cap)
    keep = cooks <= thr
    m_exc = sm.Logit(cap.loc[keep, 'ICUA'], Xd[keep]).fit(disp=0)
    print('excluding %d influential points (Cook>4/n): OR=%.3f, P=%.3f'
          % ((~keep).sum(), np.exp(m_exc.params['LymphoScore']),
             m_exc.pvalues['LymphoScore']))
    worst = np.argmax(cooks)
    m_w = sm.Logit(cap['ICUA'].drop(cap.index[worst]),
                   Xd.drop(cap.index[worst])).fit(disp=0)
    print('excluding largest Cook point: OR=%.3f, P=%.3f'
          % (np.exp(m_w.params['LymphoScore']), m_w.pvalues['LymphoScore']))

    # Hosmer-Lemeshow (5 groups)
    pr = mf.predict(Xd)
    grp = pd.qcut(pr, 5, duplicates='drop')
    hl = pd.DataFrame({'y': cap['ICUA'].values, 'p': pr, 'g': grp})
    agg = hl.groupby('g', observed=True).agg(obs=('y', 'sum'), exp=('p', 'sum'),
                                             n=('y', 'size'))
    hlstat = ((agg.obs - agg.exp) ** 2 / (agg.exp * (1 - agg.exp / agg.n))).sum()
    print('Hosmer-Lemeshow chi2=%.2f, P=%.3f'
          % (hlstat, 1 - stats.chi2.cdf(hlstat, len(agg) - 2)))

    # interaction: LymphoScore x pneumonia type, age-adjusted (n=144)
    pna['age_n'] = pd.to_numeric(pna['age'], errors='coerce')
    pnad = pna.dropna(subset=['age_n']).copy()
    pnad['is_cap'] = (pnad.pna == 'cap').astype(int)
    pnad['inter'] = pnad['LymphoScore'] * pnad['is_cap']
    m0 = sm.Logit(pnad['ICUA'],
                  sm.add_constant(pnad[['LymphoScore', 'is_cap', 'age_n']]
                                  .astype(float))).fit(disp=0)
    m1 = sm.Logit(pnad['ICUA'],
                  sm.add_constant(pnad[['LymphoScore', 'is_cap', 'inter', 'age_n']]
                                  .astype(float))).fit(disp=0)
    lrt = 2 * (m1.llf - m0.llf)
    print('interaction LRT chi2=%.2f, P=%.4f' % (lrt, 1 - stats.chi2.cdf(lrt, 1)))

    # HAP subgroup estimate (descriptive; 9 events)
    hap = pnad[pnad.pna == 'hap']
    if hap['ICUA'].sum() > 0:
        mh = sm.Logit(hap['ICUA'],
                      sm.add_constant(hap[['LymphoScore', 'age_n']]
                                      .astype(float))).fit(disp=0)
        print('HAP OR=%.3f, P=%.3f' % (np.exp(mh.params['LymphoScore']),
                                       mh.pvalues['LymphoScore']))

    # ---- 6. secondary outcome: 28-day mortality ----
    all_d = sub.dropna(subset=['death28']).copy()
    all_d['age_n'] = pd.to_numeric(all_d['age'], errors='coerce')
    all_d['male'] = (all_d['gender'] == 'male').astype(int)
    m_mort = sm.Logit(all_d['death28'],
                      sm.add_constant(all_d[['LymphoScore', 'age_n', 'male']]
                                      .astype(float))).fit(disp=0)
    print('28-day mortality: LymphoScore OR=%.3f, P=%.3f'
          % (np.exp(m_mort.params['LymphoScore']),
             m_mort.pvalues['LymphoScore']))

    # MARS paired-sample feasibility check (baseline-event pairs too few)
    paired = pheno.get('paired sampling', pd.Series(dtype=str))
    print('paired-sampling annotation counts:')
    print(paired.value_counts(dropna=False))


if __name__ == '__main__':
    main()
