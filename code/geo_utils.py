#!/usr/bin/env python3
"""Shared helpers: GEO series-matrix parsing and probe-to-gene mapping."""
import io
import gzip
import sqlite3

import pandas as pd

LYM_GENES = ['IL7R', 'CD3D', 'CD3E', 'CD247', 'LCK', 'ZAP70', 'GIMAP1',
             'GIMAP4', 'CCR7', 'TCF7', 'CD28', 'CD2', 'ITK', 'GIMAP5',
             'GIMAP6', 'SKAP1', 'THEMIS']


def parse_series_matrix(path):
    """Return (expression matrix [probe x sample], phenotype DataFrame)."""
    titles = accs = None
    chars, tbl = [], []
    intable = False
    with gzip.open(path, 'rt', errors='replace') as f:
        for line in f:
            if line.startswith('!Sample_title'):
                titles = [x.strip('"') for x in line.rstrip('\n').split('\t')[1:]]
            elif line.startswith('!Sample_geo_accession'):
                accs = [x.strip('"') for x in line.rstrip('\n').split('\t')[1:]]
            elif line.startswith('!Sample_characteristics_ch1'):
                chars.append([x.strip('"') for x in line.rstrip('\n').split('\t')[1:]])
            elif line.startswith('!series_matrix_table_begin'):
                intable = True
                header = [h.strip('"') for h in next(f).rstrip('\n').split('\t')]
            elif line.startswith('!series_matrix_table_end'):
                break
            elif intable:
                tbl.append(line)
    pheno = pd.DataFrame({'gsm': accs, 'title': titles})
    for row in chars:
        key = row[0].split(':')[0]
        pheno[key] = [x.split(':', 1)[1].strip() if ':' in x else x for x in row]
    expr = pd.read_csv(io.StringIO(''.join(tbl)), sep='\t', header=None,
                       names=header, index_col=0)
    return expr, pheno


def probe2gene(expr, sqlite_path, gene_info):
    """ChipDb probe->Entrez, gene_info Entrez->Symbol; average by symbol."""
    con = sqlite3.connect(sqlite_path)
    probes = pd.read_sql("SELECT * FROM probes", con)
    con.close()
    probes['symbol'] = probes['gene_id'].map(
        dict(zip(gene_info['GeneID'], gene_info['Symbol'])))
    pm = expr.index.to_series().map(dict(zip(probes['probe_id'], probes['symbol'])))
    return expr.astype(float).groupby(pm).mean()
