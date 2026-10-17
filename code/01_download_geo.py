#!/usr/bin/env python3
"""Download the GEO files used in this study.

GEO FTP throttles single connections, so we fetch in byte-range chunks.
Run once before the analysis scripts:

    python 01_download_geo.py --outdir /path/to/raw
"""
import os
import argparse
import urllib.request
import concurrent.futures

URLS = {
    # discovery cohort (MARS), Affymetrix U219 / GPL13667
    'GSE65682_series_matrix.txt.gz':
        'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE65nnn/GSE65682/matrix/GSE65682_series_matrix.txt.gz',
    # longitudinal cohort, Illumina HT-12 v3 / GPL6947
    'GSE54514_series_matrix.txt.gz':
        'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE54nnn/GSE54514/matrix/GSE54514_series_matrix.txt.gz',
    # severity-gradient cohort (RNA-seq)
    'GSE154918_series_matrix.txt.gz':
        'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE154nnn/GSE154918/matrix/GSE154918_series_matrix.txt.gz',
    'GSE154918_suppl.txt.gz':
        'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE154nnn/GSE154918/suppl/GSE154918_Schughart_Sepsis_200320.txt.gz',
    # single-cell anchoring cohort (10x)
    'GSE167363_RAW.tar':
        'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE167nnn/GSE167363/suppl/GSE167363_RAW.tar',
    # probe annotation: NCBI gene info + Bioconductor ChipDb sources
    'Homo_sapiens.gene_info.gz':
        'https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz',
    'hgu219.db.tar.gz':
        'https://bioconductor.org/packages/3.23/data/annotation/src/contrib/hgu219.db_3.2.3.tar.gz',
    'illuminaHumanv3.db.tar.gz':
        'https://bioconductor.org/packages/3.23/data/annotation/src/contrib/illuminaHumanv3.db_1.26.0.tar.gz',
}

HDR = {'User-Agent': 'Mozilla/5.0'}


def fetch_chunk(url, start, end, tries=6):
    h = dict(HDR)
    h['Range'] = f'bytes={start}-{end}'
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers=h)
            return urllib.request.urlopen(req, timeout=600).read()
        except Exception:
            if a == tries - 1:
                raise
    return b''


def parallel_download(url, out, nchunks=16):
    req = urllib.request.Request(url, method='HEAD', headers=HDR)
    size = int(urllib.request.urlopen(req, timeout=30).headers['Content-Length'])
    chunk = size // nchunks + 1
    ranges = [(i * chunk, min((i + 1) * chunk - 1, size - 1)) for i in range(nchunks)]
    with concurrent.futures.ThreadPoolExecutor(nchunks) as ex:
        parts = list(ex.map(lambda se: fetch_chunk(url, *se), ranges))
    with open(out, 'wb') as f:
        for p in parts:
            f.write(p)
    print(f'{out}: {size / 1e6:.1f} MB')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--nchunks', type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    for fn, url in URLS.items():
        out = os.path.join(args.outdir, fn)
        if os.path.exists(out):
            print(f'skip {fn} (exists)')
            continue
        parallel_download(url, out, args.nchunks)
    # ChipDb tarballs need unpacking so the .sqlite files are reachable
    import tarfile
    for fn in ('hgu219.db.tar.gz', 'illuminaHumanv3.db.tar.gz'):
        with tarfile.open(os.path.join(args.outdir, fn)) as tf:
            tf.extractall(args.outdir)


if __name__ == '__main__':
    main()
