# LymphoScore and ICU-acquired infection in CAP-induced sepsis

Analysis code for the manuscript:

> Lymphocyte Immunosuppression Transcriptomic Score and ICU-Acquired
> Infection in CAP-Induced Sepsis: a secondary analysis of public
> transcriptomic cohorts

The study tests whether a 17-gene lymphocyte function score (LymphoScore)
measured in whole blood at ICU admission is associated with subsequent
ICU-acquired infection (ICU-AI), using public transcriptomic cohorts.

## Data

All data are public; nothing is redistributed here.

| Cohort | Accession | Role |
|---|---|---|
| MARS | [GSE65682](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65682) | discovery cohort (802 samples; 346 patients with ICU-AI annotation) |
| — | [GSE154918](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE154918) | severity-gradient assessment |
| — | [GSE54514](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE54514) | longitudinal trajectories |
| — | [GSE167363](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE167363) | single-cell anchoring |

GSE154918, GSE54514 and GSE167363 have no ICU-AI endpoint; they are used
for endpoint-hierarchy assessment only, not as independent ICU-AI
validation.

## Layout

```
code/
  01_download_geo.py               fetch all raw files from GEO/NCBI/Bioconductor
  geo_utils.py                     series-matrix parsing, probe-to-gene mapping
  02_discovery_analysis.py         GSE65682: LymphoScore, DE, models, sensitivity analyses
  03_external_cohorts.py           GSE154918 severity gradient, GSE54514 slopes
  04_singlecell_patient_level.py   GSE167363 patient-level marker summary (+ optional UMAP)
```

## Reproduce

```bash
pip install -r requirements.txt

python code/01_download_geo.py --outdir ./raw
tar -xf raw/GSE167363_RAW.tar -C raw/sc        # single-cell matrix triples

python code/02_discovery_analysis.py  --raw ./raw --out ./output
python code/03_external_cohorts.py    --raw ./raw --out ./output
python code/04_singlecell_patient_level.py --raw ./raw --out ./output --skip-umap
```

Python 3.10+ (developed on 3.12). The single-cell UMAP step needs
scanpy/anndata and roughly 16 GB RAM; `--skip-umap` runs only the
patient-level summary, which is the part reported in the manuscript.

## Notes

- LymphoScore = mean of within-dataset z-scores across the 17 genes
  (IL7R, CD3D, CD3E, CD247, LCK, ZAP70, GIMAP1, GIMAP4, GIMAP5, GIMAP6,
  CCR7, TCF7, CD28, CD2, ITK, SKAP1, THEMIS), computed separately per
  dataset.
- The CAP subgroup has 11 events, so we report Firth-corrected estimates,
  bootstrap optimism correction (B=500), leave-one-out and Cook's
  distance sensitivity analyses, and treat the parsimonious model
  (LymphoScore + age) as primary.
- For GSE167363, sepsis patients were sampled at T0 and T6. We average
  T0/T6 within patient before any comparison, so each patient contributes
  one data point (n=2 healthy vs n=5 sepsis). At this sample size the
  single-cell results are descriptive only; no group test is reported.
- The whole-cohort analyses (differential expression and the 73-gene
  elastic-net screen) are negative by design and are reported as such.

## License

Code: MIT (see LICENSE). The public datasets remain subject to their
original terms; please cite the original publications.
