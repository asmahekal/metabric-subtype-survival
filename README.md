# metabric-subtype-survival

Code and results for: *Subtype-specific training does not improve machine-learning survival prediction in breast cancer: a paired comparison with pooled models in the METABRIC cohort.*

Data: [METABRIC_RNA_Mutation.csv (Kaggle)](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric)

```bash
pip install -r requirements.txt
python metabric_subtype_survival.py --data METABRIC_RNA_Mutation.csv
```

`results/` contains the output of the run reported in the paper (figures in `results/figures.zip`).

License: MIT
