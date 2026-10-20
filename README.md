# Targeting Wildlife Import Inspections

ADS-505 Applied Data Science for Business — Team Project

Team Members: Nandini Kasibhotla, Imelda Mendoza, Jacob Moore

## Business problem

The US Fish & Wildlife Service inspects a small fraction of the wildlife and wildlife-product shipments entering the country. Shipments that are seized or refused make up roughly 2% of all shipments, so inspecting at random wastes most of the effort spent.

This project builds a model that scores incoming shipments by the likelihood they will be seized or refused, so inspection capacity can be pointed at the shipments most likely to be refused or seized. We report how many refused or seized shipments a targeted strategy catches compared with random inspection.

## Repository contents

| File | Description |
| --- | --- |
| `Technical_Notebook_Team3.ipynb` | Final technical notebook: EDA, pre-processing, modeling, tuning, inspection simulation, and Model Brief |
| `lemis_shipments.csv` | Working dataset (207,931 shipments) used by the notebook |
| `README.md` | This file |

## Data

US Fish & Wildlife Service LEMIS wildlife import records, cleaned and published by EcoHealth Alliance.

- Source: https://zenodo.org/records/3565869 (CC BY 4.0)
- Paper: Eskew et al. 2020, *Scientific Data* — https://www.nature.com/articles/s41597-020-0354-5
- Coverage: 5,512,667 line-item records, 2000–2014

The raw file (`lemis_2000_2014_cleaned.csv`, 974 MB) is too large for GitHub and is not in this repository. Download it from Zenodo to rerun the data prep.

## Working file

`lemis_shipments.csv` — 207,931 shipments, 2.31% positive.

Built from the raw file by:

1. Keeping only dispositions C (cleared), S (seized) and R (refused). Abandoned shipments are excluded because abandonment is the importer withdrawing, not an enforcement finding.
2. Sampling 10% of shipments, selected by a CRC32 hash of `control_number` so the sample is identical on every run.
3. Aggregating line items to one row per shipment, since a shipment is the unit an inspector decides to open.
4. Labelling a shipment positive if any of its line items was seized or refused. Line items disagree in only 0.28% of shipments.
5. Excluding `action` and `disposition_date`, which are recorded at or after inspection and would leak the outcome.

### Columns

| Column | Description |
| --- | --- |
| control_number | Shipment identifier |
| n_lines | Line items on the declaration |
| n_taxa, n_class, n_species | Distinct taxa, classes, species codes |
| qty_total | Total declared quantity |
| value_total | Total declared value (USD) |
| value_missing | Share of line items with no declared value |
| taxa | Most common taxa group in the shipment |
| country_origin | Country of origin |
| country_imp_exp | Country shipped from |
| port | US port of entry |
| purpose | Declared purpose code |
| source | Declared source code |
| year, month | From the earliest line-item shipment date |
| target | 1 if seized or refused, 0 if cleared |

## Setup

Requires Python 3.11 or later and an internet connection (the notebook reads the CSV from this repository).

```
pip install pandas numpy matplotlib seaborn scikit-learn catboost jupyter
```

## How to run

1. Clone the repository: `git clone https://github.com/jacob-r-moore/ADS505-Final-Group-Project-.git`
2. Open `Technical_Notebook_Team3.ipynb` in Jupyter or VS Code.
3. Select **Restart & Run All**. A full run takes about 5–10 minutes, mostly CatBoost training.

The notebook loads `lemis_shipments.csv` directly from this repository's GitHub URL, so no local file paths need to be changed. To run offline, replace `DATA_URL` in the data-loading cell with `"lemis_shipments.csv"`.

Models use fixed random seeds. Key results (CatBoost PR-AUC 0.6511; top 1% of 2013–2014 shipments: 261 of 696 refused/seized cases found, 88.47% precision, 37.39× lift) should reproduce exactly. Minor values such as Logistic Regression scores can differ slightly across machines.

## Approach

- **Split:** train 2000–2010, validate 2011–2012 (CatBoost tuning), refit on 2000–2012, test once on 2013–2014.
- **Models:** Logistic Regression, Decision Tree, Random Forest, CatBoost (selected).
- **Evaluation:** PR-AUC, plus Precision@K, Recall@K, and Lift@K at 1%, 2%, 5%, 10%, and 20% inspection capacity.

## Team contributions

- **Jacob Moore:** Project idea; built the working dataset from the raw LEMIS file; set up the GitHub repository; made the notebook reproducible (GitHub data load, import cleanup); removed the leaking random split; added the validation period, EDA, and Discussion sections; updated the executive brief and slides to match final results.
- **Imelda Mendoza:** Data preparation and cleaning; exploratory data analysis and visualizations; notebook markdown; baseline and CatBoost modeling; inspection simulation.
- **Nandini Kasibhotla:** Data preparation analysis; additional model development; research questions; executive brief drafting; team submissions.

## References

Eskew, E. A., White, A. M., Ross, N., Smith, K. M., Smith, K. F., Rodríguez, J. P., Zambrana-Torrelio, C., Karesh, W. B., & Daszak, P. (2019). *United States LEMIS wildlife trade data curated by EcoHealth Alliance* (Version 1.1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3565869

Eskew, E. A., White, A. M., Ross, N., Smith, K. M., Smith, K. F., Rodríguez, J. P., Zambrana-Torrelio, C., Karesh, W. B., & Daszak, P. (2020). United States wildlife and wildlife product imports from 2000–2014. *Scientific Data, 7*, Article 22. https://doi.org/10.1038/s41597-020-0354-5
