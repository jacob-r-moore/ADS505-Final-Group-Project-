# Targeting Wildlife Import Inspections

ADS-505 Applied Data Science for Business — Team Project

## Business problem

The US Fish & Wildlife Service inspects a small fraction of the wildlife and
wildlife-product shipments entering the country. Shipments that are seized or
refused make up roughly 2% of all shipments, so inspecting at random wastes
most of the effort spent.

This project builds a model that scores incoming shipments by the likelihood
they will be seized or refused, so inspection capacity can be pointed at the
shipments most likely to be in violation. We report how many violations a
targeted strategy catches compared with random inspection.

## Data

US Fish & Wildlife Service LEMIS wildlife import records, cleaned and
published by EcoHealth Alliance.

- Source: https://zenodo.org/records/3565869 (CC BY 4.0)
- Paper: Eskew et al. 2020, *Scientific Data* — https://www.nature.com/articles/s41597-020-0354-5
- Coverage: 5,512,667 line-item records, 2000–2014

The raw file (`lemis_2000_2014_cleaned.csv`, 974 MB) is too large for GitHub
and is not in this repository. Download it from Zenodo to rerun the data prep.

## Working file

`lemis_shipments.csv` — 207,931 shipments, 2.31% positive.

Built from the raw file by:

1. Keeping only dispositions C (cleared), S (seized) and R (refused).
   Abandoned shipments are excluded because abandonment is the importer
   withdrawing, not an enforcement finding.
2. Sampling 10% of shipments, selected by a CRC32 hash of `control_number`
   so the sample is identical on every run.
3. Aggregating line items to one row per shipment, since a shipment is the
   unit an inspector decides to open.
4. Labelling a shipment positive if any of its line items was seized or
   refused. Line items disagree in only 0.28% of shipments.
5. Excluding `action` and `disposition_date`, which are recorded at or after
   inspection and would leak the outcome.

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
