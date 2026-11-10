# Agro Training Scope

This directory contains downloaded `data.gov.in` source files. The single Haralab agro model uses only sources confirmed to describe agriculture or agricultural markets.

## Included

- `mandi.csv`: agricultural market arrivals and prices.
- `procurement-details.json`: agricultural commodity procurement by district and season.

## Excluded from agro training

- `hmis-item-20-21-mn-sd-ap-prakasam-upto-Jun.csv`: public-health indicators.
- `datafile.csv`: administrative permit totals; the file does not identify an agricultural subject.
- `getStateWiseProjData.json`: generic project statistics without enough metadata to establish an agricultural meaning.
- `PCA_CDB_2806_F_Census.xls`, `datafile.xls`, and `datafile(1).xls`: not included until their sheets, schemas, and subject matter are verified.
- Derived `mandi_train.jsonl` and `mandi_validation.jsonl` remain included because they are generated from `mandi.csv`.

The raw downloads are intentionally preserved for provenance. Excluded files are not deleted; they are simply not converted into the agro training corpus.
