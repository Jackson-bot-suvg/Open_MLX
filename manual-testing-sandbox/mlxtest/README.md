# Sales Radar Model Training

Code for sales radar model training, used to predict sales duration quantiles.

## Environment Setup

```bash

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

Place training data in the `data/` directory:

#### Shared in Box folder

- `train_data.csv`: Training data
- `val_data.csv`: Validation data
- `test_data.csv`: Test data

## Execution Sequence

1. Train the model:

```bash
python train_lob.py
```

2. Inference:

```bash
python local_inference.py
```

## Model Structure

1. model_lob.py

- Class SalesRadar: MOME model
  - Embedding for context information (categorical and cumulative features)
  - Concat embedding and temporal features
  - MoME model - by FPH1 \* idx1
  - Output 3 quantiles
  - Quantile Loss

2. model_utils.py

- Util functions for model

## Data Format

The data should include the following features:

- Categorical features (integer encoded):

  - `rtm`: RTM
  - `sub_rtm`: sub_rtm
  - `province`: Province
  - `city_group`: City Tier
  - `district_group`: District Tier
  - `fph1`: Product type
  - `event_tier`: Event Tier
  - `event_category`: Event Type
  - `new_pos_type`: Pos Type Mapping
  - `disti_name`: Distributor
  - `program_name`: Project
  - `rtm4`: Sales channel
  - `holiday`: Holiday

- Cumulative features:

  - `days_since_latest_model_npi`: Days since latest FPH1 NPI

- Aggregated features:

  - `{metric}_{spatial_level}_avg_{n}d"`
    - metric: so, duration, traffic_cnt
    - spatial_level: city_group, province, district_group, national, pos
    - n: 1, 3, 7, 14, 30

- Label:

  - `duration`: Duration time

- Index:
  - `idx1`: Seperate sample into different model sub-task in MOME
