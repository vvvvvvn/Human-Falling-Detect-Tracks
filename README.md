# Create dataset

## UR-Dataset Prepared

執行 `download_data.py` 下載 UR 資料集

- 資料集架構 (部份)

```sh
├── UR-DATA/
│   ├── ADL_sequences/
│       ├── 31/
│       |   ├── adl-31-cam0-rgb/
│   ├── Fall_sequences/
│       ├── 1/
│       |   ├── fall-01-cam0-rgb/
│   ├── urfall-cam0-adls.csv
│   ├── urfall-cam0-falls.csv
│   ├── urfall-cam0-adls-OUR.csv   # 我們上次標記的
```

## example: 讀取 Fall_sequences 資料集

1. 讀取 `UR-DATA/Fall_sequences` 執行 create_dataset_2

- command:

```bash=
python Data/create_dataset_2.py
```

- 若遇到以下錯誤，需 export PYTHONPATH

```
ModuleNotFoundError: No module named 'DetectorLoader'
```

- 解法： example

```bash=
export PYTHONPATH="/home/penguin/Documents/vivian/Human-Falling-Detect-Tracks"
```

2. 執行 create_dataset_3

- command:

```bash=
python Data/create_dataset_3.py
```
