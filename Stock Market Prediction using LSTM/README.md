# Netflix Stock Price Prediction — Dataset and Usage

This repository contains an LSTM-based notebook for Netflix stock price prediction. It uses the dataset published on Kaggle: "Netflix Stock Price Prediction" by jainilcoder.

Kaggle dataset page:

https://www.kaggle.com/datasets/jainilcoder/netflix-stock-price-prediction

## Dataset files

The Kaggle dataset provides one or more CSV files (for example `NFLX.csv` or similar). Place the downloaded CSV file in a `data/` folder at the project root. Example path:

```
D:/ML/Stock Market Prediction using LSTM/data/NFLX.csv
```

This README assumes the notebook `Stock_Market_Predcition.ipynb` will load the CSV from `data/NFLX.csv`. If the notebook expects a different filename or path, either rename the file or update the notebook accordingly.

## How to download the dataset

Option A — Manual download (Kaggle website)

1. Open the dataset page: https://www.kaggle.com/datasets/jainilcoder/netflix-stock-price-prediction
2. Sign in to Kaggle (create an account if you don't have one).
3. Click "Download" to download the dataset ZIP.
4. Extract the ZIP and copy the CSV file(s) into the repository `data/` folder.

Option B — Download using the Kaggle CLI (recommended for reproducibility)

1. Install the Kaggle CLI and configure credentials:

   - Install the Kaggle package (requires Python and pip):

     ```powershell
     python -m pip install --upgrade pip
     pip install kaggle
     ```

   - Create API credentials on Kaggle: Go to your Kaggle account settings -> "API" -> "Create New API Token". This downloads a `kaggle.json` file.

   - Place `kaggle.json` in the location expected by the Kaggle CLI on Windows:

     ```powershell
     mkdir $env:USERPROFILE\.kaggle -ErrorAction Ignore
     copy .\kaggle.json $env:USERPROFILE\.kaggle\kaggle.json
     # secure the file
     icacls $env:USERPROFILE\.kaggle\kaggle.json /inheritance:r /grant:r "$($env:USERNAME):(R)"
     ```

2. Download the dataset and extract it into the project `data/` folder:

   ```powershell
   # create data folder
   mkdir data -ErrorAction Ignore
   # replace the dataset identifier if it changes
   kaggle datasets download -d jainilcoder/netflix-stock-price-prediction -p data --unzip
   ```

Note: On Windows PowerShell you might need to use `kaggle.exe` if the CLI is not in PATH. Adjust commands accordingly.

## Example: load dataset in Python (pandas)

Here's a short snippet you can use inside the notebook or a script to load the CSV:

```python
import pandas as pd

df = pd.read_csv("data/NFLX.csv", parse_dates=["Date"])  # adjust filename/Date column if different
df.sort_values("Date", inplace=True)
df.head()
```

If your CSV uses different column names, inspect it with `df.columns` and adapt the notebook preprocessing steps accordingly.

## Dependencies

Install the required Python packages using the provided `requirements.txt` file:

```powershell
pip install -r requirements.txt
```

This will install all necessary dependencies including pandas, numpy, matplotlib, scikit-learn, tensorflow, jupyter, and kaggle CLI.

## Git and data

Large or binary data files should not be committed to the repository. This repo includes a `.gitignore` entry to ignore the `data/` folder. If you prefer to track a small sample, add a `data/sample.csv` file and commit that instead.

## Troubleshooting

- Permission error copying `kaggle.json`: ensure PowerShell is running with enough permissions and the destination folder exists.
- Kaggle CLI not recognized: confirm Python scripts directory is in PATH or run `python -m kaggle`.
- Notebook cannot find file: confirm the CSV filename and path match `data/NFLX.csv` or update the notebook cell to point to your file.

## Next steps

- Verify `Stock_Market_Predcition.ipynb` loads `data/NFLX.csv`. If it expects a different path, I can update the notebook to use a relative path and a configurable variable.

---
Generated on 2025-10-23.
