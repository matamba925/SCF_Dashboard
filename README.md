# SCF Dashboard

A Dash dashboard for exploring World Development Indicators with K-means clustering.

## Features

- Select the number of K-means clusters.
- Compare trimmed and untrimmed indicator variance.
- View the five highest-variance indicators.
- Visualize clusters in two dimensions using PCA.
- Review inertia and silhouette-score metrics.

## Setup

From the `SCF_Dashboard` directory, create or activate the project environment and install the dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
pip install dash numpy pandas plotly scikit-learn
```

The application reads `data/WDICSV.csv` by default. The provided dataset is excluded from Git because it is larger than GitHub's 100 MB file limit. Keep a local copy at that path, or set `WDI_DATA_PATH` to another CSV before starting the app.

## Run

```powershell
.\.venv\Scripts\python.exe .\data\app.py
```

Open <http://127.0.0.1:8050> in a browser.

If Windows puts the computer to sleep or shuts it down, the local Python server stops or is suspended. After waking the computer, open a new terminal and run the command above again. A local server cannot remain available while the computer is powered off; deploy it to a hosted service for continuous availability.
