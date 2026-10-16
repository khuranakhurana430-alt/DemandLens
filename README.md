# Sales & Demand Forecasting

A machine learning project that predicts future sales and product demand from historical data. It covers data cleaning, feature engineering, and time-series modeling, and compares several forecasting models to produce accurate, actionable predictions that support inventory planning and business decisions.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Methodology](#methodology)
- [Models Compared](#models-compared)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Accurate demand forecasts help businesses avoid stockouts, reduce overstock, and plan purchasing, staffing, and promotions. This project builds an end-to-end forecasting pipeline:

1. Load and clean historical sales data
2. Engineer time-based and demand-related features
3. Train and tune multiple forecasting models
4. Compare models on a held-out, time-ordered test set
5. Generate future forecasts and visualize the results

## Features

- **Data cleaning**: handling missing values, duplicates, outliers, and inconsistent date formats
- **Feature engineering**: calendar features (day, week, month, quarter, holidays), lag features, rolling statistics, and seasonality indicators
- **Exploratory analysis**: trend, seasonality, and autocorrelation analysis
- **Multiple models**: statistical, machine learning, and deep learning approaches
- **Time-aware validation**: chronological train/test split and walk-forward cross-validation
- **Model comparison**: side-by-side metrics and forecast plots
- **Actionable output**: forecasts exported for inventory and business planning

## Tech Stack

- **Language**: Python 3.9+
- **Data**: pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Statistical models**: statsmodels (ARIMA / SARIMA / Holt-Winters)
- **Machine learning**: scikit-learn, XGBoost / LightGBM
- **Forecasting libraries**: Prophet
- **Deep learning** (optional): TensorFlow / Keras (LSTM)
- **Environment**: Jupyter Notebook / Google Colab

## Project Structure

```
sales-demand-forecasting/
│
├── data/
│   ├── raw/                  # Original, unmodified data
│   └── processed/            # Cleaned and feature-engineered data
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── features.py
│   ├── models.py
│   └── evaluate.py
│
├── models/                   # Saved trained models
├── outputs/                  # Forecasts, plots, reports
├── requirements.txt
└── README.md
```

> Adjust the structure above to match your actual folders and files.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip
- Git

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/sales-demand-forecasting.git
   cd sales-demand-forecasting
   ```

2. **Create and activate a virtual environment**

   Windows (PowerShell):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   macOS / Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Dataset

Place your historical sales data in `data/raw/`. The dataset should include at least:

| Column     | Description                              |
|------------|------------------------------------------|
| `date`     | Date of the sale                         |
| `product`  | Product or SKU identifier (if applicable)|
| `store`    | Store or region (if applicable)          |
| `sales`    | Units sold or revenue                    |

## Usage

Run the notebooks in order:

```bash
jupyter notebook
```

1. `01_data_cleaning.ipynb`: clean and validate the raw data
2. `02_eda.ipynb`: explore trends and seasonality
3. `03_feature_engineering.ipynb`: build model features
4. `04_modeling.ipynb`: train and tune models
5. `05_evaluation.ipynb`: compare models and generate forecasts

Or run the pipeline from the command line:

```bash
python src/data_preprocessing.py
python src/models.py
python src/evaluate.py
```

## Methodology

### 1. Data Cleaning
- Parse dates and set a proper datetime index
- Handle missing values and remove duplicates
- Detect and treat outliers
- Aggregate to the desired frequency (daily / weekly / monthly)

### 2. Feature Engineering
- **Calendar features**: day of week, month, quarter, year, weekend and holiday flags
- **Lag features**: sales from previous days, weeks, or months
- **Rolling statistics**: moving averages and rolling standard deviations
- **External factors** (if available): promotions, pricing, weather

### 3. Modeling
Models are trained on historical data and evaluated on a chronologically held-out test set to prevent data leakage.

### 4. Evaluation and Selection
Models are compared using the metrics below, and the best performer is used to generate future forecasts.

## Models Compared

| Category            | Models                                   |
|---------------------|------------------------------------------|
| Baseline            | Naive, Moving Average                    |
| Statistical         | ARIMA, SARIMA, Holt-Winters              |
| Additive / trend    | Prophet                                  |
| Machine learning    | Random Forest, XGBoost, LightGBM         |
| Deep learning       | LSTM (optional)                          |

## Evaluation Metrics

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination

## Results

> Replace this section with your actual results.

| Model    | MAE | RMSE | MAPE |
|----------|-----|------|------|
| Naive    | –   | –    | –    |
| SARIMA   | –   | –    | –    |
| Prophet  | –   | –    | –    |
| XGBoost  | –   | –    | –    |

**Key takeaways:**
- Best performing model: _TBD_
- Main drivers of demand: _TBD_
- Recommended use for inventory planning: _TBD_

## Future Work

- Add external regressors (promotions, holidays, weather, economic indicators)
- Hierarchical forecasting across products and regions
- Hyperparameter optimization with Optuna
- Prediction intervals and uncertainty estimates
- Deploy as an API or dashboard (Streamlit / Flask)

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

**Your Name**: [your.email@example.com](mailto:your.email@example.com)
GitHub: [@your-username](https://github.com/your-username)
