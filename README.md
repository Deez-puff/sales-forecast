# 📈 Sales & Demand Forecasting System

![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge\&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=for-the-badge\&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?style=for-the-badge\&logo=pandas)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge\&logo=streamlit)
![GitHub](https://img.shields.io/badge/GitHub-Public-black?style=for-the-badge\&logo=github)

> An ML-powered Sales & Demand Forecasting System that predicts future sales using historical data, time-based feature engineering, and machine learning models. Built for business owners, store managers, and startup founders.

---

# 📌 Table of Contents

* [About the Project](#about-the-project)
* [How It Works](#how-it-works)
* [Key Features](#key-features)
* [Project Structure](#project-structure)
* [Tech Stack](#tech-stack)
* [Setup and Installation](#setup-and-installation)
* [How to Run](#how-to-run)
* [Web App Usage](#web-app-usage)
* [How Forecasts Are Generated](#how-forecasts-are-generated)
* [Model Evaluation Metrics](#model-evaluation-metrics)
* [What the Forecast Means for Business](#what-the-forecast-means-for-business)
* [Dataset](#dataset)
* [Future Improvements](#future-improvements)

---

# 🧠 About the Project

Businesses rely on accurate sales forecasts to plan inventory, manage cash flow, prepare staffing, and avoid overstocking or losses.

This project builds a **Machine Learning-based Sales Forecasting System** that:

* Loads and cleans historical sales data
* Aggregates transactions into monthly summaries
* Engineers 15 time-based features (lag, rolling averages, seasonality)
* Trains and compares Linear Regression and Random Forest models
* Automatically selects the best model based on MAE
* Generates future sales forecasts for any number of months ahead
* Displays results in a business-friendly interactive web app

---

# ⚙️ How It Works

```text
Raw Sales CSV
     │
     ▼
┌─────────────────────┐
│  Data Loading       │
│  (data_loader.py)   │ → Parse dates, sort by date,
│                     │   aggregate to monthly totals
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Feature Engineering │
│(feature_engineer.py)│ → Year, Month, Quarter,
│                     │   Season, Lag Features,
│                     │   Rolling Averages
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Train/Test Split   │
│                     │ → Last 12 months = Test Set
│                     │   Earlier months = Training
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Train Two Models   │
│  (forecaster.py)    │ → Linear Regression
│                     │   Random Forest
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Model Evaluation    │
│  (evaluator.py)     │ → MAE, RMSE, R², MAPE
│                     │   Auto-select best model
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Future Forecasting  │
│  (forecaster.py)    │ → Generate N months ahead
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Streamlit Web App   │
│      (app.py)       │ → Dashboard + Insights
└─────────────────────┘
```

---

# ✨ Key Features

| Feature                     | Description                                              |
| --------------------------- | -------------------------------------------------------- |
| 📂 Upload Any Dataset       | Accepts any CSV with a date and sales column             |
| 📅 Time Feature Engineering | 15 features including lag, rolling averages, seasonality |
| 🤖 Two ML Models            | Linear Regression + Random Forest                        |
| 🏆 Auto Model Selection     | Best model chosen using lowest MAE                       |
| 🔮 Future Forecasting       | Predict 3–12 months into the future                      |
| 📊 Business Charts          | Actual vs Predicted, Forecast Trends, Monthly Patterns   |
| 💡 Business Insights        | Inventory and sales recommendations                      |
| 🌐 Web App                  | Interactive Streamlit dashboard                          |

---

# 📁 Project Structure

```text
sales-forecasting/
│
├── data/
│   ├── Sample - Superstore.csv
│   ├── best_model.pkl
│   ├── actual_vs_predicted.png
│   ├── future_forecast.png
│   ├── monthly_pattern.png
│   └── category_region.png
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineer.py
│   ├── forecaster.py
│   └── evaluator.py
│
├── app.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🛠️ Tech Stack

| Category         | Tools Used                                 |
| ---------------- | ------------------------------------------ |
| Language         | Python 3.14                                |
| Data Processing  | Pandas, NumPy                              |
| Machine Learning | Scikit-learn                               |
| Models           | Linear Regression, Random Forest Regressor |
| Model Storage    | Joblib                                     |
| Visualization    | Matplotlib, Seaborn                        |
| Web Framework    | Streamlit                                  |
| Version Control  | Git & GitHub                               |

---

# 📦 Setup and Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Deez-puff/sales-forecasting.git

cd sales-forecasting
```

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 3️⃣ Download the Dataset

Download the Superstore Sales Dataset from Kaggle:

👉 https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

Place:

```text
Sample - Superstore.csv
```

inside:

```text
data/
```

---

# ▶️ How to Run

## Option 1 — Streamlit Web App

```bash
python -m streamlit run app.py
```

Opens automatically at:

```text
http://localhost:8501
```

## Option 2 — Terminal Version

```bash
python main.py
```

This will:

* Train models
* Print evaluation metrics
* Generate charts
* Save outputs to the `data/` folder

---

# 🌐 Web App Usage

1. Upload a CSV dataset
2. Select the date column
3. Select the sales column
4. Choose forecast horizon (3–12 months)
5. Click **Train & Forecast**
6. View:

   * Model comparison metrics
   * Actual vs Predicted chart
   * Future Forecast chart
   * Monthly trend analysis
   * Category/Region analysis
   * Business recommendations

---

# 📐 How Forecasts Are Generated

## Step 1 — Monthly Aggregation

Raw transaction data is grouped by month, creating a cleaner time-series dataset.

## Step 2 — Feature Engineering

15 time-based features are generated.

| Feature Type     | Features                         |
| ---------------- | -------------------------------- |
| Date Features    | Year, Month, Quarter, Season     |
| Lag Features     | Previous 1, 2, 3, and 12 months  |
| Rolling Features | 3, 6, and 12 month averages      |
| Flags            | Q4 Indicator, Year-End Indicator |

## Step 3 — Train/Test Split

The final 12 months are reserved as the test set to simulate future forecasting.

## Step 4 — Model Training

Two models are trained:

### Linear Regression

* Fast
* Explainable
* Captures long-term trends

### Random Forest

* Handles seasonality
* Captures non-linear patterns
* Often produces better accuracy

The model with the lowest MAE is automatically selected.

## Step 5 — Future Forecasting

Future sales are predicted recursively, where each prediction becomes part of the next month's feature set.

---

# 📊 Model Evaluation Metrics

| Metric   | Meaning                           |
| -------- | --------------------------------- |
| MAE      | Average prediction error          |
| RMSE     | Penalizes large prediction errors |
| R² Score | Measures explained variance       |
| MAPE     | Percentage forecasting error      |

### Rule of Thumb

* MAPE < 10% → Excellent
* MAPE < 20% → Good
* MAPE < 50% → Acceptable
* MAPE > 50% → Needs Improvement

---

# 💡 What the Forecast Means for Business

Business owners can use the forecast to:

* 📦 Plan inventory purchases
* 💰 Manage cash flow
* 👥 Schedule staffing needs
* 🎯 Set realistic sales targets
* 📉 Reduce losses from overstocking
* 📈 Prepare for high-demand periods

---

# 📊 Dataset

**Superstore Sales Dataset**

Dataset Statistics:

* 9,994 transactions
* January 2014 – December 2017
* Sales
* Profit
* Category
* Region
* Segment

Dataset Link:

https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

---

# 🚀 Future Improvements

* [ ] Prophet forecasting support
* [ ] ARIMA forecasting support
* [ ] Product-level forecasting
* [ ] Category-level forecasting
* [ ] Confidence intervals
* [ ] Automated alerts for demand spikes
* [ ] CSV export of forecast results
* [ ] Weekly forecasting
* [ ] Daily forecasting
* [ ] Streamlit Cloud deployment

---

# 👨‍💻 Author
Deepak Rajesh

Built as part of **Future Interns ML Task 1 – 2026**

---

# 📄 License

This project is licensed under the **MIT License**.

Feel free to use, modify, and distribute this project for educational and commercial purposes.
