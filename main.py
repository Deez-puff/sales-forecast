import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import sys
sys.path.append('src')
from data_loader import load_data, aggregate_monthly_sales
from feature_engineer import create_time_features, get_feature_columns
from forecaster import (prepare_data, train_linear_regression,
                        train_random_forest, predict,
                        generate_future_forecast, save_model)
from evaluator import evaluate_model, print_evaluation, compare_models

# ── Load & Prepare Data ───────────────────────────────────
print("=" * 50)
print("   SALES FORECASTING SYSTEM")
print("=" * 50)

df       = load_data("data/Sample - Superstore.csv")
monthly  = aggregate_monthly_sales(df)
featured = create_time_features(monthly)
features = get_feature_columns()

print(f"\n✅ Data ready: {len(featured)} months of features")

# ── Split Data ────────────────────────────────────────────
X_train, X_test, y_train, y_test = prepare_data(featured, features)
print(f"✅ Training set: {len(X_train)} months")
print(f"✅ Testing set:  {len(X_test)} months")

# ── Train Models ──────────────────────────────────────────
print("\n⏳ Training Linear Regression...")
lr_model = train_linear_regression(X_train, y_train)
lr_preds = predict(lr_model, X_test)
print("✅ Linear Regression trained!")

print("\n⏳ Training Random Forest...")
rf_model = train_random_forest(X_train, y_train)
rf_preds = predict(rf_model, X_test)
print("✅ Random Forest trained!")

# ── Evaluate Both Models ──────────────────────────────────
print("\n⏳ Evaluating models...")
lr_results = evaluate_model(y_test, lr_preds, "Linear Regression")
rf_results = evaluate_model(y_test, rf_preds, "Random Forest")

print_evaluation(lr_results)
print_evaluation(rf_results)

# ── Compare & Select Best ─────────────────────────────────
best_result = compare_models([lr_results, rf_results])
best_model  = rf_model if best_result['model_name'] == "Random Forest" else lr_model

# ── Save Best Model ───────────────────────────────────────
save_model(best_model, "data/best_model.pkl")
print("\n✅ Best model saved!")

# ── Generate Future Forecast ──────────────────────────────
print("\n⏳ Generating 6-month future forecast...")
forecast_df = generate_future_forecast(best_model, featured, features, months_ahead=6)
print("✅ Forecast generated!")

print("\n── 6-Month Sales Forecast ──")
for _, row in forecast_df.iterrows():
    print(f"{row['Date'].strftime('%b %Y')}: ${row['Forecasted_Sales']:,.2f}")

print("\n🎉 Sales Forecasting Complete!")
print("=" * 50)

# ── Set Style ─────────────────────────────────────────────
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ── Chart 1: Actual vs Predicted Sales ───────────────────
print("\n⏳ Generating charts...")
test_dates = featured['Date'].iloc[-12:].values

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(featured['Date'], featured['Total_Sales'],
        label='Actual Sales', color='#2ecc71', linewidth=2)
ax.plot(featured['Date'].iloc[-12:], lr_preds,
        label='Linear Regression', color='#e74c3c',
        linewidth=2, linestyle='--')
ax.plot(featured['Date'].iloc[-12:], rf_preds,
        label='Random Forest', color='#3498db',
        linewidth=2, linestyle='--')
ax.set_title('Actual vs Predicted Monthly Sales',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date')
ax.set_ylabel('Sales ($)')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/actual_vs_predicted.png', dpi=150)
plt.show()

# ── Chart 2: Future Forecast ──────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

# Plot historical sales
ax.plot(featured['Date'], featured['Total_Sales'],
        label='Historical Sales', color='#2ecc71',
        linewidth=2, marker='o', markersize=4)

# Plot future forecast
ax.plot(forecast_df['Date'], forecast_df['Forecasted_Sales'],
        label='Forecasted Sales', color='#e67e22',
        linewidth=2, linestyle='--', marker='s', markersize=6)

# Add shaded forecast region
ax.axvspan(forecast_df['Date'].iloc[0],
           forecast_df['Date'].iloc[-1],
           alpha=0.1, color='#e67e22', label='Forecast Period')

# Add value labels on forecast points
for _, row in forecast_df.iterrows():
    ax.annotate(f"${row['Forecasted_Sales']:,.0f}",
                xy=(row['Date'], row['Forecasted_Sales']),
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=8, fontweight='bold')

ax.set_title('Sales Forecast — Next 6 Months',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date')
ax.set_ylabel('Sales ($)')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/future_forecast.png', dpi=150)
plt.show()

# ── Chart 3: Monthly Sales Trend ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
monthly_avg = featured.groupby('Month')['Total_Sales'].mean()
months = ['Jan','Feb','Mar','Apr','May','Jun',
          'Jul','Aug','Sep','Oct','Nov','Dec']
colors = ['#e74c3c' if m in [10,11,12] else '#3498db'
          for m in range(1, 13)]

bars = ax.bar(months, monthly_avg.values, color=colors, edgecolor='white')
ax.set_title('Average Monthly Sales Pattern\n(Red = Q4 Holiday Season)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Average Sales ($)')

for bar, val in zip(bars, monthly_avg.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 500,
            f'${val:,.0f}',
            ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('data/monthly_pattern.png', dpi=150)
plt.show()

# ── Chart 4: Sales by Category ────────────────────────────
from data_loader import get_category_sales, get_region_sales
category_sales = get_category_sales(df)
region_sales   = get_region_sales(df)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Category pie chart
cat_totals = category_sales.groupby('Category')['Total_Sales'].sum()
ax1.pie(cat_totals.values, labels=cat_totals.index,
        autopct='%1.1f%%', colors=['#3498db','#2ecc71','#e74c3c'],
        startangle=90)
ax1.set_title('Sales by Category', fontsize=13, fontweight='bold')

# Region bar chart
ax2.barh(region_sales['Region'], region_sales['Total_Sales'],
         color=['#3498db','#2ecc71','#e74c3c','#e67e22'])
ax2.set_title('Sales by Region', fontsize=13, fontweight='bold')
ax2.set_xlabel('Total Sales ($)')
for i, (val, reg) in enumerate(zip(region_sales['Total_Sales'],
                                    region_sales['Region'])):
    ax2.text(val + 5000, i, f'${val:,.0f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('data/category_region.png', dpi=150)
plt.show()

print("✅ All 4 charts saved to data/ folder!")
print("\n🎉 Sales Forecasting Complete!")
print("=" * 50)