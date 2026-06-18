import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sys
sys.path.append('src')

from data_loader import load_data, aggregate_monthly_sales, get_category_sales, get_region_sales
from feature_engineer import create_time_features, get_feature_columns
from forecaster import (prepare_data, train_linear_regression,
                        train_random_forest, predict,
                        generate_future_forecast, save_model)
from evaluator import evaluate_model, compare_models

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────
st.title("📈 Sales & Demand Forecasting System")
st.markdown("Upload your sales data, train a forecasting model, and predict future sales instantly.")
st.divider()

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
months_ahead = st.sidebar.slider("Forecast Months Ahead", 3, 12, 6)
show_lr      = st.sidebar.checkbox("Show Linear Regression", value=True)
show_rf      = st.sidebar.checkbox("Show Random Forest", value=True)

# ── Step 1: Upload Dataset ────────────────────────────────
st.header("Step 1 — Upload Sales Dataset")
uploaded_file = st.file_uploader(
    "Upload your CSV file (must have Order Date and Sales columns)",
    type=["csv"]
)

if uploaded_file is not None:
    # ── Load Data ─────────────────────────────────────────
    df = pd.read_csv(uploaded_file, encoding='windows-1252')
    st.success(f"✅ Dataset loaded — {len(df)} records found!")

    with st.expander("👀 Preview Dataset"):
        st.dataframe(df.head())

    st.divider()

    # ── Step 2: Configure Columns ─────────────────────────
    st.header("Step 2 — Configure Columns")
    col1, col2 = st.columns(2)
    with col1:
        date_column  = st.selectbox("Which column has the date?",
                                    df.columns.tolist(),
                                    index=df.columns.tolist().index('Order Date')
                                    if 'Order Date' in df.columns else 0)
    with col2:
        sales_column = st.selectbox("Which column has the sales amount?",
                                    df.columns.tolist(),
                                    index=df.columns.tolist().index('Sales')
                                    if 'Sales' in df.columns else 0)

    st.divider()

    # ── Step 3: Train Model ───────────────────────────────
    st.header("Step 3 — Train Forecasting Model")

    if st.button("🚀 Train & Forecast", type="primary", use_container_width=True):

        progress = st.progress(0, text="Starting...")

        # ── Prepare Data ──────────────────────────────────
        progress.progress(10, text="⏳ Preparing data...")
        df[date_column]  = pd.to_datetime(df[date_column])
        df               = df.sort_values(date_column).reset_index(drop=True)
        df['Order Date'] = df[date_column]
        df['Sales']      = df[sales_column]

        monthly  = aggregate_monthly_sales(df)
        featured = create_time_features(monthly)
        features = get_feature_columns()

        # ── Split Data ────────────────────────────────────
        progress.progress(30, text="⏳ Splitting data...")
        X_train, X_test, y_train, y_test = prepare_data(featured, features)

        # ── Train Models ──────────────────────────────────
        progress.progress(50, text="⏳ Training Linear Regression...")
        lr_model = train_linear_regression(X_train, y_train)
        lr_preds = predict(lr_model, X_test)

        progress.progress(70, text="⏳ Training Random Forest...")
        rf_model = train_random_forest(X_train, y_train)
        rf_preds = predict(rf_model, X_test)

        # ── Evaluate ──────────────────────────────────────
        progress.progress(85, text="⏳ Evaluating models...")
        lr_results = evaluate_model(y_test, lr_preds, "Linear Regression")
        rf_results = evaluate_model(y_test, rf_preds, "Random Forest")
        best_result = compare_models([lr_results, rf_results])
        best_model  = rf_model if best_result['model_name'] == "Random Forest" else lr_model

        # ── Forecast ──────────────────────────────────────
        progress.progress(95, text="⏳ Generating forecast...")
        forecast_df = generate_future_forecast(best_model, featured,
                                               features, months_ahead)
        progress.progress(100, text="✅ Done!")

        st.divider()

        # ── Results ───────────────────────────────────────
        st.header("📊 Forecasting Results")

        # ── Metrics Row ───────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🏆 Best Model", best_result['model_name'])
        m2.metric("MAE", f"${best_result['MAE']:,.0f}")
        m3.metric("R² Score", f"{best_result['R2']:.4f}")
        m4.metric("MAPE", f"{best_result['MAPE']:.2f}%")

        st.divider()

        # ── Model Comparison Table ─────────────────────────
        st.subheader("🔍 Model Comparison")
        comparison_df = pd.DataFrame([lr_results, rf_results])
        comparison_df = comparison_df[['model_name', 'MAE', 'RMSE', 'R2', 'MAPE']]
        comparison_df.columns = ['Model', 'MAE ($)', 'RMSE ($)', 'R² Score', 'MAPE (%)']
        comparison_df['MAE ($)']  = comparison_df['MAE ($)'].apply(lambda x: f"${x:,.2f}")
        comparison_df['RMSE ($)'] = comparison_df['RMSE ($)'].apply(lambda x: f"${x:,.2f}")
        comparison_df['MAPE (%)'] = comparison_df['MAPE (%)'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(comparison_df, use_container_width=True)

        st.divider()

        # ── Chart 1: Actual vs Predicted ──────────────────
        st.subheader("📈 Actual vs Predicted Sales")
        fig1, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(featured['Date'], featured['Total_Sales'],
                 label='Actual Sales', color='#2ecc71', linewidth=2)
        if show_lr:
            ax1.plot(featured['Date'].iloc[-12:], lr_preds,
                     label='Linear Regression', color='#e74c3c',
                     linewidth=2, linestyle='--')
        if show_rf:
            ax1.plot(featured['Date'].iloc[-12:], rf_preds,
                     label='Random Forest', color='#3498db',
                     linewidth=2, linestyle='--')
        ax1.set_title('Actual vs Predicted Monthly Sales',
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sales ($)')
        ax1.legend()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig1)

        st.divider()

        # ── Chart 2: Future Forecast ──────────────────────
        st.subheader(f"🔮 {months_ahead}-Month Sales Forecast")
        fig2, ax2 = plt.subplots(figsize=(14, 5))
        ax2.plot(featured['Date'], featured['Total_Sales'],
                 label='Historical Sales', color='#2ecc71',
                 linewidth=2, marker='o', markersize=4)
        ax2.plot(forecast_df['Date'], forecast_df['Forecasted_Sales'],
                 label='Forecasted Sales', color='#e67e22',
                 linewidth=2, linestyle='--', marker='s', markersize=6)
        ax2.axvspan(forecast_df['Date'].iloc[0],
                    forecast_df['Date'].iloc[-1],
                    alpha=0.1, color='#e67e22')
        for _, row in forecast_df.iterrows():
            ax2.annotate(f"${row['Forecasted_Sales']:,.0f}",
                         xy=(row['Date'], row['Forecasted_Sales']),
                         xytext=(0, 10), textcoords='offset points',
                         ha='center', fontsize=8, fontweight='bold')
        ax2.set_title(f'Sales Forecast — Next {months_ahead} Months',
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Sales ($)')
        ax2.legend()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)

        st.divider()

        # ── Forecast Table ────────────────────────────────
        st.subheader("📋 Forecast Table")
        forecast_display = forecast_df.copy()
        forecast_display['Month'] = forecast_display['Date'].dt.strftime('%B %Y')
        forecast_display['Forecasted Sales'] = forecast_display['Forecasted_Sales'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(forecast_display[['Month', 'Forecasted Sales']], use_container_width=True)

        st.divider()

        # ── Chart 3: Monthly Pattern ──────────────────────
        st.subheader("📊 Monthly Sales Pattern")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        monthly_avg = featured.groupby('Month')['Total_Sales'].mean()
        months_list = ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec']
        colors = ['#e74c3c' if m in [10,11,12] else '#3498db'
                  for m in range(1, 13)]
        bars = ax3.bar(months_list, monthly_avg.values,
                       color=colors, edgecolor='white')
        ax3.set_title('Average Monthly Sales Pattern\n(Red = Q4 Holiday Season)',
                      fontsize=13, fontweight='bold')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Average Sales ($)')
        for bar, val in zip(bars, monthly_avg.values):
            ax3.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 500,
                     f'${val:,.0f}',
                     ha='center', fontsize=8, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig3)

        st.divider()

        # ── Chart 4: Category & Region ────────────────────
        if 'Category' in df.columns and 'Region' in df.columns:
            st.subheader("🗂️ Sales by Category & Region")
            category_sales = get_category_sales(df)
            region_sales   = get_region_sales(df)

            fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(14, 5))
            cat_totals = category_sales.groupby('Category')['Total_Sales'].sum()
            ax4.pie(cat_totals.values, labels=cat_totals.index,
                    autopct='%1.1f%%',
                    colors=['#3498db','#2ecc71','#e74c3c'],
                    startangle=90)
            ax4.set_title('Sales by Category', fontsize=13, fontweight='bold')

            ax5.barh(region_sales['Region'], region_sales['Total_Sales'],
                     color=['#3498db','#2ecc71','#e74c3c','#e67e22'])
            ax5.set_title('Sales by Region', fontsize=13, fontweight='bold')
            ax5.set_xlabel('Total Sales ($)')
            for i, val in enumerate(region_sales['Total_Sales']):
                ax5.text(val + 5000, i, f'${val:,.0f}', va='center', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig4)

        st.divider()

        # ── Business Insights ─────────────────────────────
        st.subheader("💡 Business Insights")
        total_forecast = forecast_df['Forecasted_Sales'].sum()
        avg_forecast   = forecast_df['Forecasted_Sales'].mean()
        best_month     = forecast_df.loc[forecast_df['Forecasted_Sales'].idxmax()]
        worst_month    = forecast_df.loc[forecast_df['Forecasted_Sales'].idxmin()]

        st.markdown(f"""
        Based on the **{months_ahead}-month forecast**:

        - 📦 **Total forecasted revenue:** ${total_forecast:,.2f}
        - 📅 **Average monthly sales:** ${avg_forecast:,.2f}
        - 🔝 **Best month:** {best_month['Date'].strftime('%B %Y')} (${best_month['Forecasted_Sales']:,.2f})
        - 📉 **Slowest month:** {worst_month['Date'].strftime('%B %Y')} (${worst_month['Forecasted_Sales']:,.2f})
        - 🏆 **Model used:** {best_result['model_name']} with {best_result['MAPE']:.2f}% average error

        **Recommendations:**
        - 📦 Stock up inventory before **{best_month['Date'].strftime('%B %Y')}** (highest forecasted sales)
        - 💰 Plan cash flow to cover the slower **{worst_month['Date'].strftime('%B %Y')}** period
        - 📊 Use the {best_result['model_name']} model for future planning (best accuracy)
        """)

else:
    st.info("👆 Please upload a CSV file to get started.")
    st.markdown("""
    ### 📋 How to use this app:
    1. **Upload** your sales CSV dataset
    2. **Select** date and sales columns
    3. **Click** Train & Forecast
    4. **View** charts, metrics and business insights

    ### 📥 Need a dataset?
    Download here → [Superstore Sales Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)
    """)