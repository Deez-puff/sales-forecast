import pandas as pd

def load_data(filepath):
    """
    Loads and cleans the Superstore sales dataset.
    - Parses dates
    - Aggregates sales by month
    - Sorts chronologically
    """
    # ── Load CSV ──────────────────────────────────────────
    df = pd.read_csv(filepath, encoding='windows-1252')

    # ── Parse Dates ───────────────────────────────────────
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date']  = pd.to_datetime(df['Ship Date'])

    # ── Sort by Date ──────────────────────────────────────
    df = df.sort_values('Order Date').reset_index(drop=True)

    return df


def aggregate_monthly_sales(df):
    """
    Aggregates sales data by month.
    Returns a monthly summary dataframe.
    """
    # ── Extract Year & Month ──────────────────────────────
    df['Year']       = df['Order Date'].dt.year
    df['Month']      = df['Order Date'].dt.month
    df['YearMonth']  = df['Order Date'].dt.to_period('M')

    # ── Group by Month ────────────────────────────────────
    monthly = df.groupby('YearMonth').agg(
        Total_Sales    = ('Sales', 'sum'),
        Total_Profit   = ('Profit', 'sum'),
        Total_Orders   = ('Order ID', 'nunique'),
        Total_Quantity = ('Quantity', 'sum')
    ).reset_index()

    # ── Convert Period to Timestamp ───────────────────────
    monthly['Date'] = monthly['YearMonth'].dt.to_timestamp()

    return monthly


def get_category_sales(df):
    """
    Returns monthly sales broken down by product category.
    """
    df['YearMonth'] = df['Order Date'].dt.to_period('M')

    category_sales = df.groupby(['YearMonth', 'Category']).agg(
        Total_Sales = ('Sales', 'sum')
    ).reset_index()

    category_sales['Date'] = category_sales['YearMonth'].dt.to_timestamp()

    return category_sales


def get_region_sales(df):
    """
    Returns total sales broken down by region.
    """
    region_sales = df.groupby('Region')['Sales'].sum().reset_index()
    region_sales.columns = ['Region', 'Total_Sales']
    region_sales = region_sales.sort_values('Total_Sales', ascending=False)

    return region_sales