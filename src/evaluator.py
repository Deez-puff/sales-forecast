import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_test, y_pred, model_name="Model"):
    """
    Evaluates a forecasting model using multiple metrics.
    Returns a dictionary of evaluation results.
    """
    # ── Core Metrics ──────────────────────────────────────
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    results = {
        "model_name" : model_name,
        "MAE"        : mae,
        "RMSE"       : rmse,
        "R2"         : r2,
        "MAPE"       : mape
    }

    return results


def print_evaluation(results):
    """
    Prints evaluation results in a business-friendly format.
    """
    print(f"\n── {results['model_name']} Evaluation ──")
    print(f"MAE  (Mean Absolute Error):       ${results['MAE']:,.2f}")
    print(f"RMSE (Root Mean Squared Error):   ${results['RMSE']:,.2f}")
    print(f"R²   (Variance Explained):         {results['R2']:.4f}")
    print(f"MAPE (Mean Abs % Error):           {results['MAPE']:.2f}%")

    print(f"\n── What This Means for Business ──")
    print(f"✅ On average the model's monthly sales prediction")
    print(f"   is off by ${results['MAE']:,.2f}")

    if results['MAPE'] < 10:
        print(f"✅ MAPE of {results['MAPE']:.2f}% is EXCELLENT (under 10%)")
    elif results['MAPE'] < 20:
        print(f"✅ MAPE of {results['MAPE']:.2f}% is GOOD (under 20%)")
    elif results['MAPE'] < 30:
        print(f"⚠️  MAPE of {results['MAPE']:.2f}% is ACCEPTABLE (under 30%)")
    else:
        print(f"❌ MAPE of {results['MAPE']:.2f}% needs improvement (over 30%)")

    if results['R2'] > 0.8:
        print(f"✅ R² of {results['R2']:.4f} means the model explains")
        print(f"   {results['R2']*100:.1f}% of sales variance — STRONG fit")
    elif results['R2'] > 0.6:
        print(f"✅ R² of {results['R2']:.4f} means the model explains")
        print(f"   {results['R2']*100:.1f}% of sales variance — GOOD fit")
    else:
        print(f"⚠️  R² of {results['R2']:.4f} means the model explains")
        print(f"   {results['R2']*100:.1f}% of sales variance — MODERATE fit")


def compare_models(results_list):
    """
    Compares multiple models and returns the best one.
    """
    best = min(results_list, key=lambda x: x['MAE'])
    print(f"\n── Model Comparison ──")
    for r in results_list:
        marker = "🏆" if r['model_name'] == best['model_name'] else "  "
        print(f"{marker} {r['model_name']:25} MAE: ${r['MAE']:,.2f}  R²: {r['R2']:.4f}  MAPE: {r['MAPE']:.2f}%")
    print(f"\n🏆 Best Model: {best['model_name']} (lowest MAE)")
    return best