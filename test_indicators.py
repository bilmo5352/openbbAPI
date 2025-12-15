import os

import pandas as pd

from analysis_service import INDICATOR_CATALOG, analyze_ticker, get_available_indicators
from main import plot_ohlc_with_indicators


def main() -> None:
    print("Total indicators in catalog:", len(INDICATOR_CATALOG))
    names = sorted(INDICATOR_CATALOG.keys())
    print("Sample indicator names:", names[:25])

    info = get_available_indicators()
    print("\nRuntime library availability:")
    print("  OpenBB technical:", info.get("openbb_technical_available"))
    print("  TA-Lib:", info.get("talib_available"))
    print("  pandas-ta:", info.get("pandas_ta_available"))
    print("  Total supported:", info.get("total_supported"))
    print("  Total available now:", info.get("total_available"))

    print("\nRequesting analysis for", len(names), "indicators...")
    try:
        result = analyze_ticker(
            ticker="RELIANCE",
            start="2024-01-01",
            end="2024-03-01",
            exchange="NSE",
            use_yfinance_fallback=True,
            indicators=names,
        )
    except Exception as e:
        print("analyze_ticker failed:", e)
        return

    print("Rows returned:", result.get("rows"))
    computed = result.get("computed_indicators", [])
    skipped = result.get("skipped_indicators", [])
    print("Computed indicators count:", len(computed))
    print("First 25 computed:", computed[:25])
    print("Skipped indicators count:", len(skipped))
    print("First 25 skipped with reasons:")
    for item in skipped[:25]:
        print("  -", item)

    # Build DataFrame from result data and plot all indicators
    data = result.get("data", [])
    if not data:
        print("No data returned in result['data']; skipping chart.")
        return

    df = pd.DataFrame(data)

    # Restore datetime index if present
    if "index" in df.columns:
        df["index"] = pd.to_datetime(df["index"])
        df.set_index("index", inplace=True)
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    # Ensure numeric columns are real numbers (JSON may have converted them to strings)
    for col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # Quick debug: how many numeric indicator columns do we have?
    numeric_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in {"Open", "High", "Low", "Close", "Volume"}
    ]
    print(f"Numeric indicator columns to plot: {len(numeric_cols)}")

    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    chart_path = os.path.join(out_dir, "RELIANCE_all_indicators_chart.html")
    plot_ohlc_with_indicators(df, "RELIANCE", chart_path)
    print("Full indicator chart written to", chart_path)


if __name__ == "__main__":
    main()


