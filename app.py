from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

st.set_page_config(
    page_title="Albuquerque Crime Rate Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .main-title {font-size: 2.15rem; font-weight: 750; letter-spacing: -0.7px;}
    .subtitle {color: #64748b; font-size: 1rem; margin-bottom: 1.1rem;}
    div[data-testid="stMetric"] {
        background: #f8fafc; border: 1px solid #e2e8f0;
        padding: 15px 16px; border-radius: 14px;
    }
    div[data-testid="stMetricLabel"] {color: #475569;}
    .insight {
        padding: 14px 16px; border-left: 4px solid #0f766e;
        background: #f0fdfa; border-radius: 8px; margin: 8px 0 16px 0;
    }
    .small-note {color: #64748b; font-size: 0.85rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Albuquerque Crime Rate Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Time-series analysis and forecasting · Albuquerque, New Mexico · 1975–2015</div>',
    unsafe_allow_html=True
)

DEFAULT_CSV = Path(__file__).with_name("Crime in Context, 1975-2015(1).csv")

@st.cache_data
def read_csv_bytes(file_bytes):
    import io
    return pd.read_csv(io.BytesIO(file_bytes))

def clean_and_select(data):
    data = data.copy()
    data.columns = [str(c).strip() for c in data.columns]
    required = {"report_year", "agency_jurisdiction", "crimes_percapita"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(sorted(missing)))
    data["report_year"] = pd.to_numeric(data["report_year"], errors="coerce")
    data["crimes_percapita"] = pd.to_numeric(data["crimes_percapita"], errors="coerce")
    data = data.drop_duplicates()
    data = data.dropna(subset=["report_year", "agency_jurisdiction", "crimes_percapita"])
    data["report_year"] = data["report_year"].astype(int)
    data["agency_jurisdiction"] = data["agency_jurisdiction"].astype(str)
    return data

# ---------- Sidebar / data source ----------
with st.sidebar:
    st.header("Dashboard controls")
    st.caption("Use the supplied project dataset or upload another compatible CSV.")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])

try:
    if uploaded is not None:
        raw = read_csv_bytes(uploaded.getvalue())
        source_name = uploaded.name
    elif DEFAULT_CSV.exists():
        raw = pd.read_csv(DEFAULT_CSV)
        source_name = DEFAULT_CSV.name
    else:
        st.error("The CSV was not found beside app.py. Upload the project CSV in the sidebar.")
        st.stop()
    data = clean_and_select(raw)
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

agencies = sorted(data["agency_jurisdiction"].dropna().unique().tolist())
default_agency = next((a for a in agencies if "Albuquerque" in a), agencies[0] if agencies else None)
with st.sidebar:
    selected_agency = st.selectbox(
        "Agency / jurisdiction",
        agencies,
        index=agencies.index(default_agency) if default_agency in agencies else 0,
    )

selected = data[data["agency_jurisdiction"] == selected_agency].sort_values("report_year").copy()
selected = selected.drop_duplicates(subset=["report_year"], keep="first")
selected = selected.dropna(subset=["crimes_percapita"])
if len(selected) < 12:
    st.warning("This agency has fewer than 12 annual observations. Forecast comparisons may be unstable.")

years = selected["report_year"].astype(int)
min_year, max_year = int(years.min()), int(years.max())
with st.sidebar:
    year_range = st.slider(
        "Trend chart year range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
    )
    forecast_horizon = st.selectbox("Future forecast horizon", [3, 5, 8, 10], index=1)
    test_size = st.selectbox(
        "Hold-out test years",
        [3, 5, 7],
        index=1,
        help="The last selected number of years are held out for comparing forecast accuracy."
    )

crime = selected["crimes_percapita"].astype(float).to_numpy()
years_array = selected["report_year"].to_numpy()
series = pd.Series(crime, index=pd.Index(years_array, name="Year"), name="Crime rate per capita")
chart_data = selected[(selected["report_year"] >= year_range[0]) & (selected["report_year"] <= year_range[1])].copy()

# ---------- Models and metrics ----------
def forecast_model(model_name, train_values, steps):
    """Fit a model on a training array and return its forecast."""
    train_values = np.asarray(train_values, dtype=float)
    if model_name == "SES":
        fit = SimpleExpSmoothing(train_values, initialization_method="estimated").fit(optimized=True)
        pred = np.asarray(fit.forecast(steps), dtype=float)
    elif model_name == "Holt":
        fit = Holt(train_values, initialization_method="estimated").fit(optimized=True)
        pred = np.asarray(fit.forecast(steps), dtype=float)
    elif model_name == "AR(1)":
        fit = ARIMA(train_values, order=(1, 0, 0), trend="c").fit()
        pred = np.asarray(fit.forecast(steps), dtype=float)
    elif model_name == "AR(2)":
        fit = ARIMA(train_values, order=(2, 0, 0), trend="c").fit()
        pred = np.asarray(fit.forecast(steps), dtype=float)
    elif model_name == "MA(1)":
        fit = ARIMA(train_values, order=(0, 0, 1), trend="c").fit()
        pred = np.asarray(fit.forecast(steps), dtype=float)
    elif model_name == "ARMA(1,1)":
        fit = ARIMA(train_values, order=(1, 0, 1), trend="c").fit()
        pred = np.asarray(fit.forecast(steps), dtype=float)
    else:  # The report's automatic ARIMA selection was ARIMA(1,0,0).
        fit = ARIMA(train_values, order=(1, 0, 0), trend="c").fit()
        pred = np.asarray(fit.forecast(steps), dtype=float)
    return fit, pred

def metric_values(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = actual - predicted
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    nonzero = np.abs(actual) > 1e-12
    mape = float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100) if nonzero.any() else np.nan
    return rmse, mae, mape

MODEL_NAMES = ["SES", "Holt", "AR(1)", "AR(2)", "MA(1)", "ARMA(1,1)", "ARIMA(1,0,0)"]
MODEL_MAP = {
    "SES": "SES", "Holt": "Holt", "AR(1)": "AR(1)",
    "AR(2)": "AR(2)", "MA(1)": "MA(1)", "ARMA(1,1)": "ARMA(1,1)",
    "ARIMA(1,0,0)": "ARIMA(1,0,0)"
}

@st.cache_data(show_spinner=False)
def compare_models(values_tuple, years_tuple, holdout):
    vals = np.asarray(values_tuple, dtype=float)
    yrs = np.asarray(years_tuple, dtype=int)
    if len(vals) <= holdout + 5:
        return pd.DataFrame(), pd.DataFrame(), None
    train, test = vals[:-holdout], vals[-holdout:]
    rows, predictions = [], []
    for name in MODEL_NAMES:
        try:
            fit, pred = forecast_model(name, train, len(test))
            rmse, mae, mape = metric_values(test, pred)
            rows.append({"Model": name, "RMSE": rmse, "MAE": mae, "MAPE (%)": mape})
            for year, actual, forecast in zip(yrs[-holdout:], test, pred):
                predictions.append({"Model": name, "Year": int(year), "Actual": float(actual), "Predicted": float(forecast)})
        except Exception as exc:
            rows.append({"Model": name, "RMSE": np.nan, "MAE": np.nan, "MAPE (%)": np.nan, "Issue": str(exc)})
    return pd.DataFrame(rows).sort_values("RMSE", na_position="last"), pd.DataFrame(predictions), len(train)

with st.spinner("Fitting the forecasting models..."):
    accuracy_df, test_predictions, train_n = compare_models(tuple(crime), tuple(years_array), int(test_size))

# ---------- Navigation ----------
tabs = st.tabs([
    "Overview", "Trend & Crime Categories", "Model Comparison",
    "Forecast", "Stationarity & Diagnostics", "Data"
])

# ---------- Overview ----------
with tabs[0]:
    st.subheader("Crime rate at a glance")
    mean_rate = float(np.mean(crime))
    min_idx, max_idx = int(np.argmin(crime)), int(np.argmax(crime))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annual observations", f"{len(selected)}")
    col2.metric("Year range", f"{min_year}–{max_year}")
    col3.metric("Average crime rate", f"{mean_rate:,.2f}")
    col4.metric("Latest recorded rate", f"{crime[-1]:,.2f}", f"{crime[-1] - crime[-2]:+.2f} vs previous year" if len(crime) > 1 else None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data["report_year"], y=chart_data["crimes_percapita"],
        mode="lines+markers", name="Crime rate",
        line=dict(width=3), marker=dict(size=6),
        hovertemplate="Year: %{x}<br>Crime rate: %{y:.2f}<extra></extra>"
    ))
    fig.update_layout(
        title="Annual crimes per capita",
        xaxis_title="Year", yaxis_title="Crimes per capita",
        hovermode="x unified", template="plotly_white", height=410,
        margin=dict(l=15, r=15, t=55, b=15)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div class="insight"><b>Project insight:</b> The report describes a rise through the 1980s and early 1990s, a decline after the early-1990s peak, lower rates around 2011–2012, and an increase toward 2015. The recorded minimum in this dataset is {crime[min_idx]:,.2f} ({years_array[min_idx]}) and the maximum is {crime[max_idx]:,.2f} ({years_array[max_idx]}).</div>',
        unsafe_allow_html=True
    )
    st.caption(f"Source file: {source_name}. The selected jurisdiction is {selected_agency}.")

# ---------- EDA ----------
with tabs[1]:
    st.subheader("Explore the historical data")
    metric_options = [
        ("crimes_percapita", "Crimes per capita"),
        ("violent_crimes", "Violent crimes"),
        ("homicides", "Homicides"),
        ("rapes", "Rapes"),
        ("assaults", "Assaults"),
        ("robberies", "Robberies"),
        ("population", "Population"),
    ]
    available_metrics = [(c, label) for c, label in metric_options if c in selected.columns]
    selected_metric = st.selectbox("Choose a measure", options=[x[0] for x in available_metrics],
                                   format_func=lambda x: dict(available_metrics)[x])
    smooth = selected[["report_year", selected_metric]].copy()
    smooth[selected_metric] = pd.to_numeric(smooth[selected_metric], errors="coerce")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=smooth["report_year"], y=smooth[selected_metric], mode="lines+markers", name=dict(available_metrics)[selected_metric]))
    if selected_metric == "crimes_percapita":
        smooth["Moving average (3 years)"] = smooth[selected_metric].rolling(3, center=True).mean()
        fig.add_trace(go.Scatter(x=smooth["report_year"], y=smooth["Moving average (3 years)"], mode="lines", name="3-year moving average", line=dict(width=3, dash="dash")))
    fig.update_layout(title=f"{dict(available_metrics)[selected_metric]} by year",
                      xaxis_title="Year", yaxis_title=dict(available_metrics)[selected_metric],
                      template="plotly_white", height=420, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    if all(c in selected.columns for c in ["homicides", "rapes", "assaults", "robberies"]):
        st.markdown("#### Crime category comparison")
        category = st.radio("Display", ["Counts", "Per-capita rates"], horizontal=True)
        if category == "Counts":
            cols = ["homicides", "rapes", "assaults", "robberies"]
            labels = {"homicides": "Homicides", "rapes": "Rapes", "assaults": "Assaults", "robberies": "Robberies"}
        else:
            cols = [c for c in ["homicides_percapita", "rapes_percapita", "assaults_percapita", "robberies_percapita"] if c in selected.columns]
            labels = {c: c.replace("_percapita", "").title() for c in cols}
        long = selected.melt(id_vars="report_year", value_vars=cols, var_name="Category", value_name="Value")
        long["Category"] = long["Category"].map(labels)
        fig2 = px.line(long, x="report_year", y="Value", color="Category", markers=True,
                       title=f"Crime categories over time ({category.lower()})", template="plotly_white")
        fig2.update_layout(xaxis_title="Year", yaxis_title="Value", height=420)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Descriptive statistics")
    desc = selected[["crimes_percapita"]].describe().T.rename(columns={
        "count": "Count", "mean": "Mean", "std": "Std. deviation",
        "min": "Minimum", "25%": "25th percentile", "50%": "Median",
        "75%": "75th percentile", "max": "Maximum"
    })
    st.dataframe(desc.round(2), use_container_width=True)

# ---------- Model comparison ----------
with tabs[2]:
    st.subheader("Compare forecasting models")
    st.write(f"The last **{test_size} years** are held out as the test set. Models are trained on the earlier years and tested on the unseen years.")
    if accuracy_df.empty or accuracy_df["RMSE"].notna().sum() == 0:
        st.warning("Not enough observations to compare models using this hold-out setting.")
    else:
        best = accuracy_df.dropna(subset=["RMSE"]).iloc[0]
        a, b, c = st.columns(3)
        a.metric("Lowest test RMSE", str(best["Model"]))
        b.metric("RMSE", f'{best["RMSE"]:.3f}')
        c.metric("MAPE", f'{best["MAPE (%)"]:.3f}%')
        st.dataframe(accuracy_df.round(3), use_container_width=True, hide_index=True)
        metric_choice = st.selectbox("Metric chart", ["RMSE", "MAE", "MAPE (%)"])
        plot_df = accuracy_df.dropna(subset=[metric_choice]).sort_values(metric_choice)
        fig = px.bar(plot_df, x="Model", y=metric_choice, title=f"Hold-out model comparison by {metric_choice}",
                     text_auto=".2f", template="plotly_white")
        fig.update_layout(height=390, xaxis_title="", yaxis_title=metric_choice)
        st.plotly_chart(fig, use_container_width=True)
        if not test_predictions.empty:
            st.markdown("#### Actual versus predicted values")
            chosen_model = st.selectbox("Model to inspect", accuracy_df["Model"].tolist())
            chosen = test_predictions[test_predictions["Model"] == chosen_model]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=chosen["Year"], y=chosen["Actual"], mode="lines+markers", name="Actual"))
            fig.add_trace(go.Scatter(x=chosen["Year"], y=chosen["Predicted"], mode="lines+markers", name="Predicted"))
            fig.update_layout(template="plotly_white", height=360, xaxis_title="Year", yaxis_title="Crime rate", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        st.caption("RMSE, MAE, and MAPE are recalculated in Python. Results may differ slightly from R because the fitting libraries and optimization defaults differ.")

# ---------- Future forecast ----------
with tabs[3]:
    st.subheader(f"Forecast the next {forecast_horizon} years")
    full_model = ARIMA(crime, order=(1, 0, 0), trend="c").fit()
    fc = full_model.get_forecast(steps=int(forecast_horizon))
    pred_mean = np.asarray(fc.predicted_mean, dtype=float)
    ci = np.asarray(fc.conf_int(alpha=0.05), dtype=float)
    future_years = np.arange(max_year + 1, max_year + forecast_horizon + 1)
    forecast_table = pd.DataFrame({
        "Year": future_years,
        "Forecast": pred_mean,
        "Lower 95%": ci[:, 0],
        "Upper 95%": ci[:, 1],
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_array, y=crime, mode="lines", name="Historical rate", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=future_years, y=pred_mean, mode="lines+markers", name="Forecast", line=dict(width=3, dash="dash")))
    fig.add_trace(go.Scatter(
        x=np.concatenate([future_years, future_years[::-1]]),
        y=np.concatenate([ci[:, 1], ci[:, 0][::-1]]),
        fill="toself", fillcolor="rgba(15,118,110,0.16)",
        line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
        name="95% prediction interval"
    ))
    fig.update_layout(title="Historical crime rate and future forecast",
                      xaxis_title="Year", yaxis_title="Crimes per capita",
                      template="plotly_white", height=450, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(forecast_table.round(2), use_container_width=True, hide_index=True)
    st.download_button("Download forecast table (CSV)", forecast_table.to_csv(index=False).encode("utf-8"),
                       file_name="albuquerque_crime_rate_forecast.csv", mime="text/csv")
    st.info("The project report's final model is ARIMA(1,0,0). This dashboard refits that structure in Python. Its forecast values and intervals may differ slightly from the values in the R report.")
    with st.expander("Model details"):
        st.write(f"ARIMA(1,0,0) fitted to {len(crime)} observations.")
        st.write(f"AIC: {full_model.aic:.3f} · BIC: {full_model.bic:.3f}")
        st.write("The forecast is a statistical estimate, not a certainty or a statement about the causes of crime.")

# ---------- Stationarity and diagnostics ----------
with tabs[4]:
    st.subheader("Stationarity and residual diagnostics")
    c1, c2 = st.columns(2)
    try:
        adf_orig = adfuller(crime, autolag="AIC")
        with c1:
            st.markdown("#### Original series · ADF test")
            st.metric("ADF p-value", f"{adf_orig[1]:.4f}")
            st.write("Result:", "Stationarity supported at 5% level" if adf_orig[1] < 0.05 else "Insufficient evidence of stationarity at 5% level")
            st.caption("Python's ADF defaults can produce a different p-value from the R tseries test.")
    except Exception as exc:
        c1.warning(f"ADF test could not be calculated: {exc}")
    try:
        diff = np.diff(crime)
        adf_diff = adfuller(diff, autolag="AIC")
        with c2:
            st.markdown("#### First-differenced series · ADF test")
            st.metric("ADF p-value", f"{adf_diff[1]:.4f}")
            st.write("Result:", "Stationarity supported at 5% level" if adf_diff[1] < 0.05 else "Insufficient evidence of stationarity at 5% level")
    except Exception as exc:
        c2.warning(f"Differenced ADF test could not be calculated: {exc}")

    diff_frame = pd.DataFrame({"Year": years_array[1:], "First difference": diff})
    fig = px.line(diff_frame, x="Year", y="First difference", markers=True,
                  title="First difference of crime rate", template="plotly_white")
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### ACF and PACF")
    max_lags = max(1, min(15, len(crime) // 2 - 1))
    acf_vals = acf(crime, nlags=max_lags, fft=False)
    pacf_vals = pacf(crime, nlags=max_lags, method="ywm")
    lag_df = pd.DataFrame({"Lag": np.arange(1, max_lags + 1), "ACF": acf_vals[1:], "PACF": pacf_vals[1:]})
    ca, cp = st.columns(2)
    with ca:
        fig_acf = px.bar(lag_df, x="Lag", y="ACF", title="Autocorrelation (ACF)", template="plotly_white")
        fig_acf.update_layout(height=330)
        st.plotly_chart(fig_acf, use_container_width=True)
    with cp:
        fig_pacf = px.bar(lag_df, x="Lag", y="PACF", title="Partial autocorrelation (PACF)", template="plotly_white")
        fig_pacf.update_layout(height=330)
        st.plotly_chart(fig_pacf, use_container_width=True)

    st.markdown("#### Residual checks for the final ARIMA(1,0,0)")
    final_fit = ARIMA(crime, order=(1, 0, 0), trend="c").fit()
    residuals = np.asarray(final_fit.resid)
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        fig_res = px.scatter(x=years_array, y=residuals, labels={"x": "Year", "y": "Residual"},
                             title="Residuals over time", template="plotly_white")
        fig_res.add_hline(y=0, line_dash="dash")
        fig_res.update_layout(height=330)
        st.plotly_chart(fig_res, use_container_width=True)
    with res_col2:
        fig_hist = px.histogram(x=residuals, nbins=12, labels={"x": "Residual"},
                                title="Residual distribution", template="plotly_white")
        fig_hist.update_layout(height=330)
        st.plotly_chart(fig_hist, use_container_width=True)
    try:
        lb = acorr_ljungbox(residuals, lags=[min(10, max(1, len(residuals)//5))], return_df=True)
        st.dataframe(lb.reset_index(names="Lag").round(4), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.caption(f"Ljung–Box test unavailable: {exc}")
    st.caption("The R report's ADF and Ljung–Box values are reported in the written report. Values above are recalculated using Python implementations.")

# ---------- Raw data ----------
with tabs[5]:
    st.subheader("Selected agency data")
    st.write(f"{len(selected)} unique annual observations for {selected_agency}.")
    display_cols = [c for c in [
        "report_year", "agency_jurisdiction", "population", "violent_crimes",
        "homicides", "rapes", "assaults", "robberies", "crimes_percapita",
        "homicides_percapita", "rapes_percapita", "assaults_percapita", "robberies_percapita"
    ] if c in selected.columns]
    st.dataframe(selected[display_cols].reset_index(drop=True), use_container_width=True, hide_index=True)
    st.download_button("Download selected agency data (CSV)",
                       selected[display_cols].to_csv(index=False).encode("utf-8"),
                       file_name="selected_agency_crime_data.csv", mime="text/csv")
    st.markdown("#### Data quality checks")
    q1, q2, q3 = st.columns(3)
    q1.metric("Missing values in selected rows", int(selected[display_cols].isna().sum().sum()))
    q2.metric("Duplicate years removed", int(data[data["agency_jurisdiction"] == selected_agency].duplicated(subset=["report_year"]).sum()))
    expected_years = set(range(min_year, max_year + 1))
    missing_years = sorted(expected_years - set(selected["report_year"].astype(int)))
    q3.metric("Missing years", len(missing_years))
    if missing_years:
        st.write("Missing years:", ", ".join(map(str, missing_years)))

st.divider()
st.caption("Academic project dashboard · Based on the supplied R script, report, and Crime in Context dataset. Forecasts are exploratory and should not be treated as official crime estimates.")
