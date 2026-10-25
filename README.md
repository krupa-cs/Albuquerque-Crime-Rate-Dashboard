# Albuquerque Crime Rate Dashboard

This Python dashboard recreates the analysis workflow from the supplied R script and report.

## Files
- `app.py` — Streamlit dashboard
- `Crime in Context, 1975-2015(1).csv` — dataset used by the dashboard
- `requirements.txt` — Python packages

## Run on Windows (VS Code)

1. Extract this folder.
2. Open the folder in VS Code.
3. Open the terminal in VS Code.
4. Create a virtual environment (recommended):

   ```bash
   python -m venv .venv
   ```

5. Activate it:

   ```bash
   .venv\Scripts\activate
   ```

6. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

7. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

Streamlit will print a local URL, normally `http://localhost:8501`. Open that URL in your browser.

## Dashboard sections
- Overview and key statistics
- Crime-rate trend and three-year moving average
- Crime category charts
- Hold-out model comparison (SES, Holt, AR(1), AR(2), MA(1), ARMA(1,1), ARIMA(1,0,0))
- Future forecast with a 95% prediction interval
- ADF, ACF/PACF, residual and Ljung–Box diagnostics
- Data table and CSV downloads

## Notes
- The dashboard defaults to Albuquerque, NM, matching the project report. The sidebar also lets you choose another agency in the supplied dataset.
- The report says the R `auto.arima()` procedure selected ARIMA(1,0,0) on the full series. To keep the Python implementation aligned with that documented result, the dashboard uses ARIMA(1,0,0) for the final forecast.
- Python model estimates can differ slightly from R because the software libraries and estimation defaults differ. The dashboard labels the model metrics and forecasts it recalculates in Python.
- Forecasts are exploratory statistical estimates, not official forecasts or causal conclusions.
