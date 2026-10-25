# Albuquerque Crime Rate Analysis and Forecasting Dashboard

An interactive Python dashboard for exploring historical crime rates in Albuquerque, New Mexico, comparing time-series forecasting models, and visualizing future crime-rate predictions.

## 🚀 Live Dashboard

**[Click here to open the live dashboard](https://krupa-project-albuquerque-crime-rate-dashboard.streamlit.app/)**

## 💻 GitHub Repository

[View the source code on GitHub](https://github.com/krupa-cs/Albuquerque-Crime-Rate-Dashboard)

## 📌 Project Overview

This project analyzes crime-rate data for Albuquerque, New Mexico, from 1975 to 2015. It uses time-series analysis and forecasting techniques to understand historical patterns, examine changes over time, and compare the performance of different forecasting models.

The original analysis was developed in R and converted into an interactive Python dashboard to make the results easier to explore and understand.

## ✨ Dashboard Features

- **Crime Overview:** View key statistics, annual crime rates, and historical trends.
- **Exploratory Data Analysis:** Explore crime-rate patterns and a three-year moving average.
- **Crime Category Analysis:** Examine available crime categories, including homicides, rapes, assaults, and robberies.
- **Forecasting Model Comparison:** Compare Simple Exponential Smoothing (SES), Holt's method, AR(1), AR(2), MA(1), ARMA(1,1), and ARIMA(1,0,0).
- **Model Evaluation:** Compare forecasting performance using RMSE, MAE, and MAPE.
- **Statistical Diagnostics:** Examine stationarity using the Augmented Dickey-Fuller (ADF) test, along with ACF, PACF, and residual diagnostics.
- **Future Forecasting:** Visualize future crime-rate estimates with 95% prediction intervals.
- **Data Export:** Download forecast results and selected data as CSV files.

## 🛠️ Technologies Used

- **Python** — Data analysis and model implementation
- **Streamlit** — Interactive dashboard development
- **Pandas and NumPy** — Data cleaning and numerical analysis
- **Plotly** — Interactive charts and visualizations
- **Statsmodels** — Time-series forecasting and statistical tests
- **Git and GitHub** — Version control and source-code management

## 📊 Forecasting Models

The dashboard compares the following forecasting approaches:

1. Simple Exponential Smoothing (SES)
2. Holt's Linear Trend Method
3. Autoregressive Model AR(1)
4. Autoregressive Model AR(2)
5. Moving Average Model MA(1)
6. ARMA(1,1)
7. ARIMA(1,0,0)

Model performance is evaluated using a hold-out test set. The final ARIMA(1,0,0) model is also used to generate future forecasts, following the model choice documented in the project report.

## ⚙️ Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/krupa-cs/Albuquerque-Crime-Rate-Dashboard.git
cd Albuquerque-Crime-Rate-Dashboard
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch the dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser, usually at `http://localhost:8501`.

## 📁 Project Structure

```text
Albuquerque-Crime-Rate-Dashboard/
├── app.py
├── Crime in Context, 1975-2015(1).csv
├── requirements.txt
└── README.md
```

## 🔍 Key Project Insights

The original analysis examined long-term changes in Albuquerque's crime rate, tested the stationarity of the time series, and compared multiple forecasting models.

According to the project report, AR(2) achieved the lowest test-set RMSE among the compared models, while ARIMA(1,0,0) was selected for the final full-series forecasting stage. These are different model-selection objectives.

Forecasts are statistical estimates based on historical data and should not be interpreted as certain future outcomes or as evidence of the causes of crime.

## 👩‍💻 Author

**Krupa M.**

B.Sc. Computer Science, Statistics and Mathematics

GitHub: [krupa-cs](https://github.com/krupa-cs)

---

If you find this project useful, feel free to explore the repository and the live dashboard.
