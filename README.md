# Disease Spread Simulation (Influenza, Worldwide)

- Data: WHO FluNet weekly virological counts (worldwide)
- Model: Differentiable SEIRS + Transformer β(t)
- App: Streamlit dashboard with global map and per-country forecasts

Setup:
1. py -3 -m venv .venv
2. .\.venv\Scripts\Activate.ps1
3. pip install -r requirements.txt
4. Download FluNet CSV into data/raw/flunet_weekly.csv
5. python src\data\flunet_prepare.py