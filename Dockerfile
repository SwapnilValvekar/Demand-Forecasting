# SIOS - Smart Inventory Optimization System (Streamlit UI + GBDT model)
FROM python:3.12-slim

LABEL project="SIOS Demand Forecasting"
LABEL author="Swapnil Valvekar"

WORKDIR /app

# install python dependencies first (better layer caching)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# copy the app, the model code and the bundled dataset
COPY app.py generate_data.py train_model.py dist_charts.py gen_explain.py ./
COPY sios_demand_panel.csv sios_holidays.csv sios_items.csv ./

EXPOSE 8501

# container is healthy once Streamlit answers its health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').getcode()==200 else 1)"

ENTRYPOINT ["streamlit","run","app.py", \
            "--server.port=8501","--server.address=0.0.0.0", \
            "--browser.gatherUsageStats=false"]
