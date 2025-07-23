FROM apache/airflow:2.9.1
USER airflow
WORKDIR /opt/airflow
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt