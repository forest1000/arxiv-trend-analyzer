FROM python:3.11-slim
WORKDIR /app

COPY pyproject.toml README.md /app/
RUN mkdir -p /app/app
COPY app/__init__.py /app/app/__init__.py

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .

COPY . /app

# UIで動かすなら
ENTRYPOINT ["streamlit", "run", "app/ui_streamlit.py", "--server.address", "0.0.0.0", "--server.port", "8501"]