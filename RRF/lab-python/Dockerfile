FROM python:3.12-slim

RUN pip install --no-cache-dir \
    fastembed==0.8.0 \
    numpy==2.5.3 \
    onnxruntime==1.30.0 \
    requests==2.34.2

WORKDIR /app
COPY . /app/

CMD ["python", "-u", "benchmark.py"]
