FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "import sklearn; print('scikit-learn version:', sklearn.__version__)"

EXPOSE 5000
CMD ["python", "app.py"]
