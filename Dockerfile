
FROM python:3.10.10


WORKDIR /app


COPY . /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000


ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=127.0.0.1


CMD ["python", "app.py"]
