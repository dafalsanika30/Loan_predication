FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /code

RUN apt-get update && apt-get install -y build-essential libpq-dev wget --no-install-recommends \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000
#HEALTHCHECK --interval=10s --timeout=3s --start-period=10s CMD wget -qO- http://localhost:8000/health/ || exit 1


CMD ["sh", "-c", "python manage.py migrate --noinput && python manage.py runserver 0.0.0.0:8000"]

