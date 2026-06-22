FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLP models
RUN python -m spacy download en_core_web_sm
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

COPY . .

RUN mkdir -p saved_models data

EXPOSE 5000

ENV FLASK_ENV=production

CMD ["python", "app.py"]
