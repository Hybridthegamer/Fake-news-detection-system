import os
from datetime import timedelta


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-CHANGE-IN-PRODUCTION')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'jwt-dev-secret-CHANGE-IN-PRODUCTION')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)

    MAX_ARTICLE_WORDS = 5000
    MIN_ARTICLE_WORDS = 50
    MAX_HEADLINE_WORDS = 50
    MIN_HEADLINE_WORDS = 5
    CONFIDENCE_THRESHOLD = 0.70

    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
    TFIDF_VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
    LR_CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'lr_classifier.pkl')
    DISTILBERT_MODEL_PATH = os.path.join(MODEL_DIR, 'distilbert_finetuned')


class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DEV_DATABASE_URL', 'sqlite:///fakenews_dev.db'
    )


class TestingConfig(Config):
    DEBUG = True
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL', 'sqlite:///fakenews_prod.db'
    )


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig,
}
