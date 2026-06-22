import os
import logging

from flask import Flask
from flask_cors import CORS

from config import config
from extensions import db, jwt, bcrypt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(config_name: str = None) -> Flask:
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')

    app = Flask(__name__)
    app.config.from_object(config[config_name])

    os.makedirs(app.config['MODEL_DIR'], exist_ok=True)

    db.init_app(app)
    jwt.init_app(app)
    bcrypt.init_app(app)
    CORS(app, origins=['http://localhost:3000', 'http://localhost:5173', '*'])

    from api.auth import auth_bp
    from api.classify import classify_bp
    from api.history import history_bp

    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(classify_bp, url_prefix='/api')
    app.register_blueprint(history_bp, url_prefix='/api')

    with app.app_context():
        db.create_all()
        _init_classifier(app)

    return app


def _init_classifier(app: Flask) -> None:
    """Load NLP components once at startup and attach to app."""
    from nlp.preprocessor import TextPreprocessor
    from nlp.classifier import FakeNewsClassifier

    try:
        app.preprocessor = TextPreprocessor()
        logger.info("TextPreprocessor initialised.")
    except RuntimeError as e:
        logger.error("TextPreprocessor failed: %s", e)
        app.preprocessor = None

    app.classifier = FakeNewsClassifier(
        distilbert_path=app.config['DISTILBERT_MODEL_PATH'],
        lr_classifier_path=app.config['LR_CLASSIFIER_PATH'],
        vectorizer_path=app.config['TFIDF_VECTORIZER_PATH'],
    )

    if app.classifier.is_ready:
        logger.info("Active model: %s", app.classifier.active_model_name)
    else:
        logger.warning(
            "No trained model found. Run: python scripts/train_model.py"
        )


if __name__ == '__main__':
    application = create_app()
    application.run(host='0.0.0.0', port=5000, debug=True)
