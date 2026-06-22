from datetime import datetime, timezone
from extensions import db


def _utcnow():
    return datetime.now(timezone.utc)


class User(db.Model):
    __tablename__ = 'users'

    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False, unique=True)
    password_hash = db.Column(db.String(256), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    role = db.Column(db.String(20), default='user')
    created_at = db.Column(db.DateTime, default=_utcnow)

    def to_dict(self) -> dict:
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class Submission(db.Model):
    __tablename__ = 'submissions'

    submission_id = db.Column(db.Integer, primary_key=True)
    article_text = db.Column(db.Text, nullable=False)
    word_count = db.Column(db.Integer, nullable=False)
    input_type = db.Column(db.String(20), nullable=False)   # 'article' | 'headline'
    submitted_at = db.Column(db.DateTime, default=_utcnow)
    ip_address = db.Column(db.String(45), nullable=True)

    prediction = db.relationship(
        'Prediction', backref='submission', uselist=False, cascade='all, delete-orphan'
    )

    def to_dict(self) -> dict:
        text = self.article_text
        return {
            'submission_id': self.submission_id,
            'article_text': text[:200] + '…' if len(text) > 200 else text,
            'word_count': self.word_count,
            'input_type': self.input_type,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
        }


class Prediction(db.Model):
    __tablename__ = 'predictions'

    prediction_id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(
        db.Integer, db.ForeignKey('submissions.submission_id'), nullable=False
    )
    predicted_label = db.Column(db.String(10), nullable=False)   # 'Real' | 'Fake'
    confidence_score = db.Column(db.Float, nullable=False)
    model_used = db.Column(db.String(50), nullable=False)
    explanation_json = db.Column(db.JSON, nullable=True)
    predicted_at = db.Column(db.DateTime, default=_utcnow)

    def to_dict(self) -> dict:
        return {
            'prediction_id': self.prediction_id,
            'submission_id': self.submission_id,
            'predicted_label': self.predicted_label,
            'confidence_score': self.confidence_score,
            'model_used': self.model_used,
            'explanation_json': self.explanation_json,
            'predicted_at': self.predicted_at.isoformat() if self.predicted_at else None,
        }


class AuditLog(db.Model):
    __tablename__ = 'audit_log'

    log_id = db.Column(db.Integer, primary_key=True)
    event_type = db.Column(db.String(50), nullable=False)
    event_description = db.Column(db.Text, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=True)
    ip_address = db.Column(db.String(45), nullable=True)
    created_at = db.Column(db.DateTime, default=_utcnow)

    def to_dict(self) -> dict:
        return {
            'log_id': self.log_id,
            'event_type': self.event_type,
            'event_description': self.event_description,
            'user_id': self.user_id,
            'ip_address': self.ip_address,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
