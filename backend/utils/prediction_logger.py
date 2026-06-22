import logging
from typing import Optional

from extensions import db
from models.db_models import AuditLog, Prediction, Submission

logger = logging.getLogger(__name__)


class PredictionLogger:
    """Persists classification results and audit events to the database."""

    @staticmethod
    def log(
        article_text: str,
        word_count: int,
        input_type: str,
        predicted_label: str,
        confidence_score: float,
        model_used: str,
        explanation_json: Optional[dict],
        ip_address: Optional[str] = None,
    ) -> Prediction:
        submission = Submission(
            article_text=article_text,
            word_count=word_count,
            input_type=input_type,
            ip_address=ip_address,
        )
        db.session.add(submission)
        db.session.flush()

        prediction = Prediction(
            submission_id=submission.submission_id,
            predicted_label=predicted_label,
            confidence_score=confidence_score,
            model_used=model_used,
            explanation_json=explanation_json,
        )
        db.session.add(prediction)
        db.session.commit()
        return prediction

    @staticmethod
    def log_audit(
        event_type: str,
        description: str,
        user_id: Optional[int] = None,
        ip_address: Optional[str] = None,
    ) -> None:
        try:
            entry = AuditLog(
                event_type=event_type,
                event_description=description,
                user_id=user_id,
                ip_address=ip_address,
            )
            db.session.add(entry)
            db.session.commit()
        except Exception as exc:
            logger.error("Audit log failed: %s", exc)
            db.session.rollback()
