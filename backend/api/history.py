from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required

from models.db_models import Prediction, Submission

history_bp = Blueprint('history', __name__)


def _require_admin():
    identity = get_jwt_identity()
    if not identity or identity.get('role') != 'admin':
        return jsonify({'error': 'Admin access required.'}), 403
    return None


@history_bp.route('/history', methods=['GET'])
@jwt_required()
def get_history():
    err = _require_admin()
    if err:
        return err

    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)

    paginated = (
        Submission.query
        .order_by(Submission.submitted_at.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )

    records = []
    for sub in paginated.items:
        entry = sub.to_dict()
        if sub.prediction:
            entry['prediction'] = sub.prediction.to_dict()
        records.append(entry)

    return jsonify({
        'records': records,
        'total': paginated.total,
        'pages': paginated.pages,
        'current_page': page,
        'per_page': per_page,
    }), 200


@history_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_stats():
    err = _require_admin()
    if err:
        return err

    total = Submission.query.count()
    fake_count = Prediction.query.filter_by(predicted_label='Fake').count()
    real_count = Prediction.query.filter_by(predicted_label='Real').count()
    db_count = Prediction.query.filter_by(model_used='DistilBERT').count()
    lr_count = Prediction.query.filter_by(model_used='LR-TFIDF').count()

    avg_conf = None
    if total:
        from extensions import db
        from sqlalchemy import func
        result = db.session.query(func.avg(Prediction.confidence_score)).scalar()
        avg_conf = round(float(result) * 100, 2) if result else None

    return jsonify({
        'total_submissions': total,
        'fake_count': fake_count,
        'real_count': real_count,
        'distilbert_predictions': db_count,
        'lr_tfidf_predictions': lr_count,
        'average_confidence_pct': avg_conf,
    }), 200
