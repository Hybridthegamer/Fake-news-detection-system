from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token

from extensions import bcrypt
from models.db_models import User
from utils.prediction_logger import PredictionLogger

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True)
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': 'Username and password are required.'}), 400

    user = User.query.filter_by(username=data['username']).first()
    if not user or not bcrypt.check_password_hash(user.password_hash, data['password']):
        return jsonify({'error': 'Invalid username or password.'}), 401

    token = create_access_token(
        identity={'user_id': user.user_id, 'role': user.role}
    )
    ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    PredictionLogger.log_audit(
        'LOGIN', f'User {user.username} authenticated.', user.user_id, ip
    )
    return jsonify({'access_token': token, 'role': user.role, 'username': user.username}), 200
