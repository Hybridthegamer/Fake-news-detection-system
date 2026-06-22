"""Integration tests for the Flask REST API."""
import json
import pytest


class TestClassifyEndpoint:
    def test_missing_text_field(self, client):
        resp = client.post('/api/classify', json={})
        assert resp.status_code == 400
        assert 'error' in resp.get_json()

    def test_empty_text(self, client):
        resp = client.post('/api/classify', json={'text': '   '})
        assert resp.status_code == 400

    def test_too_short(self, client):
        resp = client.post('/api/classify', json={'text': 'Hi'})
        assert resp.status_code == 400

    def test_too_long(self, client):
        long_text = 'word ' * 5001
        resp = client.post('/api/classify', json={'text': long_text})
        assert resp.status_code == 400

    def test_valid_input_returns_503_without_model(self, client):
        """Without a trained model the endpoint returns 503, not 500."""
        article = (
            "Scientists at the University of Oxford have developed a new vaccine "
            "that shows 95% effectiveness against the latest variant of the virus. "
            "Clinical trials involving 30,000 participants were conducted across "
            "multiple countries. The results were published in the New England Journal "
            "of Medicine and have been peer-reviewed by independent experts."
        )
        resp = client.post('/api/classify', json={'text': article})
        assert resp.status_code in (200, 503)

    def test_xss_sanitisation(self, client):
        xss_payload = "<script>alert('xss')</script> " + "scientists discover new findings " * 15
        resp = client.post('/api/classify', json={'text': xss_payload})
        assert resp.status_code in (200, 400, 503)


class TestAuthEndpoint:
    def test_missing_credentials(self, client):
        resp = client.post('/api/auth/login', json={})
        assert resp.status_code == 400

    def test_invalid_credentials(self, client):
        resp = client.post('/api/auth/login', json={'username': 'nobody', 'password': 'wrong'})
        assert resp.status_code == 401

    def test_login_returns_token_for_valid_admin(self, client, app):
        from extensions import bcrypt, db
        from models.db_models import User
        with app.app_context():
            if not User.query.filter_by(username='testadmin').first():
                admin = User(
                    username='testadmin',
                    password_hash=bcrypt.generate_password_hash('testpass').decode(),
                    email='testadmin@test.com',
                    role='admin',
                )
                db.session.add(admin)
                db.session.commit()

        resp = client.post('/api/auth/login', json={'username': 'testadmin', 'password': 'testpass'})
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'access_token' in data
        assert data['role'] == 'admin'


class TestHistoryEndpoint:
    def test_requires_auth(self, client):
        resp = client.get('/api/history')
        assert resp.status_code == 401

    def test_stats_requires_auth(self, client):
        resp = client.get('/api/stats')
        assert resp.status_code == 401
