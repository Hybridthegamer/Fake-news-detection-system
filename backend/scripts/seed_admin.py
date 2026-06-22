#!/usr/bin/env python3
"""
Create the default admin user.

Usage:
    python scripts/seed_admin.py
    python scripts/seed_admin.py --username myadmin --password MyP@ssw0rd

IMPORTANT: Change the default password immediately in any non-development environment.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from extensions import bcrypt, db
from models.db_models import User


def main():
    parser = argparse.ArgumentParser(description='Seed admin user.')
    parser.add_argument('--username', default='admin')
    parser.add_argument('--password', default='admin123')
    parser.add_argument('--email', default='admin@fakenews.local')
    args = parser.parse_args()

    app = create_app('development')
    with app.app_context():
        existing = User.query.filter_by(username=args.username).first()
        if existing:
            print(f"User '{args.username}' already exists.")
            sys.exit(0)

        admin = User(
            username=args.username,
            password_hash=bcrypt.generate_password_hash(args.password).decode('utf-8'),
            email=args.email,
            role='admin',
        )
        db.session.add(admin)
        db.session.commit()
        print(f"Admin user created:")
        print(f"  Username : {args.username}")
        print(f"  Password : {args.password}")
        print(f"  Email    : {args.email}")
        if args.password == 'admin123':
            print("\n  WARNING: Change this password before deploying to production!")


if __name__ == '__main__':
    main()
