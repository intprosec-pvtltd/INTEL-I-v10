import getpass
import os

from db.database import SessionLocal
from db.model import User
from auth.auth import hash_password, validate_password_strength


def main():
    email = (os.getenv("SUPER_ADMIN_EMAIL") or input("Email: ")).strip().lower()
    name = (os.getenv("SUPER_ADMIN_NAME") or input("Full name: ")).strip()
    password = os.getenv("SUPER_ADMIN_PASSWORD") or getpass.getpass("Password: ")
    validate_password_strength(password)
    db = SessionLocal()
    try:
        if db.query(User.id).filter(User.email == email).first():
            raise SystemExit("A user with that email already exists")
        db.add(User(full_name=name, email=email, password=hash_password(password), role="super_admin",
                    department=None, manager_id=None, is_active=True, token_version=0))
        db.commit()
        print("Super Admin created")
    finally:
        db.close()


if __name__ == "__main__":
    main()
