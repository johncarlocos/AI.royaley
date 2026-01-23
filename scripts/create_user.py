#!/usr/bin/env python3
"""
Script to create a new user in the database
Usage:
  python scripts/create_user.py <email> <password> [role]

Notes:
- Uses the app's bcrypt settings for hashing.
- Idempotent by email: if the user exists, updates password + role.
"""

import asyncio
import sys
from uuid import uuid4

from sqlalchemy import text

from app.core.security import SecurityManager
from app.core.database import db_manager
from app.models.models import UserRole


def _parse_role(raw_role: str) -> UserRole:
    """
    Parse a user role from CLI.

    Accepts either:
    - Enum name labels used by Postgres (e.g. ADMIN, SUPER_ADMIN)
    - Enum values used by the app (e.g. admin, super_admin)
    """
    normalized = raw_role.strip()
    if not normalized:
        raise ValueError("Role cannot be empty")

    # 1) Match by Enum name (DB labels)
    by_name = normalized.upper()
    if by_name in UserRole.__members__:
        return UserRole[by_name]

    # 2) Match by Enum value (app-facing)
    by_value = normalized.lower()
    for r in UserRole:
        if r.value.lower() == by_value:
            return r

    allowed = sorted({r.value for r in UserRole} | set(UserRole.__members__.keys()))
    raise ValueError(f"Invalid role '{raw_role}'. Allowed: {', '.join(allowed)}")


async def create_or_update_user(email: str, password: str, role: UserRole) -> bool:
    """Create or update a user in the database (by email)."""
    security = SecurityManager()
    
    # Hash the password
    hashed_password = security.hash_password(password)
    
    # Check password strength
    is_strong, issues = security.check_password_strength(password)
    if not is_strong:
        print(f"Warning: Password strength issues: {', '.join(issues)}")
        print("User will still be created, but consider using a stronger password.")
    
    await db_manager.initialize()
    # Store role using the DB enum label (Enum member name), e.g. ADMIN
    role_db = role.name
    try:
        async with db_manager.session() as session:
            # Check if user already exists
            existing = await session.execute(
                text("SELECT id FROM users WHERE email = :email LIMIT 1"),
                {"email": email},
            )
            row = existing.first()

            if row:
                update_sql = text(
                    """
                    UPDATE users
                    SET
                        hashed_password = :hashed_password,
                        role = :role,
                        is_active = :is_active,
                        is_verified = :is_verified,
                        updated_at = NOW()
                    WHERE email = :email
                    """
                )
                await session.execute(
                    update_sql,
                    {
                        "email": email,
                        "hashed_password": hashed_password,
                        "role": role_db,
                        "is_active": True,
                        "is_verified": True,
                    },
                )

                print("✓ User updated successfully!")
                print(f"  Email: {email}")
                print(f"  Role: {role.value}")
                print("  Status: Active and Verified")
                return True

            # Create user
            user_id = uuid4()
            insert_sql = text(
                """
                INSERT INTO users (
                    id, email, hashed_password, role, is_active, is_verified,
                    two_factor_enabled, two_factor_secret, created_at, updated_at
                ) VALUES (
                    :id, :email, :hashed_password, :role, :is_active, :is_verified,
                    :two_factor_enabled, :two_factor_secret, NOW(), NOW()
                )
                """
            )

            await session.execute(
                insert_sql,
                {
                    "id": user_id,
                    "email": email,
                    "hashed_password": hashed_password,
                    "role": role_db,
                    "is_active": True,
                    "is_verified": True,
                    "two_factor_enabled": False,
                    "two_factor_secret": None,
                },
            )

            print("✓ User created successfully!")
            print(f"  Email: {email}")
            print(f"  User ID: {user_id}")
            print(f"  Role: {role.value}")
            print("  Status: Active and Verified")
            return True
    except Exception as e:
        print(f"Error creating/updating user: {e}")
        return False


async def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python scripts/create_user.py <email> <password> [role]")
        sys.exit(1)
    
    email = sys.argv[1]
    password = sys.argv[2]
    role = _parse_role(sys.argv[3]) if len(sys.argv) == 4 else UserRole.USER
    
    success = await create_or_update_user(email, password, role)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())

