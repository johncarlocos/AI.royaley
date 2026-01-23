"""
LOYALEY - API Dependencies
Phase 1: FastAPI Dependency Injection
"""

import logging
from typing import AsyncGenerator, Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import db_manager
from app.core.security import SecurityManager
from app.core.cache import cache_manager
from app.models import User, UserRole, APIKey

logger = logging.getLogger(__name__)

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
security_manager = SecurityManager()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Database session dependency.
    
    Yields a database session and ensures cleanup.
    """
    async with db_manager.session() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    session: AsyncSession = Depends(get_db),
) -> User:
    """
    Get current authenticated user from JWT token.
    
    Raises HTTPException if not authenticated.
    Supports demo token for development.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Support demo token for development
    if credentials.credentials == "demo-token":
        # Return a mock demo user object
        from app.models import UserRole
        from datetime import datetime
        
        class DemoUser:
            def __init__(self):
                self.id = UUID("00000000-0000-0000-0000-000000000000")
                self.email = "demo@aiprosports.com"
                self.hashed_password = ""  # Not needed for demo
                self.role = UserRole.ADMIN
                self.is_active = True
                self.is_verified = True
                self.two_factor_enabled = False
                self.two_factor_secret = None
                self.first_name = "Demo"
                self.last_name = "User"
                self.created_at = datetime.utcnow()
                self.updated_at = datetime.utcnow()
                self.last_login_at = None
        
        return DemoUser()
    
    try:
        payload = security_manager.decode_token(credentials.credentials)
        
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            )
        
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        
        # Get user from database
        result = await session.execute(
            select(User).where(User.id == UUID(user_id))
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is disabled",
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )


async def get_current_active_user(
    user: User = Depends(get_current_user),
) -> User:
    """Get current active user."""
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        )
    return user


async def get_current_verified_user(
    user: User = Depends(get_current_active_user),
) -> User:
    """Get current verified user."""
    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email not verified",
        )
    return user


async def get_admin_user(
    user: User = Depends(get_current_active_user),
) -> User:
    """Get current admin user."""
    if user.role not in [UserRole.ADMIN, UserRole.SUPER_ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


async def get_super_admin_user(
    user: User = Depends(get_current_active_user),
) -> User:
    """Get current super admin user."""
    if user.role != UserRole.SUPER_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin access required",
        )
    return user


def require_roles(*allowed_roles: UserRole):
    """
    Dependency factory for role-based access control.
    
    Usage:
        @router.get("/admin-only")
        async def admin_endpoint(user: User = Depends(require_roles(UserRole.ADMIN))):
            ...
    """
    async def role_checker(user: User = Depends(get_current_active_user)) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {[r.value for r in allowed_roles]}",
            )
        return user
    return role_checker


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    session: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise.
    
    Useful for endpoints that work with or without auth.
    """
    if not credentials:
        return None
    
    try:
        payload = security_manager.decode_token(credentials.credentials)
        user_id = payload.get("sub")
        
        if not user_id:
            return None
        
        result = await session.execute(
            select(User).where(User.id == UUID(user_id), User.is_active == True)
        )
        return result.scalar_one_or_none()
        
    except Exception:
        return None


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    session: AsyncSession = Depends(get_db),
) -> APIKey:
    """
    Verify API key from Authorization header.
    
    Used for programmatic API access.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
        )
    
    from app.core.security import hash_api_key
    
    key_hash = hash_api_key(credentials.credentials)
    
    result = await session.execute(
        select(APIKey).where(
            APIKey.key_hash == key_hash,
            APIKey.is_active == True,
        )
    )
    api_key = result.scalar_one_or_none()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    
    # Check expiration
    from datetime import datetime
    if api_key.expires_at and api_key.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key expired",
        )
    
    # Update last used
    api_key.last_used_at = datetime.utcnow()
    await session.commit()
    
    return api_key


class RateLimiter:
    """
    Rate limiting dependency.
    
    Uses Redis sliding window for distributed rate limiting.
    """
    
    def __init__(self, requests: int = 100, window: int = 60):
        self.requests = requests
        self.window = window
    
    async def __call__(
        self,
        user: Optional[User] = Depends(get_optional_user),
    ) -> None:
        """Check rate limit."""
        # Determine rate limit key
        if user:
            key = f"ratelimit:user:{user.id}"
            # Pro users get higher limits
            if user.role in [UserRole.PRO_USER, UserRole.ADMIN, UserRole.SUPER_ADMIN]:
                limit = self.requests * 2
            else:
                limit = self.requests
        else:
            # Anonymous users - would need IP, using default
            key = "ratelimit:anonymous"
            limit = self.requests // 2
        
        allowed = await cache.rate_limit(key, limit, self.window)
        
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(self.window)},
            )


# Pre-configured rate limiters
rate_limit_default = RateLimiter(requests=100, window=60)
rate_limit_strict = RateLimiter(requests=20, window=60)
rate_limit_auth = RateLimiter(requests=10, window=60)


class Pagination:
    """Pagination parameters dependency."""
    
    def __init__(
        self,
        page: int = 1,
        page_size: int = 20,
        max_page_size: int = 100,
    ):
        self.page = max(1, page)
        self.page_size = min(max(1, page_size), max_page_size)
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        return self.page_size


def get_pagination(
    page: int = 1,
    page_size: int = 20,
) -> Pagination:
    """Get pagination parameters."""
    return Pagination(page=page, page_size=page_size)
