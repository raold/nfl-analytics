"""
API Authentication Module

Exports:
- require_api_key: FastAPI dependency for authentication
- APIKey: Authenticated API key model
- APIKeyAuthenticator: Authentication backend
"""

from api.auth.api_key_auth import (
    require_api_key,
    APIKey,
    APIKeyAuthenticator,
    get_authenticator
)

__all__ = [
    "require_api_key",
    "APIKey",
    "APIKeyAuthenticator",
    "get_authenticator"
]
