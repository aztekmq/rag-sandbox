"""Simple authentication helpers for the Gradio UI."""

from __future__ import annotations

import logging
from typing import Literal

from app.config import ADMIN_PASS, ADMIN_USER, USER_PASS, USER_USER

logger = logging.getLogger(__name__)


def authenticate(username: str, password: str, mode: Literal["admin", "user"]) -> bool:
    """Validate user credentials for a given role.

    Args:
        username: Submitted username.
        password: Submitted password.
        mode: Either "admin" or "user".

    Returns:
        True when the provided credentials match the configured values.
    """

    logger.debug("Authenticating user %s for mode %s", username, mode)
    if mode == "admin":
        valid = username == ADMIN_USER and password == ADMIN_PASS
    else:
        valid = username == USER_USER and password == USER_PASS

    logger.info("Authentication %s for user %s", "passed" if valid else "failed", username)
    return valid
