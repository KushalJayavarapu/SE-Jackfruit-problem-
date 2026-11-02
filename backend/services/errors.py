class ValidationError(Exception):
    """A request was rejected; the message is shown to the user as is (NFR-03)."""


class NotFoundError(ValidationError):
    """The record does not exist, or belongs to another account (NFR-02)."""


class ForbiddenError(ValidationError):
    """The user is signed in but not allowed to do this."""
