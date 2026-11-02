import datetime


def now_iso():
    """Current UTC time as an ISO 8601 string ending in Z."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat() + "Z"
