# WSGI entrypoint. Elastic Beanstalk's Python platform looks for an
# `application` callable in application.py, so we re-export it from app.
from app import application  # noqa: F401
