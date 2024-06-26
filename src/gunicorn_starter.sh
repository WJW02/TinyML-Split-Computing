#!/bin/sh

gunicorn_workers="${gunicorn_workers:-1}"
echo "gunicorn_workers: ${gunicorn_workers}"
gunicorn --log-level debug --keep-alive 5 --workers ${gunicorn_workers} --b 0.0.0.0:8000 api.wsgi --timeout 120 app:app