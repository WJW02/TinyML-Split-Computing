#!/bin/sh

gunicornWorkers="${gunicorn_workers:-1}"
echo "gunicorn_workers: ${gunicorn_workers}"
gunicorn --log-level debug --keep-alive 5 --bind 0.0.0.0:8080 --workers="$gunicornWorkers" --timeout 120 app:app
