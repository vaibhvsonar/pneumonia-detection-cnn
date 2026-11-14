web: gunicorn --bind 0.0.0.0:5000 --bind 0.0.0.0:10000 --workers 1 --threads 2 --timeout 120 app:app
