import os

# Read PORT from environment variable (defaults to 10000 on Render, or 5000)
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Concurrency and timeout settings optimized for 512MB RAM free instances
workers = 1
threads = 2
timeout = 120
accesslog = "-"
errorlog = "-"
