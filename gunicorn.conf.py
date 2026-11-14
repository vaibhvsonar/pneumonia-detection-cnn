import os

# Get port from environment (if set)
env_port = os.environ.get("PORT", "5000")

# Bind to BOTH port 5000 and port 10000 simultaneously.
# This guarantees Render finds an open port regardless of whether it scans 5000 or 10000.
ports = list(dict.fromkeys([str(env_port), "5000", "10000"]))
bind = [f"0.0.0.0:{p}" for p in ports]

# Concurrency and timeout settings optimized for 512MB RAM free instances
workers = 1
threads = 2
timeout = 120
accesslog = "-"
errorlog = "-"
