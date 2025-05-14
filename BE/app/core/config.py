import os

# For Cloud Run, if using the Cloud SQL Proxy (handled automatically)
# DB_USER, DB_PASSWORD, DB_NAME, CLOUD_SQL_CONNECTION_NAME are set as env vars
# The socket path is typically /cloudsql/CONNECTION_NAME
# However, psycopg2 can often directly use the instance connection name for TCP if a proxy isn't explicitly used.
# Simpler: Cloud Run provides a direct TCP connection to the private IP of Cloud SQL.
# We'll construct the URL from individual components.

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST") # This will be the private IP of Cloud SQL
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "mydb")
CLOUD_SQL_CONNECTION_NAME = os.getenv("CLOUD_SQL_CONNECTION_NAME")

# Option 1: If Cloud SQL Proxy is used by Cloud Run sidecar (often the case, but can be direct private IP)
# For Unix Sockets (typical with Cloud SQL Proxy):
#SQLALCHEMY_DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@/{DB_NAME}?host=/cloudsql/{CLOUD_SQL_CONNECTION_NAME}"

# Option 2: For TCP connection (Cloud Run can connect to Private IP of Cloud SQL directly)
# Ensure your Cloud SQL instance allows connections from the VPC network Cloud Run uses.
# The DB_HOST will be injected by Cloud Run or you need to configure it.
# For simplicity in GitHub Actions, we will construct the full URL there and pass it as DATABASE_URL.
DATABASE_URL_FROM_ENV = os.getenv("DATABASE_URL")
if DATABASE_URL_FROM_ENV:
    SQLALCHEMY_DATABASE_URL = DATABASE_URL_FROM_ENV
elif DB_HOST: # If individual components are set
     SQLALCHEMY_DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else: # Fallback for local dev if DATABASE_URL is not set.
    SQLALCHEMY_DATABASE_URL = "postgresql://postgres:password@localhost:5432/mydb" # Your local default

# Alembic configuration (BE/alembic.ini) should also use this DATABASE_URL
# sqlalchemy.url = %(DATABASE_URL)s
# And in BE/alembic/env.py, read from os.environ.get('DATABASE_URL')

database_url = SQLALCHEMY_DATABASE_URL