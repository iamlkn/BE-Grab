from logging.config import fileConfig
import os # <--- ADDED: For accessing environment variables

from sqlalchemy import engine_from_config
from sqlalchemy import pool
# from sqlalchemy import create_engine # Alternative if you don't use engine_from_config for online

from alembic import context

# Ensure all your models are imported here so Base.metadata is populated
from app.db.base import Base # Assuming your Base is here
# Import your models to ensure they are registered with Base.metadata
import app.db.models.datasets
import app.db.models.cleaning_jobs
import app.db.models.connections
import app.db.models.automl_sessions
import app.db.models.finalized_models
import app.db.models.chatbot_sessions

# The import of 'engine' from app.db.session is no longer directly used
# for online mode in this modified version because we want to build the engine
# from the DATABASE_URL environment variable if available.
# from app.db.session import engine # <--- You might not need this direct import here anymore

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

def get_db_url():
    """
    Returns the database URL.
    Prioritizes DATABASE_URL environment variable.
    Falls back to sqlalchemy.url from alembic.ini if env var is not set.
    Provides a default placeholder if neither is found (though alembic.ini should have one).
    """
    env_db_url = os.environ.get("DATABASE_URL")
    if env_db_url:
        return env_db_url
    return config.get_main_option("sqlalchemy.url", "postgresql://user:pass@host/db") # Default if not in env or ini

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.
    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.
    Calls to context.execute() here emit the given string to the
    script output.
    """
    # url = config.get_main_option("sqlalchemy.url") # Old way
    url = get_db_url() # <--- MODIFIED: Use the new function
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.
    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    # section = config.get_section(config.config_ini_section) # Get the alembic config section
    # section['sqlalchemy.url'] = get_db_url() # <--- MODIFIED: Override URL from env
    # connectable = engine_from_config(
    #     section,
    #     prefix="sqlalchemy.",
    #     poolclass=pool.NullPool,
    # )

    # Simpler approach if you already have engine_from_config structure:
    # Get the existing configuration dictionary from alembic.ini
    configuration = config.get_section(config.config_ini_section, {})
    # Override the sqlalchemy.url with the one from our get_db_url function
    configuration["sqlalchemy.url"] = get_db_url() # <--- MODIFIED

    connectable = engine_from_config(
        configuration, # Pass the potentially modified configuration
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )


    # OR, if you prefer to create engine directly (less common with alembic.ini setup but works):
    # connectable = create_engine(get_db_url(), poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,  # optional: detects column type changes
            render_as_batch=True  # optional: useful for SQLite or certain alter cases
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()