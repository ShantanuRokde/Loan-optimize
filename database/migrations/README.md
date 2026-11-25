# Migrations

The schema is created with `Base.metadata.create_all()` on startup (`backend.database.init_db`).
It is additive-only for this release. If a future change alters existing columns, add an
Alembic environment here (`alembic init database/migrations`) and point `sqlalchemy.url`
at `backend.config.get_config().database_url()`.
