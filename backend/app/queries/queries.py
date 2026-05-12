"""
queries.py
~~~~~~~~~~
Centralised SQL query constants and helper factories.

All queries target PostgreSQL.  Identifiers that come from external input
must be validated with :class:`~app.queries.query_validator.QueryValidator`
before being interpolated.

Usage
-----
    from app.queries.queries import DatabaseQueries, CommonQueries

    sql = DatabaseQueries.CHECK_DATABASE_EXISTS
    create_sql = DatabaseQueries.get_create_database_query("my_db")
"""

from .query_validator import QueryValidator


class DatabaseQueries:
    """Queries that operate on the database / server level."""

    # ── Server-level checks ───────────────────────────────────────────────────

    CHECK_DATABASE_EXISTS: str = """
        SELECT 1
        FROM   pg_catalog.pg_database
        WHERE  datname = :db_name
    """

    TEST_CONNECTION: str = "SELECT 1 AS alive"

    LIST_ALL_DATABASES: str = """
        SELECT datname
        FROM   pg_catalog.pg_database
        WHERE  datistemplate = FALSE
        ORDER  BY datname
    """

    # ── Schema-level helpers ──────────────────────────────────────────────────

    LIST_TABLES_IN_SCHEMA: str = """
        SELECT table_name
        FROM   information_schema.tables
        WHERE  table_schema = :schema_name
          AND  table_type   = 'BASE TABLE'
        ORDER  BY table_name
    """

    CHECK_TABLE_EXISTS: str = """
        SELECT 1
        FROM   information_schema.tables
        WHERE  table_schema = :schema_name
          AND  table_name   = :table_name
    """

    @staticmethod
    def get_create_database_query(db_name: str) -> str:
        """
        Return a ``CREATE DATABASE`` statement for *db_name*.

        The name is validated via :class:`QueryValidator` before interpolation
        because ``CREATE DATABASE`` cannot use parameterised queries in
        PostgreSQL.
        """
        safe_name = QueryValidator.validate_identifier(db_name)
        return f'CREATE DATABASE "{safe_name}"'

    @staticmethod
    def get_drop_database_query(db_name: str) -> str:
        """Return a ``DROP DATABASE IF EXISTS`` statement."""
        safe_name = QueryValidator.validate_identifier(db_name)
        return f'DROP DATABASE IF EXISTS "{safe_name}"'


class CommonQueries:
    """Generic DML helpers used across the application."""

    # ── Projects ──────────────────────────────────────────────────────────────

    INSERT_PROJECT: str = """
        INSERT INTO projects (id, name, description, classes, created_at)
        VALUES (:id, :name, :description, :classes, NOW())
        RETURNING id, name, description, classes, created_at
    """

    SELECT_PROJECT_BY_ID: str = """
        SELECT id, name, description, classes, created_at
        FROM   projects
        WHERE  id = :project_id
    """

    SELECT_ALL_PROJECTS: str = """
        SELECT id, name, description, classes, created_at
        FROM   projects
        ORDER  BY created_at DESC
    """

    UPDATE_PROJECT_CLASSES: str = """
        UPDATE projects
        SET    classes = :classes
        WHERE  id = :project_id
        RETURNING id, name, description, classes, created_at
    """

    DELETE_PROJECT: str = """
        DELETE FROM projects
        WHERE  id = :project_id
    """

    # ── Images ────────────────────────────────────────────────────────────────

    INSERT_IMAGE: str = """
        INSERT INTO images (id, project_id, filename, filepath, width, height, status, created_at)
        VALUES (:id, :project_id, :filename, :filepath, :width, :height, :status, NOW())
        RETURNING id, project_id, filename, filepath, width, height, status, created_at
    """

    SELECT_IMAGES_BY_PROJECT: str = """
        SELECT id, project_id, filename, filepath, width, height, status, created_at
        FROM   images
        WHERE  project_id = :project_id
        ORDER  BY created_at DESC
    """

    SELECT_PENDING_IMAGES: str = """
        SELECT id, project_id, filename, filepath, width, height, status, created_at
        FROM   images
        WHERE  project_id = :project_id
          AND  status     = 'pending'
        ORDER  BY created_at DESC
    """

    UPDATE_IMAGE_STATUS: str = """
        UPDATE images
        SET    status = :status
        WHERE  id     = :image_id
        RETURNING id, status
    """

    # ── Annotations ───────────────────────────────────────────────────────────

    INSERT_ANNOTATION: str = """
        INSERT INTO annotations (id, image_id, class_name, bbox, source, created_at)
        VALUES (:id, :image_id, :class_name, :bbox, :source, NOW())
        RETURNING id, image_id, class_name, bbox, source, created_at
    """

    SELECT_ANNOTATIONS_BY_IMAGE: str = """
        SELECT id, image_id, class_name, bbox, source, created_at
        FROM   annotations
        WHERE  image_id = :image_id
        ORDER  BY created_at
    """

    COUNT_ANNOTATIONS_BY_IMAGE: str = """
        SELECT COUNT(*) AS annotation_count
        FROM   annotations
        WHERE  image_id = :image_id
    """

    DELETE_ANNOTATIONS_BY_IMAGE: str = """
        DELETE FROM annotations
        WHERE  image_id = :image_id
    """

    # ── Stale-annotation guard ────────────────────────────────────────────────
    # Images whose status = 'annotated' but have zero annotation rows

    SELECT_STALE_ANNOTATED_IMAGES: str = """
        SELECT i.id
        FROM   images i
        LEFT   JOIN annotations a ON a.image_id = i.id
        WHERE  i.project_id = :project_id
          AND  i.status     = 'annotated'
        GROUP  BY i.id
        HAVING COUNT(a.id) = 0
    """

    RESET_STALE_IMAGES_TO_PENDING: str = """
        UPDATE images
        SET    status = 'pending'
        WHERE  project_id = :project_id
          AND  status     = 'annotated'
          AND  id NOT IN (
              SELECT DISTINCT image_id FROM annotations
          )
    """
