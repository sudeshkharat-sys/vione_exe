"""
statedb_manager.py
~~~~~~~~~~~~~~~~~~
High-level manager that handles database and table lifecycle.

Responsibilities
----------------
* Create the target database if it does not exist (requires a connection to
  the ``postgres`` system database — CREATE DATABASE cannot run in a
  transaction).
* Create all application tables using :mod:`.table_creation`.
* Provide helpers for introspection (list tables) and teardown (drop tables).

Usage
-----
    from app.connectors.statedb_manager import StateDBManager

    manager = StateDBManager()
    manager.initialize_database()        # idempotent — safe to call on startup
"""

from __future__ import annotations

import logging
from typing import List, Optional

import psycopg2
from psycopg2 import sql as psql
from sqlalchemy import inspect, text

from ..config import get_settings
from .statedb_connector import StateDBConnector
from .table_creation import metadata
from ..queries.queries import DatabaseQueries

logger = logging.getLogger(__name__)


class StateDBManager:
    """
    Manages the full lifecycle of the PostgreSQL database used by the
    AI Vision Platform.

    Parameters
    ----------
    connector:
        Optional pre-built :class:`~.statedb_connector.StateDBConnector`.
        If ``None`` a new one is created from the current settings.
    """

    def __init__(self, connector: Optional[StateDBConnector] = None) -> None:
        self._settings = get_settings()
        self._connector = connector or StateDBConnector()

    # ── Public API ────────────────────────────────────────────────────────────

    def initialize_database(self) -> None:
        """
        Full bootstrap sequence — safe to call every time the application
        starts.

        Steps
        -----
        1. Connect to the ``postgres`` system database.
        2. Create the target database if it does not exist.
        3. Create all application tables (idempotent via ``checkfirst=True``).
        """
        self._ensure_database_exists()
        self.create_tables_if_not_exists()
        logger.info("StateDBManager: database fully initialised.")

    def create_tables_if_not_exists(self) -> None:
        """
        Create every table defined in :mod:`.table_creation` if it is absent.

        Uses SQLAlchemy's ``checkfirst=True`` so existing tables are never
        dropped or modified.
        """
        engine = self._connector.engine
        metadata.create_all(engine, checkfirst=True)
        table_names = list(metadata.tables.keys())
        logger.info(
            "StateDBManager: ensured tables exist: %s",
            ", ".join(table_names),
        )

    def drop_all_tables(self, *, confirm: bool = False) -> None:
        """
        Drop **all** application tables in dependency order.

        This is destructive and irreversible.  Pass ``confirm=True`` to
        proceed (prevents accidental calls).

        Parameters
        ----------
        confirm:
            Must be ``True`` to execute.  Raises ``RuntimeError`` otherwise.
        """
        if not confirm:
            raise RuntimeError(
                "drop_all_tables() requires confirm=True.  "
                "This will permanently delete all data."
            )
        engine = self._connector.engine
        metadata.drop_all(engine, checkfirst=True)
        logger.warning("StateDBManager: ALL application tables have been dropped.")

    def list_tables(self, schema: str = "public") -> List[str]:
        """
        Return the names of all tables currently in the database.

        Parameters
        ----------
        schema:
            PostgreSQL schema to inspect (default ``"public"``).

        Returns
        -------
        list of str
            Sorted list of table names.
        """
        engine = self._connector.engine
        inspector = inspect(engine)
        tables = inspector.get_table_names(schema=schema)
        return sorted(tables)

    def table_exists(self, table_name: str, schema: str = "public") -> bool:
        """Return ``True`` if *table_name* exists in *schema*."""
        return table_name in self.list_tables(schema=schema)

    def ping(self) -> bool:
        """Delegate to the connector's :meth:`~StateDBConnector.ping`."""
        return self._connector.ping()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_database_exists(self) -> None:
        """
        Connect to the ``postgres`` maintenance database and create the
        target database if it does not already exist.

        ``CREATE DATABASE`` cannot run inside a transaction, so we use a raw
        psycopg2 connection with ``autocommit=True``.
        """
        cfg = self._settings
        target_db = cfg.postgres_db

        # Build a DSN pointing at the *postgres* system database
        sys_dsn = (
            f"host={cfg.postgres_host} port={cfg.postgres_port} "
            f"dbname=postgres "
            f"user={cfg.postgres_user} password={cfg.postgres_password}"
        )

        try:
            conn = psycopg2.connect(sys_dsn)
            conn.autocommit = True
            cursor = conn.cursor()

            # Check existence
            cursor.execute(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
                (target_db,),
            )
            exists = cursor.fetchone() is not None

            if not exists:
                # Use psql.Identifier so the db name is properly quoted
                cursor.execute(
                    psql.SQL("CREATE DATABASE {}").format(
                        psql.Identifier(target_db)
                    )
                )
                logger.info("StateDBManager: created database '%s'.", target_db)
            else:
                logger.debug(
                    "StateDBManager: database '%s' already exists.", target_db
                )

            cursor.close()
            conn.close()

        except Exception as exc:
            logger.error(
                "StateDBManager: failed to ensure database '%s' exists: %s",
                target_db,
                exc,
            )
            raise
