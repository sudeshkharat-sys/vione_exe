"""
statedb_connector.py
~~~~~~~~~~~~~~~~~~~~
Synchronous PostgreSQL connector built on SQLAlchemy Core + psycopg2.

Why synchronous?
----------------
Celery tasks run in a regular (non-async) Python thread — using an async
engine inside ``asyncio.run()`` per task wastes resources.  A shared
:class:`~sqlalchemy.pool.QueuePool` keeps connections alive and limits
concurrency without holding them longer than necessary.

Thread safety
-------------
:class:`StateDBConnector` is thread-safe.  The underlying ``QueuePool``
hands out one connection per thread and returns it on context-manager exit.

Usage
-----
    from app.connectors.statedb_connector import StateDBConnector

    connector = StateDBConnector()        # uses settings from config.py
    with connector.get_session() as conn:
        rows = connector.execute_query(conn, "SELECT 1 AS ping")
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Sequence

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.pool import QueuePool

from ..config import get_settings

logger = logging.getLogger(__name__)


class StateDBConnector:
    """
    Thin wrapper around a SQLAlchemy synchronous engine with a
    ``QueuePool`` connection pool.

    Parameters
    ----------
    dsn:
        Full SQLAlchemy connection string.  Defaults to
        ``settings.postgres_url_sync`` from :func:`~app.config.get_settings`.
    pool_size:
        Maximum number of connections kept in the pool.
    max_overflow:
        Extra connections allowed beyond *pool_size* under heavy load.
    pool_timeout:
        Seconds to wait for an available connection before raising.
    pool_recycle:
        Seconds after which an idle connection is recycled (prevents
        "server closed the connection unexpectedly" after long idle).
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        pool_size: Optional[int] = None,
        max_overflow: Optional[int] = None,
        pool_timeout: Optional[int] = None,
        pool_recycle: Optional[int] = None,
    ) -> None:
        cfg = get_settings()
        self._dsn = dsn or cfg.postgres_url_sync
        self._pool_size = pool_size if pool_size is not None else cfg.db_pool_size
        self._max_overflow = max_overflow if max_overflow is not None else cfg.db_max_overflow
        self._pool_timeout = pool_timeout if pool_timeout is not None else cfg.db_pool_timeout
        self._pool_recycle = pool_recycle if pool_recycle is not None else cfg.db_pool_recycle

        self._engine: Engine = self._build_engine()
        logger.info(
            "StateDBConnector initialised — pool_size=%d max_overflow=%d",
            self._pool_size,
            self._max_overflow,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_engine(self) -> Engine:
        return create_engine(
            self._dsn,
            poolclass=QueuePool,
            pool_size=self._pool_size,
            max_overflow=self._max_overflow,
            pool_timeout=self._pool_timeout,
            pool_recycle=self._pool_recycle,
            pool_pre_ping=True,   # validates connection before handing it out
            echo=False,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def engine(self) -> Engine:
        """Expose the raw engine (e.g. for ``metadata.create_all``)."""
        return self._engine

    @contextmanager
    def get_session(self) -> Generator[Connection, None, None]:
        """
        Context manager that yields an open :class:`~sqlalchemy.engine.Connection`.

        The connection is committed on clean exit and rolled back on exception.
        It is always returned to the pool on exit.

        Example
        -------
        ::

            with connector.get_session() as conn:
                result = conn.execute(text("SELECT NOW()"))
        """
        with self._engine.connect() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    # ── Query helpers ─────────────────────────────────────────────────────────

    def execute_query(
        self,
        conn: Connection,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a SELECT-style query and return all rows as dicts.

        Parameters
        ----------
        conn:
            Active connection from :meth:`get_session`.
        sql:
            Raw SQL string with ``:name`` style parameters.
        params:
            Mapping of parameter names → values.

        Returns
        -------
        list of dict
            Each dict maps column name → value.
        """
        result = conn.execute(text(sql), params or {})
        keys = list(result.keys())
        return [dict(zip(keys, row)) for row in result.fetchall()]

    def execute_insert(
        self,
        conn: Connection,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute an INSERT … RETURNING query and return the first row.

        Returns ``None`` if nothing was returned (e.g. no ``RETURNING`` clause).
        """
        result = conn.execute(text(sql), params or {})
        if result.returns_rows:
            row = result.fetchone()
            if row:
                return dict(zip(result.keys(), row))
        return None

    def execute_update(
        self,
        conn: Connection,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Execute an UPDATE / DELETE statement.

        Returns
        -------
        int
            Number of rows affected.
        """
        result = conn.execute(text(sql), params or {})
        return result.rowcount

    def execute_many(
        self,
        conn: Connection,
        sql: str,
        params_list: Sequence[Dict[str, Any]],
    ) -> int:
        """
        Execute a statement repeatedly for each dict in *params_list*.

        Uses ``executemany`` under the hood for efficiency.

        Returns
        -------
        int
            Total number of rows affected.
        """
        if not params_list:
            return 0
        result = conn.execute(text(sql), list(params_list))
        return result.rowcount

    def ping(self) -> bool:
        """
        Return ``True`` if the database is reachable, ``False`` otherwise.
        Useful for health-check endpoints.
        """
        try:
            with self.get_session() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.warning("StateDBConnector.ping() failed: %s", exc)
            return False

    def dispose(self) -> None:
        """Close all pooled connections.  Call on application shutdown."""
        self._engine.dispose()
        logger.info("StateDBConnector pool disposed.")
