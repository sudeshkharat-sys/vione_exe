"""
query_validator.py
~~~~~~~~~~~~~~~~~~
Prevents SQL-injection through dynamically composed identifiers
(table names, column names, schema names).

Usage
-----
    from app.queries.query_validator import QueryValidator

    safe_table = QueryValidator.validate_identifier("my_table")   # raises on bad input
    safe_col   = QueryValidator.sanitize_identifier("col-name")   # strips/replaces bad chars
"""

import re
from typing import Set


# SQL reserved keywords that must never be used as bare identifiers.
# Not exhaustive — covers the most dangerous ones.
_SQL_KEYWORDS: Set[str] = {
    "SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
    "TABLE", "DATABASE", "SCHEMA", "INDEX", "VIEW", "TRIGGER", "FUNCTION",
    "PROCEDURE", "GRANT", "REVOKE", "TRUNCATE", "EXECUTE", "UNION",
    "INTERSECT", "EXCEPT", "FROM", "WHERE", "JOIN", "ON", "AS", "WITH",
    "INTO", "SET", "VALUES", "RETURNING", "CASCADE", "RESTRICT",
}

# Allowed pattern: starts with letter/underscore, followed by word chars only.
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Characters that are safe to keep when sanitizing.
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_]")


class QueryValidatorError(ValueError):
    """Raised when an identifier fails validation."""


class QueryValidator:
    """Static helpers for validating and sanitising SQL identifiers."""

    # Maximum allowed identifier length (PostgreSQL limit is 63).
    MAX_IDENTIFIER_LENGTH: int = 63

    @staticmethod
    def validate_identifier(identifier: str) -> str:
        """
        Assert that *identifier* is a safe SQL name.

        Rules
        -----
        * Must not be empty.
        * Must match ``^[A-Za-z_][A-Za-z0-9_]*$``.
        * Must not exceed :attr:`MAX_IDENTIFIER_LENGTH` characters.
        * Must not be an SQL reserved keyword.

        Returns the (unchanged) identifier on success, raises
        :class:`QueryValidatorError` otherwise.
        """
        if not identifier:
            raise QueryValidatorError("Identifier must not be empty.")

        if len(identifier) > QueryValidator.MAX_IDENTIFIER_LENGTH:
            raise QueryValidatorError(
                f"Identifier '{identifier}' exceeds the maximum length of "
                f"{QueryValidator.MAX_IDENTIFIER_LENGTH} characters."
            )

        if not _SAFE_IDENTIFIER_RE.match(identifier):
            raise QueryValidatorError(
                f"Identifier '{identifier}' contains invalid characters. "
                "Only letters, digits and underscores are allowed, and it "
                "must start with a letter or underscore."
            )

        if identifier.upper() in _SQL_KEYWORDS:
            raise QueryValidatorError(
                f"Identifier '{identifier}' is a reserved SQL keyword."
            )

        return identifier

    @staticmethod
    def sanitize_identifier(identifier: str) -> str:
        """
        Return a sanitised version of *identifier* that is safe to use.

        Transformation steps
        --------------------
        1. Strip leading/trailing whitespace.
        2. Replace all non-``[A-Za-z0-9_]`` characters with ``_``.
        3. Prepend ``_`` if the result starts with a digit.
        4. Truncate to :attr:`MAX_IDENTIFIER_LENGTH`.
        5. Append ``_safe`` if the result is an SQL keyword.

        Unlike :meth:`validate_identifier`, this never raises — it always
        produces a usable (if mangled) identifier.
        """
        cleaned = identifier.strip()
        cleaned = _SANITIZE_RE.sub("_", cleaned)

        # Must not start with a digit
        if cleaned and cleaned[0].isdigit():
            cleaned = f"_{cleaned}"

        # Enforce length limit
        cleaned = cleaned[: QueryValidator.MAX_IDENTIFIER_LENGTH]

        # Avoid reserved keywords
        if cleaned.upper() in _SQL_KEYWORDS:
            cleaned = f"{cleaned}_safe"

        return cleaned or "_"

    @staticmethod
    def validate_table_and_schema(
        table_name: str,
        schema: str = "public",
    ) -> tuple[str, str]:
        """
        Validate both *table_name* and *schema* together.

        Returns ``(validated_table_name, validated_schema)``.
        """
        return (
            QueryValidator.validate_identifier(table_name),
            QueryValidator.validate_identifier(schema),
        )
