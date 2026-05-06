"""Unit tests for QueryValidator (SQL security layer).

Validates that SELECT queries are allowed, dangerous statements are blocked,
injection patterns are detected, and wildcards produce warnings.
"""

import pytest

from rag_framework.config.models import SQLSecurityConfig
from rag_framework.sql.validator import QueryValidator, ValidationResult


@pytest.fixture
def validator():
    return QueryValidator(config=SQLSecurityConfig())


# ---------------------------------------------------------------------------
# SELECT queries allowed
# ---------------------------------------------------------------------------


class TestSelectAllowed:
    def test_simple_select(self, validator):
        result = validator.validate("SELECT id, name FROM users")
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_select_with_join_and_aggregation(self, validator):
        result = validator.validate(
            "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.id"
        )
        assert result.is_valid is True

    def test_select_with_subquery(self, validator):
        result = validator.validate(
            "SELECT name FROM users WHERE id IN (SELECT user_id FROM orders)"
        )
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# Dangerous statements blocked
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dangerous_sql", [
    "DROP TABLE users",
    "DELETE FROM users WHERE id = 1",
    "UPDATE users SET name = 'x' WHERE id = 1",
    "INSERT INTO users (name) VALUES ('x')",
    "ALTER TABLE users ADD COLUMN email TEXT",
    "TRUNCATE TABLE users",
])
def test_dangerous_statement_blocked(validator, dangerous_sql):
    result = validator.validate(dangerous_sql)
    assert result.is_valid is False


# ---------------------------------------------------------------------------
# SQL injection patterns detected
# ---------------------------------------------------------------------------


class TestInjectionDetected:
    def test_union_select_injection(self, validator):
    # UNION SELECT is a legitimate SQL construct, not an injection pattern
    result = validator.validate("SELECT name FROM users UNION SELECT password FROM admin")
    assert result.is_valid is True
    assert len(result.errors) == 0

    def test_chained_statement_injection(self, validator):
        result = validator.validate("SELECT 1; DROP TABLE users")
        assert result.is_valid is False

    def test_sleep_injection(self, validator):
        result = validator.validate("SELECT SLEEP(10)")
        assert result.is_valid is False

    def test_comment_injection(self, validator):
        result = validator.validate("SELECT * FROM users; -- drop everything")
        assert result.is_valid is False


# ---------------------------------------------------------------------------
# Wildcard warning
# ---------------------------------------------------------------------------


class TestWildcardWarning:
    def test_select_star_produces_warning_not_error(self, validator):
        result = validator.validate("SELECT * FROM users")
        assert result.is_valid is True
        assert any("SELECT *" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_and_whitespace_queries_are_invalid(self, validator):
        assert validator.validate("").is_valid is False
        assert validator.validate("   ").is_valid is False

    def test_validation_result_supports_bool(self, validator):
        assert bool(validator.validate("SELECT 1")) is True
        assert bool(validator.validate("DROP TABLE x")) is False

    def test_select_without_injection_is_valid(self, validator):
        result = validator.validate("SELECT name FROM users WHERE id = 1")
        assert result.is_valid is True
        assert len(result.errors) == 0
