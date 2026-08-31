"""Mobile cost cards omit token prices while desktop keeps both rate columns."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "web/styles.css").read_text()
DESKTOP, MOBILE = CSS.split("@media (max-width: 720px) {", 1)
MOBILE = MOBILE.split("\n@media", 1)[0]


def rule(css: str, selector: str) -> str:
    match = re.search(re.escape(selector) + r"\s*\{([^}]+)\}", css)
    assert match is not None, selector
    return match[1]


def test_token_price_cells_and_mobile_row_are_hidden_only_on_mobile() -> None:
    assert "display: none" in rule(MOBILE, ".budget-row .budget-price")
    assert "display: none" in rule(MOBILE, ".budget-price-mobile")
    assert "display: none" in rule(MOBILE, ".budget-table thead th:nth-child(n + 5)")
    assert "display: none" not in rule(DESKTOP, ".budget-table .budget-price")
    assert "display: none" in rule(DESKTOP, ".budget-price-mobile")
    assert ".budget-table thead th:nth-child(n + 5)" not in DESKTOP


def test_mobile_keeps_three_spend_columns_without_trailing_price_separators() -> None:
    assert "grid-template-columns: repeat(3, minmax(0, 1fr))" in MOBILE
    assert "display: flex" in rule(MOBILE, ".budget-row td")
    # Only visible trial rows and the first two spend cells receive separators.
    assert "border-bottom: 1px solid" in rule(MOBILE, ".budget-row:has(+ .budget-row)")
    assert ".budget-row:not(:last-child)" not in MOBILE
    assert ".budget-row td:not(:last-child)" not in MOBILE
    assert ".budget-row .budget-api,\n  .budget-row .budget-training" in MOBILE
    assert "display: none" not in rule(MOBILE, ".budget-table-actions")
    assert "display: none" not in rule(MOBILE, ".budget-price-sources")
