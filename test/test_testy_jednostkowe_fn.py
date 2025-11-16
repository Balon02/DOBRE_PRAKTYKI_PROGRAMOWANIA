import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from testy_jednostkowe_fn import (
    calculate_discount,
    count_wovels,
    fibonacci,
    flatten_list,
    is_palindrome,
    is_prime,
    word_frequencies,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("kajak", True),
        ("Kobyła ma mały bok", True),
        ("python", False),
        ("", True),
        ("A", True),
    ],
)
def test_is_palindrome(text, expected):
    assert is_palindrome(text) == expected


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, 0),
        (1, 1),
        (5, 5),
        (10, 55),
    ],
)
def test_fibonacci_values(n, expected):
    assert fibonacci(n) == expected


def test_fibonacci_negative_raises():
    with pytest.raises(ValueError):
        fibonacci(-1)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Python", 2),
        ("AEIOUY", 6),
        ("bcd", 0),
        ("", 0),
        ("Próba żółwia", 5),
    ],
)
def test_count_wovels(text, expected):
    assert count_wovels(text) == expected


@pytest.mark.parametrize(
    ("price", "discount", "expected"),
    [
        (100, 0.2, 80.0),
        (50, 0, 50.0),
        (200, 1, 0.0),
    ],
)
def test_calculate_discount_returns_expected_value(price, discount, expected):
    assert calculate_discount(price, discount) == expected


@pytest.mark.parametrize(
    ("price", "discount"),
    [
        (100, -0.1),
        (100, 1.5),
    ],
)
def test_calculate_discount_invalid_discounts(price, discount):
    with pytest.raises(ValueError):
        calculate_discount(price, discount)


@pytest.mark.parametrize(
    ("nested", "expected"),
    [
        ([1, 2, 3], [1, 2, 3]),
        ([1, [2, 3], [4, [5]]], [1, 2, 3, 4, 5]),
        ([], []),
        ([[[1]]], [1]),
        ([1, [2, [3, [4]]]], [1, 2, 3, 4]),
    ],
)
def test_flatten_list(nested, expected):
    assert flatten_list(nested) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("To be or not to be", {"to": 2, "be": 2, "or": 1, "not": 1}),
        ("Hello, hello!", {"hello": 2}),
        ("", {}),
        ("Python Python python", {"python": 3}),
        ("Ala ma kota, a kot ma Ale.", {"ala": 1, "ma": 2, "kota": 1, "a": 1, "kot": 1, "ale": 1}),
    ],
)
def test_word_frequencies(text, expected):
    assert word_frequencies(text) == expected


@pytest.mark.parametrize(
    ("number", "expected"),
    [
        (2, True),
        (3, True),
        (4, False),
        (0, False),
        (1, False),
        (5, True),
        (97, True),
    ],
)
def test_is_prime(number, expected):
    assert is_prime(number) == expected
