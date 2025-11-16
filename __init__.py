import os
import sys

_BASE_DIR = os.path.dirname(__file__)
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from testy_jednostkowe_fn import (
    calculate_discount,
    count_wovels,
    fibonacci,
    flatten_list,
    is_palindrome,
    is_prime,
    word_frequencies,
)

__all__ = [
    'calculate_discount',
    'count_wovels',
    'fibonacci',
    'flatten_list',
    'is_palindrome',
    'is_prime',
    'word_frequencies',
]
