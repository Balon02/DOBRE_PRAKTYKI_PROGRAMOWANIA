import re

def is_palindrome(text: str) -> bool: 
    text = text.lower().replace(' ', '')
    return text == text[::-1]

def fibonacci(n: int) -> int:
    if n < 0: raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n): a, b = b, a + b
    return a

def count_wovels(text: str) -> int: 
    text = text.lower()
    return sum([1 for l in text if l in['a', 'ą', 'e', 'ę', 'i', 'o', 'ó', 'u', 'y']])

def calculate_discount(price: float, discount: float) -> float: 
    if discount < 0 or discount > 1: raise ValueError('discount cannot be negative' if discount <0 else 'discount cannot be > 1')
    return price * (1 - discount)

def flatten_list(nested_list: list) -> list:
    result = []
    for element in nested_list:
        if isinstance(element, list): result.extend(flatten_list(element))
        else: result.append(element)
    return result

def word_frequencies(text: str) -> dict:
    words = words = re.findall(r'\w+', text.lower(), re.UNICODE)
    uniques = set(words)
    freqs = {}
    for unique in uniques: freqs[unique] = len([w for w in words if w == unique])
    return freqs

def is_prime(n: int) -> bool:
    if n < 2: return False
    result = True
    for i in range(2, n): 
        if n%i == 0: result = False
    return result