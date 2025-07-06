"""
69. Sqrt(x)

Given a non-negative integer x, return the square root of x rounded down to the nearest integer. 
The returned integer should be non-negative as well.

You must not use any built-in exponent function or operator.

For example, do not use pow(x, 0.5) or x ** 0.5.

Example 1:
Input: x = 4
Output: 2
Explanation: The square root of 4 is 2, so we return 2.

Example 2:
Input: x = 8
Output: 2
Explanation: The square root of 8 is 2.82842..., and since we round it down to the nearest integer, 2 is returned.

Constraints:
- 0 <= x <= 2^31 - 1
"""

def my_sqrt_binary_search(x):
    """
    Approach 1: Binary Search
    Time Complexity: O(log x)
    Space Complexity: O(1)
    
    Use binary search to find the largest integer whose square is <= x.
    """
    if x < 2:
        return x
    
    left, right = 1, x // 2
    
    while left <= right:
        mid = left + (right - left) // 2
        square = mid * mid
        
        if square == x:
            return mid
        elif square < x:
            left = mid + 1
        else:
            right = mid - 1
    
    return right  # right is the largest integer whose square <= x


def my_sqrt_newton_method(x):
    """
    Approach 2: Newton's Method
    Time Complexity: O(log x)
    Space Complexity: O(1)
    
    Use Newton's method for finding square root: r = (r + x/r) / 2
    """
    if x < 2:
        return x
    
    r = x // 2  # Initial guess
    
    while r * r > x:
        r = (r + x // r) // 2
    
    return r


def my_sqrt_bit_manipulation(x):
    """
    Approach 3: Bit Manipulation
    Time Complexity: O(log x)
    Space Complexity: O(1)
    
    Build the result bit by bit from the most significant bit.
    """
    if x < 2:
        return x
    
    # Find the position of the most significant bit
    msb = 0
    temp = x
    while temp:
        msb += 1
        temp >>= 1
    
    # Start from the most significant bit of the result
    result = 0
    for i in range((msb + 1) // 2, -1, -1):
        candidate = result | (1 << i)
        if candidate * candidate <= x:
            result = candidate
    
    return result


def my_sqrt_linear_search(x):
    """
    Approach 4: Linear Search (Not optimal)
    Time Complexity: O(√x)
    Space Complexity: O(1)
    
    Linear search from 1 to √x.
    """
    if x < 2:
        return x
    
    i = 1
    while i * i <= x:
        i += 1
    
    return i - 1


def my_sqrt_exponential_search(x):
    """
    Approach 5: Exponential Search + Binary Search
    Time Complexity: O(log x)
    Space Complexity: O(1)
    
    First find the range using exponential search, then binary search.
    """
    if x < 2:
        return x
    
    # Exponential search to find range
    left = 1
    while left * left <= x:
        left <<= 1
    
    # Binary search in the range [left//2, left]
    right = left
    left = left >> 1
    
    while left <= right:
        mid = left + (right - left) // 2
        square = mid * mid
        
        if square == x:
            return mid
        elif square < x:
            left = mid + 1
        else:
            right = mid - 1
    
    return right


def my_sqrt_babylonian_method(x):
    """
    Approach 6: Babylonian Method (Ancient Algorithm)
    Time Complexity: O(log x)
    Space Complexity: O(1)
    
    Ancient Babylonian method for computing square roots.
    """
    if x < 2:
        return x
    
    guess = x // 2
    
    while True:
        better_guess = (guess + x // guess) // 2
        if better_guess >= guess:
            break
        guess = better_guess
    
    return guess


def my_sqrt_continued_fraction(x):
    """
    Approach 7: Continued Fraction Method
    Time Complexity: O(log x)
    Space Complexity: O(1)
    
    Use continued fraction approximation for square root.
    """
    if x < 2:
        return x
    
    # For integer square root, we can use a simpler approach
    # based on the continued fraction representation
    result = 0
    bit = 1 << 30  # Start with the highest possible bit
    
    while bit > x:
        bit >>= 2
    
    while bit != 0:
        if x >= result + bit:
            x -= result + bit
            result = (result >> 1) + bit
        else:
            result >>= 1
        bit >>= 2
    
    return result


def my_sqrt_digit_by_digit(x):
    """
    Approach 8: Digit-by-Digit Calculation
    Time Complexity: O(log x)
    Space Complexity: O(1)
    
    Calculate square root digit by digit (like long division).
    """
    if x < 2:
        return x
    
    # Convert to string to process digit by digit
    x_str = str(x)
    if len(x_str) % 2 == 1:
        x_str = '0' + x_str
    
    result = 0
    remainder = 0
    
    for i in range(0, len(x_str), 2):
        # Take next two digits
        remainder = remainder * 100 + int(x_str[i:i+2])
        
        # Find the largest digit d such that (20*result + d) * d <= remainder
        d = 0
        while (20 * result + d + 1) * (d + 1) <= remainder:
            d += 1
        
        remainder -= (20 * result + d) * d
        result = result * 10 + d
    
    return result


def my_sqrt_optimized_binary_search(x):
    """
    Approach 9: Optimized Binary Search
    Time Complexity: O(log x)
    Space Complexity: O(1)
    
    Optimized binary search with better bounds.
    """
    if x < 2:
        return x
    
    # Better initial bounds
    if x < 16:
        left, right = 1, 4
    elif x < 256:
        left, right = 4, 16
    else:
        left, right = 16, x // 16
    
    while left <= right:
        mid = left + (right - left) // 2
        square = mid * mid
        
        if square == x:
            return mid
        elif square < x:
            left = mid + 1
        else:
            right = mid - 1
    
    return right


def test_my_sqrt():
    """Test all approaches with various test cases."""
    
    test_cases = [
        (4, 2),
        (8, 2),
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 1),
        (9, 3),
        (16, 4),
        (25, 5),
        (26, 5),
        (100, 10),
        (121, 11),
        (144, 12),
        (2147395600, 46340),  # Large number
        (2147483647, 46340),  # Max 32-bit integer
        (49, 7),
        (64, 8),
        (81, 9),
    ]
    
    approaches = [
        ("Binary Search", my_sqrt_binary_search),
        ("Newton's Method", my_sqrt_newton_method),
        ("Bit Manipulation", my_sqrt_bit_manipulation),
        ("Linear Search", my_sqrt_linear_search),
        ("Exponential Search", my_sqrt_exponential_search),
        ("Babylonian Method", my_sqrt_babylonian_method),
        ("Continued Fraction", my_sqrt_continued_fraction),
        ("Digit by Digit", my_sqrt_digit_by_digit),
        ("Optimized Binary Search", my_sqrt_optimized_binary_search),
    ]
    
    for i, (x, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: x = {x}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            result = func(x)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_my_sqrt() 