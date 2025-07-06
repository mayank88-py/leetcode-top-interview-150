"""
50. Pow(x, n)

Implement pow(x, n), which calculates x raised to the power n (i.e., x^n).

Example 1:
Input: x = 2.00000, n = 10
Output: 1024.00000

Example 2:
Input: x = 2.10000, n = 3
Output: 9.26100

Example 3:
Input: x = 2.00000, n = -2
Output: 0.25000
Explanation: 2^-2 = 1/2^2 = 1/4 = 0.25

Constraints:
- -100.0 < x < 100.0
- -2^31 <= n <= 2^31-1
- n is an integer.
- Either x is not zero or n > 0.
- -10^4 <= x^n <= 10^4
"""

def my_pow_fast_exponentiation(x, n):
    """
    Approach 1: Fast Exponentiation (Exponentiation by Squaring)
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Use the property: x^n = (x^2)^(n/2) if n is even, x * x^(n-1) if n is odd.
    """
    if n == 0:
        return 1.0
    
    if n < 0:
        x = 1 / x
        n = -n
    
    result = 1.0
    current_power = x
    
    while n > 0:
        if n % 2 == 1:  # n is odd
            result *= current_power
        current_power *= current_power
        n //= 2
    
    return result


def my_pow_recursive(x, n):
    """
    Approach 2: Recursive Fast Exponentiation
    Time Complexity: O(log n)
    Space Complexity: O(log n) - recursion depth
    
    Recursive implementation of fast exponentiation.
    """
    def power_helper(base, exp):
        if exp == 0:
            return 1.0
        if exp == 1:
            return base
        
        half_power = power_helper(base, exp // 2)
        
        if exp % 2 == 0:
            return half_power * half_power
        else:
            return half_power * half_power * base
    
    if n == 0:
        return 1.0
    
    if n < 0:
        return 1.0 / power_helper(x, -n)
    
    return power_helper(x, n)


def my_pow_iterative_simple(x, n):
    """
    Approach 3: Simple Iterative Approach
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Simple iteration - multiply x by itself n times.
    WARNING: Not efficient for large n.
    """
    if n == 0:
        return 1.0
    
    if abs(n) > 1000:  # Avoid TLE for large n
        return my_pow_fast_exponentiation(x, n)
    
    result = 1.0
    
    if n > 0:
        for _ in range(n):
            result *= x
    else:
        for _ in range(-n):
            result /= x
    
    return result


def my_pow_bit_manipulation(x, n):
    """
    Approach 4: Bit Manipulation
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Use bit manipulation to implement fast exponentiation.
    """
    if n == 0:
        return 1.0
    
    if n < 0:
        x = 1 / x
        n = -n
    
    result = 1.0
    
    while n:
        if n & 1:  # Check if least significant bit is 1
            result *= x
        x *= x
        n >>= 1  # Right shift by 1 bit
    
    return result


def my_pow_binary_representation(x, n):
    """
    Approach 5: Binary Representation Method
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Use binary representation of n to calculate power.
    """
    if n == 0:
        return 1.0
    
    negative = n < 0
    n = abs(n)
    
    result = 1.0
    power = x
    
    # Process each bit of n
    while n > 0:
        if n & 1:  # If current bit is 1
            result *= power
        power *= power  # Square the power for next bit
        n >>= 1
    
    return 1.0 / result if negative else result


def my_pow_memoization(x, n):
    """
    Approach 6: Memoization (for repeated calculations)
    Time Complexity: O(log n) first call, O(1) for repeated calls
    Space Complexity: O(log n) for memoization table
    
    Use memoization to cache intermediate results.
    """
    memo = {}
    
    def power_memo(base, exp):
        if exp == 0:
            return 1.0
        if exp == 1:
            return base
        
        if (base, exp) in memo:
            return memo[(base, exp)]
        
        half_power = power_memo(base, exp // 2)
        
        if exp % 2 == 0:
            result = half_power * half_power
        else:
            result = half_power * half_power * base
        
        memo[(base, exp)] = result
        return result
    
    if n == 0:
        return 1.0
    
    if n < 0:
        return 1.0 / power_memo(x, -n)
    
    return power_memo(x, n)


def my_pow_logarithmic_optimization(x, n):
    """
    Approach 7: Logarithmic Optimization
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Optimized version with special handling for common cases.
    """
    # Handle special cases
    if n == 0:
        return 1.0
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    if x == -1:
        return 1.0 if n % 2 == 0 else -1.0
    
    if n < 0:
        x = 1 / x
        n = -n
    
    result = 1.0
    base = x
    
    while n > 0:
        if n & 1:
            result *= base
        base *= base
        n >>= 1
    
    return result


def my_pow_tail_recursive(x, n):
    """
    Approach 8: Tail Recursive
    Time Complexity: O(log n)
    Space Complexity: O(log n) - recursion depth
    
    Tail recursive implementation.
    """
    def power_tail_recursive(base, exp, acc):
        if exp == 0:
            return acc
        if exp % 2 == 0:
            return power_tail_recursive(base * base, exp // 2, acc)
        else:
            return power_tail_recursive(base, exp - 1, acc * base)
    
    if n == 0:
        return 1.0
    
    if n < 0:
        return 1.0 / power_tail_recursive(x, -n, 1.0)
    
    return power_tail_recursive(x, n, 1.0)


def my_pow_optimized_edge_cases(x, n):
    """
    Approach 9: Optimized with Edge Cases
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Highly optimized version with comprehensive edge case handling.
    """
    # Edge cases
    if n == 0:
        return 1.0
    if n == 1:
        return x
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    if x == -1.0:
        return 1.0 if n % 2 == 0 else -1.0
    
    # Handle negative exponent
    is_negative = n < 0
    if is_negative:
        n = -n
        x = 1.0 / x
    
    # Fast exponentiation
    result = 1.0
    while n > 0:
        if n & 1:
            result *= x
        x *= x
        n >>= 1
    
    return result


def test_my_pow():
    """Test all approaches with various test cases."""
    
    test_cases = [
        (2.0, 10, 1024.0),
        (2.1, 3, 9.261),
        (2.0, -2, 0.25),
        (1.0, 2147483647, 1.0),
        (2.0, 0, 1.0),
        (0.0, 2, 0.0),
        (-1.0, 2, 1.0),
        (-1.0, 3, -1.0),
        (3.0, 4, 81.0),
        (0.5, 3, 0.125),
        (5.0, -2, 0.04),
        (1.0, -2147483648, 1.0),
        (2.0, 3, 8.0),
        (4.0, -1, 0.25),
        (-2.0, 4, 16.0),
        (-2.0, 3, -8.0),
    ]
    
    approaches = [
        ("Fast Exponentiation", my_pow_fast_exponentiation),
        ("Recursive", my_pow_recursive),
        ("Iterative Simple", my_pow_iterative_simple),
        ("Bit Manipulation", my_pow_bit_manipulation),
        ("Binary Representation", my_pow_binary_representation),
        ("Memoization", my_pow_memoization),
        ("Logarithmic Optimization", my_pow_logarithmic_optimization),
        ("Tail Recursive", my_pow_tail_recursive),
        ("Optimized Edge Cases", my_pow_optimized_edge_cases),
    ]
    
    for i, (x, n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: x = {x}, n = {n}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            result = func(x, n)
            # Use tolerance for floating point comparison
            status = "✓" if abs(result - expected) < 1e-5 else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_my_pow() 