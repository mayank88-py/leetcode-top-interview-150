"""
172. Factorial Trailing Zeroes

Given an integer n, return the number of trailing zeroes in n!.

Note that n! = n * (n - 1) * (n - 2) * ... * 3 * 2 * 1.

Example 1:
Input: n = 3
Output: 0
Explanation: 3! = 6, no trailing zero.

Example 2:
Input: n = 5
Output: 1
Explanation: 5! = 120, one trailing zero.

Example 3:
Input: n = 0
Output: 0

Constraints:
- 0 <= n <= 10^4

Follow up: Could you write a solution that works in logarithmic time complexity?
"""

def trailing_zeroes_count_fives(n):
    """
    Approach 1: Count Factors of 5
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Trailing zeroes are created by factors of 10 = 2 × 5.
    Since there are more factors of 2 than 5, we count factors of 5.
    """
    count = 0
    power_of_five = 5
    
    while power_of_five <= n:
        count += n // power_of_five
        power_of_five *= 5
    
    return count


def trailing_zeroes_iterative_division(n):
    """
    Approach 2: Iterative Division by 5
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Keep dividing n by 5 and add to count until n becomes 0.
    """
    count = 0
    
    while n >= 5:
        n //= 5
        count += n
    
    return count


def trailing_zeroes_recursive(n):
    """
    Approach 3: Recursive Approach
    Time Complexity: O(log n)
    Space Complexity: O(log n) - recursion depth
    
    Recursive implementation of counting factors of 5.
    """
    if n < 5:
        return 0
    
    return n // 5 + trailing_zeroes_recursive(n // 5)


def trailing_zeroes_mathematical_formula(n):
    """
    Approach 4: Mathematical Formula
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Use the mathematical formula for counting factors of 5 in n!
    """
    if n < 5:
        return 0
    
    # Sum of floor(n/5^i) for i = 1, 2, 3, ...
    result = 0
    i = 1
    
    while True:
        power_of_five = 5 ** i
        if power_of_five > n:
            break
        result += n // power_of_five
        i += 1
    
    return result


def trailing_zeroes_optimized_loop(n):
    """
    Approach 5: Optimized Loop
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Optimized version with better loop structure.
    """
    count = 0
    
    for i in range(5, n + 1, 5):
        temp = i
        while temp % 5 == 0:
            count += 1
            temp //= 5
    
    return count


def trailing_zeroes_bit_manipulation(n):
    """
    Approach 6: Using Bit Manipulation Concepts
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Use bit manipulation thinking (similar to counting set bits).
    """
    count = 0
    divisor = 5
    
    while divisor <= n:
        count += n // divisor
        # Check for overflow before multiplication
        if divisor > n // 5:
            break
        divisor *= 5
    
    return count


def trailing_zeroes_precomputed(n):
    """
    Approach 7: Precomputed for Small Values
    Time Complexity: O(1) for lookup, O(k) for precomputation
    Space Complexity: O(k) for lookup table
    
    Precompute results for small values of n.
    """
    # Initialize lookup table on first call
    if not hasattr(trailing_zeroes_precomputed, "lookup"):
        trailing_zeroes_precomputed.lookup = {}
        
        # Precompute for common values
        for i in range(101):  # Precompute for n <= 100
            count = 0
            temp = i
            while temp >= 5:
                temp //= 5
                count += temp
            trailing_zeroes_precomputed.lookup[i] = count
    
    # Use lookup if available, otherwise compute
    if n in trailing_zeroes_precomputed.lookup:
        return trailing_zeroes_precomputed.lookup[n]
    else:
        return trailing_zeroes_count_fives(n)


def trailing_zeroes_brute_force(n):
    """
    Approach 8: Brute Force (Not recommended for large n)
    Time Complexity: O(n log n)
    Space Complexity: O(1)
    
    Count all factors of 5 in numbers 1 to n.
    """
    if n >= 1000:  # Avoid TLE for large n
        return trailing_zeroes_count_fives(n)
    
    count = 0
    
    for i in range(5, n + 1):
        temp = i
        while temp % 5 == 0:
            count += 1
            temp //= 5
    
    return count


def trailing_zeroes_string_analysis(n):
    """
    Approach 9: String Analysis (Educational purpose)
    Time Complexity: O(n!) - factorial computation
    Space Complexity: O(n!) - storing factorial as string
    
    Actually compute factorial and count trailing zeros.
    WARNING: Only for very small n due to factorial growth.
    """
    if n > 20:  # Avoid computing large factorials
        return trailing_zeroes_count_fives(n)
    
    # Compute factorial
    factorial = 1
    for i in range(1, n + 1):
        factorial *= i
    
    # Convert to string and count trailing zeros
    factorial_str = str(factorial)
    count = 0
    
    for i in range(len(factorial_str) - 1, -1, -1):
        if factorial_str[i] == '0':
            count += 1
        else:
            break
    
    return count


def test_trailing_zeroes():
    """Test all approaches with various test cases."""
    
    test_cases = [
        (3, 0),     # 3! = 6
        (5, 1),     # 5! = 120
        (0, 0),     # 0! = 1
        (1, 0),     # 1! = 1
        (10, 2),    # 10! = 3628800
        (25, 6),    # 25! has 6 trailing zeros
        (50, 12),   # 50! has 12 trailing zeros
        (100, 24),  # 100! has 24 trailing zeros
        (125, 31),  # 125! has 31 trailing zeros
        (4, 0),     # 4! = 24
        (15, 3),    # 15! has 3 trailing zeros
        (20, 4),    # 20! has 4 trailing zeros
        (30, 7),    # 30! has 7 trailing zeros
        (1000, 249), # 1000! has 249 trailing zeros
    ]
    
    approaches = [
        ("Count Fives", trailing_zeroes_count_fives),
        ("Iterative Division", trailing_zeroes_iterative_division),
        ("Recursive", trailing_zeroes_recursive),
        ("Mathematical Formula", trailing_zeroes_mathematical_formula),
        ("Optimized Loop", trailing_zeroes_optimized_loop),
        ("Bit Manipulation", trailing_zeroes_bit_manipulation),
        ("Precomputed", trailing_zeroes_precomputed),
        ("Brute Force", trailing_zeroes_brute_force),
        ("String Analysis", trailing_zeroes_string_analysis),
    ]
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: n = {n}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            result = func(n)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_trailing_zeroes() 