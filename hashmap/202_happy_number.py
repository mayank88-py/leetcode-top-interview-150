"""
202. Happy Number

Problem:
Write an algorithm to determine if a number n is happy.

A happy number is a number defined by the following process:
- Starting with any positive integer, replace the number by the sum of the squares of its digits.
- Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
- Those numbers for which this process ends in 1 are happy.

Return true if n is a happy number, and false if not.

Example 1:
Input: n = 19
Output: true
Explanation: 
1² + 9² = 82
8² + 2² = 68
6² + 8² = 100
1² + 0² + 0² = 1

Example 2:
Input: n = 2
Output: false

Time Complexity: O(log n) for the number of digits
Space Complexity: O(log n) for the set of seen numbers
"""


def is_happy(n):
    """
    Check if a number is happy using set to detect cycles.
    
    Args:
        n: Positive integer to check
    
    Returns:
        True if n is happy, False otherwise
    """
    def get_sum_of_squares(num):
        """Calculate sum of squares of digits"""
        total = 0
        while num > 0:
            digit = num % 10
            total += digit * digit
            num //= 10
        return total
    
    seen = set()
    
    while n != 1 and n not in seen:
        seen.add(n)
        n = get_sum_of_squares(n)
    
    return n == 1


def is_happy_floyd_cycle(n):
    """
    Check if a number is happy using Floyd's cycle detection algorithm.
    
    Args:
        n: Positive integer to check
    
    Returns:
        True if n is happy, False otherwise
    """
    def get_sum_of_squares(num):
        """Calculate sum of squares of digits"""
        total = 0
        while num > 0:
            digit = num % 10
            total += digit * digit
            num //= 10
        return total
    
    # Floyd's cycle detection (tortoise and hare)
    slow = n
    fast = n
    
    while True:
        slow = get_sum_of_squares(slow)  # Move one step
        fast = get_sum_of_squares(get_sum_of_squares(fast))  # Move two steps
        
        if fast == 1:
            return True
        
        if slow == fast:  # Cycle detected
            return False


def is_happy_recursive(n):
    """
    Check if a number is happy using recursion with memoization.
    
    Args:
        n: Positive integer to check
    
    Returns:
        True if n is happy, False otherwise
    """
    def get_sum_of_squares(num):
        """Calculate sum of squares of digits"""
        total = 0
        while num > 0:
            digit = num % 10
            total += digit * digit
            num //= 10
        return total
    
    def helper(num, seen):
        """Recursive helper function"""
        if num == 1:
            return True
        if num in seen:
            return False
        
        seen.add(num)
        return helper(get_sum_of_squares(num), seen)
    
    return helper(n, set())


def is_happy_string_manipulation(n):
    """
    Check if a number is happy using string manipulation.
    
    Args:
        n: Positive integer to check
    
    Returns:
        True if n is happy, False otherwise
    """
    def get_sum_of_squares(num):
        """Calculate sum of squares of digits using string"""
        num_str = str(num)
        return sum(int(digit) ** 2 for digit in num_str)
    
    seen = set()
    
    while n != 1 and n not in seen:
        seen.add(n)
        n = get_sum_of_squares(n)
    
    return n == 1


def is_happy_mathematical_pattern(n):
    """
    Check if a number is happy using known mathematical patterns.
    
    Args:
        n: Positive integer to check
    
    Returns:
        True if n is happy, False otherwise
    """
    def get_sum_of_squares(num):
        """Calculate sum of squares of digits"""
        total = 0
        while num > 0:
            digit = num % 10
            total += digit * digit
            num //= 10
        return total
    
    # Known cycle that unhappy numbers eventually reach
    unhappy_cycle = {4, 16, 37, 58, 89, 145, 42, 20}
    
    while n != 1 and n not in unhappy_cycle:
        n = get_sum_of_squares(n)
    
    return n == 1


def is_happy_iterative_limit(n):
    """
    Check if a number is happy with iteration limit.
    
    Args:
        n: Positive integer to check
    
    Returns:
        True if n is happy, False otherwise
    """
    def get_sum_of_squares(num):
        """Calculate sum of squares of digits"""
        total = 0
        while num > 0:
            digit = num % 10
            total += digit * digit
            num //= 10
        return total
    
    # Limit iterations to avoid infinite loops
    max_iterations = 1000
    
    for _ in range(max_iterations):
        if n == 1:
            return True
        n = get_sum_of_squares(n)
    
    # If we reach here, assume it's not happy
    return False


# Test cases
if __name__ == "__main__":
    # Test case 1
    n1 = 19
    result1a = is_happy(n1)
    result1b = is_happy_floyd_cycle(n1)
    result1c = is_happy_recursive(n1)
    result1d = is_happy_string_manipulation(n1)
    result1e = is_happy_mathematical_pattern(n1)
    result1f = is_happy_iterative_limit(n1)
    print(f"Test 1 - Input: {n1}, Expected: True")
    print(f"Set: {result1a}, Floyd: {result1b}, Recursive: {result1c}, String: {result1d}, Pattern: {result1e}, Limit: {result1f}")
    print()
    
    # Test case 2
    n2 = 2
    result2a = is_happy(n2)
    result2b = is_happy_floyd_cycle(n2)
    result2c = is_happy_recursive(n2)
    result2d = is_happy_string_manipulation(n2)
    result2e = is_happy_mathematical_pattern(n2)
    result2f = is_happy_iterative_limit(n2)
    print(f"Test 2 - Input: {n2}, Expected: False")
    print(f"Set: {result2a}, Floyd: {result2b}, Recursive: {result2c}, String: {result2d}, Pattern: {result2e}, Limit: {result2f}")
    print()
    
    # Test case 3
    n3 = 1
    result3a = is_happy(n3)
    result3b = is_happy_floyd_cycle(n3)
    result3c = is_happy_recursive(n3)
    result3d = is_happy_string_manipulation(n3)
    result3e = is_happy_mathematical_pattern(n3)
    result3f = is_happy_iterative_limit(n3)
    print(f"Test 3 - Input: {n3}, Expected: True")
    print(f"Set: {result3a}, Floyd: {result3b}, Recursive: {result3c}, String: {result3d}, Pattern: {result3e}, Limit: {result3f}")
    print()
    
    # Test case 4
    n4 = 7
    result4a = is_happy(n4)
    result4b = is_happy_floyd_cycle(n4)
    result4c = is_happy_recursive(n4)
    result4d = is_happy_string_manipulation(n4)
    result4e = is_happy_mathematical_pattern(n4)
    result4f = is_happy_iterative_limit(n4)
    print(f"Test 4 - Input: {n4}, Expected: True")
    print(f"Set: {result4a}, Floyd: {result4b}, Recursive: {result4c}, String: {result4d}, Pattern: {result4e}, Limit: {result4f}")
    print()
    
    # Test case 5
    n5 = 10
    result5a = is_happy(n5)
    result5b = is_happy_floyd_cycle(n5)
    result5c = is_happy_recursive(n5)
    result5d = is_happy_string_manipulation(n5)
    result5e = is_happy_mathematical_pattern(n5)
    result5f = is_happy_iterative_limit(n5)
    print(f"Test 5 - Input: {n5}, Expected: True")
    print(f"Set: {result5a}, Floyd: {result5b}, Recursive: {result5c}, String: {result5d}, Pattern: {result5e}, Limit: {result5f}")
    print()
    
    # Test case 6
    n6 = 4
    result6a = is_happy(n6)
    result6b = is_happy_floyd_cycle(n6)
    result6c = is_happy_recursive(n6)
    result6d = is_happy_string_manipulation(n6)
    result6e = is_happy_mathematical_pattern(n6)
    result6f = is_happy_iterative_limit(n6)
    print(f"Test 6 - Input: {n6}, Expected: False")
    print(f"Set: {result6a}, Floyd: {result6b}, Recursive: {result6c}, String: {result6d}, Pattern: {result6e}, Limit: {result6f}")
    print()
    
    # Test case 7
    n7 = 100
    result7a = is_happy(n7)
    result7b = is_happy_floyd_cycle(n7)
    result7c = is_happy_recursive(n7)
    result7d = is_happy_string_manipulation(n7)
    result7e = is_happy_mathematical_pattern(n7)
    result7f = is_happy_iterative_limit(n7)
    print(f"Test 7 - Input: {n7}, Expected: True")
    print(f"Set: {result7a}, Floyd: {result7b}, Recursive: {result7c}, String: {result7d}, Pattern: {result7e}, Limit: {result7f}")
    print()
    
    # Test case 8
    n8 = 89
    result8a = is_happy(n8)
    result8b = is_happy_floyd_cycle(n8)
    result8c = is_happy_recursive(n8)
    result8d = is_happy_string_manipulation(n8)
    result8e = is_happy_mathematical_pattern(n8)
    result8f = is_happy_iterative_limit(n8)
    print(f"Test 8 - Input: {n8}, Expected: False")
    print(f"Set: {result8a}, Floyd: {result8b}, Recursive: {result8c}, String: {result8d}, Pattern: {result8e}, Limit: {result8f}")
    print()
    
    # Test case 9
    n9 = 13
    result9a = is_happy(n9)
    result9b = is_happy_floyd_cycle(n9)
    result9c = is_happy_recursive(n9)
    result9d = is_happy_string_manipulation(n9)
    result9e = is_happy_mathematical_pattern(n9)
    result9f = is_happy_iterative_limit(n9)
    print(f"Test 9 - Input: {n9}, Expected: True")
    print(f"Set: {result9a}, Floyd: {result9b}, Recursive: {result9c}, String: {result9d}, Pattern: {result9e}, Limit: {result9f}")
    print()
    
    # Test case 10
    n10 = 999
    result10a = is_happy(n10)
    result10b = is_happy_floyd_cycle(n10)
    result10c = is_happy_recursive(n10)
    result10d = is_happy_string_manipulation(n10)
    result10e = is_happy_mathematical_pattern(n10)
    result10f = is_happy_iterative_limit(n10)
    print(f"Test 10 - Input: {n10}, Expected: True")
    print(f"Set: {result10a}, Floyd: {result10b}, Recursive: {result10c}, String: {result10d}, Pattern: {result10e}, Limit: {result10f}") 