"""
9. Palindrome Number

Given an integer x, return true if x is a palindrome, and false otherwise.

Example 1:
Input: x = 121
Output: true
Explanation: 121 reads as 121 from left to right and from right to left.

Example 2:
Input: x = -121
Output: false
Explanation: From left to right, it reads -121. From right to left, it becomes 121-. 
Therefore it is not a palindrome.

Example 3:
Input: x = 10
Output: false
Explanation: Reads 01 from right to left. Therefore it is not a palindrome.

Constraints:
- -2^31 <= x <= 2^31 - 1

Follow up: Could you solve it without converting the integer to a string?
"""

def is_palindrome_string_conversion(x):
    """
    Approach 1: String Conversion
    Time Complexity: O(log x) - number of digits
    Space Complexity: O(log x) - for string storage
    
    Convert number to string and check if it's equal to its reverse.
    """
    str_x = str(x)
    return str_x == str_x[::-1]


def is_palindrome_full_reversal(x):
    """
    Approach 2: Full Number Reversal
    Time Complexity: O(log x)
    Space Complexity: O(1)
    
    Reverse the entire number and compare with original.
    """
    if x < 0:
        return False
    
    original = x
    reversed_num = 0
    
    while x > 0:
        reversed_num = reversed_num * 10 + x % 10
        x //= 10
    
    return original == reversed_num


def is_palindrome_half_reversal(x):
    """
    Approach 3: Half Number Reversal (Optimal)
    Time Complexity: O(log x)
    Space Complexity: O(1)
    
    Only reverse half of the number to save time and avoid overflow.
    """
    # Negative numbers and numbers ending with 0 (except 0) are not palindromes
    if x < 0 or (x % 10 == 0 and x != 0):
        return False
    
    reversed_half = 0
    
    # Reverse half of the number
    while x > reversed_half:
        reversed_half = reversed_half * 10 + x % 10
        x //= 10
    
    # For even number of digits: x == reversed_half
    # For odd number of digits: x == reversed_half // 10 (remove middle digit)
    return x == reversed_half or x == reversed_half // 10


def is_palindrome_digit_extraction(x):
    """
    Approach 4: Digit Extraction with List
    Time Complexity: O(log x)
    Space Complexity: O(log x)
    
    Extract all digits into a list and check palindrome.
    """
    if x < 0:
        return False
    
    if x < 10:
        return True
    
    digits = []
    while x > 0:
        digits.append(x % 10)
        x //= 10
    
    # Check if list is palindrome
    left, right = 0, len(digits) - 1
    while left < right:
        if digits[left] != digits[right]:
            return False
        left += 1
        right -= 1
    
    return True


def is_palindrome_mathematical_comparison(x):
    """
    Approach 5: Mathematical Digit Comparison
    Time Complexity: O(log x)
    Space Complexity: O(1)
    
    Compare digits from both ends without storing them.
    """
    if x < 0:
        return False
    
    if x < 10:
        return True
    
    # Find the number of digits
    temp = x
    num_digits = 0
    while temp > 0:
        num_digits += 1
        temp //= 10
    
    # Compare digits from both ends
    for i in range(num_digits // 2):
        # Get digit from the left
        left_divisor = 10 ** (num_digits - 1 - i)
        left_digit = (x // left_divisor) % 10
        
        # Get digit from the right
        right_divisor = 10 ** i
        right_digit = (x // right_divisor) % 10
        
        if left_digit != right_digit:
            return False
    
    return True


def is_palindrome_recursive(x):
    """
    Approach 6: Recursive Approach
    Time Complexity: O(log x)
    Space Complexity: O(log x) - recursion depth
    
    Recursive implementation using helper function.
    """
    if x < 0:
        return False
    
    def get_digit_count(num):
        if num == 0:
            return 1
        count = 0
        while num > 0:
            count += 1
            num //= 10
        return count
    
    def check_palindrome_recursive(num, digit_count):
        if digit_count <= 1:
            return True
        
        # Get first and last digits
        first_digit = num // (10 ** (digit_count - 1))
        last_digit = num % 10
        
        if first_digit != last_digit:
            return False
        
        # Remove first and last digits
        num = (num % (10 ** (digit_count - 1))) // 10
        
        return check_palindrome_recursive(num, digit_count - 2)
    
    digit_count = get_digit_count(x)
    return check_palindrome_recursive(x, digit_count)


def is_palindrome_bit_manipulation(x):
    """
    Approach 7: Using Bit Manipulation (Creative approach)
    Time Complexity: O(log x)
    Space Complexity: O(1)
    
    Convert to string and use XOR to check palindrome property.
    Note: This approach uses string conversion but demonstrates bit manipulation.
    """
    if x < 0:
        return False
    
    s = str(x)
    n = len(s)
    
    # Use XOR to check if characters at symmetric positions are same
    for i in range(n // 2):
        if ord(s[i]) ^ ord(s[n - 1 - i]) != 0:
            return False
    
    return True


def is_palindrome_optimized_edge_cases(x):
    """
    Approach 8: Optimized with Edge Case Handling
    Time Complexity: O(log x)
    Space Complexity: O(1)
    
    Optimized version with better edge case handling.
    """
    # Handle negative numbers and numbers ending with 0
    if x < 0 or (x > 0 and x % 10 == 0):
        return False
    
    # Single digit numbers are palindromes
    if x < 10:
        return True
    
    # Use half reversal approach
    reversed_half = 0
    while x > reversed_half:
        reversed_half = reversed_half * 10 + x % 10
        x //= 10
    
    return x == reversed_half or x == reversed_half // 10


def test_is_palindrome():
    """Test all approaches with various test cases."""
    
    test_cases = [
        (121, True),
        (-121, False),
        (10, False),
        (0, True),
        (1, True),
        (11, True),
        (12321, True),
        (123321, True),
        (1234, False),
        (-101, False),
        (1001, True),
        (12345, False),
        (9, True),
        (99, True),
        (100, False),
        (1221, True),
        (12345654321, True),
        (2147483647, False),  # Large number
        (-2147447412, False), # Large negative
    ]
    
    approaches = [
        ("String Conversion", is_palindrome_string_conversion),
        ("Full Reversal", is_palindrome_full_reversal),
        ("Half Reversal", is_palindrome_half_reversal),
        ("Digit Extraction", is_palindrome_digit_extraction),
        ("Mathematical Comparison", is_palindrome_mathematical_comparison),
        ("Recursive", is_palindrome_recursive),
        ("Bit Manipulation", is_palindrome_bit_manipulation),
        ("Optimized Edge Cases", is_palindrome_optimized_edge_cases),
    ]
    
    for i, (x, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: {x}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            result = func(x)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_is_palindrome() 