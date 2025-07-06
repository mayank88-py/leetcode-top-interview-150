"""
66. Plus One

You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer. The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading zeros.

Increment the large integer by one and return the resulting array of digits.

Example 1:
Input: digits = [1,2,3]
Output: [1,2,4]
Explanation: The array represents the integer 123.
Incrementing by one gives 123 + 1 = 124.
Thus, the result should be [1,2,4].

Example 2:
Input: digits = [4,3,2,1]
Output: [4,3,2,2]
Explanation: The array represents the integer 4321.
Incrementing by one gives 4321 + 1 = 4322.
Thus, the result should be [4,3,2,2].

Example 3:
Input: digits = [9]
Output: [1,0]
Explanation: The array represents the integer 9.
Incrementing by one gives 9 + 1 = 10.
Thus, the result should be [1,0].

Constraints:
- 1 <= digits.length <= 100
- 0 <= digits[i] <= 9
- digits does not contain any leading zeros except for the zero itself.
"""

def plus_one_simple(digits):
    """
    Approach 1: Simple Digit Addition
    Time Complexity: O(n)
    Space Complexity: O(1) - excluding output array
    
    Add 1 to the last digit and handle carry propagation.
    """
    # Start from the last digit
    for i in range(len(digits) - 1, -1, -1):
        if digits[i] < 9:
            digits[i] += 1
            return digits
        else:
            digits[i] = 0
    
    # If we reach here, all digits were 9
    return [1] + digits


def plus_one_carry_propagation(digits):
    """
    Approach 2: Explicit Carry Propagation
    Time Complexity: O(n)
    Space Complexity: O(1) - excluding output array
    
    Explicitly handle carry propagation through the digits.
    """
    carry = 1
    result = digits[:]
    
    for i in range(len(result) - 1, -1, -1):
        total = result[i] + carry
        result[i] = total % 10
        carry = total // 10
        
        if carry == 0:
            break
    
    # If there's still a carry, prepend it
    if carry:
        result = [carry] + result
    
    return result


def plus_one_recursive(digits):
    """
    Approach 3: Recursive Approach
    Time Complexity: O(n)
    Space Complexity: O(n) - recursion depth
    
    Recursive implementation of plus one.
    """
    def add_one_recursive(arr, index):
        if index < 0:
            return [1]  # Need to add a new digit
        
        if arr[index] < 9:
            arr[index] += 1
            return arr
        else:
            arr[index] = 0
            return add_one_recursive(arr, index - 1)
    
    result = digits[:]
    return add_one_recursive(result, len(result) - 1)


def plus_one_string_conversion(digits):
    """
    Approach 4: String Conversion (Not recommended for large numbers)
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Convert to string, then to int, add 1, and convert back.
    """
    # Convert digits array to string
    num_str = ''.join(map(str, digits))
    
    # Convert to int, add 1, and convert back to string
    result_str = str(int(num_str) + 1)
    
    # Convert back to digits array
    return [int(digit) for digit in result_str]


def plus_one_optimized(digits):
    """
    Approach 5: Optimized with Early Return
    Time Complexity: O(n) worst case, O(1) best case
    Space Complexity: O(1) - excluding output array
    
    Optimized version that returns early when possible.
    """
    n = len(digits)
    
    # Traverse from right to left
    for i in range(n - 1, -1, -1):
        if digits[i] < 9:
            digits[i] += 1
            return digits
        digits[i] = 0
    
    # All digits were 9, need to add a new digit at the beginning
    return [1] + [0] * n


def plus_one_in_place_with_resize(digits):
    """
    Approach 6: In-place with Conditional Resize
    Time Complexity: O(n)
    Space Complexity: O(1) if no resize needed, O(n) if resize needed
    
    Try to modify in-place, resize only if necessary.
    """
    # Check if we need to resize (all digits are 9)
    need_resize = all(digit == 9 for digit in digits)
    
    if need_resize:
        # Create new array with extra digit
        result = [1] + [0] * len(digits)
        return result
    
    # Modify in place
    carry = 1
    for i in range(len(digits) - 1, -1, -1):
        digits[i] += carry
        if digits[i] == 10:
            digits[i] = 0
            carry = 1
        else:
            carry = 0
            break
    
    return digits


def plus_one_mathematical(digits):
    """
    Approach 7: Mathematical Approach
    Time Complexity: O(n)
    Space Complexity: O(1) - excluding output array
    
    Use mathematical operations to simulate addition.
    """
    result = []
    carry = 1
    
    for i in range(len(digits) - 1, -1, -1):
        digit_sum = digits[i] + carry
        result.append(digit_sum % 10)
        carry = digit_sum // 10
    
    if carry:
        result.append(carry)
    
    # Reverse to get correct order
    return result[::-1]


def plus_one_edge_case_optimized(digits):
    """
    Approach 8: Edge Case Optimized
    Time Complexity: O(n)
    Space Complexity: O(1) - excluding output array
    
    Optimized for common edge cases.
    """
    # Edge case: empty array
    if not digits:
        return [1]
    
    # Edge case: single digit
    if len(digits) == 1:
        if digits[0] < 9:
            return [digits[0] + 1]
        else:
            return [1, 0]
    
    # General case
    for i in range(len(digits) - 1, -1, -1):
        if digits[i] < 9:
            digits[i] += 1
            return digits
        digits[i] = 0
    
    # All 9s case
    return [1] + digits


def plus_one_functional(digits):
    """
    Approach 9: Functional Programming Style
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Use functional programming concepts.
    """
    def process_digit(carry, digit):
        total = digit + carry
        return total // 10, total % 10
    
    result = []
    carry = 1
    
    for digit in reversed(digits):
        carry, new_digit = process_digit(carry, digit)
        result.append(new_digit)
    
    if carry:
        result.append(carry)
    
    return result[::-1]


def test_plus_one():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([1, 2, 3], [1, 2, 4]),
        ([4, 3, 2, 1], [4, 3, 2, 2]),
        ([9], [1, 0]),
        ([0], [1]),
        ([9, 9], [1, 0, 0]),
        ([9, 9, 9], [1, 0, 0, 0]),
        ([1, 2, 9], [1, 3, 0]),
        ([2, 9, 9], [3, 0, 0]),
        ([1, 0, 0], [1, 0, 1]),
        ([8, 9], [9, 0]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 6]),
        ([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [9, 8, 7, 6, 5, 4, 3, 2, 1, 1]),
    ]
    
    approaches = [
        ("Simple", plus_one_simple),
        ("Carry Propagation", plus_one_carry_propagation),
        ("Recursive", plus_one_recursive),
        ("String Conversion", plus_one_string_conversion),
        ("Optimized", plus_one_optimized),
        ("In-place with Resize", plus_one_in_place_with_resize),
        ("Mathematical", plus_one_mathematical),
        ("Edge Case Optimized", plus_one_edge_case_optimized),
        ("Functional", plus_one_functional),
    ]
    
    for i, (digits, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: {digits}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            # Create a copy for functions that might modify the input
            digits_copy = digits.copy()
            result = func(digits_copy)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_plus_one() 