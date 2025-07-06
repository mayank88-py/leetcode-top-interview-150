"""
201. Bitwise AND of Numbers Range

Given two integers left and right that represent the range [left, right], 
return the bitwise AND of all numbers in this range, inclusive.

Example 1:
Input: left = 5, right = 7
Output: 4
Explanation: 5 & 6 & 7 = 4

Example 2:
Input: left = 0, right = 0
Output: 0

Example 3:
Input: left = 1, right = 2147483647
Output: 0

Constraints:
- 0 <= left <= right <= 2^31 - 1
"""

def range_bitwise_and_common_prefix(left, right):
    """
    Approach 1: Find Common Prefix
    Time Complexity: O(log n) where n is the maximum value
    Space Complexity: O(1)
    
    The AND result is the common prefix of left and right.
    Keep shifting right until left == right.
    """
    shift = 0
    
    # Find the common prefix by right shifting
    while left != right:
        left >>= 1
        right >>= 1
        shift += 1
    
    # Shift back to get the common prefix
    return left << shift


def range_bitwise_and_brian_kernighan(left, right):
    """
    Approach 2: Brian Kernighan's Algorithm
    Time Complexity: O(number of different bits)
    Space Complexity: O(1)
    
    Use Brian Kernighan's algorithm to remove rightmost set bits from right
    until right <= left.
    """
    while left < right:
        # Remove the rightmost set bit from right
        right &= (right - 1)
    
    return right


def range_bitwise_and_bit_by_bit(left, right):
    """
    Approach 3: Bit-by-bit Analysis
    Time Complexity: O(32) = O(1)
    Space Complexity: O(1)
    
    Analyze each bit position from MSB to LSB.
    """
    result = 0
    
    # Check each bit from MSB to LSB (31 to 0)
    for i in range(31, -1, -1):
        bit_left = (left >> i) & 1
        bit_right = (right >> i) & 1
        
        if bit_left != bit_right:
            # Different bits means all subsequent bits will be 0 in AND
            break
        
        if bit_left == 1:  # Both bits are 1
            result |= (1 << i)
    
    return result


def range_bitwise_and_brute_force(left, right):
    """
    Approach 4: Brute Force (Not recommended for large ranges)
    Time Complexity: O(right - left)
    Space Complexity: O(1)
    
    Calculate AND of all numbers in range. Only efficient for small ranges.
    """
    if right - left > 10000:  # Avoid TLE for large ranges
        return range_bitwise_and_common_prefix(left, right)
    
    result = left
    for i in range(left + 1, right + 1):
        result &= i
        if result == 0:  # Early termination
            break
    
    return result


def range_bitwise_and_mask_approach(left, right):
    """
    Approach 5: Mask Approach
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Find the position where left and right first differ and create a mask.
    """
    # Find the most significant bit where left and right differ
    diff = left ^ right
    
    # Find the highest set bit in diff
    msb_pos = 0
    while diff:
        diff >>= 1
        msb_pos += 1
    
    # Create mask to keep only the common prefix
    if msb_pos == 0:
        return left  # left == right
    
    mask = ~((1 << msb_pos) - 1)
    return left & mask


def range_bitwise_and_recursive(left, right):
    """
    Approach 6: Recursive Approach
    Time Complexity: O(log n)
    Space Complexity: O(log n) - recursion depth
    
    Recursive implementation of the common prefix approach.
    """
    if left == right:
        return left
    
    return range_bitwise_and_recursive(left >> 1, right >> 1) << 1


def range_bitwise_and_iterative_optimization(left, right):
    """
    Approach 7: Iterative with Early Optimization
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Optimized iterative approach with early termination.
    """
    if left == 0:
        return 0
    
    shift = 0
    
    # Early termination: if the range spans more than one power of 2
    while left != right:
        if left == 0:
            return 0
        
        left >>= 1
        right >>= 1
        shift += 1
    
    return left << shift


def range_bitwise_and_mathematical(left, right):
    """
    Approach 8: Mathematical Approach
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Use mathematical properties to find the result.
    """
    # If left is 0, result is 0
    if left == 0:
        return 0
    
    # Find the highest bit position where left and right differ
    diff = left ^ right
    
    # Count leading zeros in diff
    leading_zeros = 0
    test_bit = 1 << 31
    
    while test_bit and not (diff & test_bit):
        leading_zeros += 1
        test_bit >>= 1
    
    # Create mask for common prefix
    if leading_zeros == 32:
        return left  # left == right
    
    mask_bits = 32 - leading_zeros
    mask = (0xFFFFFFFF << mask_bits) & 0xFFFFFFFF
    
    return left & mask


def range_bitwise_and_builtin_optimization(left, right):
    """
    Approach 9: Using Built-in Bit Operations
    Time Complexity: O(1) - using built-in bit length
    Space Complexity: O(1)
    
    Use Python's built-in bit_length() for optimization.
    """
    if left == 0:
        return 0
    
    # Find the bit length difference
    left_bits = left.bit_length()
    right_bits = right.bit_length()
    
    if left_bits != right_bits:
        return 0
    
    # Find common prefix
    shift = 0
    while left != right:
        left >>= 1
        right >>= 1
        shift += 1
    
    return left << shift


def test_range_bitwise_and():
    """Test all approaches with various test cases."""
    
    test_cases = [
        (5, 7, 4),           # Example 1: 5 & 6 & 7 = 4
        (0, 0, 0),           # Example 2: Single number
        (1, 2147483647, 0),  # Example 3: Large range
        (1, 1, 1),           # Same numbers
        (2, 6, 0),           # 2 & 3 & 4 & 5 & 6 = 0
        (0, 1, 0),           # Small range with 0
        (4, 7, 4),           # 4 & 5 & 6 & 7 = 4
        (1, 3, 0),           # 1 & 2 & 3 = 0
        (5, 5, 5),           # Single number (non-zero)
        (26, 30, 24),        # Medium range
        (1024, 2047, 1024),  # Power of 2 range
        (0, 2147483647, 0),  # Full range
    ]
    
    approaches = [
        ("Common Prefix", range_bitwise_and_common_prefix),
        ("Brian Kernighan", range_bitwise_and_brian_kernighan),
        ("Bit-by-bit", range_bitwise_and_bit_by_bit),
        ("Brute Force", range_bitwise_and_brute_force),
        ("Mask Approach", range_bitwise_and_mask_approach),
        ("Recursive", range_bitwise_and_recursive),
        ("Iterative Optimized", range_bitwise_and_iterative_optimization),
        ("Mathematical", range_bitwise_and_mathematical),
        ("Built-in Optimized", range_bitwise_and_builtin_optimization),
    ]
    
    for i, (left, right, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: left = {left}, right = {right}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            result = func(left, right)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_range_bitwise_and() 