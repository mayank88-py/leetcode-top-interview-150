"""
67. Add Binary

Given two binary strings a and b, return their sum as a binary string.

Example 1:
Input: a = "11", b = "1"
Output: "100"

Example 2:
Input: a = "1010", b = "1011"
Output: "10101"

Constraints:
- 1 <= a.length, b.length <= 10^4
- a and b consist only of '0' or '1' characters.
- Each string does not contain leading zeros except for the zero itself.
"""

def add_binary_string_manipulation(a, b):
    """
    Approach 1: String Manipulation with Carry
    Time Complexity: O(max(len(a), len(b)))
    Space Complexity: O(max(len(a), len(b)))
    
    Simulate binary addition with string manipulation.
    """
    result = []
    carry = 0
    i, j = len(a) - 1, len(b) - 1
    
    while i >= 0 or j >= 0 or carry:
        # Get current bits
        bit_a = int(a[i]) if i >= 0 else 0
        bit_b = int(b[j]) if j >= 0 else 0
        
        # Calculate sum
        total = bit_a + bit_b + carry
        result.append(str(total % 2))
        carry = total // 2
        
        i -= 1
        j -= 1
    
    return ''.join(reversed(result))


def add_binary_bit_manipulation(a, b):
    """
    Approach 2: Bit Manipulation (Convert to int)
    Time Complexity: O(max(len(a), len(b)))
    Space Complexity: O(max(len(a), len(b)))
    
    Convert to integers, add, and convert back to binary.
    """
    # Convert binary strings to integers
    num_a = int(a, 2)
    num_b = int(b, 2)
    
    # Add using bit manipulation
    while num_b:
        # XOR gives sum without carry
        sum_without_carry = num_a ^ num_b
        # AND and shift gives carry
        carry = (num_a & num_b) << 1
        
        num_a = sum_without_carry
        num_b = carry
    
    # Convert back to binary (remove '0b' prefix)
    return bin(num_a)[2:]


def add_binary_recursive(a, b):
    """
    Approach 3: Recursive Approach
    Time Complexity: O(max(len(a), len(b)))
    Space Complexity: O(max(len(a), len(b))) - recursion depth
    
    Recursive implementation of binary addition.
    """
    def add_helper(a, b, carry, index_a, index_b):
        # Base case: no more digits and no carry
        if index_a < 0 and index_b < 0 and carry == 0:
            return ""
        
        # Get current bits
        bit_a = int(a[index_a]) if index_a >= 0 else 0
        bit_b = int(b[index_b]) if index_b >= 0 else 0
        
        # Calculate sum
        total = bit_a + bit_b + carry
        current_bit = str(total % 2)
        new_carry = total // 2
        
        # Recursive call for remaining digits
        rest = add_helper(a, b, new_carry, index_a - 1, index_b - 1)
        
        return rest + current_bit
    
    return add_helper(a, b, 0, len(a) - 1, len(b) - 1)


def add_binary_padding(a, b):
    """
    Approach 4: Padding Approach
    Time Complexity: O(max(len(a), len(b)))
    Space Complexity: O(max(len(a), len(b)))
    
    Pad shorter string with leading zeros and add digit by digit.
    """
    # Pad shorter string
    max_len = max(len(a), len(b))
    a = a.zfill(max_len)
    b = b.zfill(max_len)
    
    result = []
    carry = 0
    
    # Process from right to left
    for i in range(max_len - 1, -1, -1):
        total = int(a[i]) + int(b[i]) + carry
        result.append(str(total % 2))
        carry = total // 2
    
    # Add final carry if exists
    if carry:
        result.append(str(carry))
    
    return ''.join(reversed(result))


def add_binary_builtin(a, b):
    """
    Approach 5: Using Built-in Functions
    Time Complexity: O(max(len(a), len(b)))
    Space Complexity: O(max(len(a), len(b)))
    
    Use Python's built-in binary conversion functions.
    """
    return bin(int(a, 2) + int(b, 2))[2:]


def add_binary_bitwise_simulation(a, b):
    """
    Approach 6: Bitwise Simulation
    Time Complexity: O(max(len(a), len(b)))
    Space Complexity: O(max(len(a), len(b)))
    
    Simulate bitwise addition without converting to int.
    """
    result = []
    carry = 0
    i, j = len(a) - 1, len(b) - 1
    
    while i >= 0 or j >= 0 or carry:
        bit_sum = carry
        
        if i >= 0:
            bit_sum += ord(a[i]) - ord('0')  # Convert '0'/'1' to 0/1
            i -= 1
        
        if j >= 0:
            bit_sum += ord(b[j]) - ord('0')
            j -= 1
        
        result.append(chr((bit_sum % 2) + ord('0')))  # Convert back to '0'/'1'
        carry = bit_sum // 2
    
    return ''.join(reversed(result))


def test_add_binary():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ("11", "1", "100"),
        ("1010", "1011", "10101"),
        ("0", "0", "0"),
        ("1", "1", "10"),
        ("1111", "1111", "11110"),
        ("1", "111", "1000"),
        ("101", "11", "1000"),
        ("1101", "101", "10010"),
        ("11", "11", "110"),
        ("1010101", "1111", "1100100"),
        ("0", "1", "1"),
        ("1000", "111", "1111"),
    ]
    
    approaches = [
        ("String Manipulation", add_binary_string_manipulation),
        ("Bit Manipulation", add_binary_bit_manipulation),
        ("Recursive", add_binary_recursive),
        ("Padding", add_binary_padding),
        ("Built-in", add_binary_builtin),
        ("Bitwise Simulation", add_binary_bitwise_simulation),
    ]
    
    for i, (a, b, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: a = \"{a}\", b = \"{b}\"")
        print(f"Expected: \"{expected}\"")
        
        for name, func in approaches:
            result = func(a, b)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: \"{result}\"")


if __name__ == "__main__":
    test_add_binary() 