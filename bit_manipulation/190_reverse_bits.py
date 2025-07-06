"""
190. Reverse Bits

Reverse bits of a given 32 bits unsigned integer.

Note:
- Note that in some languages, such as Java, there is no unsigned integer type. 
  In this case, both input and output will be given as a signed integer type. 
  They should not affect your algorithm, as the integer's internal binary representation is the same, 
  whether it is signed or unsigned.
- In Java, the compiler represents the signed integers using 2's complement notation. 
  Therefore, in Example 2 above, the input represents the signed integer -3 and the output represents the signed integer -1073741825.

Example 1:
Input: n = 00000010100101000001111010011100
Output:    964176192 (00111001011110000010100101000000)
Explanation: The input binary string 00000010100101000001111010011100 represents the unsigned integer 43261596, 
so return 964176192 which its binary representation is 00111001011110000010100101000000.

Example 2:
Input: n = 11111111111111111111111111111101
Output:   3221225471 (10111111111111111111111111111111)
Explanation: The input binary string 11111111111111111111111111111101 represents the unsigned integer 4294967293, 
so return 3221225471 which its binary representation is 10111111111111111111111111111111.

Constraints:
- The input must be a binary string of length 32

Follow up: If this function is called many times, how would you optimize it?
"""

def reverse_bits_iterative(n):
    """
    Approach 1: Bit-by-bit Reversal (Iterative)
    Time Complexity: O(32) = O(1)
    Space Complexity: O(1)
    
    Extract each bit from right and build result from left.
    """
    result = 0
    for i in range(32):
        # Extract the rightmost bit
        bit = n & 1
        # Shift result left and add the extracted bit
        result = (result << 1) | bit
        # Shift n right to process next bit
        n >>= 1
    
    return result


def reverse_bits_bit_manipulation(n):
    """
    Approach 2: Optimized Bit Manipulation
    Time Complexity: O(1)
    Space Complexity: O(1)
    
    Use bit manipulation tricks to reverse in fewer operations.
    """
    result = 0
    power = 31
    
    while n:
        # Add the rightmost bit to result at correct position
        result += (n & 1) << power
        n >>= 1
        power -= 1
    
    return result


def reverse_bits_divide_conquer(n):
    """
    Approach 3: Divide and Conquer
    Time Complexity: O(1)
    Space Complexity: O(1)
    
    Use divide and conquer approach to swap bits in groups.
    """
    # Swap odd and even bits
    n = ((n & 0xAAAAAAAA) >> 1) | ((n & 0x55555555) << 1)
    
    # Swap consecutive pairs
    n = ((n & 0xCCCCCCCC) >> 2) | ((n & 0x33333333) << 2)
    
    # Swap nibbles (4-bit groups)
    n = ((n & 0xF0F0F0F0) >> 4) | ((n & 0x0F0F0F0F) << 4)
    
    # Swap bytes
    n = ((n & 0xFF00FF00) >> 8) | ((n & 0x00FF00FF) << 8)
    
    # Swap 16-bit halves
    n = (n >> 16) | (n << 16)
    
    return n & 0xFFFFFFFF  # Ensure 32-bit result


def reverse_bits_string_conversion(n):
    """
    Approach 4: String Conversion
    Time Complexity: O(32) = O(1)
    Space Complexity: O(32) = O(1)
    
    Convert to binary string, reverse, and convert back.
    """
    # Convert to 32-bit binary string
    binary_str = format(n, '032b')
    
    # Reverse the string
    reversed_str = binary_str[::-1]
    
    # Convert back to integer
    return int(reversed_str, 2)


def reverse_bits_lookup_table(n):
    """
    Approach 5: Lookup Table (for repeated calls)
    Time Complexity: O(1) after preprocessing
    Space Complexity: O(256) for lookup table
    
    Use precomputed lookup table for 8-bit reversals.
    """
    # Precompute reverse for all 8-bit numbers
    if not hasattr(reverse_bits_lookup_table, "lookup"):
        reverse_bits_lookup_table.lookup = {}
        for i in range(256):
            reversed_val = 0
            val = i
            for _ in range(8):
                reversed_val = (reversed_val << 1) | (val & 1)
                val >>= 1
            reverse_bits_lookup_table.lookup[i] = reversed_val
    
    lookup = reverse_bits_lookup_table.lookup
    
    # Split 32-bit number into 4 bytes and reverse each
    return (lookup[n & 0xFF] << 24) | \
           (lookup[(n >> 8) & 0xFF] << 16) | \
           (lookup[(n >> 16) & 0xFF] << 8) | \
           (lookup[(n >> 24) & 0xFF])


def reverse_bits_recursive(n):
    """
    Approach 6: Recursive Implementation
    Time Complexity: O(log 32) = O(1)
    Space Complexity: O(log 32) = O(1) - recursion depth
    
    Recursive divide and conquer approach.
    """
    def reverse_helper(num, bits):
        if bits == 1:
            return num
        
        # Split into two halves
        mask = (1 << (bits // 2)) - 1
        left_half = (num >> (bits // 2)) & mask
        right_half = num & mask
        
        # Recursively reverse each half and swap them
        return (reverse_helper(right_half, bits // 2) << (bits // 2)) | \
               reverse_helper(left_half, bits // 2)
    
    return reverse_helper(n, 32)


def reverse_bits_manual_shifts(n):
    """
    Approach 7: Manual Bit Shifts
    Time Complexity: O(1)
    Space Complexity: O(1)
    
    Manually shift and combine bits in specific patterns.
    """
    # Step 1: Swap odd and even bits
    n = ((n & 0x55555555) << 1) | ((n & 0xAAAAAAAA) >> 1)
    
    # Step 2: Swap consecutive pairs of bits
    n = ((n & 0x33333333) << 2) | ((n & 0xCCCCCCCC) >> 2)
    
    # Step 3: Swap nibbles
    n = ((n & 0x0F0F0F0F) << 4) | ((n & 0xF0F0F0F0) >> 4)
    
    # Step 4: Swap bytes
    n = ((n & 0x00FF00FF) << 8) | ((n & 0xFF00FF00) >> 8)
    
    # Step 5: Swap 16-bit halves
    n = (n << 16) | (n >> 16)
    
    return n & 0xFFFFFFFF


def test_reverse_bits():
    """Test all approaches with various test cases."""
    
    test_cases = [
        (0b00000010100101000001111010011100, 964176192),  # Example 1
        (0b11111111111111111111111111111101, 3221225471),  # Example 2
        (0, 0),  # All zeros
        (0xFFFFFFFF, 0xFFFFFFFF),  # All ones
        (1, 0x80000000),  # Single bit at LSB
        (0x80000000, 1),  # Single bit at MSB
        (0x12345678, 0x1E6A2C48),  # Mixed pattern
        (0xAAAAAAAA, 0x55555555),  # Alternating pattern
        (0x0F0F0F0F, 0xF0F0F0F0),  # Nibble pattern
        (0x00FF00FF, 0xFF00FF00),  # Byte pattern
    ]
    
    approaches = [
        ("Iterative", reverse_bits_iterative),
        ("Bit Manipulation", reverse_bits_bit_manipulation),
        ("Divide & Conquer", reverse_bits_divide_conquer),
        ("String Conversion", reverse_bits_string_conversion),
        ("Lookup Table", reverse_bits_lookup_table),
        ("Recursive", reverse_bits_recursive),
        ("Manual Shifts", reverse_bits_manual_shifts),
    ]
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: {n:032b} ({n})")
        print(f"Expected: {expected:032b} ({expected})")
        
        for name, func in approaches:
            result = func(n)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result:032b} ({result})")


if __name__ == "__main__":
    test_reverse_bits() 