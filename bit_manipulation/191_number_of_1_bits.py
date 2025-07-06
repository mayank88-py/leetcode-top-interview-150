"""
191. Number of 1 Bits

Write a function that takes the binary representation of an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).

Note:
- Note that in some languages, such as Java, there is no unsigned integer type. In this case, the input will be given as a signed integer type. It should not affect your algorithm, as the integer's internal binary representation is the same, whether it is signed or unsigned.
- In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 3, the input represents the signed integer. -3.

Example 1:
Input: n = 00000000000000000000000000001011
Output: 3
Explanation: The input binary string 00000000000000000000000000001011 has a total of three '1' bits.

Example 2:
Input: n = 00000000000000000000000010000000
Output: 1
Explanation: The input binary string 00000000000000000000000010000000 has a total of one '1' bit.

Example 3:
Input: n = 11111111111111111111111111111101
Output: 31
Explanation: The input binary string 11111111111111111111111111111101 has a total of thirty one '1' bits.

Constraints:
- The input must be a binary string of length 32.

Follow up: If this function is called many times, how would you optimize it?
"""

def hamming_weight_brian_kernighan(n):
    """
    Approach 1: Brian Kernighan's Algorithm
    Time Complexity: O(number of 1 bits)
    Space Complexity: O(1)
    
    Each iteration removes the rightmost 1 bit.
    n & (n-1) removes the rightmost set bit.
    """
    count = 0
    while n:
        count += 1
        n &= n - 1  # Remove the rightmost 1 bit
    return count


def hamming_weight_bit_shift(n):
    """
    Approach 2: Bit Shifting
    Time Complexity: O(32) = O(1)
    Space Complexity: O(1)
    
    Check each bit position by shifting.
    """
    count = 0
    while n:
        count += n & 1  # Add 1 if current bit is set
        n >>= 1         # Shift right to check next bit
    return count


def hamming_weight_bit_manipulation(n):
    """
    Approach 3: Bit Manipulation with Mask
    Time Complexity: O(32) = O(1)
    Space Complexity: O(1)
    
    Use a mask to check each bit position.
    """
    count = 0
    mask = 1
    for _ in range(32):
        if n & mask:
            count += 1
        mask <<= 1
    return count


def hamming_weight_builtin(n):
    """
    Approach 4: Using Built-in Function
    Time Complexity: O(1) - typically optimized
    Space Complexity: O(1)
    
    Use Python's built-in bin() function.
    """
    return bin(n).count('1')


def hamming_weight_lookup_table(n):
    """
    Approach 5: Lookup Table (8-bit chunks)
    Time Complexity: O(1) after preprocessing
    Space Complexity: O(256) for lookup table
    
    Precompute bit counts for all 8-bit numbers.
    """
    # Initialize lookup table on first call
    if not hasattr(hamming_weight_lookup_table, "lookup"):
        hamming_weight_lookup_table.lookup = [0] * 256
        for i in range(256):
            hamming_weight_lookup_table.lookup[i] = (i & 1) + hamming_weight_lookup_table.lookup[i >> 1]
    
    lookup = hamming_weight_lookup_table.lookup
    
    # Sum bit counts from each byte
    return lookup[n & 0xFF] + \
           lookup[(n >> 8) & 0xFF] + \
           lookup[(n >> 16) & 0xFF] + \
           lookup[(n >> 24) & 0xFF]


def hamming_weight_parallel_counting(n):
    """
    Approach 6: Parallel Bit Counting
    Time Complexity: O(1)
    Space Complexity: O(1)
    
    Use bit manipulation to count bits in parallel.
    """
    # Count bits in pairs
    n = n - ((n >> 1) & 0x55555555)
    
    # Count bits in groups of 4
    n = (n & 0x33333333) + ((n >> 2) & 0x33333333)
    
    # Count bits in groups of 8
    n = (n + (n >> 4)) & 0x0F0F0F0F
    
    # Count bits in groups of 16
    n = n + (n >> 8)
    
    # Count bits in groups of 32
    n = n + (n >> 16)
    
    return n & 0x3F  # Return lower 6 bits (max count is 32)


def hamming_weight_divide_conquer(n):
    """
    Approach 7: Divide and Conquer
    Time Complexity: O(log 32) = O(1)
    Space Complexity: O(1)
    
    Recursively count bits by dividing the number.
    """
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    # Split into two halves and count recursively
    lower_half = n & 0xFFFF
    upper_half = n >> 16
    
    return hamming_weight_divide_conquer(lower_half) + hamming_weight_divide_conquer(upper_half)


def hamming_weight_nibble_lookup(n):
    """
    Approach 8: Nibble Lookup Table
    Time Complexity: O(1)
    Space Complexity: O(16) for nibble lookup
    
    Use 4-bit (nibble) lookup table.
    """
    # Nibble lookup table (4 bits = 16 entries)
    nibble_count = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]
    
    count = 0
    while n:
        count += nibble_count[n & 0xF]  # Count bits in lower 4 bits
        n >>= 4  # Shift right by 4 bits
    
    return count


def hamming_weight_string_conversion(n):
    """
    Approach 9: String Conversion
    Time Complexity: O(32) = O(1)
    Space Complexity: O(32) = O(1)
    
    Convert to binary string and count '1's.
    """
    binary_str = format(n, '032b')
    return binary_str.count('1')


def test_hamming_weight():
    """Test all approaches with various test cases."""
    
    test_cases = [
        (0b00000000000000000000000000001011, 3),   # Example 1
        (0b00000000000000000000000010000000, 1),   # Example 2
        (0b11111111111111111111111111111101, 31),  # Example 3
        (0, 0),                                     # All zeros
        (0xFFFFFFFF, 32),                          # All ones
        (1, 1),                                     # Single bit
        (0x80000000, 1),                           # MSB only
        (0x55555555, 16),                          # Alternating bits (0101...)
        (0xAAAAAAAA, 16),                          # Alternating bits (1010...)
        (0x0F0F0F0F, 16),                          # Nibble pattern
        (0x12345678, 13),                          # Mixed pattern
        (0x7FFFFFFF, 31),                          # All but MSB
    ]
    
    approaches = [
        ("Brian Kernighan", hamming_weight_brian_kernighan),
        ("Bit Shift", hamming_weight_bit_shift),
        ("Bit Manipulation", hamming_weight_bit_manipulation),
        ("Built-in", hamming_weight_builtin),
        ("Lookup Table", hamming_weight_lookup_table),
        ("Parallel Counting", hamming_weight_parallel_counting),
        ("Divide & Conquer", hamming_weight_divide_conquer),
        ("Nibble Lookup", hamming_weight_nibble_lookup),
        ("String Conversion", hamming_weight_string_conversion),
    ]
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: {n:032b} ({n})")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            result = func(n)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_hamming_weight() 