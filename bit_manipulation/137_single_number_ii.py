"""
137. Single Number II

Given an integer array nums where every element appears three times except for one, which appears exactly once. Find the single element and return it.

You must implement a solution with a linear runtime complexity and use only constant extra space.

Example 1:
Input: nums = [2,2,3,2]
Output: 3

Example 2:
Input: nums = [0,1,0,1,0,1,99]
Output: 99

Constraints:
- 1 <= nums.length <= 3 * 10^4
- -2^31 <= nums[i] <= 2^31 - 1
- Each element in the array appears exactly three times except for one element which appears once.
"""

def single_number_ii_bit_counting(nums):
    """
    Approach 1: Bit Counting
    Time Complexity: O(32n) = O(n)
    Space Complexity: O(1)
    
    Count set bits at each position. If count % 3 != 0, that bit belongs to single number.
    """
    result = 0
    
    # Check each bit position (32 bits for integer)
    for i in range(32):
        bit_count = 0
        
        # Count set bits at position i
        for num in nums:
            if num & (1 << i):
                bit_count += 1
        
        # If count is not divisible by 3, single number has this bit set
        if bit_count % 3 != 0:
            result |= (1 << i)
    
    # Handle negative numbers (two's complement)
    if result >= 2**31:
        result -= 2**32
    
    return result


def single_number_ii_state_machine(nums):
    """
    Approach 2: State Machine (Two variables)
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Use two variables to track the state of each bit (appears 0, 1, or 2 times).
    """
    ones = twos = 0
    
    for num in nums:
        # Update twos: bits that have appeared twice
        twos |= ones & num
        
        # Update ones: bits that have appeared once
        ones ^= num
        
        # Remove bits that have appeared three times
        threes = ones & twos
        ones &= ~threes
        twos &= ~threes
    
    return ones


def single_number_ii_mathematical(nums):
    """
    Approach 3: Mathematical Approach
    Time Complexity: O(n)
    Space Complexity: O(n) for set
    
    3 * (sum of unique numbers) - sum of all numbers = 2 * single number
    """
    unique_nums = set(nums)
    return (3 * sum(unique_nums) - sum(nums)) // 2


def single_number_ii_hash_map(nums):
    """
    Approach 4: Hash Map
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Count frequency of each number and return the one with count 1.
    """
    count = {}
    for num in nums:
        count[num] = count.get(num, 0) + 1
    
    for num, freq in count.items():
        if freq == 1:
            return num


def single_number_ii_sorting(nums):
    """
    Approach 5: Sorting
    Time Complexity: O(n log n)
    Space Complexity: O(1) if in-place sort
    
    Sort array and check triplets.
    """
    nums.sort()
    
    # Check triplets
    for i in range(0, len(nums) - 2, 3):
        if nums[i] != nums[i + 1]:
            return nums[i]
        if nums[i + 1] != nums[i + 2]:
            return nums[i + 1]
    
    # If no mismatch found, last element is the single one
    return nums[-1]


def single_number_ii_three_state_bits(nums):
    """
    Approach 6: Three State Bit Tracking
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Track each bit's state using modular arithmetic.
    """
    a = b = 0
    
    for num in nums:
        # Calculate new states
        new_a = (~a & b & num) | (a & ~b & ~num)
        new_b = ~a & (b ^ num)
        
        a, b = new_a, new_b
    
    return b


def single_number_ii_bit_manipulation_optimized(nums):
    """
    Approach 7: Optimized Bit Manipulation
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Use XOR and AND operations to simulate modulo 3 counting.
    """
    ones = twos = 0
    
    for num in nums:
        ones = (ones ^ num) & ~twos
        twos = (twos ^ num) & ~ones
    
    return ones


def single_number_ii_digital_circuit(nums):
    """
    Approach 8: Digital Circuit Simulation
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Simulate a digital circuit that counts modulo 3.
    """
    x1 = x2 = 0
    
    for num in nums:
        # First bit of the counter
        x1 = x1 ^ num & ~x2
        # Second bit of the counter
        x2 = x2 ^ num & ~x1
    
    return x1


def single_number_ii_generic_k_times(nums, k=3):
    """
    Approach 9: Generic Solution for K Times
    Time Complexity: O(n * log k)
    Space Complexity: O(log k)
    
    Generic solution that works for any k (here k=3).
    """
    # Number of bits needed to represent k states
    bits_needed = 0
    temp = k
    while temp:
        bits_needed += 1
        temp >>= 1
    
    # Initialize bit counters
    counters = [0] * bits_needed
    
    for num in nums:
        # Update counters for each bit position
        for i in range(32):  # 32-bit integers
            if num & (1 << i):
                # Increment counter for this bit position
                carry = 1
                for j in range(bits_needed):
                    sum_bit = (counters[j] >> i) & 1
                    new_bit = sum_bit ^ carry
                    carry = sum_bit & carry
                    
                    # Update the bit
                    if new_bit:
                        counters[j] |= (1 << i)
                    else:
                        counters[j] &= ~(1 << i)
                    
                    # Check if we've reached k
                    if carry == 0:
                        break
                
                # Reset if count reaches k
                count = 0
                for j in range(bits_needed):
                    if (counters[j] >> i) & 1:
                        count += (1 << j)
                
                if count == k:
                    for j in range(bits_needed):
                        counters[j] &= ~(1 << i)
    
    # The result is the first counter
    result = counters[0]
    
    # Handle negative numbers
    if result >= 2**31:
        result -= 2**32
    
    return result


def test_single_number_ii():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([2, 2, 3, 2], 3),
        ([0, 1, 0, 1, 0, 1, 99], 99),
        ([43, 16, 45, 89, 45, -2, 45, 2, 16, -1, 16, 67, -2, -2, 2], 89),
        ([1], 1),
        ([5, 5, 5, 8], 8),
        ([-1, -1, -1, 0], 0),
        ([1, 1, 1, 2, 2, 2, 3], 3),
        ([21, 21, 21, 11], 11),
        ([43, 16, 45, 89, 45, -2, 45], 89),  # Mixed positive and negative
        ([0, 0, 0, 1], 1),
    ]
    
    approaches = [
        ("Bit Counting", single_number_ii_bit_counting),
        ("State Machine", single_number_ii_state_machine),
        ("Mathematical", single_number_ii_mathematical),
        ("Hash Map", single_number_ii_hash_map),
        ("Sorting", single_number_ii_sorting),
        ("Three State Bits", single_number_ii_three_state_bits),
        ("Bit Manipulation Optimized", single_number_ii_bit_manipulation_optimized),
        ("Digital Circuit", single_number_ii_digital_circuit),
        ("Generic K Times", lambda nums: single_number_ii_generic_k_times(nums, 3)),
    ]
    
    for i, (nums, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: {nums}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            # Create a copy for functions that modify the array
            nums_copy = nums.copy()
            result = func(nums_copy)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_single_number_ii() 