"""
136. Single Number

Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.

You must implement a solution with a linear runtime complexity and use only constant extra space.

Example 1:
Input: nums = [2,2,1]
Output: 1

Example 2:
Input: nums = [4,1,2,1,2]
Output: 4

Example 3:
Input: nums = [1]
Output: 1

Constraints:
- 1 <= nums.length <= 3 * 10^4
- -3 * 10^4 <= nums[i] <= 3 * 10^4
- Each element in the array appears twice except for one element which appears only once.
"""

def single_number_xor(nums):
    """
    Approach 1: XOR Operation (Optimal)
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    XOR properties: a ^ a = 0, a ^ 0 = a, XOR is commutative and associative
    All duplicate numbers will cancel out, leaving only the single number.
    """
    result = 0
    for num in nums:
        result ^= num
    return result


def single_number_hash_set(nums):
    """
    Approach 2: Hash Set
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Use set to track seen numbers. Remove if seen before, add if new.
    """
    seen = set()
    for num in nums:
        if num in seen:
            seen.remove(num)
        else:
            seen.add(num)
    
    return seen.pop()  # Only one element remains


def single_number_hash_map(nums):
    """
    Approach 3: Hash Map (Count frequency)
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


def single_number_sorting(nums):
    """
    Approach 4: Sorting
    Time Complexity: O(n log n)
    Space Complexity: O(1) if in-place sort
    
    Sort array and check adjacent pairs.
    """
    nums.sort()
    
    # Check pairs
    for i in range(0, len(nums) - 1, 2):
        if nums[i] != nums[i + 1]:
            return nums[i]
    
    # If no mismatch found, last element is the single one
    return nums[-1]


def single_number_mathematical(nums):
    """
    Approach 5: Mathematical Approach
    Time Complexity: O(n)
    Space Complexity: O(n) for set
    
    2 * (sum of unique numbers) - sum of all numbers = single number
    """
    unique_nums = set(nums)
    return 2 * sum(unique_nums) - sum(nums)


def single_number_bit_manipulation(nums):
    """
    Approach 6: Bit Manipulation (Alternative XOR)
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Another way to implement XOR using bit operations.
    """
    result = 0
    for num in nums:
        result = result ^ num
    return result


def single_number_reduce(nums):
    """
    Approach 7: Using Reduce (Functional approach)
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Use reduce function with XOR operator.
    """
    from functools import reduce
    import operator
    
    return reduce(operator.xor, nums, 0)


def single_number_linear_search(nums):
    """
    Approach 8: Linear Search (Brute Force)
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    
    For each number, count its occurrences in the array.
    """
    for i, num in enumerate(nums):
        count = 0
        for j, other in enumerate(nums):
            if num == other:
                count += 1
        
        if count == 1:
            return num


def single_number_stack(nums):
    """
    Approach 9: Stack-based Approach
    Time Complexity: O(n log n) due to sorting
    Space Complexity: O(n)
    
    Use stack to match pairs after sorting.
    """
    nums.sort()
    stack = []
    
    for num in nums:
        if stack and stack[-1] == num:
            stack.pop()  # Remove pair
        else:
            stack.append(num)
    
    return stack[0]  # Only one element remains


def test_single_number():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([2, 2, 1], 1),
        ([4, 1, 2, 1, 2], 4),
        ([1], 1),
        ([0, 1, 0], 1),
        ([1, 2, 1, 3, 2, 5], 3),  # Not mentioned in problem but valid
        ([-1, -1, 2], 2),
        ([5, 7, 5, 4, 7], 4),
        ([1, 0, 1], 0),
        ([2, 1, 4, 9, 6, 9, 6, 2, 1], 4),
        ([100], 100),
    ]
    
    approaches = [
        ("XOR", single_number_xor),
        ("Hash Set", single_number_hash_set),
        ("Hash Map", single_number_hash_map),
        ("Sorting", single_number_sorting),
        ("Mathematical", single_number_mathematical),
        ("Bit Manipulation", single_number_bit_manipulation),
        ("Reduce", single_number_reduce),
        ("Linear Search", single_number_linear_search),
        ("Stack", single_number_stack),
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
    test_single_number() 