"""
1. Two Sum

Problem:
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.

Example 1:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:
Input: nums = [3,2,4], target = 6
Output: [1,2]

Example 3:
Input: nums = [3,3], target = 6
Output: [0,1]

Time Complexity: O(n) for optimal solution
Space Complexity: O(n) for optimal solution
"""


def two_sum(nums, target):
    """
    Find two indices where nums[i] + nums[j] = target using hash map.
    
    Args:
        nums: List of integers
        target: Target sum
    
    Returns:
        List of two indices [i, j] where nums[i] + nums[j] = target
    """
    if not nums or len(nums) < 2:
        return []
    
    # Hash map to store value -> index mapping
    num_to_index = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        
        # Check if complement exists in hash map
        if complement in num_to_index:
            return [num_to_index[complement], i]
        
        # Add current number to hash map
        num_to_index[num] = i
    
    return []


def two_sum_brute_force(nums, target):
    """
    Find two indices using brute force approach.
    
    Args:
        nums: List of integers
        target: Target sum
    
    Returns:
        List of two indices [i, j] where nums[i] + nums[j] = target
    """
    if not nums or len(nums) < 2:
        return []
    
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    
    return []


def two_sum_two_pass(nums, target):
    """
    Find two indices using two-pass hash map approach.
    
    Args:
        nums: List of integers
        target: Target sum
    
    Returns:
        List of two indices [i, j] where nums[i] + nums[j] = target
    """
    if not nums or len(nums) < 2:
        return []
    
    # First pass: build hash map
    num_to_index = {}
    for i, num in enumerate(nums):
        num_to_index[num] = i
    
    # Second pass: find complement
    for i, num in enumerate(nums):
        complement = target - num
        
        if complement in num_to_index and num_to_index[complement] != i:
            return [i, num_to_index[complement]]
    
    return []


def two_sum_sorted_two_pointers(nums, target):
    """
    Find two indices using sorting and two pointers (returns values, not indices).
    Note: This approach doesn't preserve original indices.
    
    Args:
        nums: List of integers
        target: Target sum
    
    Returns:
        List of two values [val1, val2] where val1 + val2 = target
    """
    if not nums or len(nums) < 2:
        return []
    
    # Create array of (value, original_index) pairs
    indexed_nums = [(nums[i], i) for i in range(len(nums))]
    indexed_nums.sort()
    
    left, right = 0, len(indexed_nums) - 1
    
    while left < right:
        current_sum = indexed_nums[left][0] + indexed_nums[right][0]
        
        if current_sum == target:
            # Return original indices
            return [indexed_nums[left][1], indexed_nums[right][1]]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []


# Test cases
if __name__ == "__main__":
    # Test case 1
    nums1 = [2, 7, 11, 15]
    target1 = 9
    result1a = two_sum(nums1, target1)
    result1b = two_sum_brute_force(nums1, target1)
    result1c = two_sum_two_pass(nums1, target1)
    result1d = two_sum_sorted_two_pointers(nums1, target1)
    print(f"Test 1 - Expected: [0,1], HashMap: {result1a}, Brute: {result1b}, Two-pass: {result1c}, Sorted: {result1d}")
    
    # Test case 2
    nums2 = [3, 2, 4]
    target2 = 6
    result2a = two_sum(nums2, target2)
    result2b = two_sum_brute_force(nums2, target2)
    result2c = two_sum_two_pass(nums2, target2)
    result2d = two_sum_sorted_two_pointers(nums2, target2)
    print(f"Test 2 - Expected: [1,2], HashMap: {result2a}, Brute: {result2b}, Two-pass: {result2c}, Sorted: {result2d}")
    
    # Test case 3
    nums3 = [3, 3]
    target3 = 6
    result3a = two_sum(nums3, target3)
    result3b = two_sum_brute_force(nums3, target3)
    result3c = two_sum_two_pass(nums3, target3)
    result3d = two_sum_sorted_two_pointers(nums3, target3)
    print(f"Test 3 - Expected: [0,1], HashMap: {result3a}, Brute: {result3b}, Two-pass: {result3c}, Sorted: {result3d}")
    
    # Test case 4 - Negative numbers
    nums4 = [-1, -2, -3, -4, -5]
    target4 = -8
    result4a = two_sum(nums4, target4)
    result4b = two_sum_brute_force(nums4, target4)
    result4c = two_sum_two_pass(nums4, target4)
    result4d = two_sum_sorted_two_pointers(nums4, target4)
    print(f"Test 4 - Expected: [2,4], HashMap: {result4a}, Brute: {result4b}, Two-pass: {result4c}, Sorted: {result4d}")
    
    # Test case 5 - Zero target
    nums5 = [0, 4, 3, 0]
    target5 = 0
    result5a = two_sum(nums5, target5)
    result5b = two_sum_brute_force(nums5, target5)
    result5c = two_sum_two_pass(nums5, target5)
    result5d = two_sum_sorted_two_pointers(nums5, target5)
    print(f"Test 5 - Expected: [0,3], HashMap: {result5a}, Brute: {result5b}, Two-pass: {result5c}, Sorted: {result5d}")
    
    # Test case 6 - Single element (should return empty)
    nums6 = [1]
    target6 = 1
    result6a = two_sum(nums6, target6)
    result6b = two_sum_brute_force(nums6, target6)
    result6c = two_sum_two_pass(nums6, target6)
    result6d = two_sum_sorted_two_pointers(nums6, target6)
    print(f"Test 6 - Expected: [], HashMap: {result6a}, Brute: {result6b}, Two-pass: {result6c}, Sorted: {result6d}")
    
    # Test case 7 - Large numbers
    nums7 = [1000000000, 1000000001, 1000000002]
    target7 = 2000000001
    result7a = two_sum(nums7, target7)
    result7b = two_sum_brute_force(nums7, target7)
    result7c = two_sum_two_pass(nums7, target7)
    result7d = two_sum_sorted_two_pointers(nums7, target7)
    print(f"Test 7 - Expected: [0,1], HashMap: {result7a}, Brute: {result7b}, Two-pass: {result7c}, Sorted: {result7d}")
    
    # Test case 8 - Mixed positive and negative
    nums8 = [-3, 4, 3, 90]
    target8 = 0
    result8a = two_sum(nums8, target8)
    result8b = two_sum_brute_force(nums8, target8)
    result8c = two_sum_two_pass(nums8, target8)
    result8d = two_sum_sorted_two_pointers(nums8, target8)
    print(f"Test 8 - Expected: [0,2], HashMap: {result8a}, Brute: {result8b}, Two-pass: {result8c}, Sorted: {result8d}") 