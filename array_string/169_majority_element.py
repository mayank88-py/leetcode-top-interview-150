"""
169. Majority Element

Problem:
Given an array nums of size n, return the majority element.
The majority element is the element that appears more than ⌊n / 2⌋ times.
You may assume that the majority element always exists in the array.

Example 1:
Input: nums = [3,2,3]
Output: 3

Example 2:
Input: nums = [2,2,1,1,1,2,2]
Output: 2

Time Complexity: O(n)
Space Complexity: O(1)
"""


def majority_element_boyer_moore(nums):
    """
    Find majority element using Boyer-Moore Voting Algorithm.
    
    Args:
        nums: List of integers
    
    Returns:
        The majority element
    """
    candidate = None
    count = 0
    
    # Phase 1: Find candidate
    for num in nums:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)
    
    # Phase 2: Verify candidate (not needed as problem guarantees majority exists)
    return candidate


def majority_element_hashmap(nums):
    """
    Find majority element using HashMap approach.
    
    Args:
        nums: List of integers
    
    Returns:
        The majority element
    """
    counts = {}
    majority_count = len(nums) // 2
    
    for num in nums:
        counts[num] = counts.get(num, 0) + 1
        if counts[num] > majority_count:
            return num
    
    return None


def majority_element_sorting(nums):
    """
    Find majority element using sorting approach.
    
    Args:
        nums: List of integers
    
    Returns:
        The majority element
    """
    nums.sort()
    return nums[len(nums) // 2]


# Test cases
if __name__ == "__main__":
    # Test case 1
    nums1 = [3, 2, 3]
    result1a = majority_element_boyer_moore(nums1)
    result1b = majority_element_hashmap(nums1)
    result1c = majority_element_sorting(nums1.copy())
    print(f"Test 1 - Expected: 3, Boyer-Moore: {result1a}, HashMap: {result1b}, Sorting: {result1c}")
    
    # Test case 2
    nums2 = [2, 2, 1, 1, 1, 2, 2]
    result2a = majority_element_boyer_moore(nums2)
    result2b = majority_element_hashmap(nums2)
    result2c = majority_element_sorting(nums2.copy())
    print(f"Test 2 - Expected: 2, Boyer-Moore: {result2a}, HashMap: {result2b}, Sorting: {result2c}")
    
    # Test case 3 - Single element
    nums3 = [1]
    result3a = majority_element_boyer_moore(nums3)
    result3b = majority_element_hashmap(nums3)
    result3c = majority_element_sorting(nums3.copy())
    print(f"Test 3 - Expected: 1, Boyer-Moore: {result3a}, HashMap: {result3b}, Sorting: {result3c}")
    
    # Test case 4 - All elements are the same
    nums4 = [5, 5, 5, 5, 5]
    result4a = majority_element_boyer_moore(nums4)
    result4b = majority_element_hashmap(nums4)
    result4c = majority_element_sorting(nums4.copy())
    print(f"Test 4 - Expected: 5, Boyer-Moore: {result4a}, HashMap: {result4b}, Sorting: {result4c}")
    
    # Test case 5 - Large array
    nums5 = [1, 2, 1, 2, 1, 2, 1]
    result5a = majority_element_boyer_moore(nums5)
    result5b = majority_element_hashmap(nums5)
    result5c = majority_element_sorting(nums5.copy())
    print(f"Test 5 - Expected: 1, Boyer-Moore: {result5a}, HashMap: {result5b}, Sorting: {result5c}") 