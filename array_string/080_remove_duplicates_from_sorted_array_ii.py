"""
80. Remove Duplicates from Sorted Array II

Problem:
Given an integer array nums sorted in non-decreasing order, remove some duplicates in-place
such that each unique element appears at most twice. The relative order of the elements should be kept the same.

Since it is impossible to change the length of the array in some languages, you must instead
have the result be placed in the first part of the array nums. More formally, if there are k
elements after removing the duplicates, then the first k elements of nums should hold the final result.
It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Example 1:
Input: nums = [1,1,1,2,2,3]
Output: 5, nums = [1,1,2,2,3,_]

Example 2:
Input: nums = [0,0,1,1,1,1,2,3,3]
Output: 7, nums = [0,0,1,1,2,3,3,_,_]

Time Complexity: O(n)
Space Complexity: O(1)
"""


def remove_duplicates(nums):
    """
    Remove duplicates from sorted array allowing at most 2 duplicates.
    
    Args:
        nums: Sorted list of integers
    
    Returns:
        Length of the array after removing duplicates
    """
    if len(nums) <= 2:
        return len(nums)
    
    # Two-pointer approach
    write_index = 2  # Start from index 2 since first 2 elements are always valid
    
    for read_index in range(2, len(nums)):
        # Check if current element is different from element at write_index-2
        # This ensures at most 2 duplicates
        if nums[read_index] != nums[write_index - 2]:
            nums[write_index] = nums[read_index]
            write_index += 1
    
    return write_index


# Test cases
if __name__ == "__main__":
    # Test case 1
    nums1 = [1, 1, 1, 2, 2, 3]
    result1 = remove_duplicates(nums1)
    print(f"Test 1 - Expected: 5, Got: {result1}, Array: {nums1[:result1]}")
    
    # Test case 2
    nums2 = [0, 0, 1, 1, 1, 1, 2, 3, 3]
    result2 = remove_duplicates(nums2)
    print(f"Test 2 - Expected: 7, Got: {result2}, Array: {nums2[:result2]}")
    
    # Test case 3 - All elements are the same
    nums3 = [1, 1, 1, 1, 1]
    result3 = remove_duplicates(nums3)
    print(f"Test 3 - Expected: 2, Got: {result3}, Array: {nums3[:result3]}")
    
    # Test case 4 - No duplicates
    nums4 = [1, 2, 3, 4, 5]
    result4 = remove_duplicates(nums4)
    print(f"Test 4 - Expected: 5, Got: {result4}, Array: {nums4[:result4]}")
    
    # Test case 5 - Two elements
    nums5 = [1, 1]
    result5 = remove_duplicates(nums5)
    print(f"Test 5 - Expected: 2, Got: {result5}, Array: {nums5[:result5]}")
    
    # Test case 6 - Single element
    nums6 = [1]
    result6 = remove_duplicates(nums6)
    print(f"Test 6 - Expected: 1, Got: {result6}, Array: {nums6[:result6]}")
    
    # Test case 7 - Each element appears exactly twice
    nums7 = [1, 1, 2, 2, 3, 3]
    result7 = remove_duplicates(nums7)
    print(f"Test 7 - Expected: 6, Got: {result7}, Array: {nums7[:result7]}") 