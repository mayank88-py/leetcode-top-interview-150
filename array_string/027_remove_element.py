"""
27. Remove Element

Problem:
Given an integer array nums and an integer val, remove all occurrences of val in nums in-place.
The relative order of the elements may be changed.

Since it is impossible to change the length of the array in some languages, you must instead
have the result be placed in the first part of the array nums. More formally, if there are k
elements after removing the duplicates, then the first k elements of nums should hold the final result.
It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Example 1:
Input: nums = [3,2,2,3], val = 3
Output: 2, nums = [2,2,_,_]

Example 2:
Input: nums = [0,1,2,2,3,0,4,2], val = 2
Output: 5, nums = [0,1,4,0,3,_,_,_]

Time Complexity: O(n)
Space Complexity: O(1)
"""


def remove_element(nums, val):
    """
    Remove all occurrences of val in nums in-place.
    
    Args:
        nums: List of integers
        val: Value to remove
    
    Returns:
        Length of the array after removal
    """
    # Two-pointer approach
    write_index = 0
    
    for read_index in range(len(nums)):
        if nums[read_index] != val:
            nums[write_index] = nums[read_index]
            write_index += 1
    
    return write_index


# Test cases
if __name__ == "__main__":
    # Test case 1
    nums1 = [3, 2, 2, 3]
    val1 = 3
    result1 = remove_element(nums1, val1)
    print(f"Test 1 - Expected: 2, Got: {result1}, Array: {nums1[:result1]}")
    
    # Test case 2
    nums2 = [0, 1, 2, 2, 3, 0, 4, 2]
    val2 = 2
    result2 = remove_element(nums2, val2)
    print(f"Test 2 - Expected: 5, Got: {result2}, Array: {nums2[:result2]}")
    
    # Test case 3 - All elements are the same as val
    nums3 = [3, 3, 3, 3]
    val3 = 3
    result3 = remove_element(nums3, val3)
    print(f"Test 3 - Expected: 0, Got: {result3}, Array: {nums3[:result3]}")
    
    # Test case 4 - No elements match val
    nums4 = [1, 2, 3, 4]
    val4 = 5
    result4 = remove_element(nums4, val4)
    print(f"Test 4 - Expected: 4, Got: {result4}, Array: {nums4[:result4]}")
    
    # Test case 5 - Empty array
    nums5 = []
    val5 = 1
    result5 = remove_element(nums5, val5)
    print(f"Test 5 - Expected: 0, Got: {result5}, Array: {nums5[:result5]}") 