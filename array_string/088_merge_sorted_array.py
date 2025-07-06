"""
88. Merge Sorted Array

Problem:
You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, 
and two integers m and n, representing the number of elements in nums1 and nums2 respectively.

Merge nums1 and nums2 into a single array sorted in non-decreasing order.

The final sorted array should not be returned by the function, but instead be stored inside the array nums1.
To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, 
and the last n elements are set to 0 and should be ignored. nums2 has a length of n.

Example 1:
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]

Example 2:
Input: nums1 = [1], m = 1, nums2 = [], n = 0
Output: [1]

Example 3:
Input: nums1 = [0], m = 0, nums2 = [1], n = 1
Output: [1]

Time Complexity: O(m + n)
Space Complexity: O(1)
"""


def merge(nums1, m, nums2, n):
    """
    Merge two sorted arrays in place.
    
    Args:
        nums1: First sorted array with extra space at the end
        m: Number of elements in nums1
        nums2: Second sorted array
        n: Number of elements in nums2
    """
    # Start from the end of both arrays
    p1 = m - 1  # Pointer for nums1
    p2 = n - 1  # Pointer for nums2
    p = m + n - 1  # Pointer for merged array position
    
    # Merge from the end to avoid overwriting elements
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1
    
    # Copy remaining elements from nums2 if any
    while p2 >= 0:
        nums1[p] = nums2[p2]
        p2 -= 1
        p -= 1


# Test cases
if __name__ == "__main__":
    # Test case 1
    nums1 = [1, 2, 3, 0, 0, 0]
    m = 3
    nums2 = [2, 5, 6]
    n = 3
    merge(nums1, m, nums2, n)
    print(f"Test 1 - Expected: [1, 2, 2, 3, 5, 6], Got: {nums1}")
    
    # Test case 2
    nums1 = [1]
    m = 1
    nums2 = []
    n = 0
    merge(nums1, m, nums2, n)
    print(f"Test 2 - Expected: [1], Got: {nums1}")
    
    # Test case 3
    nums1 = [0]
    m = 0
    nums2 = [1]
    n = 1
    merge(nums1, m, nums2, n)
    print(f"Test 3 - Expected: [1], Got: {nums1}")
    
    # Test case 4 - Edge case
    nums1 = [4, 5, 6, 0, 0, 0]
    m = 3
    nums2 = [1, 2, 3]
    n = 3
    merge(nums1, m, nums2, n)
    print(f"Test 4 - Expected: [1, 2, 3, 4, 5, 6], Got: {nums1}") 