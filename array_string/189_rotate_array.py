"""
189. Rotate Array

Problem:
Given an array, rotate the array to the right by k steps, where k is non-negative.

Example 1:
Input: nums = [1,2,3,4,5,6,7], k = 3
Output: [5,6,7,1,2,3,4]

Example 2:
Input: nums = [-1,-100,3,99], k = 2
Output: [3,99,-1,-100]

Time Complexity: O(n)
Space Complexity: O(1)
"""


def rotate_reverse(nums, k):
    """
    Rotate array using reverse approach.
    
    Args:
        nums: List of integers to rotate
        k: Number of steps to rotate right
    """
    n = len(nums)
    k = k % n  # Handle k > n
    
    # Reverse entire array
    reverse(nums, 0, n - 1)
    # Reverse first k elements
    reverse(nums, 0, k - 1)
    # Reverse remaining elements
    reverse(nums, k, n - 1)


def reverse(nums, start, end):
    """
    Reverse array elements from start to end.
    """
    while start < end:
        nums[start], nums[end] = nums[end], nums[start]
        start += 1
        end -= 1


def rotate_cyclic(nums, k):
    """
    Rotate array using cyclic replacement.
    
    Args:
        nums: List of integers to rotate
        k: Number of steps to rotate right
    """
    n = len(nums)
    k = k % n
    
    if k == 0:
        return
    
    count = 0
    start = 0
    
    while count < n:
        current = start
        prev = nums[start]
        
        # Cycle through positions
        while True:
            next_idx = (current + k) % n
            nums[next_idx], prev = prev, nums[next_idx]
            current = next_idx
            count += 1
            
            if start == current:
                break
        
        start += 1


def rotate_extra_space(nums, k):
    """
    Rotate array using extra space.
    
    Args:
        nums: List of integers to rotate
        k: Number of steps to rotate right
    """
    n = len(nums)
    k = k % n
    
    # Create a copy of the array
    nums_copy = nums[:]
    
    # Place each element at its new position
    for i in range(n):
        nums[(i + k) % n] = nums_copy[i]


# Test cases
if __name__ == "__main__":
    # Test case 1
    nums1 = [1, 2, 3, 4, 5, 6, 7]
    k1 = 3
    rotate_reverse(nums1, k1)
    print(f"Test 1 - Expected: [5,6,7,1,2,3,4], Got: {nums1}")
    
    # Test case 2
    nums2 = [-1, -100, 3, 99]
    k2 = 2
    rotate_reverse(nums2, k2)
    print(f"Test 2 - Expected: [3,99,-1,-100], Got: {nums2}")
    
    # Test case 3 - k > n
    nums3 = [1, 2, 3]
    k3 = 4  # Same as k = 1
    rotate_reverse(nums3, k3)
    print(f"Test 3 - Expected: [3,1,2], Got: {nums3}")
    
    # Test case 4 - k = 0
    nums4 = [1, 2, 3, 4]
    k4 = 0
    rotate_reverse(nums4, k4)
    print(f"Test 4 - Expected: [1,2,3,4], Got: {nums4}")
    
    # Test case 5 - Single element
    nums5 = [1]
    k5 = 1
    rotate_reverse(nums5, k5)
    print(f"Test 5 - Expected: [1], Got: {nums5}")
    
    # Test cyclic approach
    print("\nTesting cyclic approach:")
    nums6 = [1, 2, 3, 4, 5, 6, 7]
    k6 = 3
    rotate_cyclic(nums6, k6)
    print(f"Test 6 - Expected: [5,6,7,1,2,3,4], Got: {nums6}")
    
    # Test extra space approach
    print("\nTesting extra space approach:")
    nums7 = [1, 2, 3, 4, 5, 6, 7]
    k7 = 3
    rotate_extra_space(nums7, k7)
    print(f"Test 7 - Expected: [5,6,7,1,2,3,4], Got: {nums7}") 