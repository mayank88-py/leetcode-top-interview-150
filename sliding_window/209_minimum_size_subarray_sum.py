"""
209. Minimum Size Subarray Sum

Problem:
Given an array of positive integers nums and a positive integer target, return the minimal length
of a contiguous subarray [numsl, numsl+1, ..., numsr-1, numsr] of which the sum is greater than
or equal to target. If there is no such subarray, return 0 instead.

Example 1:
Input: target = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: The subarray [4,3] has the minimal length under the problem constraint.

Example 2:
Input: target = 4, nums = [1,4,4]
Output: 1

Example 3:
Input: target = 11, nums = [1,1,1,1,1,1,1,1]
Output: 0

Time Complexity: O(n)
Space Complexity: O(1)
"""


def min_subarray_len(target, nums):
    """
    Find minimum size subarray with sum >= target using sliding window.
    
    Args:
        target: Target sum
        nums: List of positive integers
    
    Returns:
        Minimum length of subarray with sum >= target, or 0 if none exists
    """
    if not nums:
        return 0
    
    left = 0
    current_sum = 0
    min_length = float('inf')
    
    for right in range(len(nums)):
        # Expand window by including nums[right]
        current_sum += nums[right]
        
        # Contract window while sum >= target
        while current_sum >= target:
            min_length = min(min_length, right - left + 1)
            current_sum -= nums[left]
            left += 1
    
    return min_length if min_length != float('inf') else 0


def min_subarray_len_brute_force(target, nums):
    """
    Find minimum size subarray using brute force approach.
    
    Args:
        target: Target sum
        nums: List of positive integers
    
    Returns:
        Minimum length of subarray with sum >= target, or 0 if none exists
    """
    if not nums:
        return 0
    
    min_length = float('inf')
    
    for i in range(len(nums)):
        current_sum = 0
        for j in range(i, len(nums)):
            current_sum += nums[j]
            if current_sum >= target:
                min_length = min(min_length, j - i + 1)
                break
    
    return min_length if min_length != float('inf') else 0


def min_subarray_len_binary_search(target, nums):
    """
    Find minimum size subarray using binary search on answer.
    
    Args:
        target: Target sum
        nums: List of positive integers
    
    Returns:
        Minimum length of subarray with sum >= target, or 0 if none exists
    """
    if not nums:
        return 0
    
    def can_achieve_length(length):
        """Check if we can achieve target sum with given length."""
        current_sum = sum(nums[:length])
        if current_sum >= target:
            return True
        
        for i in range(length, len(nums)):
            current_sum = current_sum - nums[i - length] + nums[i]
            if current_sum >= target:
                return True
        
        return False
    
    left, right = 1, len(nums)
    result = 0
    
    while left <= right:
        mid = (left + right) // 2
        
        if can_achieve_length(mid):
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    
    return result


# Test cases
if __name__ == "__main__":
    # Test case 1
    target1 = 7
    nums1 = [2, 3, 1, 2, 4, 3]
    result1a = min_subarray_len(target1, nums1)
    result1b = min_subarray_len_brute_force(target1, nums1)
    result1c = min_subarray_len_binary_search(target1, nums1)
    print(f"Test 1 - Expected: 2, Sliding Window: {result1a}, Brute Force: {result1b}, Binary Search: {result1c}")
    
    # Test case 2
    target2 = 4
    nums2 = [1, 4, 4]
    result2a = min_subarray_len(target2, nums2)
    result2b = min_subarray_len_brute_force(target2, nums2)
    result2c = min_subarray_len_binary_search(target2, nums2)
    print(f"Test 2 - Expected: 1, Sliding Window: {result2a}, Brute Force: {result2b}, Binary Search: {result2c}")
    
    # Test case 3
    target3 = 11
    nums3 = [1, 1, 1, 1, 1, 1, 1, 1]
    result3a = min_subarray_len(target3, nums3)
    result3b = min_subarray_len_brute_force(target3, nums3)
    result3c = min_subarray_len_binary_search(target3, nums3)
    print(f"Test 3 - Expected: 0, Sliding Window: {result3a}, Brute Force: {result3b}, Binary Search: {result3c}")
    
    # Test case 4 - Single element equal to target
    target4 = 5
    nums4 = [5]
    result4a = min_subarray_len(target4, nums4)
    result4b = min_subarray_len_brute_force(target4, nums4)
    result4c = min_subarray_len_binary_search(target4, nums4)
    print(f"Test 4 - Expected: 1, Sliding Window: {result4a}, Brute Force: {result4b}, Binary Search: {result4c}")
    
    # Test case 5 - All elements needed
    target5 = 15
    nums5 = [1, 2, 3, 4, 5]
    result5a = min_subarray_len(target5, nums5)
    result5b = min_subarray_len_brute_force(target5, nums5)
    result5c = min_subarray_len_binary_search(target5, nums5)
    print(f"Test 5 - Expected: 5, Sliding Window: {result5a}, Brute Force: {result5b}, Binary Search: {result5c}")
    
    # Test case 6 - Large single element
    target6 = 3
    nums6 = [1, 1, 1, 1, 1, 1, 1, 10]
    result6a = min_subarray_len(target6, nums6)
    result6b = min_subarray_len_brute_force(target6, nums6)
    result6c = min_subarray_len_binary_search(target6, nums6)
    print(f"Test 6 - Expected: 1, Sliding Window: {result6a}, Brute Force: {result6b}, Binary Search: {result6c}")
    
    # Test case 7 - Empty array
    target7 = 5
    nums7 = []
    result7a = min_subarray_len(target7, nums7)
    result7b = min_subarray_len_brute_force(target7, nums7)
    result7c = min_subarray_len_binary_search(target7, nums7)
    print(f"Test 7 - Expected: 0, Sliding Window: {result7a}, Brute Force: {result7b}, Binary Search: {result7c}")
    
    # Test case 8 - Exact match
    target8 = 6
    nums8 = [2, 1, 3]
    result8a = min_subarray_len(target8, nums8)
    result8b = min_subarray_len_brute_force(target8, nums8)
    result8c = min_subarray_len_binary_search(target8, nums8)
    print(f"Test 8 - Expected: 3, Sliding Window: {result8a}, Brute Force: {result8b}, Binary Search: {result8c}") 