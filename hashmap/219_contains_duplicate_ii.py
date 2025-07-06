"""
219. Contains Duplicate II

Problem:
Given an integer array nums and an integer k, return true if there are two distinct indices i and j 
in the array such that nums[i] == nums[j] and abs(i - j) <= k.

Example 1:
Input: nums = [1,2,3,1], k = 3
Output: true

Example 2:
Input: nums = [1,0,1,1], k = 1
Output: true

Example 3:
Input: nums = [1,2,3,1,2,3], k = 2
Output: false

Time Complexity: O(n) for hash map approach
Space Complexity: O(min(n, k)) for the sliding window approach
"""


def contains_nearby_duplicate(nums, k):
    """
    Check if array contains duplicate within k distance using hash map.
    
    Args:
        nums: List of integers
        k: Maximum distance between duplicates
    
    Returns:
        True if duplicate exists within k distance, False otherwise
    """
    num_indices = {}
    
    for i, num in enumerate(nums):
        if num in num_indices:
            # Check if the distance is within k
            if i - num_indices[num] <= k:
                return True
        
        # Update the most recent index for this number
        num_indices[num] = i
    
    return False


def contains_nearby_duplicate_sliding_window(nums, k):
    """
    Check using sliding window with set.
    
    Args:
        nums: List of integers
        k: Maximum distance between duplicates
    
    Returns:
        True if duplicate exists within k distance, False otherwise
    """
    if k == 0:
        return False
    
    window = set()
    
    for i, num in enumerate(nums):
        # Remove element that's outside the window
        if i > k:
            window.remove(nums[i - k - 1])
        
        # Check if current number is in the window
        if num in window:
            return True
        
        # Add current number to the window
        window.add(num)
    
    return False


def contains_nearby_duplicate_brute_force(nums, k):
    """
    Check using brute force approach.
    
    Args:
        nums: List of integers
        k: Maximum distance between duplicates
    
    Returns:
        True if duplicate exists within k distance, False otherwise
    """
    n = len(nums)
    
    for i in range(n):
        for j in range(i + 1, min(i + k + 1, n)):
            if nums[i] == nums[j]:
                return True
    
    return False


def contains_nearby_duplicate_all_indices(nums, k):
    """
    Check by storing all indices for each number.
    
    Args:
        nums: List of integers
        k: Maximum distance between duplicates
    
    Returns:
        True if duplicate exists within k distance, False otherwise
    """
    num_indices = {}
    
    # Store all indices for each number
    for i, num in enumerate(nums):
        if num not in num_indices:
            num_indices[num] = []
        num_indices[num].append(i)
    
    # Check if any number has indices within k distance
    for indices in num_indices.values():
        if len(indices) > 1:
            for i in range(len(indices) - 1):
                if indices[i + 1] - indices[i] <= k:
                    return True
    
    return False


def contains_nearby_duplicate_deque(nums, k):
    """
    Check using deque to maintain sliding window.
    
    Args:
        nums: List of integers
        k: Maximum distance between duplicates
    
    Returns:
        True if duplicate exists within k distance, False otherwise
    """
    from collections import deque
    
    if k == 0:
        return False
    
    # Use deque to maintain indices
    window = deque()
    seen = set()
    
    for i, num in enumerate(nums):
        # Remove indices that are outside the window
        while window and i - window[0] > k:
            old_idx = window.popleft()
            # Only remove from set if it's not in the current window
            if nums[old_idx] not in [nums[j] for j in window]:
                seen.discard(nums[old_idx])
        
        # Check if current number is in the window
        if num in seen:
            return True
        
        # Add current number and index to window
        window.append(i)
        seen.add(num)
    
    return False


def contains_nearby_duplicate_optimized(nums, k):
    """
    Optimized approach using hash map with early termination.
    
    Args:
        nums: List of integers
        k: Maximum distance between duplicates
    
    Returns:
        True if duplicate exists within k distance, False otherwise
    """
    if k == 0:
        return False
    
    num_indices = {}
    
    for i, num in enumerate(nums):
        if num in num_indices and i - num_indices[num] <= k:
            return True
        num_indices[num] = i
    
    return False


# Test cases
if __name__ == "__main__":
    # Test case 1
    nums1, k1 = [1,2,3,1], 3
    result1a = contains_nearby_duplicate(nums1, k1)
    result1b = contains_nearby_duplicate_sliding_window(nums1, k1)
    result1c = contains_nearby_duplicate_brute_force(nums1, k1)
    result1d = contains_nearby_duplicate_all_indices(nums1, k1)
    result1e = contains_nearby_duplicate_deque(nums1, k1)
    result1f = contains_nearby_duplicate_optimized(nums1, k1)
    print(f"Test 1 - Input: {nums1}, k={k1}, Expected: True")
    print(f"HashMap: {result1a}, SlidingWindow: {result1b}, BruteForce: {result1c}, AllIndices: {result1d}, Deque: {result1e}, Optimized: {result1f}")
    print()
    
    # Test case 2
    nums2, k2 = [1,0,1,1], 1
    result2a = contains_nearby_duplicate(nums2, k2)
    result2b = contains_nearby_duplicate_sliding_window(nums2, k2)
    result2c = contains_nearby_duplicate_brute_force(nums2, k2)
    result2d = contains_nearby_duplicate_all_indices(nums2, k2)
    result2e = contains_nearby_duplicate_deque(nums2, k2)
    result2f = contains_nearby_duplicate_optimized(nums2, k2)
    print(f"Test 2 - Input: {nums2}, k={k2}, Expected: True")
    print(f"HashMap: {result2a}, SlidingWindow: {result2b}, BruteForce: {result2c}, AllIndices: {result2d}, Deque: {result2e}, Optimized: {result2f}")
    print()
    
    # Test case 3
    nums3, k3 = [1,2,3,1,2,3], 2
    result3a = contains_nearby_duplicate(nums3, k3)
    result3b = contains_nearby_duplicate_sliding_window(nums3, k3)
    result3c = contains_nearby_duplicate_brute_force(nums3, k3)
    result3d = contains_nearby_duplicate_all_indices(nums3, k3)
    result3e = contains_nearby_duplicate_deque(nums3, k3)
    result3f = contains_nearby_duplicate_optimized(nums3, k3)
    print(f"Test 3 - Input: {nums3}, k={k3}, Expected: False")
    print(f"HashMap: {result3a}, SlidingWindow: {result3b}, BruteForce: {result3c}, AllIndices: {result3d}, Deque: {result3e}, Optimized: {result3f}")
    print()
    
    # Test case 4 - k = 0
    nums4, k4 = [1,2,3,1], 0
    result4a = contains_nearby_duplicate(nums4, k4)
    result4b = contains_nearby_duplicate_sliding_window(nums4, k4)
    result4c = contains_nearby_duplicate_brute_force(nums4, k4)
    result4d = contains_nearby_duplicate_all_indices(nums4, k4)
    result4e = contains_nearby_duplicate_deque(nums4, k4)
    result4f = contains_nearby_duplicate_optimized(nums4, k4)
    print(f"Test 4 - Input: {nums4}, k={k4}, Expected: False")
    print(f"HashMap: {result4a}, SlidingWindow: {result4b}, BruteForce: {result4c}, AllIndices: {result4d}, Deque: {result4e}, Optimized: {result4f}")
    print()
    
    # Test case 5 - Single element
    nums5, k5 = [1], 1
    result5a = contains_nearby_duplicate(nums5, k5)
    result5b = contains_nearby_duplicate_sliding_window(nums5, k5)
    result5c = contains_nearby_duplicate_brute_force(nums5, k5)
    result5d = contains_nearby_duplicate_all_indices(nums5, k5)
    result5e = contains_nearby_duplicate_deque(nums5, k5)
    result5f = contains_nearby_duplicate_optimized(nums5, k5)
    print(f"Test 5 - Input: {nums5}, k={k5}, Expected: False")
    print(f"HashMap: {result5a}, SlidingWindow: {result5b}, BruteForce: {result5c}, AllIndices: {result5d}, Deque: {result5e}, Optimized: {result5f}")
    print()
    
    # Test case 6 - Adjacent duplicates
    nums6, k6 = [1,1], 1
    result6a = contains_nearby_duplicate(nums6, k6)
    result6b = contains_nearby_duplicate_sliding_window(nums6, k6)
    result6c = contains_nearby_duplicate_brute_force(nums6, k6)
    result6d = contains_nearby_duplicate_all_indices(nums6, k6)
    result6e = contains_nearby_duplicate_deque(nums6, k6)
    result6f = contains_nearby_duplicate_optimized(nums6, k6)
    print(f"Test 6 - Input: {nums6}, k={k6}, Expected: True")
    print(f"HashMap: {result6a}, SlidingWindow: {result6b}, BruteForce: {result6c}, AllIndices: {result6d}, Deque: {result6e}, Optimized: {result6f}")
    print()
    
    # Test case 7 - No duplicates
    nums7, k7 = [1,2,3,4,5], 2
    result7a = contains_nearby_duplicate(nums7, k7)
    result7b = contains_nearby_duplicate_sliding_window(nums7, k7)
    result7c = contains_nearby_duplicate_brute_force(nums7, k7)
    result7d = contains_nearby_duplicate_all_indices(nums7, k7)
    result7e = contains_nearby_duplicate_deque(nums7, k7)
    result7f = contains_nearby_duplicate_optimized(nums7, k7)
    print(f"Test 7 - Input: {nums7}, k={k7}, Expected: False")
    print(f"HashMap: {result7a}, SlidingWindow: {result7b}, BruteForce: {result7c}, AllIndices: {result7d}, Deque: {result7e}, Optimized: {result7f}")
    print()
    
    # Test case 8 - Large k
    nums8, k8 = [1,2,3,1], 10
    result8a = contains_nearby_duplicate(nums8, k8)
    result8b = contains_nearby_duplicate_sliding_window(nums8, k8)
    result8c = contains_nearby_duplicate_brute_force(nums8, k8)
    result8d = contains_nearby_duplicate_all_indices(nums8, k8)
    result8e = contains_nearby_duplicate_deque(nums8, k8)
    result8f = contains_nearby_duplicate_optimized(nums8, k8)
    print(f"Test 8 - Input: {nums8}, k={k8}, Expected: True")
    print(f"HashMap: {result8a}, SlidingWindow: {result8b}, BruteForce: {result8c}, AllIndices: {result8d}, Deque: {result8e}, Optimized: {result8f}")
    print()
    
    # Test case 9 - Multiple duplicates
    nums9, k9 = [1,2,1,2,1], 2
    result9a = contains_nearby_duplicate(nums9, k9)
    result9b = contains_nearby_duplicate_sliding_window(nums9, k9)
    result9c = contains_nearby_duplicate_brute_force(nums9, k9)
    result9d = contains_nearby_duplicate_all_indices(nums9, k9)
    result9e = contains_nearby_duplicate_deque(nums9, k9)
    result9f = contains_nearby_duplicate_optimized(nums9, k9)
    print(f"Test 9 - Input: {nums9}, k={k9}, Expected: True")
    print(f"HashMap: {result9a}, SlidingWindow: {result9b}, BruteForce: {result9c}, AllIndices: {result9d}, Deque: {result9e}, Optimized: {result9f}")
    print()
    
    # Test case 10 - Empty array
    nums10, k10 = [], 1
    result10a = contains_nearby_duplicate(nums10, k10)
    result10b = contains_nearby_duplicate_sliding_window(nums10, k10)
    result10c = contains_nearby_duplicate_brute_force(nums10, k10)
    result10d = contains_nearby_duplicate_all_indices(nums10, k10)
    result10e = contains_nearby_duplicate_deque(nums10, k10)
    result10f = contains_nearby_duplicate_optimized(nums10, k10)
    print(f"Test 10 - Input: {nums10}, k={k10}, Expected: False")
    print(f"HashMap: {result10a}, SlidingWindow: {result10b}, BruteForce: {result10c}, AllIndices: {result10d}, Deque: {result10e}, Optimized: {result10f}") 