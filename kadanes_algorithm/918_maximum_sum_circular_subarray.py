"""
918. Maximum Sum Circular Subarray

Problem:
Given a circular integer array nums of length n, return the maximum possible sum of a non-empty subarray of nums.

A circular array means the end of the array connects to the beginning of the array. 
Formally, the next element of nums[i] is nums[(i + 1) % n] and the previous element of nums[i] is nums[(i - 1 + n) % n].

A subarray may only include each element of the fixed buffer nums at most once. 
Formally, for a subarray nums[i], nums[i + 1], ..., nums[j], there are no indices k1, k2 where i <= k1, k2 <= j and k1 % n == k2 % n.

Example 1:
Input: nums = [1,-2,3,-2]
Output: 3
Explanation: Subarray [3] has maximum sum 3.

Example 2:
Input: nums = [5,-3,5]
Output: 10
Explanation: Subarray [5,5] has maximum sum 5 + 5 = 10.

Example 3:
Input: nums = [-3,-2,-3]
Output: -2
Explanation: Subarray [-2] has maximum sum -2.

Time Complexity: O(n)
Space Complexity: O(1)
"""


def max_subarray_sum_circular_optimal(nums):
    """
    Optimal solution using Kadane's algorithm for both max and min subarrays.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Algorithm:
    1. Case 1: Maximum subarray is non-circular (standard Kadane's)
    2. Case 2: Maximum subarray is circular = total_sum - minimum_subarray
    3. Return max of both cases, but handle all-negative case
    """
    def kadane_max(arr):
        """Standard Kadane's algorithm for maximum subarray."""
        max_current = max_global = arr[0]
        for i in range(1, len(arr)):
            max_current = max(arr[i], max_current + arr[i])
            max_global = max(max_global, max_current)
        return max_global
    
    def kadane_min(arr):
        """Modified Kadane's algorithm for minimum subarray."""
        min_current = min_global = arr[0]
        for i in range(1, len(arr)):
            min_current = min(arr[i], min_current + arr[i])
            min_global = min(min_global, min_current)
        return min_global
    
    if not nums:
        return 0
    
    # Case 1: Maximum subarray is non-circular
    max_normal = kadane_max(nums)
    
    # Case 2: Maximum subarray is circular
    total_sum = sum(nums)
    min_subarray = kadane_min(nums)
    max_circular = total_sum - min_subarray
    
    # Special case: if all elements are negative, max_circular would be 0
    # But we need at least one element, so return max_normal
    if max_circular == 0:
        return max_normal
    
    return max(max_normal, max_circular)


def max_subarray_sum_circular_prefix_suffix(nums):
    """
    Prefix and suffix maximum approach.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Algorithm:
    1. Compute prefix maximums and suffix maximums
    2. For each split point, check prefix_max + suffix_max
    3. Also compute standard Kadane's for non-circular case
    4. Return maximum of all cases
    """
    n = len(nums)
    if n == 1:
        return nums[0]
    
    # Standard Kadane's for non-circular case
    def kadane(arr):
        max_current = max_global = arr[0]
        for i in range(1, len(arr)):
            max_current = max(arr[i], max_current + arr[i])
            max_global = max(max_global, max_current)
        return max_global
    
    max_non_circular = kadane(nums)
    
    # Compute prefix maximums
    prefix_sum = [0] * n
    prefix_max = [0] * n
    prefix_sum[0] = nums[0]
    prefix_max[0] = nums[0]
    
    for i in range(1, n):
        prefix_sum[i] = prefix_sum[i-1] + nums[i]
        prefix_max[i] = max(prefix_max[i-1], prefix_sum[i])
    
    # Compute suffix maximums
    suffix_sum = [0] * n
    suffix_max = [0] * n
    suffix_sum[n-1] = nums[n-1]
    suffix_max[n-1] = nums[n-1]
    
    for i in range(n-2, -1, -1):
        suffix_sum[i] = suffix_sum[i+1] + nums[i]
        suffix_max[i] = max(suffix_max[i+1], suffix_sum[i])
    
    # Find maximum circular subarray
    max_circular = float('-inf')
    for i in range(n-1):
        max_circular = max(max_circular, prefix_max[i] + suffix_max[i+1])
    
    return max(max_non_circular, max_circular)


def max_subarray_sum_circular_deque(nums):
    """
    Sliding window maximum using deque approach.
    
    Time Complexity: O(n^2) in worst case, but often better in practice
    Space Complexity: O(n)
    
    Algorithm:
    1. Try all possible window sizes from 1 to n
    2. For each window size, find maximum sum using sliding window
    3. Use deque to maintain window efficiently
    """
    from collections import deque
    
    n = len(nums)
    max_sum = float('-inf')
    
    # Try all window sizes
    for window_size in range(1, n + 1):
        current_sum = sum(nums[:window_size])
        window_max = current_sum
        
        # Slide the window
        for start in range(1, n):
            # Remove element going out of window
            current_sum -= nums[(start - 1) % n]
            # Add element coming into window
            current_sum += nums[(start + window_size - 1) % n]
            window_max = max(window_max, current_sum)
        
        max_sum = max(max_sum, window_max)
    
    return max_sum


def max_subarray_sum_circular_brute_force(nums):
    """
    Brute force approach - check all possible circular subarrays.
    
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    
    Algorithm:
    1. Try all possible starting positions
    2. For each start, try all possible lengths
    3. Calculate sum of each circular subarray
    4. Keep track of maximum
    """
    n = len(nums)
    max_sum = float('-inf')
    
    for start in range(n):
        current_sum = 0
        for length in range(1, n + 1):
            current_sum += nums[(start + length - 1) % n]
            max_sum = max(max_sum, current_sum)
    
    return max_sum


def max_subarray_sum_circular_dp(nums):
    """
    Dynamic programming approach with state tracking.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Algorithm:
    1. Track maximum subarray ending at each position for both directions
    2. Consider wrapping around the array
    3. Use DP to efficiently compute all possibilities
    """
    n = len(nums)
    if n == 1:
        return nums[0]
    
    # DP arrays for forward and backward traversal
    forward_max = [0] * n
    backward_max = [0] * n
    
    # Forward pass (standard Kadane's)
    forward_max[0] = nums[0]
    for i in range(1, n):
        forward_max[i] = max(nums[i], forward_max[i-1] + nums[i])
    
    # Backward pass
    backward_max[n-1] = nums[n-1]
    for i in range(n-2, -1, -1):
        backward_max[i] = max(nums[i], backward_max[i+1] + nums[i])
    
    # Find maximum non-circular subarray
    max_non_circular = max(forward_max)
    
    # Find maximum circular subarray
    # This would involve prefix sums + suffix sums
    total_sum = sum(nums)
    
    # Use min subarray approach for circular case
    min_current = min_global = nums[0]
    for i in range(1, n):
        min_current = min(nums[i], min_current + nums[i])
        min_global = min(min_global, min_current)
    
    max_circular = total_sum - min_global
    
    # Handle all negative case
    if max_circular == 0:
        return max_non_circular
    
    return max(max_non_circular, max_circular)


def test_max_subarray_sum_circular():
    """Test all implementations with various test cases."""
    
    test_cases = [
        ([1,-2,3,-2], 3),
        ([5,-3,5], 10),
        ([-3,-2,-3], -2),
        ([3,-1,2,-1], 4),
        ([3,-2,2,-3], 3),
        ([-2,-3,-1], -1),
        ([1,2,3], 6),
        ([5], 5),
        ([1,-1,1,-1], 2)
    ]
    
    implementations = [
        ("Optimal (Kadane)", max_subarray_sum_circular_optimal),
        ("Prefix-Suffix", max_subarray_sum_circular_prefix_suffix),
        ("Sliding Window", max_subarray_sum_circular_deque),
        ("Brute Force", max_subarray_sum_circular_brute_force),
        ("Dynamic Programming", max_subarray_sum_circular_dp)
    ]
    
    print("Testing Maximum Sum Circular Subarray...")
    
    for i, (nums, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: nums = {nums}")
        print(f"Expected: {expected}")
        
        for impl_name, impl_func in implementations:
            # Make a copy since some implementations might modify the array
            nums_copy = nums[:]
            result = impl_func(nums_copy)
            
            is_correct = result == expected
            print(f"{impl_name:20} | Result: {result:3} | {'✓' if is_correct else '✗'}")


if __name__ == "__main__":
    test_max_subarray_sum_circular() 