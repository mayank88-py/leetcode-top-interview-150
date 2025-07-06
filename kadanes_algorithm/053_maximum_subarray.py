"""
53. Maximum Subarray

Problem:
Given an integer array nums, find the contiguous subarray (containing at least one number) 
which has the largest sum and return its sum.

A subarray is a contiguous part of an array.

Example 1:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.

Example 2:
Input: nums = [1]
Output: 1

Example 3:
Input: nums = [5,4,-1,7,8]
Output: 23

Time Complexity: O(n) for optimal solution
Space Complexity: O(1) for optimal solution
"""


def max_subarray_kadane(nums):
    """
    Kadane's Algorithm - optimal solution.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Algorithm:
    1. Keep track of maximum sum ending at current position
    2. At each position, decide whether to extend previous subarray or start new one
    3. Update global maximum if current maximum is larger
    """
    if not nums:
        return 0
    
    max_current = nums[0]  # Maximum sum ending at current position
    max_global = nums[0]   # Maximum sum found so far
    
    for i in range(1, len(nums)):
        # Either extend previous subarray or start new one
        max_current = max(nums[i], max_current + nums[i])
        max_global = max(max_global, max_current)
    
    return max_global


def max_subarray_divide_conquer(nums):
    """
    Divide and conquer approach.
    
    Time Complexity: O(n log n)
    Space Complexity: O(log n) for recursion stack
    
    Algorithm:
    1. Divide array into two halves
    2. Recursively find max subarray in left and right halves
    3. Find max subarray that crosses the middle
    4. Return maximum of the three
    """
    def max_crossing_sum(left, mid, right):
        """Find maximum sum of subarray crossing the middle."""
        # Maximum sum ending at mid (going left)
        left_sum = float('-inf')
        current_sum = 0
        for i in range(mid, left - 1, -1):
            current_sum += nums[i]
            left_sum = max(left_sum, current_sum)
        
        # Maximum sum starting at mid+1 (going right)
        right_sum = float('-inf')
        current_sum = 0
        for i in range(mid + 1, right + 1):
            current_sum += nums[i]
            right_sum = max(right_sum, current_sum)
        
        return left_sum + right_sum
    
    def max_subarray_helper(left, right):
        if left == right:
            return nums[left]
        
        mid = (left + right) // 2
        
        # Maximum subarray in left half
        left_max = max_subarray_helper(left, mid)
        
        # Maximum subarray in right half
        right_max = max_subarray_helper(mid + 1, right)
        
        # Maximum subarray crossing the middle
        cross_max = max_crossing_sum(left, mid, right)
        
        return max(left_max, right_max, cross_max)
    
    if not nums:
        return 0
    
    return max_subarray_helper(0, len(nums) - 1)


def max_subarray_dp(nums):
    """
    Dynamic programming approach.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Algorithm:
    1. dp[i] = maximum sum of subarray ending at position i
    2. dp[i] = max(nums[i], dp[i-1] + nums[i])
    3. Return maximum value in dp array
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    max_sum = dp[0]
    
    for i in range(1, n):
        dp[i] = max(nums[i], dp[i-1] + nums[i])
        max_sum = max(max_sum, dp[i])
    
    return max_sum


def max_subarray_brute_force(nums):
    """
    Brute force approach - check all subarrays.
    
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    
    Algorithm:
    1. Check all possible subarrays
    2. Calculate sum of each subarray
    3. Keep track of maximum sum
    """
    if not nums:
        return 0
    
    max_sum = float('-inf')
    
    for i in range(len(nums)):
        current_sum = 0
        for j in range(i, len(nums)):
            current_sum += nums[j]
            max_sum = max(max_sum, current_sum)
    
    return max_sum


def max_subarray_with_indices(nums):
    """
    Kadane's algorithm that also returns the indices of the maximum subarray.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Returns:
    (max_sum, start_index, end_index)
    """
    if not nums:
        return 0, 0, 0
    
    max_sum = nums[0]
    current_sum = nums[0]
    start = 0
    end = 0
    temp_start = 0
    
    for i in range(1, len(nums)):
        if current_sum < 0:
            current_sum = nums[i]
            temp_start = i
        else:
            current_sum += nums[i]
        
        if current_sum > max_sum:
            max_sum = current_sum
            start = temp_start
            end = i
    
    return max_sum, start, end


def max_subarray_circular(nums):
    """
    Maximum subarray sum in circular array.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Algorithm:
    1. Find maximum subarray using Kadane's algorithm
    2. Find minimum subarray using modified Kadane's algorithm
    3. Maximum circular = total_sum - minimum_subarray
    4. Return max(maximum_subarray, maximum_circular)
    """
    def kadane_max(arr):
        max_current = max_global = arr[0]
        for i in range(1, len(arr)):
            max_current = max(arr[i], max_current + arr[i])
            max_global = max(max_global, max_current)
        return max_global
    
    def kadane_min(arr):
        min_current = min_global = arr[0]
        for i in range(1, len(arr)):
            min_current = min(arr[i], min_current + arr[i])
            min_global = min(min_global, min_current)
        return min_global
    
    if not nums:
        return 0
    
    # Case 1: Maximum subarray is non-circular
    max_kadane = kadane_max(nums)
    
    # Case 2: Maximum subarray is circular
    total_sum = sum(nums)
    min_kadane = kadane_min(nums)
    max_circular = total_sum - min_kadane
    
    # If all elements are negative, max_circular would be 0 (empty array)
    # In this case, return max_kadane
    if max_circular == 0:
        return max_kadane
    
    return max(max_kadane, max_circular)


def test_maximum_subarray():
    """Test all implementations with various test cases."""
    
    test_cases = [
        ([-2,1,-3,4,-1,2,1,-5,4], 6),
        ([1], 1),
        ([5,4,-1,7,8], 23),
        ([-1], -1),
        ([-2,-1], -1),
        ([1,2,3,4,5], 15),
        ([-1,-2,-3,-4], -1),
        ([0], 0),
        ([1,-1,1], 1)
    ]
    
    implementations = [
        ("Kadane's Algorithm", max_subarray_kadane),
        ("Divide & Conquer", max_subarray_divide_conquer),
        ("Dynamic Programming", max_subarray_dp),
        ("Brute Force", max_subarray_brute_force)
    ]
    
    print("Testing Maximum Subarray...")
    
    for i, (nums, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: nums = {nums}")
        print(f"Expected: {expected}")
        
        for impl_name, impl_func in implementations:
            # Make a copy since some implementations might modify the array
            nums_copy = nums[:]
            result = impl_func(nums_copy)
            
            is_correct = result == expected
            print(f"{impl_name:20} | Result: {result:3} | {'✓' if is_correct else '✗'}")
        
        # Test with indices for first few cases
        if i < 3:
            max_sum, start, end = max_subarray_with_indices(nums[:])
            subarray = nums[start:end+1]
            print(f"{'With indices':20} | Subarray: {subarray} | Sum: {max_sum}")


if __name__ == "__main__":
    test_maximum_subarray() 