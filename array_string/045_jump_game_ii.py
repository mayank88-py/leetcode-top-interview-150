"""
45. Jump Game II

Problem:
You are given a 0-indexed array of integers nums of length n. You are initially positioned at nums[0].
Each element nums[i] represents the maximum length of a forward jump from index i.
In other words, if you are at nums[i], you can jump to any nums[i + j] where:
- 0 <= j <= nums[i] 
- i + j < n

Return the minimum number of jumps to reach nums[n - 1].
The test cases are generated such that you can reach nums[n - 1].

Example 1:
Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2.
Jump 1 step from index 0 to 1, then 3 steps to the last index.

Example 2:
Input: nums = [2,3,0,1,4]
Output: 2

Time Complexity: O(n)
Space Complexity: O(1)
"""


def jump(nums):
    """
    Find minimum number of jumps using greedy approach.
    
    Args:
        nums: List of jump lengths
    
    Returns:
        Minimum number of jumps to reach the end
    """
    if len(nums) <= 1:
        return 0
    
    jumps = 0
    current_reach = 0
    farthest_reach = 0
    
    # We don't need to consider the last element
    for i in range(len(nums) - 1):
        # Update the farthest we can reach
        farthest_reach = max(farthest_reach, i + nums[i])
        
        # If we've reached the end of current jump range
        if i == current_reach:
            jumps += 1
            current_reach = farthest_reach
            
            # Early termination if we can reach the end
            if current_reach >= len(nums) - 1:
                break
    
    return jumps


def jump_bfs(nums):
    """
    Find minimum number of jumps using BFS approach.
    
    Args:
        nums: List of jump lengths
    
    Returns:
        Minimum number of jumps to reach the end
    """
    if len(nums) <= 1:
        return 0
    
    n = len(nums)
    jumps = 0
    current_level_end = 0
    farthest = 0
    
    for i in range(n - 1):
        farthest = max(farthest, i + nums[i])
        
        if i == current_level_end:
            jumps += 1
            current_level_end = farthest
    
    return jumps


def jump_dp(nums):
    """
    Find minimum number of jumps using dynamic programming.
    
    Args:
        nums: List of jump lengths
    
    Returns:
        Minimum number of jumps to reach the end
    """
    if len(nums) <= 1:
        return 0
    
    n = len(nums)
    dp = [float('inf')] * n
    dp[0] = 0
    
    for i in range(n):
        for j in range(min(nums[i] + 1, n - i)):
            if i + j < n:
                dp[i + j] = min(dp[i + j], dp[i] + 1)
    
    return dp[n - 1]


# Test cases
if __name__ == "__main__":
    # Test case 1
    nums1 = [2, 3, 1, 1, 4]
    result1a = jump(nums1)
    result1b = jump_bfs(nums1)
    result1c = jump_dp(nums1)
    print(f"Test 1 - Expected: 2, Greedy: {result1a}, BFS: {result1b}, DP: {result1c}")
    
    # Test case 2
    nums2 = [2, 3, 0, 1, 4]
    result2a = jump(nums2)
    result2b = jump_bfs(nums2)
    result2c = jump_dp(nums2)
    print(f"Test 2 - Expected: 2, Greedy: {result2a}, BFS: {result2b}, DP: {result2c}")
    
    # Test case 3 - Single element
    nums3 = [0]
    result3a = jump(nums3)
    result3b = jump_bfs(nums3)
    result3c = jump_dp(nums3)
    print(f"Test 3 - Expected: 0, Greedy: {result3a}, BFS: {result3b}, DP: {result3c}")
    
    # Test case 4 - Two elements
    nums4 = [1, 1]
    result4a = jump(nums4)
    result4b = jump_bfs(nums4)
    result4c = jump_dp(nums4)
    print(f"Test 4 - Expected: 1, Greedy: {result4a}, BFS: {result4b}, DP: {result4c}")
    
    # Test case 5 - Large jumps
    nums5 = [1, 3, 2]
    result5a = jump(nums5)
    result5b = jump_bfs(nums5)
    result5c = jump_dp(nums5)
    print(f"Test 5 - Expected: 2, Greedy: {result5a}, BFS: {result5b}, DP: {result5c}")
    
    # Test case 6 - All ones
    nums6 = [1, 1, 1, 1]
    result6a = jump(nums6)
    result6b = jump_bfs(nums6)
    result6c = jump_dp(nums6)
    print(f"Test 6 - Expected: 3, Greedy: {result6a}, BFS: {result6b}, DP: {result6c}")
    
    # Test case 7 - Large array
    nums7 = [3, 4, 3, 2, 5, 4, 3]
    result7a = jump(nums7)
    result7b = jump_bfs(nums7)
    result7c = jump_dp(nums7)
    print(f"Test 7 - Expected: 3, Greedy: {result7a}, BFS: {result7b}, DP: {result7c}")
    
    # Test case 8 - Maximum jump at start
    nums8 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 0]
    result8a = jump(nums8)
    result8b = jump_bfs(nums8)
    result8c = jump_dp(nums8)
    print(f"Test 8 - Expected: 2, Greedy: {result8a}, BFS: {result8b}, DP: {result8c}") 