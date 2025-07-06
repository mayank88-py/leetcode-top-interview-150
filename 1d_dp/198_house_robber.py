"""
198. House Robber

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

Example 1:
Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.

Example 2:
Input: nums = [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
Total amount you can rob = 2 + 9 + 1 = 12.

Constraints:
- 1 <= nums.length <= 100
- 0 <= nums[i] <= 400
"""

def rob_dp_bottom_up(nums):
    """
    Approach 1: Dynamic Programming (Bottom-up)
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    dp[i] = maximum money that can be robbed from houses 0 to i
    dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    
    for i in range(2, n):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    
    return dp[n-1]


def rob_dp_optimized(nums):
    """
    Approach 2: Space-Optimized DP
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Only keep track of the last two values since we only need them.
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2 = nums[0]  # dp[i-2]
    prev1 = max(nums[0], nums[1])  # dp[i-1]
    
    for i in range(2, len(nums)):
        current = max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = current
    
    return prev1


def rob_memoization(nums):
    """
    Approach 3: Memoization (Top-down DP)
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Use memoization to avoid recalculating subproblems.
    """
    if not nums:
        return 0
    
    memo = {}
    
    def rob_helper(index):
        if index >= len(nums):
            return 0
        if index == len(nums) - 1:
            return nums[index]
        
        if index in memo:
            return memo[index]
        
        # Choice: rob current house or skip it
        rob_current = nums[index] + rob_helper(index + 2)
        skip_current = rob_helper(index + 1)
        
        memo[index] = max(rob_current, skip_current)
        return memo[index]
    
    return rob_helper(0)


def rob_recursive_naive(nums):
    """
    Approach 4: Naive Recursion (Not efficient)
    Time Complexity: O(2^n)
    Space Complexity: O(n) - recursion depth
    
    Simple recursive solution without memoization.
    """
    if len(nums) > 20:  # Avoid TLE for large arrays
        return rob_dp_optimized(nums)
    
    def rob_helper(index):
        if index >= len(nums):
            return 0
        if index == len(nums) - 1:
            return nums[index]
        
        # Choice: rob current house or skip it
        rob_current = nums[index] + rob_helper(index + 2)
        skip_current = rob_helper(index + 1)
        
        return max(rob_current, skip_current)
    
    return rob_helper(0)


def rob_alternative_dp(nums):
    """
    Approach 5: Alternative DP Formulation
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Different way to think about the DP state.
    rob = max money if we rob current house
    not_rob = max money if we don't rob current house
    """
    if not nums:
        return 0
    
    rob = 0      # Max money if we rob current house
    not_rob = 0  # Max money if we don't rob current house
    
    for money in nums:
        # If we rob current house, we can't rob previous house
        new_rob = not_rob + money
        # If we don't rob current house, take max from previous state
        new_not_rob = max(rob, not_rob)
        
        rob = new_rob
        not_rob = new_not_rob
    
    return max(rob, not_rob)


def rob_dp_with_path(nums):
    """
    Approach 6: DP with Path Reconstruction
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Track which houses were robbed for the optimal solution.
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    n = len(nums)
    dp = [0] * n
    parent = [-1] * n  # To track the path
    
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    parent[1] = 0 if nums[0] > nums[1] else 1
    
    for i in range(2, n):
        if dp[i-1] > dp[i-2] + nums[i]:
            dp[i] = dp[i-1]
            parent[i] = i - 1
        else:
            dp[i] = dp[i-2] + nums[i]
            parent[i] = i - 2
    
    return dp[n-1]


def rob_iterative_max_tracking(nums):
    """
    Approach 7: Iterative with Max Tracking
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Keep track of maximum at each step using two variables.
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    max_including_prev = nums[0]  # Max including previous element
    max_excluding_prev = 0        # Max excluding previous element
    
    for i in range(1, len(nums)):
        # Current max excluding current element
        new_max_excluding = max(max_including_prev, max_excluding_prev)
        
        # Current max including current element
        max_including_prev = max_excluding_prev + nums[i]
        
        # Update max excluding for next iteration
        max_excluding_prev = new_max_excluding
    
    return max(max_including_prev, max_excluding_prev)


def rob_sliding_window_concept(nums):
    """
    Approach 8: Sliding Window Concept
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Think of it as maintaining a sliding window of optimal solutions.
    """
    if not nums:
        return 0
    if len(nums) <= 2:
        return max(nums)
    
    # Window of size 2: [prev_prev, prev]
    prev_prev = nums[0]
    prev = max(nums[0], nums[1])
    
    for i in range(2, len(nums)):
        # Either rob current house + best from 2 houses ago,
        # or keep the best from previous house
        current = max(prev, prev_prev + nums[i])
        prev_prev = prev
        prev = current
    
    return prev


def rob_mathematical_approach(nums):
    """
    Approach 9: Mathematical Approach
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Use mathematical insight about the problem structure.
    """
    if not nums:
        return 0
    
    # At each house, we have two choices:
    # 1. Rob this house + optimal solution for houses[0..i-2]
    # 2. Don't rob this house + optimal solution for houses[0..i-1]
    
    incl = nums[0]  # Maximum money including the previous house
    excl = 0        # Maximum money excluding the previous house
    
    for i in range(1, len(nums)):
        # Current max excluding current house is max of previous two
        new_excl = max(incl, excl)
        
        # Current max including current house
        incl = excl + nums[i]
        
        # Update excl for next iteration
        excl = new_excl
    
    return max(incl, excl)


def test_rob():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([1, 2, 3, 1], 4),
        ([2, 7, 9, 3, 1], 12),
        ([1], 1),
        ([2, 1], 2),
        ([5, 1, 3, 9], 14),
        ([2, 1, 1, 2], 4),
        ([1, 2, 3, 1], 4),
        ([4, 1, 2, 9], 13),
        ([2, 7, 9, 3, 1], 12),
        ([1, 3, 1, 3, 100], 103),
        ([100, 1, 1, 100], 200),
        ([5, 5, 10, 100, 10, 5], 110),
        ([1, 2, 9, 9, 1], 18),
        ([2, 3, 2], 3),
        ([5, 1, 3, 9], 14),
    ]
    
    approaches = [
        ("DP Bottom-up", rob_dp_bottom_up),
        ("DP Optimized", rob_dp_optimized),
        ("Memoization", rob_memoization),
        ("Recursive Naive", rob_recursive_naive),
        ("Alternative DP", rob_alternative_dp),
        ("DP with Path", rob_dp_with_path),
        ("Iterative Max Tracking", rob_iterative_max_tracking),
        ("Sliding Window", rob_sliding_window_concept),
        ("Mathematical", rob_mathematical_approach),
    ]
    
    for i, (nums, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: {nums}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            result = func(nums.copy())  # Use copy to avoid modifying original
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_rob() 