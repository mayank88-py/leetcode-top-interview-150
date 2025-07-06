"""
55. Jump Game

Problem:
You are given an integer array nums. You are initially positioned at the array's first index,
and each element in the array represents your maximum jump length at that position.

Return true if you can reach the last index, or false otherwise.

Example 1:
Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.

Example 2:
Input: nums = [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0,
which makes it impossible to reach the last index.

Time Complexity: O(n)
Space Complexity: O(1)
"""


def can_jump(nums):
    """
    Check if we can reach the last index using greedy approach.
    
    Args:
        nums: List of jump lengths
    
    Returns:
        True if we can reach the last index, False otherwise
    """
    if not nums or len(nums) == 1:
        return True
    
    max_reach = 0
    
    for i in range(len(nums)):
        # If current index is beyond our reach, we can't proceed
        if i > max_reach:
            return False
        
        # Update maximum reachable index
        max_reach = max(max_reach, i + nums[i])
        
        # If we can reach or exceed the last index, return True
        if max_reach >= len(nums) - 1:
            return True
    
    return False


def can_jump_backward(nums):
    """
    Check if we can reach the last index using backward approach.
    
    Args:
        nums: List of jump lengths
    
    Returns:
        True if we can reach the last index, False otherwise
    """
    if not nums or len(nums) == 1:
        return True
    
    # Start from the last index
    last_good_index = len(nums) - 1
    
    # Work backwards
    for i in range(len(nums) - 2, -1, -1):
        # If we can reach the last good index from current position
        if i + nums[i] >= last_good_index:
            last_good_index = i
    
    # If we can reach the last good index from the first position
    return last_good_index == 0


# Test cases
if __name__ == "__main__":
    # Test case 1
    nums1 = [2, 3, 1, 1, 4]
    result1a = can_jump(nums1)
    result1b = can_jump_backward(nums1)
    print(f"Test 1 - Expected: True, Greedy: {result1a}, Backward: {result1b}")
    
    # Test case 2
    nums2 = [3, 2, 1, 0, 4]
    result2a = can_jump(nums2)
    result2b = can_jump_backward(nums2)
    print(f"Test 2 - Expected: False, Greedy: {result2a}, Backward: {result2b}")
    
    # Test case 3 - Single element
    nums3 = [0]
    result3a = can_jump(nums3)
    result3b = can_jump_backward(nums3)
    print(f"Test 3 - Expected: True, Greedy: {result3a}, Backward: {result3b}")
    
    # Test case 4 - All zeros except first
    nums4 = [1, 0, 0, 0]
    result4a = can_jump(nums4)
    result4b = can_jump_backward(nums4)
    print(f"Test 4 - Expected: False, Greedy: {result4a}, Backward: {result4b}")
    
    # Test case 5 - Large jumps
    nums5 = [2, 0, 0]
    result5a = can_jump(nums5)
    result5b = can_jump_backward(nums5)
    print(f"Test 5 - Expected: True, Greedy: {result5a}, Backward: {result5b}")
    
    # Test case 6 - Can't reach due to zero
    nums6 = [1, 0, 1, 0]
    result6a = can_jump(nums6)
    result6b = can_jump_backward(nums6)
    print(f"Test 6 - Expected: False, Greedy: {result6a}, Backward: {result6b}")
    
    # Test case 7 - Two elements
    nums7 = [1, 1]
    result7a = can_jump(nums7)
    result7b = can_jump_backward(nums7)
    print(f"Test 7 - Expected: True, Greedy: {result7a}, Backward: {result7b}")
    
    # Test case 8 - Empty array
    nums8 = []
    result8a = can_jump(nums8)
    result8b = can_jump_backward(nums8)
    print(f"Test 8 - Expected: True, Greedy: {result8a}, Backward: {result8b}") 