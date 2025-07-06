"""
35. Search Insert Position

Given a sorted array of distinct integers and a target value, return the index if the target is found. 
If not, return the index where it would be if it were inserted in order.

You must write an algorithm with O(log n) runtime complexity.

Example 1:
Input: nums = [1,3,5,6], target = 5
Output: 2

Example 2:
Input: nums = [1,3,5,6], target = 2
Output: 1

Example 3:
Input: nums = [1,3,5,6], target = 7
Output: 4

Constraints:
- 1 <= nums.length <= 10^4
- -10^4 <= nums[i] <= 10^4
- nums contains distinct values sorted in ascending order.
- -10^4 <= target <= 10^4
"""

def search_insert_position_binary_search(nums, target):
    """
    Approach 1: Standard Binary Search
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Use binary search to find the target or the position where it should be inserted.
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    # If target not found, left is the insertion position
    return left


def search_insert_position_binary_search_template(nums, target):
    """
    Approach 2: Binary Search Template (Find leftmost position)
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Use a template that finds the leftmost position where target can be inserted.
    """
    left, right = 0, len(nums)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left


def search_insert_position_linear(nums, target):
    """
    Approach 3: Linear Search
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Simple linear search for comparison (not optimal for sorted array).
    """
    for i, num in enumerate(nums):
        if num >= target:
            return i
    
    # If target is larger than all elements
    return len(nums)


def search_insert_position_builtin(nums, target):
    """
    Approach 4: Using Built-in Bisect Logic
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Implementing bisect_left logic manually without importing bisect.
    """
    def bisect_left_manual(arr, x):
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] < x:
                left = mid + 1
            else:
                right = mid
        return left
    
    return bisect_left_manual(nums, target)


def test_search_insert_position():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([1, 3, 5, 6], 5, 2),      # Target found
        ([1, 3, 5, 6], 2, 1),      # Insert in middle
        ([1, 3, 5, 6], 7, 4),      # Insert at end
        ([1, 3, 5, 6], 0, 0),      # Insert at beginning
        ([1], 1, 0),               # Single element, found
        ([1], 0, 0),               # Single element, insert before
        ([1], 2, 1),               # Single element, insert after
        ([1, 3], 2, 1),            # Two elements
        ([1, 2, 3, 4, 5], 3, 2),   # Multiple elements, found
        ([2, 7, 8, 9, 10], 1, 0),  # Insert at start
    ]
    
    approaches = [
        ("Binary Search", search_insert_position_binary_search),
        ("Binary Search Template", search_insert_position_binary_search_template),
        ("Linear Search", search_insert_position_linear),
        ("Manual Bisect", search_insert_position_builtin),
    ]
    
    for i, (nums, target, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: nums = {nums}, target = {target}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            result = func(nums, target)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_search_insert_position() 