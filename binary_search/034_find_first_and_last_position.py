"""
34. Find First and Last Position of Element in Sorted Array

Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.

If target is not found in the array, return [-1, -1].

You must write an algorithm with O(log n) runtime complexity.

Example 1:
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]

Example 2:
Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]

Example 3:
Input: nums = [], target = 0
Output: [-1,-1]

Constraints:
- 0 <= nums.length <= 10^5
- -10^9 <= nums[i] <= 10^9
- nums is a non-decreasing array.
- -10^9 <= target <= 10^9
"""

def search_range_two_binary_searches(nums, target):
    """
    Approach 1: Two Separate Binary Searches
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Use two binary searches: one to find the leftmost occurrence,
    and another to find the rightmost occurrence.
    """
    def find_left_bound():
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left
    
    def find_right_bound():
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
        return right
    
    if not nums:
        return [-1, -1]
    
    left_bound = find_left_bound()
    right_bound = find_right_bound()
    
    # Check if target exists
    if left_bound <= right_bound and left_bound < len(nums) and nums[left_bound] == target:
        return [left_bound, right_bound]
    
    return [-1, -1]


def search_range_template_approach(nums, target):
    """
    Approach 2: Binary Search Template
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Use a cleaner binary search template for finding bounds.
    """
    def find_leftmost():
        left, right = 0, len(nums)
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        return left
    
    def find_rightmost():
        left, right = 0, len(nums)
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] <= target:
                left = mid + 1
            else:
                right = mid
        return left - 1
    
    if not nums:
        return [-1, -1]
    
    left_idx = find_leftmost()
    right_idx = find_rightmost()
    
    # Check if target exists
    if left_idx < len(nums) and nums[left_idx] == target:
        return [left_idx, right_idx]
    
    return [-1, -1]


def search_range_unified_approach(nums, target):
    """
    Approach 3: Unified Binary Search Function
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Use a single binary search function with a parameter to find left or right bound.
    """
    def binary_search(find_left):
        left, right = 0, len(nums) - 1
        idx = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                idx = mid
                if find_left:
                    right = mid - 1  # Continue searching left
                else:
                    left = mid + 1   # Continue searching right
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return idx
    
    if not nums:
        return [-1, -1]
    
    left_bound = binary_search(True)
    if left_bound == -1:
        return [-1, -1]
    
    right_bound = binary_search(False)
    return [left_bound, right_bound]


def search_range_linear_expansion(nums, target):
    """
    Approach 4: Binary Search + Linear Expansion
    Time Complexity: O(log n + k) where k is the number of occurrences
    Space Complexity: O(1)
    
    Find any occurrence with binary search, then expand linearly.
    This is efficient when the number of duplicates is small.
    """
    if not nums:
        return [-1, -1]
    
    # Find any occurrence of target
    left, right = 0, len(nums) - 1
    found_idx = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            found_idx = mid
            break
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    if found_idx == -1:
        return [-1, -1]
    
    # Expand to find leftmost occurrence
    left_bound = found_idx
    while left_bound > 0 and nums[left_bound - 1] == target:
        left_bound -= 1
    
    # Expand to find rightmost occurrence
    right_bound = found_idx
    while right_bound < len(nums) - 1 and nums[right_bound + 1] == target:
        right_bound += 1
    
    return [left_bound, right_bound]


def search_range_recursive(nums, target):
    """
    Approach 5: Recursive Binary Search
    Time Complexity: O(log n)
    Space Complexity: O(log n) - recursion depth
    
    Recursive implementation of the binary search approach.
    """
    def find_left_recursive(left, right):
        if left > right:
            return -1
        
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            # Check if this is the leftmost occurrence
            if mid == 0 or nums[mid - 1] != target:
                return mid
            else:
                return find_left_recursive(left, mid - 1)
        elif nums[mid] < target:
            return find_left_recursive(mid + 1, right)
        else:
            return find_left_recursive(left, mid - 1)
    
    def find_right_recursive(left, right):
        if left > right:
            return -1
        
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            # Check if this is the rightmost occurrence
            if mid == len(nums) - 1 or nums[mid + 1] != target:
                return mid
            else:
                return find_right_recursive(mid + 1, right)
        elif nums[mid] < target:
            return find_right_recursive(mid + 1, right)
        else:
            return find_right_recursive(left, mid - 1)
    
    if not nums:
        return [-1, -1]
    
    left_bound = find_left_recursive(0, len(nums) - 1)
    if left_bound == -1:
        return [-1, -1]
    
    right_bound = find_right_recursive(0, len(nums) - 1)
    return [left_bound, right_bound]


def test_search_range():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([5, 7, 7, 8, 8, 10], 8, [3, 4]),
        ([5, 7, 7, 8, 8, 10], 6, [-1, -1]),
        ([], 0, [-1, -1]),
        ([1], 1, [0, 0]),
        ([1], 0, [-1, -1]),
        ([1, 1, 1, 1, 1], 1, [0, 4]),
        ([1, 2, 3, 4, 5], 3, [2, 2]),
        ([1, 3, 3, 3, 3, 3, 3, 5], 3, [1, 6]),
        ([2, 2], 2, [0, 1]),
        ([1, 2, 3], 0, [-1, -1]),
        ([1, 4, 6, 7, 8, 8, 8, 8, 9], 8, [4, 7]),
    ]
    
    approaches = [
        ("Two Binary Searches", search_range_two_binary_searches),
        ("Template Approach", search_range_template_approach),
        ("Unified Approach", search_range_unified_approach),
        ("Linear Expansion", search_range_linear_expansion),
        ("Recursive", search_range_recursive),
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
    test_search_range() 