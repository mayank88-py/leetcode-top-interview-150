"""
33. Search in Rotated Sorted Array

There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is possibly rotated at some pivot index k 
(1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). 
For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.

Example 1:
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Example 2:
Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1

Example 3:
Input: nums = [1], target = 0
Output: -1

Constraints:
- 1 <= nums.length <= 5000
- -10^4 <= nums[i] <= 10^4
- All values of nums are unique.
- nums is an ascending array that is possibly rotated.
- -10^4 <= target <= 10^4
"""

def search_rotated_array_one_pass(nums, target):
    """
    Approach 1: One-Pass Binary Search
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Modified binary search that works directly on rotated array.
    At each step, determine which half is sorted and search accordingly.
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Determine which half is sorted
        if nums[left] <= nums[mid]:  # Left half is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1


def search_rotated_array_find_pivot(nums, target):
    """
    Approach 2: Find Pivot First, Then Binary Search
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    First find the rotation pivot, then determine which part to search.
    """
    def find_pivot():
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid
        return left
    
    def binary_search(start, end):
        while start <= end:
            mid = start + (end - start) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                start = mid + 1
            else:
                end = mid - 1
        return -1
    
    n = len(nums)
    if n == 1:
        return 0 if nums[0] == target else -1
    
    pivot = find_pivot()
    
    # Search in the appropriate half
    if pivot == 0:  # Array is not rotated
        return binary_search(0, n - 1)
    
    if target >= nums[0]:
        return binary_search(0, pivot - 1)
    else:
        return binary_search(pivot, n - 1)


def search_rotated_array_recursive(nums, target):
    """
    Approach 3: Recursive Binary Search
    Time Complexity: O(log n)
    Space Complexity: O(log n) - recursion depth
    
    Recursive implementation of the binary search approach.
    """
    def search_recursive(left, right):
        if left > right:
            return -1
        
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Check which half is sorted
        if nums[left] <= nums[mid]:  # Left half is sorted
            if nums[left] <= target < nums[mid]:
                return search_recursive(left, mid - 1)
            else:
                return search_recursive(mid + 1, right)
        else:  # Right half is sorted
            if nums[mid] < target <= nums[right]:
                return search_recursive(mid + 1, right)
            else:
                return search_recursive(left, mid - 1)
    
    return search_recursive(0, len(nums) - 1)


def search_rotated_array_optimized(nums, target):
    """
    Approach 4: Optimized with Edge Case Handling
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Handle edge cases early and use optimized comparisons.
    """
    n = len(nums)
    if n == 0:
        return -1
    if n == 1:
        return 0 if nums[0] == target else -1
    
    left, right = 0, n - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Handle duplicate edge case (though problem states distinct values)
        if nums[left] == nums[mid] == nums[right]:
            left += 1
            right -= 1
            continue
        
        # Check if left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1


def search_rotated_array_alternative(nums, target):
    """
    Approach 5: Alternative Implementation
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Different way to handle the sorted half determination.
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if nums[mid] == target:
            return mid
        
        # If left part is sorted
        if nums[left] < nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # If right part is sorted
        elif nums[mid] < nums[right]:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
        # Special case: nums[left] == nums[mid]
        else:
            if nums[mid] == nums[left]:
                left += 1
            else:
                right -= 1
    
    return -1


def test_search_rotated_array():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([4, 5, 6, 7, 0, 1, 2], 0, 4),
        ([4, 5, 6, 7, 0, 1, 2], 3, -1),
        ([1], 0, -1),
        ([1], 1, 0),
        ([1, 3], 3, 1),
        ([3, 1], 1, 1),
        ([5, 1, 3], 3, 2),
        ([4, 5, 6, 7, 8, 1, 2, 3], 8, 4),
        ([6, 7, 1, 2, 3, 4, 5], 6, 0),
        ([2, 3, 4, 5, 6, 7, 1], 1, 6),
        ([1, 2, 3, 4, 5], 3, 2),  # No rotation
        ([2, 1], 1, 1),
    ]
    
    approaches = [
        ("One-Pass Binary Search", search_rotated_array_one_pass),
        ("Find Pivot First", search_rotated_array_find_pivot),
        ("Recursive Binary Search", search_rotated_array_recursive),
        ("Optimized Edge Cases", search_rotated_array_optimized),
        ("Alternative Implementation", search_rotated_array_alternative),
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
    test_search_rotated_array() 