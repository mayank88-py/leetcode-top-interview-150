"""
153. Find Minimum in Rotated Sorted Array

Suppose an array of length n sorted in ascending order is rotated between 1 and n times. 
For example, the array nums = [0,1,2,4,5,6,7] might become:
- [4,5,6,7,0,1,2] if it was rotated 4 times.
- [0,1,2,4,5,6,7] if it was rotated 7 times.

Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array 
[a[n-1], a[0], a[1], a[2], ..., a[n-2]].

Given the sorted rotated array nums of unique elements, return the minimum element of this array.

You must write an algorithm that runs in O(log n) time.

Example 1:
Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.

Example 2:
Input: nums = [4,5,6,7,0,1,2]
Output: 0
Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.

Example 3:
Input: nums = [11,13,15,17]
Output: 11
Explanation: The original array was [11,13,15,17] and it was rotated 4 times.

Constraints:
- n == nums.length
- 1 <= n <= 5000
- -5000 <= nums[i] <= 5000
- All the integers of nums are unique.
- nums is sorted and rotated between 1 and n times.
"""

def find_min_binary_search(nums):
    """
    Approach 1: Binary Search (Compare with Right)
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Compare middle element with rightmost element to determine which half contains minimum.
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        # If mid element is greater than right element,
        # minimum must be in the right half
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            # Minimum is in the left half or at mid
            right = mid
    
    return nums[left]


def find_min_compare_with_left(nums):
    """
    Approach 2: Binary Search (Compare with Left)
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Alternative approach comparing middle element with leftmost element.
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        # If array is sorted (no rotation in current range)
        if nums[left] <= nums[right]:
            return nums[left]
        
        # If mid element is greater than or equal to left element,
        # minimum must be in the right half
        if nums[mid] >= nums[left]:
            left = mid + 1
        else:
            # Minimum is in the left half including mid
            right = mid
    
    return nums[left]


def find_min_recursive(nums):
    """
    Approach 3: Recursive Binary Search
    Time Complexity: O(log n)
    Space Complexity: O(log n) - recursion depth
    
    Recursive implementation of binary search.
    """
    def binary_search_recursive(left, right):
        # Base case: only one element
        if left == right:
            return nums[left]
        
        # If only two elements
        if right - left == 1:
            return min(nums[left], nums[right])
        
        mid = left + (right - left) // 2
        
        # If the array is not rotated
        if nums[left] < nums[right]:
            return nums[left]
        
        # If mid is greater than right, minimum is in right half
        if nums[mid] > nums[right]:
            return binary_search_recursive(mid + 1, right)
        else:
            # Minimum is in left half or at mid
            return binary_search_recursive(left, mid)
    
    return binary_search_recursive(0, len(nums) - 1)


def find_min_find_pivot(nums):
    """
    Approach 4: Find Rotation Pivot
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Find the pivot point (where rotation occurred) and return the element at that position.
    """
    n = len(nums)
    
    # Edge case: array is not rotated
    if nums[0] <= nums[n - 1]:
        return nums[0]
    
    left, right = 0, n - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        # Check if mid is the pivot
        if mid > 0 and nums[mid] < nums[mid - 1]:
            return nums[mid]
        
        # Check if mid + 1 is the pivot
        if mid < n - 1 and nums[mid] > nums[mid + 1]:
            return nums[mid + 1]
        
        # Decide which half to search
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid - 1
    
    return nums[left]


def find_min_optimized(nums):
    """
    Approach 5: Optimized with Early Returns
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Optimized version with early return conditions.
    """
    n = len(nums)
    
    # Edge cases
    if n == 1:
        return nums[0]
    
    # Check if array is not rotated
    if nums[0] < nums[n - 1]:
        return nums[0]
    
    left, right = 0, n - 1
    
    while left <= right:
        # If we have narrowed down to a small range
        if right - left <= 1:
            return min(nums[left], nums[right])
        
        mid = left + (right - left) // 2
        
        # Check if we found the minimum
        if mid > 0 and nums[mid] < nums[mid - 1]:
            return nums[mid]
        
        if mid < n - 1 and nums[mid] > nums[mid + 1]:
            return nums[mid + 1]
        
        # Decide which half to explore
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid - 1
    
    return nums[left]


def find_min_linear(nums):
    """
    Approach 6: Linear Search (for comparison)
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Simple linear search - not optimal but useful for verification.
    """
    return min(nums)


def test_find_min():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([3, 4, 5, 1, 2], 1),
        ([4, 5, 6, 7, 0, 1, 2], 0),
        ([11, 13, 15, 17], 11),
        ([1], 1),
        ([2, 1], 1),
        ([1, 2], 1),
        ([3, 1, 2], 1),
        ([2, 3, 1], 1),
        ([5, 1, 2, 3, 4], 1),
        ([1, 2, 3, 4, 5], 1),  # No rotation
        ([2, 3, 4, 5, 1], 1),
        ([4, 5, 1, 2, 3], 1),
    ]
    
    approaches = [
        ("Binary Search (Right)", find_min_binary_search),
        ("Binary Search (Left)", find_min_compare_with_left),
        ("Recursive", find_min_recursive),
        ("Find Pivot", find_min_find_pivot),
        ("Optimized", find_min_optimized),
        ("Linear Search", find_min_linear),
    ]
    
    for i, (nums, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: {nums}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            result = func(nums)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_find_min() 