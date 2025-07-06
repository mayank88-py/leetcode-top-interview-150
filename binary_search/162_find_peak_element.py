"""
162. Find Peak Element

A peak element is an element that is strictly greater than its neighbors.

Given a 0-indexed integer array nums, find a peak element, and return its index. 
If the array contains multiple peaks, return the index to any of the peaks.

You may imagine that nums[-1] = nums[n] = -∞. In other words, an element is always 
considered to be strictly greater than a neighbor that is outside the array.

You must write an algorithm that runs in O(log n) time.

Example 1:
Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.

Example 2:
Input: nums = [1,2,1,3,5,6,4]
Output: 5
Explanation: Your function can return either index number 1 where the peak element is 2, 
or index number 5 where the peak element is 6.

Constraints:
- 1 <= nums.length <= 1000
- -2^31 <= nums[i] <= 2^31 - 1
- nums[i] != nums[i + 1] for all valid i.
"""

def find_peak_element_binary_search(nums):
    """
    Approach 1: Binary Search
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Use binary search. If mid element is smaller than its right neighbor,
    there must be a peak on the right side. Otherwise, peak is on left side or mid itself.
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        # Compare with right neighbor
        if nums[mid] < nums[mid + 1]:
            # Peak must be on the right side
            left = mid + 1
        else:
            # Peak is on the left side or at mid
            right = mid
    
    return left


def find_peak_element_linear(nums):
    """
    Approach 2: Linear Search
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Simple linear search to find any peak element.
    """
    n = len(nums)
    
    for i in range(n):
        left_smaller = (i == 0) or (nums[i] > nums[i - 1])
        right_smaller = (i == n - 1) or (nums[i] > nums[i + 1])
        
        if left_smaller and right_smaller:
            return i
    
    return -1  # Should never reach here given constraints


def find_peak_element_recursive(nums):
    """
    Approach 3: Recursive Binary Search
    Time Complexity: O(log n)
    Space Complexity: O(log n) - due to recursion
    
    Recursive implementation of binary search approach.
    """
    def binary_search_recursive(left, right):
        if left == right:
            return left
        
        mid = left + (right - left) // 2
        
        if nums[mid] < nums[mid + 1]:
            return binary_search_recursive(mid + 1, right)
        else:
            return binary_search_recursive(left, mid)
    
    return binary_search_recursive(0, len(nums) - 1)


def find_peak_element_optimized(nums):
    """
    Approach 4: Optimized Edge Case Handling
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Handle edge cases first, then use binary search.
    """
    n = len(nums)
    
    # Handle single element
    if n == 1:
        return 0
    
    # Check if first or last element is peak
    if nums[0] > nums[1]:
        return 0
    if nums[n - 1] > nums[n - 2]:
        return n - 1
    
    # Binary search in the middle elements
    left, right = 1, n - 2
    
    while left <= right:
        mid = left + (right - left) // 2
        
        # Check if mid is a peak
        if nums[mid] > nums[mid - 1] and nums[mid] > nums[mid + 1]:
            return mid
        
        # If mid is smaller than right neighbor, go right
        if nums[mid] < nums[mid + 1]:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Should never reach here


def find_peak_element_template(nums):
    """
    Approach 5: Binary Search Template
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Using a different binary search template that's easier to understand.
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        # Check if mid is a peak
        left_ok = (mid == 0) or (nums[mid] > nums[mid - 1])
        right_ok = (mid == len(nums) - 1) or (nums[mid] > nums[mid + 1])
        
        if left_ok and right_ok:
            return mid
        elif not left_ok:
            # Peak is on the left side
            right = mid - 1
        else:
            # Peak is on the right side
            left = mid + 1
    
    return -1  # Should never reach here


def test_find_peak_element():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([1, 2, 3, 1], [1, 2]),         # Multiple valid peaks
        ([1, 2, 1, 3, 5, 6, 4], [1, 5]), # Multiple valid peaks
        ([1], [0]),                      # Single element
        ([1, 2], [1]),                   # Two elements, ascending
        ([2, 1], [0]),                   # Two elements, descending
        ([1, 3, 2], [1]),               # Three elements, peak in middle
        ([1, 2, 3, 4, 5], [4]),         # Strictly increasing
        ([5, 4, 3, 2, 1], [0]),         # Strictly decreasing
        ([1, 3, 2, 4, 1], [1, 3]),     # Multiple peaks
        ([6, 5, 4, 3, 2, 3, 2], [0, 5]), # Multiple peaks
    ]
    
    approaches = [
        ("Binary Search", find_peak_element_binary_search),
        ("Linear Search", find_peak_element_linear),
        ("Recursive Binary Search", find_peak_element_recursive),
        ("Optimized Edge Cases", find_peak_element_optimized),
        ("Binary Search Template", find_peak_element_template),
    ]
    
    for i, (nums, expected_peaks) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: {nums}")
        print(f"Expected peaks at indices: {expected_peaks}")
        
        for name, func in approaches:
            result = func(nums)
            # Check if result is a valid peak
            is_valid = result in expected_peaks
            status = "✓" if is_valid else "✗"
            peak_value = nums[result] if 0 <= result < len(nums) else "N/A"
            print(f"{status} {name}: index {result} (value: {peak_value})")


if __name__ == "__main__":
    test_find_peak_element() 