"""
228. Summary Ranges

Problem:
You are given a sorted unique integer array nums.

A range [a,b] is the set of all integers from a to b (inclusive).

Return the smallest sorted list of ranges that cover all the numbers in the array exactly. That is, each element of nums is covered by exactly one of the ranges, and there is no integer x such that x is in one of the ranges but not in nums.

Each range [a,b] in the list should be output as:
- "a->b" if a != b
- "a" if a == b

Example 1:
Input: nums = [0,1,2,4,5,7]
Output: ["0->2","4->5","7"]
Explanation: The ranges are:
[0,2] --> "0->2"
[4,5] --> "4->5"
[7,7] --> "7"

Example 2:
Input: nums = [0,2,3,4,6,8,9]
Output: ["0","2->4","6","8->9"]
Explanation: The ranges are:
[0,0] --> "0"
[2,4] --> "2->4"
[6,6] --> "6"
[8,9] --> "8->9"

Time Complexity: O(n) where n is the length of nums
Space Complexity: O(1) excluding the output array
"""


def summary_ranges(nums):
    """
    Find summary ranges using two pointers approach.
    
    Args:
        nums: Sorted unique integer array
    
    Returns:
        List of range strings
    """
    if not nums:
        return []
    
    result = []
    start = 0
    
    for end in range(len(nums)):
        # Check if current number breaks the sequence
        if end == len(nums) - 1 or nums[end] + 1 != nums[end + 1]:
            if start == end:
                # Single number range
                result.append(str(nums[start]))
            else:
                # Multi-number range
                result.append(f"{nums[start]}->{nums[end]}")
            start = end + 1
    
    return result


def summary_ranges_iterative(nums):
    """
    Find summary ranges using iterative approach.
    
    Args:
        nums: Sorted unique integer array
    
    Returns:
        List of range strings
    """
    if not nums:
        return []
    
    result = []
    i = 0
    
    while i < len(nums):
        start = i
        
        # Find the end of current range
        while i + 1 < len(nums) and nums[i] + 1 == nums[i + 1]:
            i += 1
        
        # Add range to result
        if start == i:
            result.append(str(nums[start]))
        else:
            result.append(f"{nums[start]}->{nums[i]}")
        
        i += 1
    
    return result


def summary_ranges_stack(nums):
    """
    Find summary ranges using stack approach.
    
    Args:
        nums: Sorted unique integer array
    
    Returns:
        List of range strings
    """
    if not nums:
        return []
    
    stack = []
    result = []
    
    for num in nums:
        if not stack:
            stack.append(num)
        elif stack[-1] + 1 == num:
            # Continue current range
            pass
        else:
            # End current range and start new one
            if len(stack) == 1:
                result.append(str(stack[0]))
            else:
                result.append(f"{stack[0]}->{stack[-1]}")
            stack = [num]
        
        stack.append(num)
    
    # Handle last range
    if stack:
        unique_stack = list(dict.fromkeys(stack))  # Remove duplicates while preserving order
        if len(unique_stack) == 1:
            result.append(str(unique_stack[0]))
        else:
            result.append(f"{unique_stack[0]}->{unique_stack[-1]}")
    
    return result


def summary_ranges_simple_stack(nums):
    """
    Find summary ranges using simplified stack approach.
    
    Args:
        nums: Sorted unique integer array
    
    Returns:
        List of range strings
    """
    if not nums:
        return []
    
    result = []
    start = nums[0]
    end = nums[0]
    
    for i in range(1, len(nums)):
        if nums[i] == end + 1:
            # Extend current range
            end = nums[i]
        else:
            # Complete current range and start new one
            if start == end:
                result.append(str(start))
            else:
                result.append(f"{start}->{end}")
            start = end = nums[i]
    
    # Add the last range
    if start == end:
        result.append(str(start))
    else:
        result.append(f"{start}->{end}")
    
    return result


def summary_ranges_functional(nums):
    """
    Find summary ranges using functional programming approach.
    
    Args:
        nums: Sorted unique integer array
    
    Returns:
        List of range strings
    """
    if not nums:
        return []
    
    def group_consecutive():
        """Group consecutive numbers"""
        groups = []
        current_group = [nums[0]]
        
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1:
                current_group.append(nums[i])
            else:
                groups.append(current_group)
                current_group = [nums[i]]
        
        groups.append(current_group)
        return groups
    
    def format_group(group):
        """Format a group into range string"""
        if len(group) == 1:
            return str(group[0])
        else:
            return f"{group[0]}->{group[-1]}"
    
    groups = group_consecutive()
    return [format_group(group) for group in groups]


def summary_ranges_recursive(nums):
    """
    Find summary ranges using recursive approach.
    
    Args:
        nums: Sorted unique integer array
    
    Returns:
        List of range strings
    """
    if not nums:
        return []
    
    def find_ranges(start_idx):
        """Recursively find ranges starting from given index"""
        if start_idx >= len(nums):
            return []
        
        # Find end of current range
        end_idx = start_idx
        while end_idx + 1 < len(nums) and nums[end_idx] + 1 == nums[end_idx + 1]:
            end_idx += 1
        
        # Format current range
        if start_idx == end_idx:
            current_range = str(nums[start_idx])
        else:
            current_range = f"{nums[start_idx]}->{nums[end_idx]}"
        
        # Recursively find remaining ranges
        remaining_ranges = find_ranges(end_idx + 1)
        
        return [current_range] + remaining_ranges
    
    return find_ranges(0)


def summary_ranges_enumerate(nums):
    """
    Find summary ranges using enumerate approach.
    
    Args:
        nums: Sorted unique integer array
    
    Returns:
        List of range strings
    """
    if not nums:
        return []
    
    result = []
    start_idx = 0
    
    for i, num in enumerate(nums):
        # Check if this is the end of a range
        if i == len(nums) - 1 or nums[i + 1] != num + 1:
            if start_idx == i:
                result.append(str(nums[start_idx]))
            else:
                result.append(f"{nums[start_idx]}->{nums[i]}")
            start_idx = i + 1
    
    return result


def summary_ranges_sliding_window(nums):
    """
    Find summary ranges using sliding window approach.
    
    Args:
        nums: Sorted unique integer array
    
    Returns:
        List of range strings
    """
    if not nums:
        return []
    
    result = []
    left = 0
    
    for right in range(len(nums)):
        # If next number breaks the sequence or we're at the end
        if right == len(nums) - 1 or nums[right + 1] != nums[right] + 1:
            if left == right:
                result.append(str(nums[left]))
            else:
                result.append(f"{nums[left]}->{nums[right]}")
            left = right + 1
    
    return result


# Test cases
if __name__ == "__main__":
    # Test case 1
    nums1 = [0,1,2,4,5,7]
    result1a = summary_ranges(nums1)
    result1b = summary_ranges_iterative(nums1)
    result1c = summary_ranges_stack(nums1)
    result1d = summary_ranges_simple_stack(nums1)
    result1e = summary_ranges_functional(nums1)
    result1f = summary_ranges_recursive(nums1)
    result1g = summary_ranges_enumerate(nums1)
    result1h = summary_ranges_sliding_window(nums1)
    print(f"Test 1 - Input: {nums1}, Expected: ['0->2','4->5','7']")
    print(f"TwoPointers: {result1a}, Iterative: {result1b}, Stack: {result1c}, SimpleStack: {result1d}, Functional: {result1e}, Recursive: {result1f}, Enumerate: {result1g}, SlidingWindow: {result1h}")
    print()
    
    # Test case 2
    nums2 = [0,2,3,4,6,8,9]
    result2a = summary_ranges(nums2)
    result2b = summary_ranges_iterative(nums2)
    result2c = summary_ranges_stack(nums2)
    result2d = summary_ranges_simple_stack(nums2)
    result2e = summary_ranges_functional(nums2)
    result2f = summary_ranges_recursive(nums2)
    result2g = summary_ranges_enumerate(nums2)
    result2h = summary_ranges_sliding_window(nums2)
    print(f"Test 2 - Input: {nums2}, Expected: ['0','2->4','6','8->9']")
    print(f"TwoPointers: {result2a}, Iterative: {result2b}, Stack: {result2c}, SimpleStack: {result2d}, Functional: {result2e}, Recursive: {result2f}, Enumerate: {result2g}, SlidingWindow: {result2h}")
    print()
    
    # Test case 3 - Empty array
    nums3 = []
    result3a = summary_ranges(nums3)
    result3b = summary_ranges_iterative(nums3)
    result3c = summary_ranges_stack(nums3)
    result3d = summary_ranges_simple_stack(nums3)
    result3e = summary_ranges_functional(nums3)
    result3f = summary_ranges_recursive(nums3)
    result3g = summary_ranges_enumerate(nums3)
    result3h = summary_ranges_sliding_window(nums3)
    print(f"Test 3 - Input: {nums3}, Expected: []")
    print(f"TwoPointers: {result3a}, Iterative: {result3b}, Stack: {result3c}, SimpleStack: {result3d}, Functional: {result3e}, Recursive: {result3f}, Enumerate: {result3g}, SlidingWindow: {result3h}")
    print()
    
    # Test case 4 - Single element
    nums4 = [1]
    result4a = summary_ranges(nums4)
    result4b = summary_ranges_iterative(nums4)
    result4c = summary_ranges_stack(nums4)
    result4d = summary_ranges_simple_stack(nums4)
    result4e = summary_ranges_functional(nums4)
    result4f = summary_ranges_recursive(nums4)
    result4g = summary_ranges_enumerate(nums4)
    result4h = summary_ranges_sliding_window(nums4)
    print(f"Test 4 - Input: {nums4}, Expected: ['1']")
    print(f"TwoPointers: {result4a}, Iterative: {result4b}, Stack: {result4c}, SimpleStack: {result4d}, Functional: {result4e}, Recursive: {result4f}, Enumerate: {result4g}, SlidingWindow: {result4h}")
    print()
    
    # Test case 5 - All consecutive
    nums5 = [1,2,3,4,5]
    result5a = summary_ranges(nums5)
    result5b = summary_ranges_iterative(nums5)
    result5c = summary_ranges_stack(nums5)
    result5d = summary_ranges_simple_stack(nums5)
    result5e = summary_ranges_functional(nums5)
    result5f = summary_ranges_recursive(nums5)
    result5g = summary_ranges_enumerate(nums5)
    result5h = summary_ranges_sliding_window(nums5)
    print(f"Test 5 - Input: {nums5}, Expected: ['1->5']")
    print(f"TwoPointers: {result5a}, Iterative: {result5b}, Stack: {result5c}, SimpleStack: {result5d}, Functional: {result5e}, Recursive: {result5f}, Enumerate: {result5g}, SlidingWindow: {result5h}")
    print()
    
    # Test case 6 - All separate
    nums6 = [1,3,5,7,9]
    result6a = summary_ranges(nums6)
    result6b = summary_ranges_iterative(nums6)
    result6c = summary_ranges_stack(nums6)
    result6d = summary_ranges_simple_stack(nums6)
    result6e = summary_ranges_functional(nums6)
    result6f = summary_ranges_recursive(nums6)
    result6g = summary_ranges_enumerate(nums6)
    result6h = summary_ranges_sliding_window(nums6)
    print(f"Test 6 - Input: {nums6}, Expected: ['1','3','5','7','9']")
    print(f"TwoPointers: {result6a}, Iterative: {result6b}, Stack: {result6c}, SimpleStack: {result6d}, Functional: {result6e}, Recursive: {result6f}, Enumerate: {result6g}, SlidingWindow: {result6h}")
    print()
    
    # Test case 7 - Negative numbers
    nums7 = [-3,-2,-1,1,2,4]
    result7a = summary_ranges(nums7)
    result7b = summary_ranges_iterative(nums7)
    result7c = summary_ranges_stack(nums7)
    result7d = summary_ranges_simple_stack(nums7)
    result7e = summary_ranges_functional(nums7)
    result7f = summary_ranges_recursive(nums7)
    result7g = summary_ranges_enumerate(nums7)
    result7h = summary_ranges_sliding_window(nums7)
    print(f"Test 7 - Input: {nums7}, Expected: ['-3->-1','1->2','4']")
    print(f"TwoPointers: {result7a}, Iterative: {result7b}, Stack: {result7c}, SimpleStack: {result7d}, Functional: {result7e}, Recursive: {result7f}, Enumerate: {result7g}, SlidingWindow: {result7h}")
    print()
    
    # Test case 8 - Two elements consecutive
    nums8 = [1,2]
    result8a = summary_ranges(nums8)
    result8b = summary_ranges_iterative(nums8)
    result8c = summary_ranges_stack(nums8)
    result8d = summary_ranges_simple_stack(nums8)
    result8e = summary_ranges_functional(nums8)
    result8f = summary_ranges_recursive(nums8)
    result8g = summary_ranges_enumerate(nums8)
    result8h = summary_ranges_sliding_window(nums8)
    print(f"Test 8 - Input: {nums8}, Expected: ['1->2']")
    print(f"TwoPointers: {result8a}, Iterative: {result8b}, Stack: {result8c}, SimpleStack: {result8d}, Functional: {result8e}, Recursive: {result8f}, Enumerate: {result8g}, SlidingWindow: {result8h}")
    print()
    
    # Test case 9 - Two elements separate
    nums9 = [1,3]
    result9a = summary_ranges(nums9)
    result9b = summary_ranges_iterative(nums9)
    result9c = summary_ranges_stack(nums9)
    result9d = summary_ranges_simple_stack(nums9)
    result9e = summary_ranges_functional(nums9)
    result9f = summary_ranges_recursive(nums9)
    result9g = summary_ranges_enumerate(nums9)
    result9h = summary_ranges_sliding_window(nums9)
    print(f"Test 9 - Input: {nums9}, Expected: ['1','3']")
    print(f"TwoPointers: {result9a}, Iterative: {result9b}, Stack: {result9c}, SimpleStack: {result9d}, Functional: {result9e}, Recursive: {result9f}, Enumerate: {result9g}, SlidingWindow: {result9h}")
    print()
    
    # Test case 10 - Large numbers
    nums10 = [2147483646,2147483647]
    result10a = summary_ranges(nums10)
    result10b = summary_ranges_iterative(nums10)
    result10c = summary_ranges_stack(nums10)
    result10d = summary_ranges_simple_stack(nums10)
    result10e = summary_ranges_functional(nums10)
    result10f = summary_ranges_recursive(nums10)
    result10g = summary_ranges_enumerate(nums10)
    result10h = summary_ranges_sliding_window(nums10)
    print(f"Test 10 - Input: {nums10}, Expected: ['2147483646->2147483647']")
    print(f"TwoPointers: {result10a}, Iterative: {result10b}, Stack: {result10c}, SimpleStack: {result10d}, Functional: {result10e}, Recursive: {result10f}, Enumerate: {result10g}, SlidingWindow: {result10h}") 