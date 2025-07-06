"""
128. Longest Consecutive Sequence

Problem:
Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

You must write an algorithm that runs in O(n) time.

Example 1:
Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.

Example 2:
Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9

Time Complexity: O(n) for optimal solution
Space Complexity: O(n) for the hash set
"""


def longest_consecutive(nums):
    """
    Find longest consecutive sequence using hash set.
    
    Args:
        nums: List of integers
    
    Returns:
        Length of longest consecutive sequence
    """
    if not nums:
        return 0
    
    num_set = set(nums)
    longest_length = 0
    
    for num in num_set:
        # Only start counting if this is the beginning of a sequence
        if num - 1 not in num_set:
            current_num = num
            current_length = 1
            
            # Count consecutive numbers
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1
            
            longest_length = max(longest_length, current_length)
    
    return longest_length


def longest_consecutive_sorting(nums):
    """
    Find longest consecutive sequence by sorting first.
    
    Args:
        nums: List of integers
    
    Returns:
        Length of longest consecutive sequence
    """
    if not nums:
        return 0
    
    # Sort and remove duplicates
    sorted_nums = sorted(set(nums))
    
    longest_length = 1
    current_length = 1
    
    for i in range(1, len(sorted_nums)):
        if sorted_nums[i] == sorted_nums[i-1] + 1:
            current_length += 1
        else:
            longest_length = max(longest_length, current_length)
            current_length = 1
    
    return max(longest_length, current_length)


def longest_consecutive_union_find(nums):
    """
    Find longest consecutive sequence using Union-Find.
    
    Args:
        nums: List of integers
    
    Returns:
        Length of longest consecutive sequence
    """
    if not nums:
        return 0
    
    # Simple Union-Find implementation
    parent = {}
    size = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
            size[x] = 1
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            if size[px] < size[py]:
                px, py = py, px
            parent[py] = px
            size[px] += size[py]
    
    # Initialize all numbers
    for num in set(nums):
        find(num)
    
    # Union consecutive numbers
    for num in set(nums):
        if num + 1 in parent:
            union(num, num + 1)
    
    # Find maximum component size
    return max(size[find(num)] for num in set(nums))


def longest_consecutive_dynamic_programming(nums):
    """
    Find longest consecutive sequence using dynamic programming approach.
    
    Args:
        nums: List of integers
    
    Returns:
        Length of longest consecutive sequence
    """
    if not nums:
        return 0
    
    # dp[num] = length of consecutive sequence ending at num
    dp = {}
    max_length = 0
    
    for num in nums:
        if num in dp:
            continue
        
        # Check if we can extend existing sequences
        left_length = dp.get(num - 1, 0)
        right_length = dp.get(num + 1, 0)
        
        # Current sequence length
        current_length = left_length + right_length + 1
        
        # Update boundaries of the sequence
        dp[num] = current_length
        dp[num - left_length] = current_length
        dp[num + right_length] = current_length
        
        max_length = max(max_length, current_length)
    
    return max_length


def longest_consecutive_recursive(nums):
    """
    Find longest consecutive sequence using recursive approach with memoization.
    
    Args:
        nums: List of integers
    
    Returns:
        Length of longest consecutive sequence
    """
    if not nums:
        return 0
    
    num_set = set(nums)
    memo = {}
    
    def dfs(num):
        if num in memo:
            return memo[num]
        
        if num not in num_set:
            return 0
        
        # Length starting from current number
        memo[num] = 1 + dfs(num + 1)
        return memo[num]
    
    max_length = 0
    for num in num_set:
        # Only start from numbers that are beginning of sequences
        if num - 1 not in num_set:
            max_length = max(max_length, dfs(num))
    
    return max_length


def longest_consecutive_range_tracking(nums):
    """
    Find longest consecutive sequence by tracking ranges.
    
    Args:
        nums: List of integers
    
    Returns:
        Length of longest consecutive sequence
    """
    if not nums:
        return 0
    
    # Track ranges as [start, end]
    ranges = {}  # start -> end
    max_length = 0
    
    for num in set(nums):
        if num in ranges:
            continue
        
        # Find if this number extends existing ranges
        start = end = num
        
        # Check if we can extend to the left
        if num - 1 in ranges:
            # Find the start of the range ending at num-1
            for s, e in ranges.items():
                if e == num - 1:
                    start = s
                    del ranges[s]
                    break
        
        # Check if we can extend to the right  
        if num + 1 in ranges:
            end = ranges[num + 1]
            del ranges[num + 1]
        
        # Add the new range
        ranges[start] = end
        max_length = max(max_length, end - start + 1)
    
    return max_length


# Test cases
if __name__ == "__main__":
    # Test case 1
    nums1 = [100,4,200,1,3,2]
    result1a = longest_consecutive(nums1)
    result1b = longest_consecutive_sorting(nums1)
    result1c = longest_consecutive_union_find(nums1)
    result1d = longest_consecutive_dynamic_programming(nums1)
    result1e = longest_consecutive_recursive(nums1)
    result1f = longest_consecutive_range_tracking(nums1)
    print(f"Test 1 - Input: {nums1}, Expected: 4")
    print(f"HashSet: {result1a}, Sorting: {result1b}, UnionFind: {result1c}, DP: {result1d}, Recursive: {result1e}, Range: {result1f}")
    print()
    
    # Test case 2
    nums2 = [0,3,7,2,5,8,4,6,0,1]
    result2a = longest_consecutive(nums2)
    result2b = longest_consecutive_sorting(nums2)
    result2c = longest_consecutive_union_find(nums2)
    result2d = longest_consecutive_dynamic_programming(nums2)
    result2e = longest_consecutive_recursive(nums2)
    result2f = longest_consecutive_range_tracking(nums2)
    print(f"Test 2 - Input: {nums2}, Expected: 9")
    print(f"HashSet: {result2a}, Sorting: {result2b}, UnionFind: {result2c}, DP: {result2d}, Recursive: {result2e}, Range: {result2f}")
    print()
    
    # Test case 3 - Empty array
    nums3 = []
    result3a = longest_consecutive(nums3)
    result3b = longest_consecutive_sorting(nums3)
    result3c = longest_consecutive_union_find(nums3)
    result3d = longest_consecutive_dynamic_programming(nums3)
    result3e = longest_consecutive_recursive(nums3)
    result3f = longest_consecutive_range_tracking(nums3)
    print(f"Test 3 - Input: {nums3}, Expected: 0")
    print(f"HashSet: {result3a}, Sorting: {result3b}, UnionFind: {result3c}, DP: {result3d}, Recursive: {result3e}, Range: {result3f}")
    print()
    
    # Test case 4 - Single element
    nums4 = [1]
    result4a = longest_consecutive(nums4)
    result4b = longest_consecutive_sorting(nums4)
    result4c = longest_consecutive_union_find(nums4)
    result4d = longest_consecutive_dynamic_programming(nums4)
    result4e = longest_consecutive_recursive(nums4)
    result4f = longest_consecutive_range_tracking(nums4)
    print(f"Test 4 - Input: {nums4}, Expected: 1")
    print(f"HashSet: {result4a}, Sorting: {result4b}, UnionFind: {result4c}, DP: {result4d}, Recursive: {result4e}, Range: {result4f}")
    print()
    
    # Test case 5 - All same numbers
    nums5 = [1,1,1,1]
    result5a = longest_consecutive(nums5)
    result5b = longest_consecutive_sorting(nums5)
    result5c = longest_consecutive_union_find(nums5)
    result5d = longest_consecutive_dynamic_programming(nums5)
    result5e = longest_consecutive_recursive(nums5)
    result5f = longest_consecutive_range_tracking(nums5)
    print(f"Test 5 - Input: {nums5}, Expected: 1")
    print(f"HashSet: {result5a}, Sorting: {result5b}, UnionFind: {result5c}, DP: {result5d}, Recursive: {result5e}, Range: {result5f}")
    print()
    
    # Test case 6 - No consecutive numbers
    nums6 = [1,3,5,7,9]
    result6a = longest_consecutive(nums6)
    result6b = longest_consecutive_sorting(nums6)
    result6c = longest_consecutive_union_find(nums6)
    result6d = longest_consecutive_dynamic_programming(nums6)
    result6e = longest_consecutive_recursive(nums6)
    result6f = longest_consecutive_range_tracking(nums6)
    print(f"Test 6 - Input: {nums6}, Expected: 1")
    print(f"HashSet: {result6a}, Sorting: {result6b}, UnionFind: {result6c}, DP: {result6d}, Recursive: {result6e}, Range: {result6f}")
    print()
    
    # Test case 7 - Negative numbers
    nums7 = [-1,0,1,2]
    result7a = longest_consecutive(nums7)
    result7b = longest_consecutive_sorting(nums7)
    result7c = longest_consecutive_union_find(nums7)
    result7d = longest_consecutive_dynamic_programming(nums7)
    result7e = longest_consecutive_recursive(nums7)
    result7f = longest_consecutive_range_tracking(nums7)
    print(f"Test 7 - Input: {nums7}, Expected: 4")
    print(f"HashSet: {result7a}, Sorting: {result7b}, UnionFind: {result7c}, DP: {result7d}, Recursive: {result7e}, Range: {result7f}")
    print()
    
    # Test case 8 - Large gaps
    nums8 = [1,2,3,100,101,102,103,104]
    result8a = longest_consecutive(nums8)
    result8b = longest_consecutive_sorting(nums8)
    result8c = longest_consecutive_union_find(nums8)
    result8d = longest_consecutive_dynamic_programming(nums8)
    result8e = longest_consecutive_recursive(nums8)
    result8f = longest_consecutive_range_tracking(nums8)
    print(f"Test 8 - Input: {nums8}, Expected: 5")
    print(f"HashSet: {result8a}, Sorting: {result8b}, UnionFind: {result8c}, DP: {result8d}, Recursive: {result8e}, Range: {result8f}")
    print()
    
    # Test case 9 - Duplicates with gaps
    nums9 = [1,2,2,3,4,4,5]
    result9a = longest_consecutive(nums9)
    result9b = longest_consecutive_sorting(nums9)
    result9c = longest_consecutive_union_find(nums9)
    result9d = longest_consecutive_dynamic_programming(nums9)
    result9e = longest_consecutive_recursive(nums9)
    result9f = longest_consecutive_range_tracking(nums9)
    print(f"Test 9 - Input: {nums9}, Expected: 5")
    print(f"HashSet: {result9a}, Sorting: {result9b}, UnionFind: {result9c}, DP: {result9d}, Recursive: {result9e}, Range: {result9f}")
    print()
    
    # Test case 10 - Reverse order
    nums10 = [5,4,3,2,1]
    result10a = longest_consecutive(nums10)
    result10b = longest_consecutive_sorting(nums10)
    result10c = longest_consecutive_union_find(nums10)
    result10d = longest_consecutive_dynamic_programming(nums10)
    result10e = longest_consecutive_recursive(nums10)
    result10f = longest_consecutive_range_tracking(nums10)
    print(f"Test 10 - Input: {nums10}, Expected: 5")
    print(f"HashSet: {result10a}, Sorting: {result10b}, UnionFind: {result10c}, DP: {result10d}, Recursive: {result10e}, Range: {result10f}") 