"""
300. Longest Increasing Subsequence

Given an integer array nums, return the length of the longest strictly increasing subsequence.

Example 1:
Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,18], therefore the length is 4.

Example 2:
Input: nums = [0,1,0,3,2,3]
Output: 4

Example 3:
Input: nums = [7,7,7,7,7,7,7]
Output: 1

Constraints:
- 1 <= nums.length <= 2500
- -10^4 <= nums[i] <= 10^4

Follow up: Can you come up with an algorithm that runs in O(n log n) time complexity?
"""

def length_of_lis_dp_n_squared(nums):
    """
    Approach 1: Dynamic Programming O(n^2)
    Time Complexity: O(n^2)
    Space Complexity: O(n)
    
    dp[i] = length of LIS ending at index i
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n  # Each element forms a subsequence of length 1
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)


def length_of_lis_binary_search(nums):
    """
    Approach 2: Binary Search + DP O(n log n)
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Maintain an array where tails[i] is the smallest tail of all increasing 
    subsequences of length i+1.
    """
    if not nums:
        return 0
    
    def binary_search(tails, target):
        """Find the leftmost position to insert target."""
        left, right = 0, len(tails) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if tails[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return left
    
    tails = []
    
    for num in nums:
        pos = binary_search(tails, num)
        
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    
    return len(tails)


def length_of_lis_patience_sorting(nums):
    """
    Approach 3: Patience Sorting Algorithm
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Use patience sorting concept - maintain piles of cards.
    """
    import bisect
    
    if not nums:
        return 0
    
    piles = []
    
    for num in nums:
        # Find the leftmost pile where we can place this number
        pos = bisect.bisect_left(piles, num)
        
        if pos == len(piles):
            piles.append(num)
        else:
            piles[pos] = num
    
    return len(piles)


def length_of_lis_with_path(nums):
    """
    Approach 4: DP with Path Reconstruction
    Time Complexity: O(n^2)
    Space Complexity: O(n)
    
    Track the actual LIS, not just its length.
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n
    parent = [-1] * n  # To reconstruct the path
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    
    # Find the index with maximum LIS length
    max_length = max(dp)
    max_index = dp.index(max_length)
    
    # Reconstruct the LIS (optional)
    lis = []
    current = max_index
    while current != -1:
        lis.append(nums[current])
        current = parent[current]
    lis.reverse()
    
    return max_length


def length_of_lis_memoization(nums):
    """
    Approach 5: Memoization (Top-down DP)
    Time Complexity: O(n^2)
    Space Complexity: O(n^2)
    
    Use memoization to avoid recalculating subproblems.
    """
    if not nums:
        return 0
    
    memo = {}
    
    def lis_from_index(index, prev_index):
        if index == len(nums):
            return 0
        
        if (index, prev_index) in memo:
            return memo[(index, prev_index)]
        
        # Option 1: Skip current element
        result = lis_from_index(index + 1, prev_index)
        
        # Option 2: Include current element if it's valid
        if prev_index == -1 or nums[index] > nums[prev_index]:
            result = max(result, 1 + lis_from_index(index + 1, index))
        
        memo[(index, prev_index)] = result
        return result
    
    return lis_from_index(0, -1)


def length_of_lis_segment_tree(nums):
    """
    Approach 6: Coordinate Compression + Segment Tree
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Use coordinate compression and segment tree for range maximum queries.
    """
    if not nums:
        return 0
    
    # Coordinate compression
    sorted_nums = sorted(set(nums))
    coord_map = {num: i for i, num in enumerate(sorted_nums)}
    
    # Simple array instead of segment tree for this problem size
    max_length = [0] * len(sorted_nums)
    
    result = 0
    for num in nums:
        coord = coord_map[num]
        
        # Find maximum length for all coordinates < current coordinate
        current_max = 0
        for i in range(coord):
            current_max = max(current_max, max_length[i])
        
        max_length[coord] = max(max_length[coord], current_max + 1)
        result = max(result, max_length[coord])
    
    return result


def length_of_lis_fenwick_tree(nums):
    """
    Approach 7: Coordinate Compression + Fenwick Tree
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Use Fenwick Tree (Binary Indexed Tree) for efficient range queries.
    """
    if not nums:
        return 0
    
    # Coordinate compression
    sorted_nums = sorted(set(nums))
    coord_map = {num: i + 1 for i, num in enumerate(sorted_nums)}  # 1-indexed
    
    class FenwickTree:
        def __init__(self, size):
            self.size = size
            self.tree = [0] * (size + 1)
        
        def update(self, index, value):
            while index <= self.size:
                self.tree[index] = max(self.tree[index], value)
                index += index & (-index)
        
        def query(self, index):
            result = 0
            while index > 0:
                result = max(result, self.tree[index])
                index -= index & (-index)
            return result
    
    ft = FenwickTree(len(sorted_nums))
    result = 0
    
    for num in nums:
        coord = coord_map[num]
        current_max = ft.query(coord - 1)  # Query for all smaller numbers
        new_length = current_max + 1
        ft.update(coord, new_length)
        result = max(result, new_length)
    
    return result


def length_of_lis_greedy_construction(nums):
    """
    Approach 8: Greedy Construction
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Greedy approach: always try to extend the longest subsequence.
    """
    if not nums:
        return 0
    
    from bisect import bisect_left
    
    subsequence = []
    
    for num in nums:
        # Find position to insert/replace
        pos = bisect_left(subsequence, num)
        
        if pos == len(subsequence):
            subsequence.append(num)
        else:
            subsequence[pos] = num
    
    return len(subsequence)


def length_of_lis_optimized_space(nums):
    """
    Approach 9: Space-Optimized DP
    Time Complexity: O(n^2)
    Space Complexity: O(1) extra space
    
    Optimize space by using the input array for DP storage (if modification allowed).
    """
    if not nums:
        return 0
    
    # Note: This modifies the input array
    # In practice, you'd create a copy if needed
    n = len(nums)
    original_nums = nums.copy()  # Keep original for reference
    
    # Use nums array to store DP values (destructive)
    for i in range(n):
        nums[i] = 1  # Initialize DP values
    
    for i in range(1, n):
        for j in range(i):
            if original_nums[j] < original_nums[i]:
                nums[i] = max(nums[i], nums[j] + 1)
    
    result = max(nums)
    
    # Restore original array
    nums[:] = original_nums
    
    return result


def test_length_of_lis():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([10, 9, 2, 5, 3, 7, 101, 18], 4),
        ([0, 1, 0, 3, 2, 3], 4),
        ([7, 7, 7, 7, 7, 7, 7], 1),
        ([1, 3, 6, 7, 9, 4, 10, 5, 6], 6),
        ([1], 1),
        ([1, 2], 2),
        ([2, 1], 1),
        ([1, 2, 3, 4, 5], 5),
        ([5, 4, 3, 2, 1], 1),
        ([10, 22, 9, 33, 21, 50, 41, 60], 5),
        ([3, 10, 2, 1, 20], 3),
        ([50, 3, 10, 7, 40, 80], 4),
        ([], 0),
        ([4, 10, 4, 3, 8, 9], 3),
        ([2, 15, 3, 7, 8, 6, 18], 5),
    ]
    
    approaches = [
        ("DP O(n^2)", length_of_lis_dp_n_squared),
        ("Binary Search O(n log n)", length_of_lis_binary_search),
        ("Patience Sorting", length_of_lis_patience_sorting),
        ("DP with Path", length_of_lis_with_path),
        ("Memoization", length_of_lis_memoization),
        ("Segment Tree", length_of_lis_segment_tree),
        ("Fenwick Tree", length_of_lis_fenwick_tree),
        ("Greedy Construction", length_of_lis_greedy_construction),
        ("Optimized Space", length_of_lis_optimized_space),
    ]
    
    for i, (nums, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: {nums}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            # Create copy to avoid modifying original
            nums_copy = nums.copy()
            result = func(nums_copy)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_length_of_lis() 