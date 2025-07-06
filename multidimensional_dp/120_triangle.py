"""
120. Triangle

Given a triangle array, return the minimum path sum from top to bottom.

For each step, you may move to an adjacent number of the row below. More formally, if you are on index i on the current row, you may move to either index i or index i + 1 on the next row.

Example 1:
Input: triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
Output: 11
Explanation: The triangle looks like:
   2
  3 4
 6 5 7
4 1 8 3
The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).

Example 2:
Input: triangle = [[-10]]
Output: -10

Constraints:
- 1 <= triangle.length <= 200
- triangle[0].length == 1
- triangle[i].length == triangle[i - 1].length + 1
- -10^4 <= triangle[i][j] <= 10^4

Follow up: Could you do this using only O(n) extra space, where n is the total number of rows in the triangle?
"""

def minimum_total_dp_bottom_up(triangle):
    """
    Approach 1: Bottom-up DP
    Time Complexity: O(n^2) where n is number of rows
    Space Complexity: O(n^2)
    
    Build DP table from bottom to top.
    """
    if not triangle:
        return 0
    
    n = len(triangle)
    # Create DP table with same structure as triangle
    dp = [[0] * len(row) for row in triangle]
    
    # Initialize bottom row
    dp[n-1] = triangle[n-1][:]
    
    # Fill DP table from bottom to top
    for i in range(n-2, -1, -1):
        for j in range(len(triangle[i])):
            dp[i][j] = triangle[i][j] + min(dp[i+1][j], dp[i+1][j+1])
    
    return dp[0][0]


def minimum_total_dp_top_down(triangle):
    """
    Approach 2: Top-down DP with Memoization
    Time Complexity: O(n^2)
    Space Complexity: O(n^2)
    
    Use memoization to avoid recalculating subproblems.
    """
    if not triangle:
        return 0
    
    memo = {}
    
    def min_path(row, col):
        if row == len(triangle) - 1:
            return triangle[row][col]
        
        if (row, col) in memo:
            return memo[(row, col)]
        
        # Two choices: go to (row+1, col) or (row+1, col+1)
        left = min_path(row + 1, col)
        right = min_path(row + 1, col + 1)
        
        memo[(row, col)] = triangle[row][col] + min(left, right)
        return memo[(row, col)]
    
    return min_path(0, 0)


def minimum_total_space_optimized(triangle):
    """
    Approach 3: Space-Optimized DP O(n)
    Time Complexity: O(n^2)
    Space Complexity: O(n)
    
    Use only O(n) extra space.
    """
    if not triangle:
        return 0
    
    n = len(triangle)
    # Use only one array for DP
    dp = triangle[-1][:]  # Start with bottom row
    
    # Process from second last row to top
    for i in range(n-2, -1, -1):
        for j in range(len(triangle[i])):
            dp[j] = triangle[i][j] + min(dp[j], dp[j+1])
    
    return dp[0]


def minimum_total_in_place(triangle):
    """
    Approach 4: In-place Modification
    Time Complexity: O(n^2)
    Space Complexity: O(1) - only modifies input
    
    Modify the triangle in place to save space.
    """
    if not triangle:
        return 0
    
    n = len(triangle)
    
    # Process from second last row to top
    for i in range(n-2, -1, -1):
        for j in range(len(triangle[i])):
            triangle[i][j] += min(triangle[i+1][j], triangle[i+1][j+1])
    
    return triangle[0][0]


def minimum_total_recursive_naive(triangle):
    """
    Approach 5: Naive Recursion (Not efficient)
    Time Complexity: O(2^n)
    Space Complexity: O(n) - recursion depth
    
    Simple recursive solution without memoization.
    """
    if not triangle:
        return 0
    
    def min_path(row, col):
        if row == len(triangle) - 1:
            return triangle[row][col]
        
        # Two choices: go down or go down-right
        left = min_path(row + 1, col)
        right = min_path(row + 1, col + 1)
        
        return triangle[row][col] + min(left, right)
    
    # Avoid TLE for large triangles
    if len(triangle) > 10:
        return minimum_total_space_optimized(triangle)
    
    return min_path(0, 0)


def minimum_total_dp_rolling_array(triangle):
    """
    Approach 6: Rolling Array DP
    Time Complexity: O(n^2)
    Space Complexity: O(n)
    
    Use rolling array technique to optimize space.
    """
    if not triangle:
        return 0
    
    n = len(triangle)
    prev = triangle[-1][:]  # Previous row
    
    for i in range(n-2, -1, -1):
        curr = [0] * len(triangle[i])
        for j in range(len(triangle[i])):
            curr[j] = triangle[i][j] + min(prev[j], prev[j+1])
        prev = curr
    
    return prev[0]


def minimum_total_iterative_improvement(triangle):
    """
    Approach 7: Iterative Improvement
    Time Complexity: O(n^2)
    Space Complexity: O(n)
    
    Iteratively improve the path sums.
    """
    if not triangle:
        return 0
    
    # Work with a copy of the last row
    current_sums = triangle[-1][:]
    
    # Process each row from bottom to top
    for row_idx in range(len(triangle) - 2, -1, -1):
        next_sums = []
        for col_idx in range(len(triangle[row_idx])):
            min_below = min(current_sums[col_idx], current_sums[col_idx + 1])
            next_sums.append(triangle[row_idx][col_idx] + min_below)
        current_sums = next_sums
    
    return current_sums[0]


def minimum_total_path_tracking(triangle):
    """
    Approach 8: DP with Path Tracking
    Time Complexity: O(n^2)
    Space Complexity: O(n^2)
    
    Track the actual path taken for minimum sum.
    """
    if not triangle:
        return 0
    
    n = len(triangle)
    dp = [[float('inf')] * len(row) for row in triangle]
    path = [[[] for _ in range(len(row))] for row in triangle]
    
    # Initialize bottom row
    for j in range(len(triangle[n-1])):
        dp[n-1][j] = triangle[n-1][j]
        path[n-1][j] = [triangle[n-1][j]]
    
    # Fill DP table from bottom to top
    for i in range(n-2, -1, -1):
        for j in range(len(triangle[i])):
            if dp[i+1][j] < dp[i+1][j+1]:
                dp[i][j] = triangle[i][j] + dp[i+1][j]
                path[i][j] = [triangle[i][j]] + path[i+1][j]
            else:
                dp[i][j] = triangle[i][j] + dp[i+1][j+1]
                path[i][j] = [triangle[i][j]] + path[i+1][j+1]
    
    return dp[0][0]


def minimum_total_bfs_approach(triangle):
    """
    Approach 9: BFS-like Approach
    Time Complexity: O(n^2)
    Space Complexity: O(n)
    
    Process level by level like BFS.
    """
    if not triangle:
        return 0
    
    from collections import deque
    
    # Initialize with first row
    current_level = deque([triangle[0][0]])
    
    for row_idx in range(1, len(triangle)):
        next_level = deque()
        current_row = triangle[row_idx]
        
        for col_idx in range(len(current_row)):
            min_from_above = float('inf')
            
            # Check possible previous positions
            if col_idx < len(current_level):
                min_from_above = min(min_from_above, current_level[col_idx])
            if col_idx > 0 and col_idx - 1 < len(current_level):
                min_from_above = min(min_from_above, current_level[col_idx - 1])
            
            next_level.append(current_row[col_idx] + min_from_above)
        
        current_level = next_level
    
    return min(current_level)


def test_minimum_total():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]], 11),
        ([[-10]], -10),
        ([[1], [2, 3]], 3),
        ([[1], [2, 3], [4, 5, 6]], 6),
        ([[-1], [2, 3], [1, -1, -3]], -1),
        ([[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]], 11),
        ([[1]], 1),
        ([[1], [1, 1]], 2),
        ([[5], [7, 3], [2, 3, 1]], 9),
        ([[-1], [-2, -3]], -4),
    ]
    
    approaches = [
        ("DP Bottom-up", minimum_total_dp_bottom_up),
        ("DP Top-down", minimum_total_dp_top_down),
        ("Space Optimized", minimum_total_space_optimized),
        ("In-place", minimum_total_in_place),
        ("Recursive Naive", minimum_total_recursive_naive),
        ("Rolling Array", minimum_total_dp_rolling_array),
        ("Iterative Improvement", minimum_total_iterative_improvement),
        ("Path Tracking", minimum_total_path_tracking),
        ("BFS Approach", minimum_total_bfs_approach),
    ]
    
    for i, (triangle, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: {triangle}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            # Create deep copy to avoid modifying original
            triangle_copy = [row[:] for row in triangle]
            result = func(triangle_copy)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_minimum_total() 