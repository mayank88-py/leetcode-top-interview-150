"""
64. Minimum Path Sum

Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

Example 1:
Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
Output: 7
Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.

Example 2:
Input: grid = [[1,2,3],[4,5,6]]
Output: 12

Constraints:
- m == grid.length
- n == grid[i].length
- 1 <= m, n <= 200
- 0 <= grid[i][j] <= 100
"""

def min_path_sum_2d_dp(grid):
    """
    Approach 1: 2D Dynamic Programming
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    dp[i][j] = minimum path sum to reach cell (i, j)
    """
    if not grid or not grid[0]:
        return 0
    
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    
    # Initialize first cell
    dp[0][0] = grid[0][0]
    
    # Initialize first row
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    
    # Initialize first column
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    
    # Fill the DP table
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
    
    return dp[m-1][n-1]


def min_path_sum_space_optimized(grid):
    """
    Approach 2: Space Optimized DP (1D array)
    Time Complexity: O(m * n)
    Space Complexity: O(n)
    
    Use only one row to store DP values.
    """
    if not grid or not grid[0]:
        return 0
    
    m, n = len(grid), len(grid[0])
    dp = [0] * n
    
    # Initialize first row
    dp[0] = grid[0][0]
    for j in range(1, n):
        dp[j] = dp[j-1] + grid[0][j]
    
    # Process remaining rows
    for i in range(1, m):
        dp[0] += grid[i][0]  # Update first column
        for j in range(1, n):
            dp[j] = grid[i][j] + min(dp[j], dp[j-1])
    
    return dp[n-1]


def min_path_sum_in_place(grid):
    """
    Approach 3: In-place Modification
    Time Complexity: O(m * n)
    Space Complexity: O(1)
    
    Modify the input grid in place.
    """
    if not grid or not grid[0]:
        return 0
    
    m, n = len(grid), len(grid[0])
    
    # Initialize first row
    for j in range(1, n):
        grid[0][j] += grid[0][j-1]
    
    # Initialize first column
    for i in range(1, m):
        grid[i][0] += grid[i-1][0]
    
    # Fill the grid
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += min(grid[i-1][j], grid[i][j-1])
    
    return grid[m-1][n-1]


def min_path_sum_memoization(grid):
    """
    Approach 4: Top-down DP with Memoization
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Use memoization to avoid recalculating subproblems.
    """
    if not grid or not grid[0]:
        return 0
    
    m, n = len(grid), len(grid[0])
    memo = {}
    
    def min_path(i, j):
        if i == m - 1 and j == n - 1:
            return grid[i][j]
        if i >= m or j >= n:
            return float('inf')
        
        if (i, j) in memo:
            return memo[(i, j)]
        
        # Two choices: go right or go down
        right = min_path(i, j + 1)
        down = min_path(i + 1, j)
        
        memo[(i, j)] = grid[i][j] + min(right, down)
        return memo[(i, j)]
    
    return min_path(0, 0)


def min_path_sum_rolling_array(grid):
    """
    Approach 5: Rolling Array
    Time Complexity: O(m * n)
    Space Complexity: O(min(m, n))
    
    Use rolling array technique for space optimization.
    """
    if not grid or not grid[0]:
        return 0
    
    m, n = len(grid), len(grid[0])
    
    # Choose the smaller dimension for rolling array
    if m < n:
        # Roll by rows
        prev = grid[0][:]
        for j in range(1, n):
            prev[j] += prev[j-1]
        
        for i in range(1, m):
            curr = [0] * n
            curr[0] = prev[0] + grid[i][0]
            for j in range(1, n):
                curr[j] = grid[i][j] + min(prev[j], curr[j-1])
            prev = curr
        
        return prev[n-1]
    else:
        # Roll by columns
        prev = [grid[i][0] for i in range(m)]
        for i in range(1, m):
            prev[i] += prev[i-1]
        
        for j in range(1, n):
            curr = [0] * m
            curr[0] = prev[0] + grid[0][j]
            for i in range(1, m):
                curr[i] = grid[i][j] + min(prev[i], curr[i-1])
            prev = curr
        
        return prev[m-1]


def min_path_sum_dijkstra(grid):
    """
    Approach 6: Dijkstra's Algorithm
    Time Complexity: O(mn log(mn))
    Space Complexity: O(mn)
    
    Treat as shortest path problem in a graph.
    """
    if not grid or not grid[0]:
        return 0
    
    import heapq
    
    m, n = len(grid), len(grid[0])
    # Priority queue: (distance, row, col)
    pq = [(grid[0][0], 0, 0)]
    visited = set()
    
    while pq:
        dist, row, col = heapq.heappop(pq)
        
        if (row, col) in visited:
            continue
        
        visited.add((row, col))
        
        if row == m - 1 and col == n - 1:
            return dist
        
        # Explore neighbors (right and down)
        for dr, dc in [(0, 1), (1, 0)]:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < m and 0 <= new_col < n and 
                (new_row, new_col) not in visited):
                new_dist = dist + grid[new_row][new_col]
                heapq.heappush(pq, (new_dist, new_row, new_col))
    
    return -1


def min_path_sum_bfs(grid):
    """
    Approach 7: BFS with Priority Queue
    Time Complexity: O(mn log(mn))
    Space Complexity: O(mn)
    
    Use BFS to explore paths in order of increasing cost.
    """
    if not grid or not grid[0]:
        return 0
    
    import heapq
    from collections import deque
    
    m, n = len(grid), len(grid[0])
    # Priority queue: (cost, row, col)
    pq = [(grid[0][0], 0, 0)]
    visited = set()
    
    while pq:
        cost, row, col = heapq.heappop(pq)
        
        if row == m - 1 and col == n - 1:
            return cost
        
        if (row, col) in visited:
            continue
        
        visited.add((row, col))
        
        # Add neighbors
        for dr, dc in [(0, 1), (1, 0)]:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < m and 0 <= new_col < n and 
                (new_row, new_col) not in visited):
                new_cost = cost + grid[new_row][new_col]
                heapq.heappush(pq, (new_cost, new_row, new_col))
    
    return -1


def min_path_sum_recursive_naive(grid):
    """
    Approach 8: Naive Recursion (Not efficient)
    Time Complexity: O(2^(m+n))
    Space Complexity: O(m+n)
    
    Simple recursive solution without memoization.
    """
    if not grid or not grid[0]:
        return 0
    
    m, n = len(grid), len(grid[0])
    
    def min_path(i, j):
        if i == m - 1 and j == n - 1:
            return grid[i][j]
        if i >= m or j >= n:
            return float('inf')
        
        # Two choices: go right or go down
        right = min_path(i, j + 1)
        down = min_path(i + 1, j)
        
        return grid[i][j] + min(right, down)
    
    # Avoid TLE for large grids
    if m * n > 100:
        return min_path_sum_space_optimized(grid)
    
    return min_path(0, 0)


def test_min_path_sum():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([[1, 3, 1], [1, 5, 1], [4, 2, 1]], 7),
        ([[1, 2, 3], [4, 5, 6]], 12),
        ([[1]], 1),
        ([[1, 2], [3, 4]], 8),
        ([[5, 1, 2], [2, 3, 1]], 9),
        ([[1, 2, 5], [3, 2, 1]], 6),
        ([[1, 4, 8, 6, 2, 2, 1, 7], [4, 7, 3, 1, 4, 5, 5, 1]], 23),
        ([[7, 1, 3, 5, 8, 9, 9, 2, 1, 9, 0, 8, 3, 1, 6, 6, 9, 5]], 85),
        ([[9, 1, 4, 8]], 22),
    ]
    
    approaches = [
        ("2D DP", min_path_sum_2d_dp),
        ("Space Optimized", min_path_sum_space_optimized),
        ("In-place", min_path_sum_in_place),
        ("Memoization", min_path_sum_memoization),
        ("Rolling Array", min_path_sum_rolling_array),
        ("Dijkstra", min_path_sum_dijkstra),
        ("BFS", min_path_sum_bfs),
        ("Recursive Naive", min_path_sum_recursive_naive),
    ]
    
    for i, (grid, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: {grid}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            # Create deep copy to avoid modifying original
            grid_copy = [row[:] for row in grid]
            result = func(grid_copy)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_min_path_sum() 