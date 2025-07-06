"""
63. Unique Paths II

You are given an m x n integer array grid. There is a robot initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m-1][n-1]). The robot can only move either down or right at any point in time.

An obstacle and space are marked as 1 and 0 respectively in grid. A path that the robot takes cannot include any square that is an obstacle.

Return the number of unique paths that the robot can take to reach the bottom-right corner.

Example 1:
Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
Output: 2
Explanation: There is one obstacle in the middle of the 3x3 grid.
There are two ways to reach the bottom-right corner:
1. Right -> Right -> Down -> Down
2. Down -> Down -> Right -> Right

Example 2:
Input: obstacleGrid = [[0,1],[0,0]]
Output: 1

Constraints:
- m == obstacleGrid.length
- n == obstacleGrid[i].length
- 1 <= m, n <= 100
- obstacleGrid[i][j] is 0 or 1.
"""

def unique_paths_with_obstacles_2d_dp(obstacle_grid):
    """
    Approach 1: 2D Dynamic Programming
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    dp[i][j] = number of unique paths to reach cell (i, j)
    """
    if not obstacle_grid or not obstacle_grid[0] or obstacle_grid[0][0] == 1:
        return 0
    
    m, n = len(obstacle_grid), len(obstacle_grid[0])
    dp = [[0] * n for _ in range(m)]
    
    # Initialize starting position
    dp[0][0] = 1
    
    # Initialize first row
    for j in range(1, n):
        if obstacle_grid[0][j] == 0:
            dp[0][j] = dp[0][j-1]
        else:
            dp[0][j] = 0
    
    # Initialize first column
    for i in range(1, m):
        if obstacle_grid[i][0] == 0:
            dp[i][0] = dp[i-1][0]
        else:
            dp[i][0] = 0
    
    # Fill the DP table
    for i in range(1, m):
        for j in range(1, n):
            if obstacle_grid[i][j] == 0:
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
            else:
                dp[i][j] = 0
    
    return dp[m-1][n-1]


def unique_paths_with_obstacles_space_optimized(obstacle_grid):
    """
    Approach 2: Space Optimized DP
    Time Complexity: O(m * n)
    Space Complexity: O(n)
    
    Use only one row to store DP values.
    """
    if not obstacle_grid or not obstacle_grid[0] or obstacle_grid[0][0] == 1:
        return 0
    
    m, n = len(obstacle_grid), len(obstacle_grid[0])
    dp = [0] * n
    
    # Initialize first row
    dp[0] = 1 if obstacle_grid[0][0] == 0 else 0
    for j in range(1, n):
        if obstacle_grid[0][j] == 0:
            dp[j] = dp[j-1]
        else:
            dp[j] = 0
    
    # Process remaining rows
    for i in range(1, m):
        # Update first column
        if obstacle_grid[i][0] == 1:
            dp[0] = 0
        # dp[0] remains the same if obstacle_grid[i][0] == 0
        
        for j in range(1, n):
            if obstacle_grid[i][j] == 0:
                dp[j] = dp[j] + dp[j-1]
            else:
                dp[j] = 0
    
    return dp[n-1]


def unique_paths_with_obstacles_in_place(obstacle_grid):
    """
    Approach 3: In-place Modification
    Time Complexity: O(m * n)
    Space Complexity: O(1)
    
    Modify the input grid in place.
    """
    if not obstacle_grid or not obstacle_grid[0] or obstacle_grid[0][0] == 1:
        return 0
    
    m, n = len(obstacle_grid), len(obstacle_grid[0])
    
    # Set starting position
    obstacle_grid[0][0] = 1
    
    # Initialize first row
    for j in range(1, n):
        if obstacle_grid[0][j] == 0:
            obstacle_grid[0][j] = obstacle_grid[0][j-1]
        else:
            obstacle_grid[0][j] = 0
    
    # Initialize first column
    for i in range(1, m):
        if obstacle_grid[i][0] == 0:
            obstacle_grid[i][0] = obstacle_grid[i-1][0]
        else:
            obstacle_grid[i][0] = 0
    
    # Fill the grid
    for i in range(1, m):
        for j in range(1, n):
            if obstacle_grid[i][j] == 0:
                obstacle_grid[i][j] = obstacle_grid[i-1][j] + obstacle_grid[i][j-1]
            else:
                obstacle_grid[i][j] = 0
    
    return obstacle_grid[m-1][n-1]


def unique_paths_with_obstacles_memoization(obstacle_grid):
    """
    Approach 4: Top-down DP with Memoization
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Use memoization to avoid recalculating subproblems.
    """
    if not obstacle_grid or not obstacle_grid[0] or obstacle_grid[0][0] == 1:
        return 0
    
    m, n = len(obstacle_grid), len(obstacle_grid[0])
    memo = {}
    
    def count_paths(i, j):
        if i >= m or j >= n or obstacle_grid[i][j] == 1:
            return 0
        if i == m - 1 and j == n - 1:
            return 1
        
        if (i, j) in memo:
            return memo[(i, j)]
        
        # Two directions: right and down
        right = count_paths(i, j + 1)
        down = count_paths(i + 1, j)
        
        memo[(i, j)] = right + down
        return memo[(i, j)]
    
    return count_paths(0, 0)


def unique_paths_with_obstacles_rolling_array(obstacle_grid):
    """
    Approach 5: Rolling Array
    Time Complexity: O(m * n)
    Space Complexity: O(min(m, n))
    
    Use rolling array technique for space optimization.
    """
    if not obstacle_grid or not obstacle_grid[0] or obstacle_grid[0][0] == 1:
        return 0
    
    m, n = len(obstacle_grid), len(obstacle_grid[0])
    
    # Choose the smaller dimension for rolling array
    if m < n:
        # Roll by rows
        prev = [0] * n
        prev[0] = 1 if obstacle_grid[0][0] == 0 else 0
        
        for j in range(1, n):
            if obstacle_grid[0][j] == 0:
                prev[j] = prev[j-1]
            else:
                prev[j] = 0
        
        for i in range(1, m):
            curr = [0] * n
            curr[0] = prev[0] if obstacle_grid[i][0] == 0 else 0
            
            for j in range(1, n):
                if obstacle_grid[i][j] == 0:
                    curr[j] = prev[j] + curr[j-1]
                else:
                    curr[j] = 0
            
            prev = curr
        
        return prev[n-1]
    else:
        # Roll by columns - transpose the problem
        prev = [0] * m
        prev[0] = 1 if obstacle_grid[0][0] == 0 else 0
        
        for i in range(1, m):
            if obstacle_grid[i][0] == 0:
                prev[i] = prev[i-1]
            else:
                prev[i] = 0
        
        for j in range(1, n):
            curr = [0] * m
            curr[0] = prev[0] if obstacle_grid[0][j] == 0 else 0
            
            for i in range(1, m):
                if obstacle_grid[i][j] == 0:
                    curr[i] = prev[i] + curr[i-1]
                else:
                    curr[i] = 0
            
            prev = curr
        
        return prev[m-1]


def unique_paths_with_obstacles_bfs(obstacle_grid):
    """
    Approach 6: BFS Approach
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Use BFS to count all possible paths.
    """
    if not obstacle_grid or not obstacle_grid[0] or obstacle_grid[0][0] == 1:
        return 0
    
    from collections import deque
    
    m, n = len(obstacle_grid), len(obstacle_grid[0])
    
    # BFS with path counting
    queue = deque([(0, 0, 1)])  # (row, col, path_count)
    visited = {}  # Store maximum paths to each cell
    
    while queue:
        row, col, paths = queue.popleft()
        
        if row == m - 1 and col == n - 1:
            return visited.get((row, col), 0) + paths
        
        # Update path count for current cell
        if (row, col) in visited:
            visited[(row, col)] += paths
            continue
        else:
            visited[(row, col)] = paths
        
        # Explore neighbors (right and down)
        for dr, dc in [(0, 1), (1, 0)]:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < m and 0 <= new_col < n and 
                obstacle_grid[new_row][new_col] == 0):
                queue.append((new_row, new_col, paths))
    
    return visited.get((m-1, n-1), 0)


def unique_paths_with_obstacles_dfs(obstacle_grid):
    """
    Approach 7: DFS with Path Counting
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Use DFS to count unique paths.
    """
    if not obstacle_grid or not obstacle_grid[0] or obstacle_grid[0][0] == 1:
        return 0
    
    m, n = len(obstacle_grid), len(obstacle_grid[0])
    visited = {}
    
    def dfs(i, j):
        if i >= m or j >= n or obstacle_grid[i][j] == 1:
            return 0
        if i == m - 1 and j == n - 1:
            return 1
        
        if (i, j) in visited:
            return visited[(i, j)]
        
        # Explore both directions
        paths = dfs(i + 1, j) + dfs(i, j + 1)
        visited[(i, j)] = paths
        
        return paths
    
    return dfs(0, 0)


def unique_paths_with_obstacles_iterative_bottom_up(obstacle_grid):
    """
    Approach 8: Iterative Bottom-up
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Build solution from bottom-right to top-left.
    """
    if not obstacle_grid or not obstacle_grid[0]:
        return 0
    
    m, n = len(obstacle_grid), len(obstacle_grid[0])
    
    if obstacle_grid[m-1][n-1] == 1 or obstacle_grid[0][0] == 1:
        return 0
    
    dp = [[0] * n for _ in range(m)]
    
    # Initialize destination
    dp[m-1][n-1] = 1
    
    # Fill from bottom-right to top-left
    for i in range(m-1, -1, -1):
        for j in range(n-1, -1, -1):
            if obstacle_grid[i][j] == 1:
                dp[i][j] = 0
            elif i == m-1 and j == n-1:
                continue  # Already set
            else:
                right = dp[i][j+1] if j+1 < n else 0
                down = dp[i+1][j] if i+1 < m else 0
                dp[i][j] = right + down
    
    return dp[0][0]


def test_unique_paths_with_obstacles():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([[0, 0, 0], [0, 1, 0], [0, 0, 0]], 2),
        ([[0, 1], [0, 0]], 1),
        ([[1]], 0),
        ([[0]], 1),
        ([[0, 0], [1, 1], [0, 0]], 0),
        ([[0, 0], [0, 1]], 1),
        ([[0, 1, 0], [0, 0, 0], [1, 0, 0]], 2),
        ([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]], 4),
        ([[1, 0]], 0),
        ([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]], 0),
    ]
    
    approaches = [
        ("2D DP", unique_paths_with_obstacles_2d_dp),
        ("Space Optimized", unique_paths_with_obstacles_space_optimized),
        ("In-place", unique_paths_with_obstacles_in_place),
        ("Memoization", unique_paths_with_obstacles_memoization),
        ("Rolling Array", unique_paths_with_obstacles_rolling_array),
        ("BFS", unique_paths_with_obstacles_bfs),
        ("DFS", unique_paths_with_obstacles_dfs),
        ("Bottom-up", unique_paths_with_obstacles_iterative_bottom_up),
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
    test_unique_paths_with_obstacles() 