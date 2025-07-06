"""
221. Maximal Square

Given an m x n binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

Example 1:
Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
Output: 4

Example 2:
Input: matrix = [["0","1"],["1","0"]]
Output: 1

Example 3:
Input: matrix = [["0"]]
Output: 0

Constraints:
- m == matrix.length
- n == matrix[i].length
- 1 <= m, n <= 300
- matrix[i][j] is '0' or '1'.
"""

def maximal_square_2d_dp(matrix):
    """
    Approach 1: 2D Dynamic Programming
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    dp[i][j] = side length of largest square with bottom-right corner at (i, j)
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * n for _ in range(m)]
    max_side = 0
    
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                max_side = max(max_side, dp[i][j])
    
    return max_side * max_side


def maximal_square_space_optimized(matrix):
    """
    Approach 2: Space Optimized DP
    Time Complexity: O(m * n)
    Space Complexity: O(n)
    
    Use only one row for DP computation.
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    prev = [0] * n
    max_side = 0
    
    for i in range(m):
        curr = [0] * n
        for j in range(n):
            if matrix[i][j] == '1':
                if i == 0 or j == 0:
                    curr[j] = 1
                else:
                    curr[j] = min(prev[j], curr[j-1], prev[j-1]) + 1
                max_side = max(max_side, curr[j])
        prev = curr
    
    return max_side * max_side


def maximal_square_in_place(matrix):
    """
    Approach 3: In-place Modification
    Time Complexity: O(m * n)
    Space Complexity: O(1)
    
    Modify the input matrix in place.
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    max_side = 0
    
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                if i == 0 or j == 0:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = min(int(matrix[i-1][j]), int(matrix[i][j-1]), int(matrix[i-1][j-1])) + 1
                max_side = max(max_side, int(matrix[i][j]))
    
    return max_side * max_side


def maximal_square_stack_based(matrix):
    """
    Approach 4: Stack-based (Histogram approach)
    Time Complexity: O(m * n)
    Space Complexity: O(n)
    
    Convert to largest rectangle in histogram for each row.
    """
    if not matrix or not matrix[0]:
        return 0
    
    def largest_square_in_histogram(heights):
        stack = []
        max_area = 0
        
        for i, h in enumerate(heights):
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                width = i if not stack else i - stack[-1] - 1
                side = min(height, width)
                max_area = max(max_area, side * side)
            stack.append(i)
        
        while stack:
            height = heights[stack.pop()]
            width = len(heights) if not stack else len(heights) - stack[-1] - 1
            side = min(height, width)
            max_area = max(max_area, side * side)
        
        return max_area
    
    m, n = len(matrix), len(matrix[0])
    heights = [0] * n
    max_area = 0
    
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                heights[j] += 1
            else:
                heights[j] = 0
        
        max_area = max(max_area, largest_square_in_histogram(heights))
    
    return max_area


def maximal_square_memoization(matrix):
    """
    Approach 5: Top-down DP with Memoization
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Use memoization to avoid recalculating subproblems.
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    memo = {}
    
    def max_square_at(i, j):
        if i >= m or j >= n or matrix[i][j] == '0':
            return 0
        
        if (i, j) in memo:
            return memo[(i, j)]
        
        if i == m - 1 or j == n - 1:
            result = 1 if matrix[i][j] == '1' else 0
        else:
            result = min(max_square_at(i+1, j), max_square_at(i, j+1), max_square_at(i+1, j+1)) + 1
        
        memo[(i, j)] = result
        return result
    
    max_side = 0
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                max_side = max(max_side, max_square_at(i, j))
    
    return max_side * max_side


def maximal_square_rolling_variables(matrix):
    """
    Approach 6: Rolling Variables
    Time Complexity: O(m * n)
    Space Complexity: O(1)
    
    Use only a few variables instead of arrays.
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    max_side = 0
    prev = 0  # dp[i-1][j-1]
    
    # Create a row to store dp[i-1][j]
    dp = [0] * (n + 1)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            temp = dp[j]  # Store dp[i-1][j] before updating
            
            if matrix[i-1][j-1] == '1':
                dp[j] = min(dp[j], dp[j-1], prev) + 1
                max_side = max(max_side, dp[j])
            else:
                dp[j] = 0
            
            prev = temp
    
    return max_side * max_side


def maximal_square_brute_force(matrix):
    """
    Approach 7: Brute Force
    Time Complexity: O(m * n * min(m,n)^2)
    Space Complexity: O(1)
    
    Check all possible squares.
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    max_side = 0
    
    def is_square(top, left, size):
        for i in range(top, top + size):
            for j in range(left, left + size):
                if i >= m or j >= n or matrix[i][j] == '0':
                    return False
        return True
    
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                max_possible = min(m - i, n - j)
                for size in range(1, max_possible + 1):
                    if is_square(i, j, size):
                        max_side = max(max_side, size)
                    else:
                        break
    
    return max_side * max_side


def maximal_square_diagonal_approach(matrix):
    """
    Approach 8: Diagonal Scanning
    Time Complexity: O(m * n)
    Space Complexity: O(min(m, n))
    
    Process matrix diagonally.
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    max_side = 0
    
    # Process each diagonal
    for diag in range(m + n - 1):
        # Determine starting position for this diagonal
        if diag < m:
            start_row, start_col = diag, 0
        else:
            start_row, start_col = m - 1, diag - m + 1
        
        # Process diagonal
        dp = {}
        i, j = start_row, start_col
        
        while i >= 0 and j < n:
            if matrix[i][j] == '1':
                if i == m - 1 or j == 0:
                    dp[(i, j)] = 1
                else:
                    dp[(i, j)] = min(
                        dp.get((i+1, j), 0),
                        dp.get((i, j-1), 0),
                        dp.get((i+1, j-1), 0)
                    ) + 1
                max_side = max(max_side, dp[(i, j)])
            else:
                dp[(i, j)] = 0
            
            i -= 1
            j += 1
    
    return max_side * max_side


def maximal_square_prefix_sum(matrix):
    """
    Approach 9: Prefix Sum Approach
    Time Complexity: O(m * n * min(m,n))
    Space Complexity: O(m * n)
    
    Use prefix sums to quickly check square validity.
    """
    if not matrix or not matrix[0]:
        return 0
    
    m, n = len(matrix), len(matrix[0])
    
    # Build prefix sum matrix
    prefix = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            prefix[i][j] = (int(matrix[i-1][j-1]) + 
                          prefix[i-1][j] + 
                          prefix[i][j-1] - 
                          prefix[i-1][j-1])
    
    def get_sum(r1, c1, r2, c2):
        return (prefix[r2+1][c2+1] - 
                prefix[r1][c2+1] - 
                prefix[r2+1][c1] + 
                prefix[r1][c1])
    
    max_side = 0
    
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                max_possible = min(m - i, n - j)
                for size in range(1, max_possible + 1):
                    expected_sum = size * size
                    actual_sum = get_sum(i, j, i + size - 1, j + size - 1)
                    if actual_sum == expected_sum:
                        max_side = max(max_side, size)
                    else:
                        break
    
    return max_side * max_side


def test_maximal_square():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]], 4),
        ([["0","1"],["1","0"]], 1),
        ([["0"]], 0),
        ([["1"]], 1),
        ([["1","1"],["1","1"]], 4),
        ([["0","0","0"],["0","0","0"],["0","0","0"]], 0),
        ([["1","1","1"],["1","1","1"],["1","1","1"]], 9),
        ([["1","0","1","1","1"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]], 9),
        ([["1","1","1","1","0"],["1","1","1","1","0"],["1","1","1","1","1"],["1","1","1","1","1"],["0","0","1","1","1"]], 16),
    ]
    
    approaches = [
        ("2D DP", maximal_square_2d_dp),
        ("Space Optimized", maximal_square_space_optimized),
        ("In-place", maximal_square_in_place),
        ("Stack Based", maximal_square_stack_based),
        ("Memoization", maximal_square_memoization),
        ("Rolling Variables", maximal_square_rolling_variables),
        ("Brute Force", maximal_square_brute_force),
        ("Diagonal", maximal_square_diagonal_approach),
        ("Prefix Sum", maximal_square_prefix_sum),
    ]
    
    for i, (matrix, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: {matrix}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            # Create deep copy to avoid modifying original
            matrix_copy = [row[:] for row in matrix]
            result = func(matrix_copy)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_maximal_square() 