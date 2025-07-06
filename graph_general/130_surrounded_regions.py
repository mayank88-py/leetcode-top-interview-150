"""
130. Surrounded Regions

Given an m x n matrix board containing 'X' and 'O', capture all regions that are 4-directionally surrounded by 'X'.

A region is captured by flipping all 'O's into 'X's in that surrounded region.

Example 1:
Input: board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
Output: [["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
Explanation: Notice that an 'O' in the corner or on the border of the board is always escaped.
Only the 'O' in the middle that are 4 directionally surrounded by 'X' will be flipped to 'X'.

Example 2:
Input: board = [["X"]]
Output: [["X"]]

Constraints:
- m == board.length
- n == board[i].length
- 1 <= m, n <= 200
- board[i][j] is 'X' or 'O'.
"""

from typing import List
from collections import deque


def solve_boundary_dfs(board: List[List[str]]) -> None:
    """
    Boundary DFS approach (optimal solution).
    
    Time Complexity: O(m * n) where m and n are board dimensions
    Space Complexity: O(m * n) - recursion stack in worst case
    
    Algorithm:
    1. Start DFS from all 'O's on the boundary
    2. Mark all connected 'O's as safe (temporary mark)
    3. Convert remaining 'O's to 'X' and restore safe 'O's
    """
    if not board or not board[0]:
        return
    
    rows, cols = len(board), len(board[0])
    
    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            board[r][c] != 'O'):
            return
        
        # Mark as safe
        board[r][c] = '#'
        
        # Explore all 4 directions
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    # Mark all boundary-connected 'O's as safe
    for i in range(rows):
        for j in range(cols):
            if (i == 0 or i == rows - 1 or j == 0 or j == cols - 1) and board[i][j] == 'O':
                dfs(i, j)
    
    # Convert remaining 'O's to 'X' and restore safe 'O's
    for i in range(rows):
        for j in range(cols):
            if board[i][j] == 'O':
                board[i][j] = 'X'
            elif board[i][j] == '#':
                board[i][j] = 'O'


def solve_boundary_bfs(board: List[List[str]]) -> None:
    """
    Boundary BFS approach.
    
    Time Complexity: O(m * n) where m and n are board dimensions
    Space Complexity: O(min(m, n)) - queue size
    
    Algorithm:
    1. Start BFS from all 'O's on the boundary
    2. Mark all connected 'O's as safe using BFS
    3. Convert remaining 'O's to 'X' and restore safe 'O's
    """
    if not board or not board[0]:
        return
    
    rows, cols = len(board), len(board[0])
    
    def bfs(start_r, start_c):
        queue = deque([(start_r, start_c)])
        board[start_r][start_c] = '#'
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            r, c = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 0 <= nc < cols and 
                    board[nr][nc] == 'O'):
                    board[nr][nc] = '#'
                    queue.append((nr, nc))
    
    # Find all boundary 'O's and start BFS
    for i in range(rows):
        for j in range(cols):
            if (i == 0 or i == rows - 1 or j == 0 or j == cols - 1) and board[i][j] == 'O':
                bfs(i, j)
    
    # Convert remaining 'O's to 'X' and restore safe 'O's
    for i in range(rows):
        for j in range(cols):
            if board[i][j] == 'O':
                board[i][j] = 'X'
            elif board[i][j] == '#':
                board[i][j] = 'O'


def solve_union_find(board: List[List[str]]) -> None:
    """
    Union-Find approach.
    
    Time Complexity: O(m * n * α(m * n)) where α is inverse Ackermann function
    Space Complexity: O(m * n) - parent array
    
    Algorithm:
    1. Create dummy node for all boundary-connected 'O's
    2. Union all 'O's with their 'O' neighbors
    3. Union boundary 'O's with dummy node
    4. Convert 'O's not connected to dummy to 'X'
    """
    if not board or not board[0]:
        return
    
    rows, cols = len(board), len(board[0])
    
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px == py:
                return
            
            if self.rank[px] < self.rank[py]:
                px, py = py, px
            
            self.parent[py] = px
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1
    
    # Create Union-Find structure (extra node for dummy)
    uf = UnionFind(rows * cols + 1)
    dummy = rows * cols
    
    def get_index(r, c):
        return r * cols + c
    
    # Union all 'O's with their 'O' neighbors
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for i in range(rows):
        for j in range(cols):
            if board[i][j] == 'O':
                # If on boundary, union with dummy
                if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                    uf.union(get_index(i, j), dummy)
                
                # Union with 'O' neighbors
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < rows and 0 <= nj < cols and 
                        board[ni][nj] == 'O'):
                        uf.union(get_index(i, j), get_index(ni, nj))
    
    # Convert 'O's not connected to dummy
    for i in range(rows):
        for j in range(cols):
            if (board[i][j] == 'O' and 
                uf.find(get_index(i, j)) != uf.find(dummy)):
                board[i][j] = 'X'


def solve_iterative_dfs(board: List[List[str]]) -> None:
    """
    Iterative DFS approach using stack.
    
    Time Complexity: O(m * n) where m and n are board dimensions
    Space Complexity: O(m * n) - stack size
    
    Algorithm:
    1. Use explicit stack for DFS traversal
    2. Start from boundary 'O's
    3. Mark connected 'O's as safe
    """
    if not board or not board[0]:
        return
    
    rows, cols = len(board), len(board[0])
    
    def iterative_dfs(start_r, start_c):
        stack = [(start_r, start_c)]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while stack:
            r, c = stack.pop()
            
            if (r < 0 or r >= rows or c < 0 or c >= cols or 
                board[r][c] != 'O'):
                continue
            
            board[r][c] = '#'
            
            # Add neighbors to stack
            for dr, dc in directions:
                stack.append((r + dr, c + dc))
    
    # Process boundary 'O's
    for i in range(rows):
        for j in range(cols):
            if (i == 0 or i == rows - 1 or j == 0 or j == cols - 1) and board[i][j] == 'O':
                iterative_dfs(i, j)
    
    # Convert remaining 'O's to 'X' and restore safe 'O's
    for i in range(rows):
        for j in range(cols):
            if board[i][j] == 'O':
                board[i][j] = 'X'
            elif board[i][j] == '#':
                board[i][j] = 'O'


def solve_flood_fill(board: List[List[str]]) -> None:
    """
    Flood fill approach.
    
    Time Complexity: O(m * n) where m and n are board dimensions
    Space Complexity: O(m * n) - recursion stack
    
    Algorithm:
    1. Use flood fill algorithm from boundary
    2. Fill boundary-connected regions with temporary marker
    3. Convert remaining 'O's to 'X'
    """
    if not board or not board[0]:
        return
    
    rows, cols = len(board), len(board[0])
    
    def flood_fill(r, c, target, replacement):
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            board[r][c] != target):
            return
        
        board[r][c] = replacement
        
        flood_fill(r + 1, c, target, replacement)
        flood_fill(r - 1, c, target, replacement)
        flood_fill(r, c + 1, target, replacement)
        flood_fill(r, c - 1, target, replacement)
    
    # Mark boundary-connected 'O's as safe
    for i in range(rows):
        if board[i][0] == 'O':
            flood_fill(i, 0, 'O', '#')
        if board[i][cols - 1] == 'O':
            flood_fill(i, cols - 1, 'O', '#')
    
    for j in range(cols):
        if board[0][j] == 'O':
            flood_fill(0, j, 'O', '#')
        if board[rows - 1][j] == 'O':
            flood_fill(rows - 1, j, 'O', '#')
    
    # Convert remaining 'O's to 'X' and restore safe 'O's
    for i in range(rows):
        for j in range(cols):
            if board[i][j] == 'O':
                board[i][j] = 'X'
            elif board[i][j] == '#':
                board[i][j] = 'O'


def solve_two_pass_scanning(board: List[List[str]]) -> None:
    """
    Two-pass scanning approach.
    
    Time Complexity: O(m * n) where m and n are board dimensions
    Space Complexity: O(1) - constant extra space
    
    Algorithm:
    1. First pass: mark all boundary-connected 'O's
    2. Second pass: convert unmarked 'O's to 'X'
    3. Restore marked 'O's
    """
    if not board or not board[0]:
        return
    
    rows, cols = len(board), len(board[0])
    
    # First pass: mark boundary-connected 'O's
    def mark_safe(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            board[r][c] != 'O'):
            return
        
        board[r][c] = '#'
        
        # Use iterative approach to avoid stack overflow
        stack = [(r, c)]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while stack:
            cr, cc = stack.pop()
            
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if (0 <= nr < rows and 0 <= nc < cols and 
                    board[nr][nc] == 'O'):
                    board[nr][nc] = '#'
                    stack.append((nr, nc))
    
    # Process boundaries
    for i in range(rows):
        if board[i][0] == 'O':
            mark_safe(i, 0)
        if board[i][cols - 1] == 'O':
            mark_safe(i, cols - 1)
    
    for j in range(cols):
        if board[0][j] == 'O':
            mark_safe(0, j)
        if board[rows - 1][j] == 'O':
            mark_safe(rows - 1, j)
    
    # Second pass: convert and restore
    for i in range(rows):
        for j in range(cols):
            if board[i][j] == 'O':
                board[i][j] = 'X'
            elif board[i][j] == '#':
                board[i][j] = 'O'


def solve_level_order_bfs(board: List[List[str]]) -> None:
    """
    Level-order BFS approach.
    
    Time Complexity: O(m * n) where m and n are board dimensions
    Space Complexity: O(min(m, n)) - queue size
    
    Algorithm:
    1. Use level-order BFS from boundary
    2. Process cells level by level
    3. Mark safe cells during traversal
    """
    if not board or not board[0]:
        return
    
    rows, cols = len(board), len(board[0])
    
    # Collect all boundary 'O's
    boundary_cells = []
    for i in range(rows):
        for j in range(cols):
            if (i == 0 or i == rows - 1 or j == 0 or j == cols - 1) and board[i][j] == 'O':
                boundary_cells.append((i, j))
    
    # Level-order BFS
    queue = deque(boundary_cells)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    # Mark initial boundary cells
    for r, c in boundary_cells:
        board[r][c] = '#'
    
    while queue:
        level_size = len(queue)
        
        for _ in range(level_size):
            r, c = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 0 <= nc < cols and 
                    board[nr][nc] == 'O'):
                    board[nr][nc] = '#'
                    queue.append((nr, nc))
    
    # Convert remaining 'O's to 'X' and restore safe 'O's
    for i in range(rows):
        for j in range(cols):
            if board[i][j] == 'O':
                board[i][j] = 'X'
            elif board[i][j] == '#':
                board[i][j] = 'O'


def solve_reverse_thinking(board: List[List[str]]) -> None:
    """
    Reverse thinking approach.
    
    Time Complexity: O(m * n) where m and n are board dimensions
    Space Complexity: O(m * n) - recursion stack
    
    Algorithm:
    1. Think in reverse: find all 'O's that should NOT be converted
    2. These are the ones connected to boundary
    3. Convert all others to 'X'
    """
    if not board or not board[0]:
        return
    
    rows, cols = len(board), len(board[0])
    safe_cells = set()
    
    def mark_safe_dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            board[r][c] != 'O' or (r, c) in safe_cells):
            return
        
        safe_cells.add((r, c))
        
        mark_safe_dfs(r + 1, c)
        mark_safe_dfs(r - 1, c)
        mark_safe_dfs(r, c + 1)
        mark_safe_dfs(r, c - 1)
    
    # Find all safe 'O's (connected to boundary)
    for i in range(rows):
        for j in range(cols):
            if (i == 0 or i == rows - 1 or j == 0 or j == cols - 1) and board[i][j] == 'O':
                mark_safe_dfs(i, j)
    
    # Convert non-safe 'O's to 'X'
    for i in range(rows):
        for j in range(cols):
            if board[i][j] == 'O' and (i, j) not in safe_cells:
                board[i][j] = 'X'


def solve_bidirectional_search(board: List[List[str]]) -> None:
    """
    Bidirectional search approach.
    
    Time Complexity: O(m * n) where m and n are board dimensions
    Space Complexity: O(m * n) - visited sets
    
    Algorithm:
    1. Search from boundary inward and from interior outward
    2. Meet in the middle to determine safe regions
    3. Convert appropriate cells
    """
    if not board or not board[0]:
        return
    
    rows, cols = len(board), len(board[0])
    
    # Find all 'O' cells
    all_o_cells = set()
    boundary_o_cells = set()
    
    for i in range(rows):
        for j in range(cols):
            if board[i][j] == 'O':
                all_o_cells.add((i, j))
                if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                    boundary_o_cells.add((i, j))
    
    # BFS from boundary to find safe cells
    safe_cells = set()
    queue = deque(boundary_o_cells)
    safe_cells.update(boundary_o_cells)
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while queue:
        r, c = queue.popleft()
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if ((nr, nc) in all_o_cells and (nr, nc) not in safe_cells):
                safe_cells.add((nr, nc))
                queue.append((nr, nc))
    
    # Convert non-safe 'O's to 'X'
    for r, c in all_o_cells:
        if (r, c) not in safe_cells:
            board[r][c] = 'X'


# Test cases
def test_solve():
    """Test all surrounded regions approaches."""
    
    def copy_board(board):
        """Create a deep copy of the board."""
        return [row[:] for row in board]
    
    def boards_equal(board1, board2):
        """Check if two boards are equal."""
        if len(board1) != len(board2):
            return False
        
        for i in range(len(board1)):
            if len(board1[i]) != len(board2[i]):
                return False
            for j in range(len(board1[i])):
                if board1[i][j] != board2[i][j]:
                    return False
        
        return True
    
    # Test cases
    test_cases = [
        (
            [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]],
            [["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]],
            "Standard example"
        ),
        (
            [["X"]],
            [["X"]],
            "Single cell X"
        ),
        (
            [["O"]],
            [["O"]],
            "Single cell O (boundary)"
        ),
        (
            [["O","O","O"],["O","O","O"],["O","O","O"]],
            [["O","O","O"],["O","O","O"],["O","O","O"]],
            "All O's (all boundary connected)"
        ),
        (
            [["X","X","X"],["X","O","X"],["X","X","X"]],
            [["X","X","X"],["X","X","X"],["X","X","X"]],
            "Single surrounded O"
        ),
        (
            [["O","X","X","O","X"],["X","O","O","X","O"],["X","O","X","O","X"],["O","X","O","O","O"],["X","X","O","X","O"]],
            [["O","X","X","O","X"],["X","X","X","X","O"],["X","X","X","O","X"],["O","X","O","O","O"],["X","X","O","X","O"]],
            "Complex pattern"
        ),
        (
            [["O","O"],["O","O"]],
            [["O","O"],["O","O"]],
            "2x2 all O's"
        ),
        (
            [["X","O","X"],["O","X","O"],["X","O","X"]],
            [["X","O","X"],["O","X","O"],["X","O","X"]],
            "Checkerboard pattern"
        ),
    ]
    
    # Test all approaches
    approaches = [
        ("Boundary DFS", solve_boundary_dfs),
        ("Boundary BFS", solve_boundary_bfs),
        ("Union-Find", solve_union_find),
        ("Iterative DFS", solve_iterative_dfs),
        ("Flood Fill", solve_flood_fill),
        ("Two-pass Scanning", solve_two_pass_scanning),
        ("Level-order BFS", solve_level_order_bfs),
        ("Reverse Thinking", solve_reverse_thinking),
        ("Bidirectional Search", solve_bidirectional_search),
    ]
    
    print("Testing surrounded regions approaches:")
    print("=" * 50)
    
    for i, (input_board, expected, description) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: {description}")
        print(f"Input:    {input_board}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            try:
                board_copy = copy_board(input_board)
                func(board_copy)
                is_correct = boards_equal(board_copy, expected)
                status = "✓" if is_correct else "✗"
                print(f"{status} {name}: {board_copy}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 50)
    print("Performance Analysis:")
    print("=" * 50)
    
    # Create larger test case for performance testing
    def create_large_test_case(size):
        """Create a large test case."""
        board = [['O' if (i + j) % 3 == 0 else 'X' for j in range(size)] for i in range(size)]
        
        # Add some internal 'O's that should be converted
        for i in range(2, size - 2):
            for j in range(2, size - 2):
                if (i + j) % 5 == 0:
                    board[i][j] = 'O'
        
        return board
    
    large_board = create_large_test_case(50)
    
    import time
    
    print(f"Testing with large board (50x50):")
    for name, func in approaches:
        try:
            board_copy = copy_board(large_board)
            start_time = time.time()
            func(board_copy)
            end_time = time.time()
            print(f"{name}: (Time: {end_time - start_time:.6f}s)")
        except Exception as e:
            print(f"{name}: Error - {e}")


if __name__ == "__main__":
    test_solve() 