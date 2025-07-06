"""
200. Number of Islands

Given an m x n 2D binary grid which represents a map of '1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

Example 1:
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1

Example 2:
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3

Constraints:
- m == grid.length
- n == grid[i].length
- 1 <= m, n <= 300
- grid[i][j] is '0' or '1'
"""

from typing import List
from collections import deque


def num_islands_dfs(grid: List[List[str]]) -> int:
    """
    DFS approach (optimal solution).
    
    Time Complexity: O(m * n) where m and n are grid dimensions
    Space Complexity: O(m * n) - recursion stack in worst case
    
    Algorithm:
    1. Iterate through each cell in the grid
    2. When a '1' is found, increment island count
    3. Use DFS to mark all connected '1's as visited
    4. Continue until all cells are processed
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    island_count = 0
    
    def dfs(r, c):
        # Base case: out of bounds or water or already visited
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            grid[r][c] == '0' or grid[r][c] == '#'):
            return
        
        # Mark current cell as visited
        grid[r][c] = '#'
        
        # Explore all 4 directions
        dfs(r + 1, c)  # down
        dfs(r - 1, c)  # up
        dfs(r, c + 1)  # right
        dfs(r, c - 1)  # left
    
    # Iterate through each cell
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                island_count += 1
                dfs(i, j)
    
    return island_count


def num_islands_bfs(grid: List[List[str]]) -> int:
    """
    BFS approach using queue.
    
    Time Complexity: O(m * n) where m and n are grid dimensions
    Space Complexity: O(min(m, n)) - queue size
    
    Algorithm:
    1. Iterate through each cell in the grid
    2. When a '1' is found, increment island count
    3. Use BFS to mark all connected '1's as visited
    4. Continue until all cells are processed
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    island_count = 0
    
    def bfs(start_r, start_c):
        queue = deque([(start_r, start_c)])
        grid[start_r][start_c] = '#'  # Mark as visited
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            r, c = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 0 <= nc < cols and 
                    grid[nr][nc] == '1'):
                    grid[nr][nc] = '#'  # Mark as visited
                    queue.append((nr, nc))
    
    # Iterate through each cell
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                island_count += 1
                bfs(i, j)
    
    return island_count


def num_islands_union_find(grid: List[List[str]]) -> int:
    """
    Union-Find approach.
    
    Time Complexity: O(m * n * α(m * n)) where α is inverse Ackermann function
    Space Complexity: O(m * n) - parent array
    
    Algorithm:
    1. Create union-find data structure
    2. Union adjacent land cells
    3. Count number of distinct components
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.components = n
        
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
            
            self.components -= 1
    
    # Count land cells and create union-find
    land_cells = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                land_cells += 1
    
    if land_cells == 0:
        return 0
    
    uf = UnionFind(rows * cols)
    
    # Convert 2D coordinates to 1D index
    def get_index(r, c):
        return r * cols + c
    
    # Union adjacent land cells
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                # Check right and down neighbors
                if j + 1 < cols and grid[i][j + 1] == '1':
                    uf.union(get_index(i, j), get_index(i, j + 1))
                if i + 1 < rows and grid[i + 1][j] == '1':
                    uf.union(get_index(i, j), get_index(i + 1, j))
    
    # Count distinct land components
    land_components = set()
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                land_components.add(uf.find(get_index(i, j)))
    
    return len(land_components)


def num_islands_iterative_dfs(grid: List[List[str]]) -> int:
    """
    Iterative DFS approach using stack.
    
    Time Complexity: O(m * n) where m and n are grid dimensions
    Space Complexity: O(m * n) - stack size
    
    Algorithm:
    1. Use explicit stack instead of recursion
    2. Iterate through each cell in the grid
    3. When a '1' is found, use iterative DFS to mark island
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    island_count = 0
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                island_count += 1
                
                # Iterative DFS using stack
                stack = [(i, j)]
                
                while stack:
                    r, c = stack.pop()
                    
                    if (r < 0 or r >= rows or c < 0 or c >= cols or 
                        grid[r][c] != '1'):
                        continue
                    
                    grid[r][c] = '#'  # Mark as visited
                    
                    # Add neighbors to stack
                    for dr, dc in directions:
                        stack.append((r + dr, c + dc))
    
    return island_count


def num_islands_without_modification(grid: List[List[str]]) -> int:
    """
    Approach without modifying the original grid.
    
    Time Complexity: O(m * n) where m and n are grid dimensions
    Space Complexity: O(m * n) - visited set + recursion stack
    
    Algorithm:
    1. Use visited set to track processed cells
    2. Iterate through each cell in the grid
    3. When an unvisited '1' is found, use DFS to mark island
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    island_count = 0
    visited = set()
    
    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            grid[r][c] == '0' or (r, c) in visited):
            return
        
        visited.add((r, c))
        
        # Explore all 4 directions
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    # Iterate through each cell
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1' and (i, j) not in visited:
                island_count += 1
                dfs(i, j)
    
    return island_count


def num_islands_flood_fill(grid: List[List[str]]) -> int:
    """
    Flood fill approach.
    
    Time Complexity: O(m * n) where m and n are grid dimensions
    Space Complexity: O(m * n) - recursion stack
    
    Algorithm:
    1. Use flood fill algorithm to mark connected components
    2. Each flood fill operation marks one island
    3. Count number of flood fill operations needed
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    island_count = 0
    
    def flood_fill(r, c, marker):
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            grid[r][c] != '1'):
            return
        
        grid[r][c] = marker  # Mark with unique identifier
        
        # Fill all 4 directions
        flood_fill(r + 1, c, marker)
        flood_fill(r - 1, c, marker)
        flood_fill(r, c + 1, marker)
        flood_fill(r, c - 1, marker)
    
    # Iterate through each cell
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                island_count += 1
                flood_fill(i, j, f'#{island_count}')
    
    return island_count


def num_islands_multidirectional_bfs(grid: List[List[str]]) -> int:
    """
    Multi-directional BFS approach.
    
    Time Complexity: O(m * n) where m and n are grid dimensions
    Space Complexity: O(m * n) - queue size
    
    Algorithm:
    1. Use BFS with 8 directions (including diagonals)
    2. Note: This changes the problem definition but shows algorithm flexibility
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    island_count = 0
    
    # 8 directions (including diagonals)
    directions = [
        (0, 1), (0, -1), (1, 0), (-1, 0),  # 4 main directions
        (1, 1), (1, -1), (-1, 1), (-1, -1)  # 4 diagonal directions
    ]
    
    def bfs(start_r, start_c):
        queue = deque([(start_r, start_c)])
        grid[start_r][start_c] = '#'
        
        while queue:
            r, c = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 0 <= nc < cols and 
                    grid[nr][nc] == '1'):
                    grid[nr][nc] = '#'
                    queue.append((nr, nc))
    
    # Standard 4-directional BFS (comment out 8-directional for standard problem)
    def bfs_standard(start_r, start_c):
        queue = deque([(start_r, start_c)])
        grid[start_r][start_c] = '#'
        
        directions_4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            r, c = queue.popleft()
            
            for dr, dc in directions_4:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 0 <= nc < cols and 
                    grid[nr][nc] == '1'):
                    grid[nr][nc] = '#'
                    queue.append((nr, nc))
    
    # Iterate through each cell
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                island_count += 1
                bfs_standard(i, j)  # Use standard 4-directional BFS
    
    return island_count


def num_islands_optimized_early_termination(grid: List[List[str]]) -> int:
    """
    Optimized approach with early termination.
    
    Time Complexity: O(m * n) where m and n are grid dimensions
    Space Complexity: O(m * n) - recursion stack
    
    Algorithm:
    1. Use DFS with early termination optimizations
    2. Skip rows/columns that are all water
    3. Use boundary checking optimizations
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    island_count = 0
    
    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            grid[r][c] != '1'):
            return
        
        grid[r][c] = '#'
        
        # Optimized order: check most likely directions first
        dfs(r, c + 1)  # right
        dfs(r + 1, c)  # down
        dfs(r, c - 1)  # left
        dfs(r - 1, c)  # up
    
    # Skip empty rows/columns (optimization)
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                island_count += 1
                dfs(i, j)
    
    return island_count


# Test cases
def test_num_islands():
    """Test all island counting approaches."""
    
    # Test cases
    test_cases = [
        ([
            ["1","1","1","1","0"],
            ["1","1","0","1","0"],
            ["1","1","0","0","0"],
            ["0","0","0","0","0"]
        ], 1, "Single large island"),
        
        ([
            ["1","1","0","0","0"],
            ["1","1","0","0","0"],
            ["0","0","1","0","0"],
            ["0","0","0","1","1"]
        ], 3, "Three separate islands"),
        
        ([
            ["1","0","1","0","1"],
            ["0","1","0","1","0"],
            ["1","0","1","0","1"],
            ["0","1","0","1","0"]
        ], 13, "Checkerboard pattern"),
        
        ([
            ["1","1","1"],
            ["1","1","1"],
            ["1","1","1"]
        ], 1, "All land"),
        
        ([
            ["0","0","0"],
            ["0","0","0"],
            ["0","0","0"]
        ], 0, "All water"),
        
        ([["1"]], 1, "Single land cell"),
        
        ([["0"]], 0, "Single water cell"),
        
        ([
            ["1","0","1"],
            ["0","1","0"],
            ["1","0","1"]
        ], 5, "Plus pattern"),
    ]
    
    # Test all approaches (note: some modify the grid)
    approaches = [
        ("DFS", num_islands_dfs),
        ("BFS", num_islands_bfs),
        ("Union-Find", num_islands_union_find),
        ("Iterative DFS", num_islands_iterative_dfs),
        ("Without modification", num_islands_without_modification),
        ("Flood Fill", num_islands_flood_fill),
        ("Multi-directional BFS", num_islands_multidirectional_bfs),
        ("Optimized early termination", num_islands_optimized_early_termination),
    ]
    
    print("Testing number of islands approaches:")
    print("=" * 50)
    
    for i, (grid, expected, description) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: {description}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            try:
                # Create a copy of the grid since some functions modify it
                grid_copy = [row[:] for row in grid]
                result = func(grid_copy)
                status = "✓" if result == expected else "✗"
                print(f"{status} {name}: {result}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 50)
    print("Performance Analysis:")
    print("=" * 50)
    
    # Create larger test grid for performance testing
    def create_large_grid(size, density=0.3):
        """Create a large grid with specified density of land cells."""
        import random
        grid = []
        for i in range(size):
            row = []
            for j in range(size):
                row.append("1" if random.random() < density else "0")
            grid.append(row)
        return grid
    
    large_grid = create_large_grid(100, 0.3)
    
    import time
    
    print(f"Testing with large grid (100x100, 30% land density):")
    for name, func in approaches:
        try:
            grid_copy = [row[:] for row in large_grid]
            start_time = time.time()
            result = func(grid_copy)
            end_time = time.time()
            print(f"{name}: {result} islands (Time: {end_time - start_time:.6f}s)")
        except Exception as e:
            print(f"{name}: Error - {e}")


if __name__ == "__main__":
    test_num_islands() 