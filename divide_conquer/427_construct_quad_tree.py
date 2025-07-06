"""
427. Construct Quad Tree

Problem:
Given a n * n matrix grid of 0's and 1's only, we want to represent the grid with a Quad Tree.

Return the root of the Quad Tree representing the grid.

A Quad Tree is a tree data structure in which each internal node has exactly four children. 
Besides, each node has two attributes:
- val: True if the node represents a grid of 1's or False if the node represents a grid of 0's.
- isLeaf: True if the node is leaf node on the tree or False if the node has four children.

Example 1:
Input: grid = [[0,1],[1,0]]
Output: [[0,1],[1,0],[1,1],[1,1],[1,0]]

Example 2:
Input: grid = [[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0]]
Output: [[0,1],[1,1],[0,1],[1,1],[1,0],null,null,null,null,[1,0],[1,0],[1,1],[1,1]]

Time Complexity: O(n^2 * log n) where n is the side length
Space Complexity: O(n^2) for the tree structure
"""


class Node:
    """Quad Tree Node definition."""
    def __init__(self, val, isLeaf, topLeft=None, topRight=None, bottomLeft=None, bottomRight=None):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight


def construct_quad_tree_recursive(grid):
    """
    Recursive divide and conquer approach.
    
    Time Complexity: O(n^2 * log n)
    Space Complexity: O(n^2) for tree + O(log n) for recursion stack
    
    Algorithm:
    1. Check if current region is uniform (all 0s or all 1s)
    2. If uniform, create leaf node
    3. If not uniform, divide into 4 quadrants and recurse
    4. Create internal node with 4 children
    """
    def is_uniform(r1, c1, r2, c2):
        """Check if region is uniform (all same value)."""
        val = grid[r1][c1]
        for r in range(r1, r2):
            for c in range(c1, c2):
                if grid[r][c] != val:
                    return False, -1
        return True, val
    
    def construct(r1, c1, r2, c2):
        """Construct quad tree for region [r1,c1) to [r2,c2)."""
        uniform, val = is_uniform(r1, c1, r2, c2)
        
        if uniform:
            # Create leaf node
            return Node(val == 1, True)
        
        # Divide into 4 quadrants
        mid_r = (r1 + r2) // 2
        mid_c = (c1 + c2) // 2
        
        top_left = construct(r1, c1, mid_r, mid_c)
        top_right = construct(r1, mid_c, mid_r, c2)
        bottom_left = construct(mid_r, c1, r2, mid_c)
        bottom_right = construct(mid_r, mid_c, r2, c2)
        
        return Node(False, False, top_left, top_right, bottom_left, bottom_right)
    
    n = len(grid)
    return construct(0, 0, n, n)


def construct_quad_tree_optimized(grid):
    """
    Optimized with early uniform detection using prefix sums.
    
    Time Complexity: O(n^2 * log n) but faster in practice
    Space Complexity: O(n^2)
    
    Algorithm:
    1. Precompute prefix sums for O(1) region sum queries
    2. Use prefix sums to quickly check if region is uniform
    3. If sum equals 0, all zeros; if sum equals area, all ones
    4. Otherwise, divide and conquer
    """
    n = len(grid)
    
    # Build prefix sum array
    prefix = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            prefix[i][j] = grid[i-1][j-1] + prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1]
    
    def get_sum(r1, c1, r2, c2):
        """Get sum of region [r1,c1) to [r2,c2) using prefix sums."""
        return prefix[r2][c2] - prefix[r1][c2] - prefix[r2][c1] + prefix[r1][c1]
    
    def construct(r1, c1, r2, c2):
        area = (r2 - r1) * (c2 - c1)
        region_sum = get_sum(r1, c1, r2, c2)
        
        if region_sum == 0:
            # All zeros
            return Node(False, True)
        elif region_sum == area:
            # All ones
            return Node(True, True)
        else:
            # Mixed, need to divide
            mid_r = (r1 + r2) // 2
            mid_c = (c1 + c2) // 2
            
            top_left = construct(r1, c1, mid_r, mid_c)
            top_right = construct(r1, mid_c, mid_r, c2)
            bottom_left = construct(mid_r, c1, r2, mid_c)
            bottom_right = construct(mid_r, mid_c, r2, c2)
            
            return Node(False, False, top_left, top_right, bottom_left, bottom_right)
    
    return construct(0, 0, n, n)


def construct_quad_tree_iterative(grid):
    """
    Iterative approach using stack.
    
    Time Complexity: O(n^2 * log n)
    Space Complexity: O(n^2)
    
    Algorithm:
    1. Use stack to simulate recursion
    2. Process regions from stack
    3. If uniform, create leaf; otherwise add subregions to stack
    """
    n = len(grid)
    
    def is_uniform(r1, c1, r2, c2):
        val = grid[r1][c1]
        for r in range(r1, r2):
            for c in range(c1, c2):
                if grid[r][c] != val:
                    return False, -1
        return True, val
    
    # Stack stores (node_ref, r1, c1, r2, c2)
    root = Node(False, False)
    stack = [(root, 0, 0, n, n)]
    
    while stack:
        node, r1, c1, r2, c2 = stack.pop()
        
        uniform, val = is_uniform(r1, c1, r2, c2)
        
        if uniform:
            node.val = val == 1
            node.isLeaf = True
        else:
            node.isLeaf = False
            mid_r = (r1 + r2) // 2
            mid_c = (c1 + c2) // 2
            
            # Create children
            node.topLeft = Node(False, False)
            node.topRight = Node(False, False)
            node.bottomLeft = Node(False, False)
            node.bottomRight = Node(False, False)
            
            # Add children to stack
            stack.append((node.topLeft, r1, c1, mid_r, mid_c))
            stack.append((node.topRight, r1, mid_c, mid_r, c2))
            stack.append((node.bottomLeft, mid_r, c1, r2, mid_c))
            stack.append((node.bottomRight, mid_r, mid_c, r2, c2))
    
    return root


def construct_quad_tree_memorized(grid):
    """
    Approach with memoization of uniform regions.
    
    Time Complexity: O(n^2 * log n)
    Space Complexity: O(n^2)
    
    Algorithm:
    1. Memoize results of uniform checks
    2. Avoid recomputing uniform status for same regions
    3. Particularly useful when there are many repeated patterns
    """
    n = len(grid)
    memo = {}
    
    def is_uniform(r1, c1, r2, c2):
        key = (r1, c1, r2, c2)
        if key in memo:
            return memo[key]
        
        val = grid[r1][c1]
        for r in range(r1, r2):
            for c in range(c1, c2):
                if grid[r][c] != val:
                    memo[key] = (False, -1)
                    return False, -1
        
        memo[key] = (True, val)
        return True, val
    
    def construct(r1, c1, r2, c2):
        uniform, val = is_uniform(r1, c1, r2, c2)
        
        if uniform:
            return Node(val == 1, True)
        
        mid_r = (r1 + r2) // 2
        mid_c = (c1 + c2) // 2
        
        top_left = construct(r1, c1, mid_r, mid_c)
        top_right = construct(r1, mid_c, mid_r, c2)
        bottom_left = construct(mid_r, c1, r2, mid_c)
        bottom_right = construct(mid_r, mid_c, r2, c2)
        
        return Node(False, False, top_left, top_right, bottom_left, bottom_right)
    
    return construct(0, 0, n, n)


def quad_tree_to_list(root):
    """Convert quad tree to list representation for testing."""
    if not root:
        return []
    
    result = []
    queue = [root]
    
    while queue:
        node = queue.pop(0)
        if node:
            result.append([int(node.val), int(node.isLeaf)])
            if not node.isLeaf:
                queue.extend([node.topLeft, node.topRight, node.bottomLeft, node.bottomRight])
            else:
                # For leaf nodes, add null children
                queue.extend([None, None, None, None])
        else:
            result.append(None)
    
    # Remove trailing nulls
    while result and result[-1] is None:
        result.pop()
    
    return result


def validate_quad_tree(root, grid, r1=0, c1=0, r2=None, c2=None):
    """Validate that quad tree correctly represents the grid."""
    if r2 is None:
        r2 = len(grid)
    if c2 is None:
        c2 = len(grid[0])
    
    if not root:
        return False
    
    if root.isLeaf:
        # Check if all cells in region have same value as node
        for r in range(r1, r2):
            for c in range(c1, c2):
                if (grid[r][c] == 1) != root.val:
                    return False
        return True
    else:
        # Check children
        mid_r = (r1 + r2) // 2
        mid_c = (c1 + c2) // 2
        
        return (validate_quad_tree(root.topLeft, grid, r1, c1, mid_r, mid_c) and
                validate_quad_tree(root.topRight, grid, r1, mid_c, mid_r, c2) and
                validate_quad_tree(root.bottomLeft, grid, mid_r, c1, r2, mid_c) and
                validate_quad_tree(root.bottomRight, grid, mid_r, mid_c, r2, c2))


def test_construct_quad_tree():
    """Test all implementations with various test cases."""
    
    test_cases = [
        [[0,1],[1,0]],
        [[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0]],
        [[1,1],[1,1]],
        [[0,0],[0,0]],
        [[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]]
    ]
    
    implementations = [
        ("Recursive", construct_quad_tree_recursive),
        ("Optimized (prefix)", construct_quad_tree_optimized),
        ("Iterative", construct_quad_tree_iterative),
        ("Memoized", construct_quad_tree_memorized)
    ]
    
    print("Testing Construct Quad Tree...")
    
    for i, grid in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: {len(grid)}x{len(grid[0])} grid")
        print(f"Grid: {grid}")
        
        for impl_name, impl_func in implementations:
            result_tree = impl_func(grid)
            
            # Validate the tree
            is_valid = validate_quad_tree(result_tree, grid)
            
            print(f"{impl_name:20} | Valid: {'✓' if is_valid else '✗'}")
            
            # Show tree structure for small grids
            if len(grid) <= 4:
                tree_list = quad_tree_to_list(result_tree)
                print(f"                       Tree: {tree_list}")


if __name__ == "__main__":
    test_construct_quad_tree() 