"""
LeetCode 543: Diameter of Binary Tree

Given the root of a binary tree, return the length of the diameter of the tree.

The diameter of a binary tree is the length of the longest path between any two nodes 
in a tree. This path may or may not pass through the root.

The length of a path between two nodes is represented by the number of edges between them.

Example 1:
Input: root = [1,2,3,4,5]
Output: 3

Example 2:
Input: root = [1,2]
Output: 1

Constraints:
- The number of nodes in the tree is in the range [1, 10^4].
- -100 <= Node.val <= 100
"""

from typing import Optional, List, Tuple
from collections import deque, defaultdict


class TreeNode:
    """Binary tree node definition."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def diameter_recursive_naive(root: Optional[TreeNode]) -> int:
    """
    Naive recursive approach.
    
    Time Complexity: O(n²)
    Space Complexity: O(h) where h is tree height
    
    Algorithm:
    1. For each node, calculate max diameter passing through it
    2. Diameter = left_height + right_height
    3. Return maximum of all diameters
    """
    if not root:
        return 0
    
    def height(node):
        """Calculate height of tree."""
        if not node:
            return 0
        return 1 + max(height(node.left), height(node.right))
    
    def diameter_at_node(node):
        """Calculate diameter passing through given node."""
        if not node:
            return 0
        
        left_height = height(node.left)
        right_height = height(node.right)
        
        return left_height + right_height
    
    def max_diameter(node):
        """Find maximum diameter in tree."""
        if not node:
            return 0
        
        # Diameter passing through current node
        current_diameter = diameter_at_node(node)
        
        # Maximum diameter in left and right subtrees
        left_diameter = max_diameter(node.left)
        right_diameter = max_diameter(node.right)
        
        return max(current_diameter, left_diameter, right_diameter)
    
    return max_diameter(root)


def diameter_optimized(root: Optional[TreeNode]) -> int:
    """
    Optimized recursive approach.
    
    Time Complexity: O(n)
    Space Complexity: O(h)
    
    Algorithm:
    1. Calculate height and diameter in single traversal
    2. For each node, update global maximum diameter
    3. Return height to parent, maintain diameter globally
    """
    max_diameter = [0]
    
    def height(node):
        """Calculate height and update diameter."""
        if not node:
            return 0
        
        left_height = height(node.left)
        right_height = height(node.right)
        
        # Update maximum diameter
        max_diameter[0] = max(max_diameter[0], left_height + right_height)
        
        # Return height of current subtree
        return 1 + max(left_height, right_height)
    
    height(root)
    return max_diameter[0]


def diameter_with_path(root: Optional[TreeNode]) -> Tuple[int, List[int]]:
    """
    Find diameter and the actual path.
    
    Time Complexity: O(n)
    Space Complexity: O(h)
    
    Algorithm:
    1. Track the path that gives maximum diameter
    2. Store path endpoints when diameter is updated
    3. Return both diameter and path
    """
    max_diameter = [0]
    diameter_path = [None, None]  # [start_node, end_node]
    
    def dfs(node):
        """Returns (height, deepest_node)."""
        if not node:
            return 0, None
        
        left_height, left_deepest = dfs(node.left)
        right_height, right_deepest = dfs(node.right)
        
        # Check if current diameter is maximum
        current_diameter = left_height + right_height
        if current_diameter > max_diameter[0]:
            max_diameter[0] = current_diameter
            diameter_path[0] = left_deepest
            diameter_path[1] = right_deepest
        
        # Return height and deepest node
        if left_height > right_height:
            return left_height + 1, left_deepest if left_deepest else node
        else:
            return right_height + 1, right_deepest if right_deepest else node
    
    dfs(root)
    return max_diameter[0], [diameter_path[0].val if diameter_path[0] else None,
                            diameter_path[1].val if diameter_path[1] else None]


def diameter_iterative_dfs(root: Optional[TreeNode]) -> int:
    """
    Iterative DFS approach.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Algorithm:
    1. Use stack for DFS traversal
    2. Calculate heights iteratively
    3. Track maximum diameter during traversal
    """
    if not root:
        return 0
    
    # First pass: calculate heights
    heights = {}
    stack = []
    visited = set()
    
    def calculate_heights():
        stack.append(root)
        
        while stack:
            node = stack[-1]
            
            if node in visited:
                # Post-order processing
                stack.pop()
                left_height = heights.get(node.left, 0)
                right_height = heights.get(node.right, 0)
                heights[node] = 1 + max(left_height, right_height)
            else:
                visited.add(node)
                if node.right:
                    stack.append(node.right)
                if node.left:
                    stack.append(node.left)
    
    calculate_heights()
    
    # Second pass: calculate maximum diameter
    max_diameter = 0
    stack = [root]
    visited = set()
    
    while stack:
        node = stack.pop()
        
        if node not in visited:
            visited.add(node)
            
            left_height = heights.get(node.left, 0)
            right_height = heights.get(node.right, 0)
            diameter = left_height + right_height
            max_diameter = max(max_diameter, diameter)
            
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
    
    return max_diameter


def diameter_level_order(root: Optional[TreeNode]) -> int:
    """
    Level order approach with bottom-up processing.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Algorithm:
    1. Use BFS to process nodes level by level
    2. Calculate heights bottom-up
    3. Track diameter during calculation
    """
    if not root:
        return 0
    
    # BFS to collect all nodes and their levels
    queue = deque([(root, 0)])
    levels = defaultdict(list)
    max_level = 0
    
    while queue:
        node, level = queue.popleft()
        levels[level].append(node)
        max_level = max(max_level, level)
        
        if node.left:
            queue.append((node.left, level + 1))
        if node.right:
            queue.append((node.right, level + 1))
    
    # Bottom-up processing
    heights = {}
    max_diameter = 0
    
    for level in range(max_level, -1, -1):
        for node in levels[level]:
            left_height = heights.get(node.left, 0)
            right_height = heights.get(node.right, 0)
            
            heights[node] = 1 + max(left_height, right_height)
            max_diameter = max(max_diameter, left_height + right_height)
    
    return max_diameter


def diameter_morris_inspired(root: Optional[TreeNode]) -> int:
    """
    Morris traversal inspired approach.
    
    Time Complexity: O(n)
    Space Complexity: O(n) for height storage
    
    Algorithm:
    1. Use Morris-like traversal
    2. Calculate heights without recursion
    3. Track diameter during traversal
    """
    if not root:
        return 0
    
    # Use iterative post-order traversal
    stack = []
    last_visited = None
    current = root
    heights = {}
    max_diameter = 0
    
    while stack or current:
        if current:
            stack.append(current)
            current = current.left
        else:
            peek_node = stack[-1]
            
            if peek_node.right and last_visited != peek_node.right:
                current = peek_node.right
            else:
                # Process node in post-order
                left_height = heights.get(peek_node.left, 0)
                right_height = heights.get(peek_node.right, 0)
                
                heights[peek_node] = 1 + max(left_height, right_height)
                max_diameter = max(max_diameter, left_height + right_height)
                
                stack.pop()
                last_visited = peek_node
    
    return max_diameter


def diameter_memoization(root: Optional[TreeNode]) -> int:
    """
    Memoization approach.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Algorithm:
    1. Memoize height calculations
    2. Avoid redundant computations
    3. Cache results for repeated subtrees
    """
    height_memo = {}
    max_diameter = [0]
    
    def height(node):
        """Calculate height with memoization."""
        if not node:
            return 0
        
        if node in height_memo:
            return height_memo[node]
        
        left_height = height(node.left)
        right_height = height(node.right)
        
        # Update diameter
        max_diameter[0] = max(max_diameter[0], left_height + right_height)
        
        # Memoize and return height
        height_memo[node] = 1 + max(left_height, right_height)
        return height_memo[node]
    
    height(root)
    return max_diameter[0]


def diameter_two_pass(root: Optional[TreeNode]) -> int:
    """
    Two-pass approach: explicit height calculation then diameter.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Algorithm:
    1. First pass: calculate all heights
    2. Second pass: calculate diameter using heights
    3. More explicit separation of concerns
    """
    if not root:
        return 0
    
    heights = {}
    
    def calculate_heights(node):
        """First pass: calculate heights."""
        if not node:
            return 0
        
        left_height = calculate_heights(node.left)
        right_height = calculate_heights(node.right)
        
        heights[node] = 1 + max(left_height, right_height)
        return heights[node]
    
    def calculate_diameter(node):
        """Second pass: calculate diameter."""
        if not node:
            return 0
        
        left_height = heights.get(node.left, 0)
        right_height = heights.get(node.right, 0)
        current_diameter = left_height + right_height
        
        left_diameter = calculate_diameter(node.left)
        right_diameter = calculate_diameter(node.right)
        
        return max(current_diameter, left_diameter, right_diameter)
    
    calculate_heights(root)
    return calculate_diameter(root)


def diameter_bottom_up(root: Optional[TreeNode]) -> int:
    """
    Bottom-up approach with explicit stack.
    
    Time Complexity: O(n)
    Space Complexity: O(h)
    
    Algorithm:
    1. Use explicit stack for bottom-up processing
    2. Process children before parents
    3. Calculate diameter during bottom-up traversal
    """
    if not root:
        return 0
    
    # Post-order traversal to process bottom-up
    stack = []
    last_visited = None
    current = root
    heights = {}
    max_diameter = 0
    
    while stack or current:
        if current:
            stack.append(current)
            current = current.left
        else:
            peek = stack[-1]
            
            if peek.right and last_visited != peek.right:
                current = peek.right
            else:
                # Process current node
                left_height = heights.get(peek.left, 0)
                right_height = heights.get(peek.right, 0)
                
                heights[peek] = 1 + max(left_height, right_height)
                max_diameter = max(max_diameter, left_height + right_height)
                
                stack.pop()
                last_visited = peek
    
    return max_diameter


def diameter_functional(root: Optional[TreeNode]) -> int:
    """
    Functional programming approach.
    
    Time Complexity: O(n)
    Space Complexity: O(h)
    
    Algorithm:
    1. Use functional programming concepts
    2. Pure functions for height and diameter
    3. Immutable state management
    """
    def solve(node):
        """Returns (height, max_diameter_in_subtree)."""
        if not node:
            return 0, 0
        
        left_height, left_diameter = solve(node.left)
        right_height, right_diameter = solve(node.right)
        
        current_height = 1 + max(left_height, right_height)
        current_diameter = left_height + right_height
        max_diameter = max(current_diameter, left_diameter, right_diameter)
        
        return current_height, max_diameter
    
    _, diameter = solve(root)
    return diameter


def build_test_tree(nodes: List[Optional[int]]) -> Optional[TreeNode]:
    """Build tree from list representation."""
    if not nodes or nodes[0] is None:
        return None
    
    root = TreeNode(nodes[0])
    queue = deque([root])
    i = 1
    
    while queue and i < len(nodes):
        node = queue.popleft()
        
        if i < len(nodes) and nodes[i] is not None:
            node.left = TreeNode(nodes[i])
            queue.append(node.left)
        i += 1
        
        if i < len(nodes) and nodes[i] is not None:
            node.right = TreeNode(nodes[i])
            queue.append(node.right)
        i += 1
    
    return root


def test_diameter():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([1], 0),
        ([1, 2], 1),
        ([1, None, 2], 1),
        ([1, 2, 3], 2),
        
        # Given examples
        ([1, 2, 3, 4, 5], 3),  # Path: 4->2->1->3 or 5->2->1->3
        
        # Left skewed
        ([1, 2, None, 3, None, None, None, 4], 3),
        
        # Right skewed
        ([1, None, 2, None, 3, None, 4], 3),
        
        # Balanced trees
        ([1, 2, 3, 4, 5, 6, 7], 4),  # Many possible longest paths
        
        # Diameter not through root
        ([1, 2, 3, 4, 5, None, None, 6, 7], 4),  # Path: 6->4->2->5 or 7->4->2->5
        
        # Complex cases
        ([1, 2, 3, 4, None, None, 5, 6, 7, None, None, None, None, None, 8], 5),
        
        # Single child patterns
        ([1, 2, None, 3, 4, None, None, 5], 3),
        ([1, None, 2, 3, None, 4], 2),
        
        # Negative values (should not affect diameter calculation)
        ([-1, -2, -3], 2),
        ([1, -2, 3, -4, -5], 3),
        
        # Large values
        ([100, 50, 150, 25, 75, 125, 175], 4),
        
        # Deep tree (complete binary tree)
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 6),
        
        # Unbalanced tree
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, None, 11, None, None, None, None, 16], 5),
        
        # Path through different levels
        ([5, 4, 6, 3, None, None, 7, 2, None, None, None, None, None, None, 8], 4),
    ]
    
    # Test all implementations
    implementations = [
        ("Recursive Naive", diameter_recursive_naive),
        ("Optimized", diameter_optimized),
        ("Iterative DFS", diameter_iterative_dfs),
        ("Level Order", diameter_level_order),
        ("Morris Inspired", diameter_morris_inspired),
        ("Memoization", diameter_memoization),
        ("Two Pass", diameter_two_pass),
        ("Bottom Up", diameter_bottom_up),
        ("Functional", diameter_functional),
    ]
    
    print("Testing Diameter of Binary Tree implementations:")
    print("=" * 75)
    
    for i, (nodes, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Tree:     {nodes}")
        print(f"Expected: {expected}")
        
        tree = build_test_tree(nodes)
        
        for name, func in implementations:
            try:
                result = func(tree)
                status = "✓" if result == expected else "✗"
                print(f"{status} {name}: {result}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Test diameter with path
    print(f"\n{'='*75}")
    print("Testing Diameter with Path:")
    print("="*75)
    
    test_tree = build_test_tree([1, 2, 3, 4, 5])
    diameter, path = diameter_with_path(test_tree)
    print(f"Tree: [1,2,3,4,5]")
    print(f"Diameter: {diameter}")
    print(f"Path endpoints: {path}")
    
    # Performance comparison
    print("\n" + "=" * 75)
    print("Performance Analysis:")
    print("=" * 75)
    
    import time
    
    # Generate large test cases
    def generate_test_tree(height):
        """Generate a complete binary tree of given height."""
        nodes = []
        for i in range(2**height - 1):
            nodes.append(i + 1)
        return nodes
    
    def generate_skewed_tree(size):
        """Generate a left-skewed tree."""
        nodes = [1]
        current = 1
        for i in range(size - 1):
            nodes.extend([current + 1, None])
            current += 1
        return nodes
    
    test_scenarios = [
        ("Small complete", generate_test_tree(4)),      # 15 nodes
        ("Medium complete", generate_test_tree(6)),     # 63 nodes
        ("Large complete", generate_test_tree(8)),      # 255 nodes
        ("Small skewed", generate_skewed_tree(20)),     # 20 nodes
        ("Medium skewed", generate_skewed_tree(50)),    # 50 nodes
        ("Large skewed", generate_skewed_tree(100)),    # 100 nodes
    ]
    
    for scenario_name, nodes in test_scenarios:
        print(f"\n{scenario_name} ({len([n for n in nodes if n is not None])} nodes):")
        tree = build_test_tree(nodes)
        
        # Test efficient implementations for large cases
        test_implementations = implementations[1:6] if len(nodes) > 100 else implementations
        
        for name, func in test_implementations:
            try:
                start_time = time.time()
                result = func(tree)
                end_time = time.time()
                
                elapsed = (end_time - start_time) * 1000  # Convert to milliseconds
                print(f"  {name}: {elapsed:.2f} ms (diameter: {result})")
            except Exception as e:
                print(f"  {name}: Error - {e}")
    
    print(f"\n{'='*75}")
    print("Algorithm Complexity Analysis:")
    print("="*75)
    print("1. Recursive Naive:    O(n²) time, O(h) space")
    print("2. Optimized:          O(n) time, O(h) space")
    print("3. Iterative DFS:      O(n) time, O(n) space")
    print("4. Level Order:        O(n) time, O(n) space")
    print("5. Morris Inspired:    O(n) time, O(n) space")
    print("6. Memoization:        O(n) time, O(n) space")
    print("7. Two Pass:           O(n) time, O(n) space")
    print("8. Bottom Up:          O(n) time, O(h) space")
    print("9. Functional:         O(n) time, O(h) space")
    
    print(f"\n{'='*75}")
    print("Key Insights:")
    print("="*75)
    print("• Diameter is the longest path between any two nodes")
    print("• Path may or may not pass through root")
    print("• Optimized approach calculates height and diameter together")
    print("• For each node: diameter = left_height + right_height")
    print("• Global maximum diameter needs to be tracked")
    print("• Single traversal is sufficient for O(n) solution")


if __name__ == "__main__":
    test_diameter() 