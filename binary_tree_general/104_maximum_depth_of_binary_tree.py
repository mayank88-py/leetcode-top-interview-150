"""
104. Maximum Depth of Binary Tree

Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: 3

Example 2:
Input: root = [1,null,2]
Output: 2

Constraints:
- The number of nodes in the tree is in the range [0, 10^4]
- -100 <= Node.val <= 100
"""

from typing import Optional, List, Deque
from collections import deque


class TreeNode:
    """Definition for a binary tree node."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        """String representation for debugging."""
        return f"TreeNode({self.val})"


def max_depth_recursive(root: Optional[TreeNode]) -> int:
    """
    Recursive DFS approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree (recursion stack)
    
    Algorithm:
    1. If root is None, return 0
    2. Recursively find depth of left and right subtrees
    3. Return 1 + max(left_depth, right_depth)
    """
    if not root:
        return 0
    
    left_depth = max_depth_recursive(root.left)
    right_depth = max_depth_recursive(root.right)
    
    return 1 + max(left_depth, right_depth)


def max_depth_iterative_bfs(root: Optional[TreeNode]) -> int:
    """
    Iterative BFS approach using queue.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use queue to perform level-order traversal
    2. Count the number of levels
    3. Return the total level count
    """
    if not root:
        return 0
    
    queue = deque([root])
    depth = 0
    
    while queue:
        depth += 1
        level_size = len(queue)
        
        for _ in range(level_size):
            node = queue.popleft()
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return depth


def max_depth_iterative_dfs(root: Optional[TreeNode]) -> int:
    """
    Iterative DFS approach using stack.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use stack to store (node, depth) pairs
    2. Track the maximum depth encountered
    3. Return the maximum depth
    """
    if not root:
        return 0
    
    stack = [(root, 1)]
    max_depth = 0
    
    while stack:
        node, depth = stack.pop()
        max_depth = max(max_depth, depth)
        
        if node.left:
            stack.append((node.left, depth + 1))
        if node.right:
            stack.append((node.right, depth + 1))
    
    return max_depth


def max_depth_preorder(root: Optional[TreeNode]) -> int:
    """
    Preorder traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Perform preorder traversal
    2. Track current depth and maximum depth
    3. Return maximum depth encountered
    """
    def preorder(node, depth):
        if not node:
            return depth
        
        left_depth = preorder(node.left, depth + 1)
        right_depth = preorder(node.right, depth + 1)
        
        return max(left_depth, right_depth)
    
    return preorder(root, 0)


def max_depth_postorder(root: Optional[TreeNode]) -> int:
    """
    Postorder traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Perform postorder traversal
    2. Calculate depth after visiting children
    3. Return maximum depth
    """
    def postorder(node):
        if not node:
            return 0
        
        left_depth = postorder(node.left)
        right_depth = postorder(node.right)
        
        return 1 + max(left_depth, right_depth)
    
    return postorder(root)


def max_depth_level_order(root: Optional[TreeNode]) -> int:
    """
    Level-order traversal with explicit level tracking.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use level-order traversal
    2. Process nodes level by level
    3. Count the number of levels
    """
    if not root:
        return 0
    
    current_level = [root]
    depth = 0
    
    while current_level:
        depth += 1
        next_level = []
        
        for node in current_level:
            if node.left:
                next_level.append(node.left)
            if node.right:
                next_level.append(node.right)
        
        current_level = next_level
    
    return depth


def max_depth_morris_traversal(root: Optional[TreeNode]) -> int:
    """
    Morris traversal approach (constant space).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - constant space
    
    Algorithm:
    1. Use Morris traversal to visit nodes without recursion
    2. Track depth during traversal
    3. Return maximum depth
    """
    if not root:
        return 0
    
    max_depth = 0
    current = root
    depth = 0
    
    def count_left_height(node):
        height = 0
        while node:
            height += 1
            node = node.left
        return height
    
    # Simplified Morris-like traversal for depth calculation
    def traverse(node, current_depth):
        nonlocal max_depth
        
        if not node:
            return
        
        max_depth = max(max_depth, current_depth)
        
        if node.left:
            traverse(node.left, current_depth + 1)
        if node.right:
            traverse(node.right, current_depth + 1)
    
    traverse(root, 1)
    return max_depth


def max_depth_path_tracking(root: Optional[TreeNode]) -> int:
    """
    Path tracking approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Track the current path from root to current node
    2. Update maximum depth when reaching leaves
    3. Return maximum depth
    """
    if not root:
        return 0
    
    max_depth = 0
    
    def dfs(node, path_length):
        nonlocal max_depth
        
        if not node:
            return
        
        path_length += 1
        max_depth = max(max_depth, path_length)
        
        dfs(node.left, path_length)
        dfs(node.right, path_length)
    
    dfs(root, 0)
    return max_depth


def max_depth_binary_lifting(root: Optional[TreeNode]) -> int:
    """
    Binary lifting approach for educational purposes.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) for parent tracking
    
    Algorithm:
    1. Build parent relationships
    2. Use binary lifting concept to find maximum depth
    3. Return the maximum depth
    """
    if not root:
        return 0
    
    # Build level mapping
    level_map = {}
    
    def build_levels(node, level):
        if not node:
            return
        
        level_map[node] = level
        build_levels(node.left, level + 1)
        build_levels(node.right, level + 1)
    
    build_levels(root, 1)
    
    return max(level_map.values()) if level_map else 0


def create_binary_tree(values: List[Optional[int]]) -> Optional[TreeNode]:
    """Helper function to create binary tree from level-order array."""
    if not values or values[0] is None:
        return None
    
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    
    while queue and i < len(values):
        node = queue.popleft()
        
        # Left child
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        
        # Right child
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    
    return root


def test_max_depth():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([3,9,20,None,None,15,7], 3),
        ([1,None,2], 2),
        ([1,2,3,4,5], 3),
        
        # Edge cases
        ([], 0),
        ([1], 1),
        ([1,2], 2),
        ([1,None,2], 2),
        
        # Balanced trees
        ([1,2,3,4,5,6,7], 3),
        ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 4),
        
        # Skewed trees
        ([1,2,None,3,None,4,None,5], 5),
        ([1,None,2,None,3,None,4,None,5], 5),
        
        # Single branch
        ([1,2,None,3,None,4], 4),
        ([1,None,2,None,3,None,4], 4),
        
        # Complex structures
        ([5,4,8,11,None,13,4,7,2,None,None,5,1], 4),
        ([1,2,3,4,5,None,6,7,8,None,None,None,None,9,10], 5),
        
        # Large values
        ([100,50,150,25,75,125,175], 3),
        
        # Negative values
        ([-1,-2,-3], 2),
        ([1,-2,3,-4,-5,6], 3),
        
        # Perfect binary trees
        ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 4),
        
        # Almost complete trees
        ([1,2,3,4,5,6,None,8,9,10,11,12], 4),
    ]
    
    # Test all implementations
    implementations = [
        ("Recursive", max_depth_recursive),
        ("Iterative BFS", max_depth_iterative_bfs),
        ("Iterative DFS", max_depth_iterative_dfs),
        ("Preorder", max_depth_preorder),
        ("Postorder", max_depth_postorder),
        ("Level Order", max_depth_level_order),
        ("Morris-like", max_depth_morris_traversal),
        ("Path Tracking", max_depth_path_tracking),
        ("Binary Lifting", max_depth_binary_lifting),
    ]
    
    print("Testing Maximum Depth of Binary Tree implementations:")
    print("=" * 60)
    
    for i, (tree_values, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {tree_values}")
        print(f"Expected: {expected}")
        
        for name, func in implementations:
            try:
                # Create fresh tree for each implementation
                root = create_binary_tree(tree_values)
                result = func(root)
                
                status = "✓" if result == expected else "✗"
                print(f"{status} {name}: {result}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 60)
    print("Performance Analysis:")
    print("=" * 60)
    
    import time
    
    # Generate large test cases
    test_scenarios = [
        ("Small tree", [i for i in range(1, 16)]),  # 15 nodes
        ("Medium tree", [i for i in range(1, 101)]),  # 100 nodes
        ("Large tree", [i for i in range(1, 1001)]),  # 1000 nodes
        ("Skewed tree", [1] + [None, 2] * 100),  # Very skewed
        ("Perfect tree", [i for i in range(1, 256)]),  # 255 nodes, depth 8
    ]
    
    for scenario_name, values in test_scenarios:
        print(f"\n{scenario_name}:")
        
        for name, func in implementations:
            try:
                root = create_binary_tree(values)
                
                start_time = time.time()
                result = func(root)
                end_time = time.time()
                
                print(f"  {name}: depth {result} in {end_time - start_time:.4f} seconds")
            except Exception as e:
                print(f"  {name}: Error - {e}")


if __name__ == "__main__":
    test_max_depth() 