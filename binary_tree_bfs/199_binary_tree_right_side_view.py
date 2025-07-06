"""
199. Binary Tree Right Side View

Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

Example 1:
Input: root = [1,2,3,null,5,null,4]
Output: [1,3,4]

Example 2:
Input: root = [1,null,3]
Output: [1,3]

Example 3:
Input: root = []
Output: []

Constraints:
- The number of nodes in the tree is in the range [0, 100]
- -100 <= Node.val <= 100
"""

from typing import Optional, List
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


def right_side_view_bfs_level_order(root: Optional[TreeNode]) -> List[int]:
    """
    BFS level order traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Perform level order traversal
    2. Take the last (rightmost) node from each level
    3. Add to result
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        
        for i in range(level_size):
            node = queue.popleft()
            
            # If this is the last node in the level (rightmost)
            if i == level_size - 1:
                result.append(node.val)
            
            # Add children for next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return result


def right_side_view_bfs_rightmost_only(root: Optional[TreeNode]) -> List[int]:
    """
    BFS approach storing only rightmost value per level.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Process each level completely
    2. Keep track of rightmost value seen in current level
    3. Add rightmost value to result after processing each level
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        rightmost_value = None
        
        for _ in range(level_size):
            node = queue.popleft()
            rightmost_value = node.val  # Keep updating to get rightmost
            
            # Add children for next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        if rightmost_value is not None:
            result.append(rightmost_value)
    
    return result


def right_side_view_dfs_recursive(root: Optional[TreeNode]) -> List[int]:
    """
    DFS recursive approach (preorder traversal, right first).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use DFS with level tracking
    2. Visit right subtree before left subtree
    3. Add first node encountered at each level
    """
    result = []
    
    def dfs(node, level):
        if not node:
            return
        
        # If this is the first node we see at this level
        if level == len(result):
            result.append(node.val)
        
        # Visit right first, then left
        dfs(node.right, level + 1)
        dfs(node.left, level + 1)
    
    dfs(root, 0)
    return result


def right_side_view_dfs_reverse_preorder(root: Optional[TreeNode]) -> List[int]:
    """
    DFS with reverse preorder (root, right, left).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use reverse preorder traversal
    2. Process right subtree before left
    3. Track levels and add first node seen at each level
    """
    if not root:
        return []
    
    result = []
    
    def reverse_preorder(node, level):
        if not node:
            return
        
        # Extend result if we've reached a new level
        if level >= len(result):
            result.append(node.val)
        
        # Process right subtree first
        reverse_preorder(node.right, level + 1)
        reverse_preorder(node.left, level + 1)
    
    reverse_preorder(root, 0)
    return result


def right_side_view_iterative_stack(root: Optional[TreeNode]) -> List[int]:
    """
    Iterative approach using stack with level tracking.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use stack to store (node, level) pairs
    2. Process nodes in reverse preorder
    3. Track levels and rightmost nodes
    """
    if not root:
        return []
    
    result = []
    stack = [(root, 0)]
    
    while stack:
        node, level = stack.pop()
        
        # If this is the first node we see at this level
        if level == len(result):
            result.append(node.val)
        
        # Push left first (processed later)
        if node.left:
            stack.append((node.left, level + 1))
        
        # Push right second (processed first)
        if node.right:
            stack.append((node.right, level + 1))
    
    return result


def right_side_view_bfs_deque_reverse(root: Optional[TreeNode]) -> List[int]:
    """
    BFS approach processing levels from right to left.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Process each level from right to left
    2. Take the first node encountered (rightmost)
    3. Use deque for efficient operations
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        
        # Process level from right to left conceptually
        level_nodes = []
        for _ in range(level_size):
            node = queue.popleft()
            level_nodes.append(node)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        # Take rightmost node (last in level_nodes)
        if level_nodes:
            result.append(level_nodes[-1].val)
    
    return result


def right_side_view_morris_inspired(root: Optional[TreeNode]) -> List[int]:
    """
    Morris-inspired approach (educational).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) ideally, but O(h) for level tracking
    
    Algorithm:
    1. Attempt Morris-like traversal
    2. Track levels during traversal
    3. Identify rightmost nodes
    
    Note: True Morris traversal is complex for this problem
    """
    # Morris traversal is complex for right side view
    # Fall back to DFS approach
    return right_side_view_dfs_recursive(root)


def right_side_view_level_map(root: Optional[TreeNode]) -> List[int]:
    """
    Level mapping approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Map each level to its rightmost node value
    2. Use DFS to populate the map
    3. Extract values in level order
    """
    if not root:
        return []
    
    level_map = {}
    max_level = [0]
    
    def dfs(node, level):
        if not node:
            return
        
        max_level[0] = max(max_level[0], level)
        
        # Always update level map (rightmost will be the last update)
        level_map[level] = node.val
        
        # Visit left first, then right (so right overwrites left)
        dfs(node.left, level + 1)
        dfs(node.right, level + 1)
    
    dfs(root, 0)
    
    # Extract values in level order
    return [level_map[i] for i in range(max_level[0] + 1)]


def right_side_view_two_pass(root: Optional[TreeNode]) -> List[int]:
    """
    Two-pass approach: first find levels, then find rightmost.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) to store all nodes with levels
    
    Algorithm:
    1. First pass: collect all nodes with their levels
    2. Second pass: find rightmost node at each level
    3. Build result from rightmost nodes
    """
    if not root:
        return []
    
    # First pass: collect all nodes with levels
    node_levels = []
    
    def collect_nodes(node, level):
        if not node:
            return
        
        node_levels.append((node.val, level))
        collect_nodes(node.left, level + 1)
        collect_nodes(node.right, level + 1)
    
    collect_nodes(root, 0)
    
    # Second pass: find rightmost node at each level
    level_rightmost = {}
    for val, level in node_levels:
        level_rightmost[level] = val  # Later nodes at same level overwrite
    
    # Build result
    max_level = max(level for _, level in node_levels)
    return [level_rightmost[i] for i in range(max_level + 1)]


def right_side_view_queue_with_levels(root: Optional[TreeNode]) -> List[int]:
    """
    Queue approach storing levels explicitly.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Store (node, level) pairs in queue
    2. Track current level being processed
    3. Update rightmost value for current level
    """
    if not root:
        return []
    
    result = []
    queue = deque([(root, 0)])
    current_level = 0
    current_rightmost = root.val
    
    while queue:
        node, level = queue.popleft()
        
        if level > current_level:
            # Moved to new level, save previous level's rightmost
            result.append(current_rightmost)
            current_level = level
            current_rightmost = node.val
        else:
            # Same level, update rightmost
            current_rightmost = node.val
        
        # Add children
        if node.left:
            queue.append((node.left, level + 1))
        if node.right:
            queue.append((node.right, level + 1))
    
    # Don't forget the last level
    result.append(current_rightmost)
    
    return result


def right_side_view_bfs_right_to_left(root: Optional[TreeNode]) -> List[int]:
    """
    BFS approach adding children right to left.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Add children to queue in right-to-left order
    2. First node at each level will be rightmost
    3. Take first node from each level
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        
        for i in range(level_size):
            node = queue.popleft()
            
            # First node in level is rightmost (due to right-first addition)
            if i == 0:
                result.append(node.val)
            
            # Add right child first, then left
            if node.right:
                queue.append(node.right)
            if node.left:
                queue.append(node.left)
    
    return result


def right_side_view_recursive_with_depth(root: Optional[TreeNode]) -> List[int]:
    """
    Recursive approach with explicit depth tracking.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Track depth explicitly during recursion
    2. Maintain maximum depth seen so far
    3. Add nodes when reaching new maximum depth
    """
    result = []
    max_depth = [0]
    
    def dfs_with_depth(node, depth):
        if not node:
            return
        
        # If this is a new maximum depth, this is rightmost at this level
        if depth > max_depth[0]:
            result.append(node.val)
            max_depth[0] = depth
        
        # Visit right first to ensure rightmost nodes are found first
        dfs_with_depth(node.right, depth + 1)
        dfs_with_depth(node.left, depth + 1)
    
    if root:
        dfs_with_depth(root, 1)
    
    return result


def right_side_view_iterative_level_by_level(root: Optional[TreeNode]) -> List[int]:
    """
    Iterative level-by-level processing.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Process one complete level at a time
    2. Keep track of all nodes at current level
    3. Extract rightmost node from each level
    """
    if not root:
        return []
    
    result = []
    current_level = [root]
    
    while current_level:
        # Rightmost node in current level
        result.append(current_level[-1].val)
        
        # Build next level
        next_level = []
        for node in current_level:
            if node.left:
                next_level.append(node.left)
            if node.right:
                next_level.append(node.right)
        
        current_level = next_level
    
    return result


def create_tree_from_list(values: List[Optional[int]]) -> Optional[TreeNode]:
    """Helper function to create tree from level-order list."""
    if not values or values[0] is None:
        return None
    
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    
    while queue and i < len(values):
        node = queue.popleft()
        
        # Add left child
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])  # type: ignore
            queue.append(node.left)
        i += 1
        
        # Add right child
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])  # type: ignore
            queue.append(node.right)
        i += 1
    
    return root


def test_right_side_view():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([1,2,3,None,5,None,4], [1,3,4]),
        ([1,None,3], [1,3]),
        ([], []),
        ([1], [1]),
        
        # Two nodes
        ([1,2], [1,2]),
        ([1,None,2], [1,2]),
        
        # Three nodes
        ([1,2,3], [1,3]),
        ([1,None,2,None,3], [1,2,3]),
        ([1,2,None,3], [1,2,3]),
        
        # Complete binary tree
        ([1,2,3,4,5,6,7], [1,3,7]),
        
        # Left heavy tree
        ([1,2,None,3,None,4,None,5], [1,2,3,4,5]),
        
        # Right heavy tree
        ([1,None,2,None,3,None,4,None,5], [1,2,3,4,5]),
        
        # Mixed structure
        ([1,2,3,4,None,None,5], [1,3,5]),
        ([1,2,3,None,4,5,None], [1,3,4]),
        
        # Large tree
        ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [1,3,7,15]),
        
        # Unbalanced
        ([1,2,3,4,5,None,6,7,8,None,9], [1,3,6,9]),
        
        # Only left children
        ([1,2,None,3,None,4], [1,2,3,4]),
        
        # Only right children
        ([1,None,2,None,3,None,4], [1,2,3,4]),
        
        # Alternating structure
        ([1,2,None,None,3,None,4], [1,2,3,4]),
        
        # Negative values
        ([-1,2,-3], [-1,-3]),
        ([1,-2,3], [1,3]),
        ([-1,-2,-3], [-1,-3]),
        
        # Zero values
        ([0,1,2], [0,2]),
        ([1,0,2], [1,2]),
        ([0], [0]),
        
        # Large values
        ([100,50,150,25,75,125,175], [100,150,175]),
        
        # Complex structure with gaps
        ([1,2,3,None,None,4,5,None,None,None,None,6,7], [1,3,5,7]),
        
        # Deep left branch
        ([1,2,None,3,None,4,None,5,None,6], [1,2,3,4,5,6]),
        
        # Wide tree
        ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], [1,3,7,15,17]),
        
        # Sparse tree
        ([1,None,2,3,None,4,None,5], [1,2,4,5]),
        
        # Same values
        ([1,1,1,1,1,1,1], [1,1,1]),
        
        # Missing right nodes
        ([1,2,3,4,5], [1,3,5]),
        
        # Missing left nodes
        ([1,2,3,None,None,6,7], [1,3,7]),
    ]
    
    # Test all implementations
    implementations = [
        ("BFS Level Order", right_side_view_bfs_level_order),
        ("BFS Rightmost Only", right_side_view_bfs_rightmost_only),
        ("DFS Recursive", right_side_view_dfs_recursive),
        ("DFS Reverse Preorder", right_side_view_dfs_reverse_preorder),
        ("Iterative Stack", right_side_view_iterative_stack),
        ("BFS Deque Reverse", right_side_view_bfs_deque_reverse),
        ("Morris Inspired", right_side_view_morris_inspired),
        ("Level Map", right_side_view_level_map),
        ("Two Pass", right_side_view_two_pass),
        ("Queue with Levels", right_side_view_queue_with_levels),
        ("BFS Right to Left", right_side_view_bfs_right_to_left),
        ("Recursive with Depth", right_side_view_recursive_with_depth),
        ("Iterative Level by Level", right_side_view_iterative_level_by_level),
    ]
    
    print("Testing Binary Tree Right Side View implementations:")
    print("=" * 60)
    
    for i, (tree_values, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Tree: {tree_values}")
        print(f"Expected: {expected}")
        
        for name, func in implementations:
            try:
                # Create fresh tree for each implementation
                root = create_tree_from_list(tree_values)
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
    
    def generate_balanced_tree(depth):
        """Generate a balanced binary tree."""
        if depth == 0:
            return []
        
        size = 2 ** depth - 1
        return list(range(1, size + 1))
    
    def generate_right_skewed_tree(size):
        """Generate a right-skewed tree."""
        result = []
        for i in range(1, size + 1):
            result.extend([i, None] if i < size else [i])
        return result
    
    test_scenarios = [
        ("Small balanced", generate_balanced_tree(4)),
        ("Medium balanced", generate_balanced_tree(6)),
        ("Large balanced", generate_balanced_tree(8)),
        ("Right skewed", generate_right_skewed_tree(50)),
        ("Complex tree", [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]),
    ]
    
    for scenario_name, tree_values in test_scenarios:
        print(f"\n{scenario_name}:")
        
        for name, func in implementations:
            try:
                root = create_tree_from_list(tree_values)
                
                start_time = time.time()
                result = func(root)
                end_time = time.time()
                
                print(f"  {name}: {result} in {end_time - start_time:.4f} seconds")
            except Exception as e:
                print(f"  {name}: Error - {e}")


if __name__ == "__main__":
    test_right_side_view() 