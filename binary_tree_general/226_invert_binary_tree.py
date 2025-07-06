"""
226. Invert Binary Tree

Given the root of a binary tree, invert the tree, and return its root.

Example 1:
Input: root = [4,2,7,1,3,6,9]
Output: [4,7,2,9,6,3,1]

Example 2:
Input: root = [2,1,3]
Output: [2,3,1]

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


def invert_tree_recursive(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Recursive approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree (recursion stack)
    
    Algorithm:
    1. Base case: if root is None, return None
    2. Swap left and right children
    3. Recursively invert left and right subtrees
    4. Return root
    """
    if not root:
        return None
    
    # Swap left and right children
    root.left, root.right = root.right, root.left
    
    # Recursively invert subtrees
    invert_tree_recursive(root.left)
    invert_tree_recursive(root.right)
    
    return root


def invert_tree_iterative_bfs(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Iterative BFS approach using queue.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use queue for level-order traversal
    2. For each node, swap its left and right children
    3. Add children to queue for further processing
    4. Return root
    """
    if not root:
        return None
    
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        
        # Swap left and right children
        node.left, node.right = node.right, node.left
        
        # Add children to queue
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return root


def invert_tree_iterative_dfs(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Iterative DFS approach using stack.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use stack for DFS traversal
    2. For each node, swap its left and right children
    3. Add children to stack for further processing
    4. Return root
    """
    if not root:
        return None
    
    stack = [root]
    
    while stack:
        node = stack.pop()
        
        # Swap left and right children
        node.left, node.right = node.right, node.left
        
        # Add children to stack
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    
    return root


def invert_tree_preorder(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Preorder traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Process current node (swap children)
    2. Recursively process left subtree
    3. Recursively process right subtree
    """
    def preorder(node):
        if not node:
            return
        
        # Process current node
        node.left, node.right = node.right, node.left
        
        # Process subtrees
        preorder(node.left)
        preorder(node.right)
    
    preorder(root)
    return root


def invert_tree_postorder(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Postorder traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Recursively process left subtree
    2. Recursively process right subtree
    3. Process current node (swap children)
    """
    def postorder(node):
        if not node:
            return
        
        # Process subtrees first
        postorder(node.left)
        postorder(node.right)
        
        # Then process current node
        node.left, node.right = node.right, node.left
    
    postorder(root)
    return root


def invert_tree_inorder(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Inorder traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Recursively process left subtree
    2. Process current node (swap children)
    3. Recursively process right subtree (now the original left)
    """
    def inorder(node):
        if not node:
            return
        
        # Process left subtree
        inorder(node.left)
        
        # Process current node
        node.left, node.right = node.right, node.left
        
        # Process right subtree (which is now the original left)
        inorder(node.left)
    
    inorder(root)
    return root


def invert_tree_level_order(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Level-order traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Process nodes level by level
    2. For each level, swap children of all nodes
    3. Continue until all levels are processed
    """
    if not root:
        return None
    
    current_level = [root]
    
    while current_level:
        next_level = []
        
        for node in current_level:
            # Swap children
            node.left, node.right = node.right, node.left
            
            # Add children to next level
            if node.left:
                next_level.append(node.left)
            if node.right:
                next_level.append(node.right)
        
        current_level = next_level
    
    return root


def invert_tree_morris_traversal(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Morris traversal approach (constant space).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - constant space
    
    Algorithm:
    1. Use Morris traversal to visit nodes without recursion
    2. Swap children during traversal
    3. Return root
    """
    if not root:
        return None
    
    current = root
    
    while current:
        if not current.left:
            # Swap children
            current.left, current.right = current.right, current.left
            current = current.right  # Now the original left
        else:
            # Find inorder predecessor
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                predecessor = predecessor.right
            
            if not predecessor.right:
                # Make current the right child of predecessor
                predecessor.right = current
                
                # Swap children
                current.left, current.right = current.right, current.left
                
                current = current.left  # Now the original right
            else:
                # Restore the tree structure
                predecessor.right = None
                current = current.right
    
    return root


def invert_tree_recursive_swap_first(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Recursive approach with swap-first strategy.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Swap children first
    2. Then recursively process the swapped children
    """
    if not root:
        return None
    
    # Swap children first
    root.left, root.right = root.right, root.left
    
    # Then recursively process (now swapped) children
    invert_tree_recursive_swap_first(root.left)
    invert_tree_recursive_swap_first(root.right)
    
    return root


def invert_tree_functional(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    Functional programming approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use functional programming style
    2. Map inversion operation over the tree
    """
    def invert_node(node):
        if not node:
            return None
        
        return TreeNode(
            node.val,
            invert_node(node.right),  # Swap: right becomes left
            invert_node(node.left)    # Swap: left becomes right
        )
    
    return invert_node(root)


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


def tree_to_list(root: Optional[TreeNode]) -> List[Optional[int]]:
    """Helper function to convert binary tree to level-order array."""
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        
        if node:
            result.append(node.val)
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append(None)
    
    # Remove trailing None values
    while result and result[-1] is None:
        result.pop()
    
    return result


def is_inverted(original: List[Optional[int]], inverted: List[Optional[int]]) -> bool:
    """Check if the inverted tree is correct."""
    if not original and not inverted:
        return True
    if not original or not inverted:
        return False
    
    # For level-order representation, we need to check if the tree structure is properly inverted
    # This is a simplified check - in practice, we'd need more sophisticated verification
    orig_tree = create_binary_tree(original)
    inv_tree = create_binary_tree(inverted)
    
    def check_inversion(node1, node2):
        if not node1 and not node2:
            return True
        if not node1 or not node2:
            return False
        if node1.val != node2.val:
            return False
        
        # Check if left child of node1 matches right child of node2
        # and right child of node1 matches left child of node2
        return (check_inversion(node1.left, node2.right) and 
                check_inversion(node1.right, node2.left))
    
    return check_inversion(orig_tree, inv_tree)


def test_invert_tree():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([4,2,7,1,3,6,9], [4,7,2,9,6,3,1]),
        ([2,1,3], [2,3,1]),
        ([], []),
        ([1], [1]),
        
        # Two nodes
        ([1,2], [1,None,2]),
        ([1,None,2], [1,2]),
        
        # Three nodes
        ([1,2,3], [1,3,2]),
        ([1,2,None], [1,None,2]),
        ([1,None,3], [1,3]),
        
        # Perfect binary trees
        ([1,2,3,4,5,6,7], [1,3,2,7,6,5,4]),
        ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [1,3,2,7,6,5,4,15,14,13,12,11,10,9,8]),
        
        # Skewed trees
        ([1,2,None,3,None,4], [1,None,2,None,3,None,4]),
        ([1,None,2,None,3,None,4], [1,2,None,3,None,4]),
        
        # Complex structures
        ([1,2,3,4,5,None,6,7,8,None,None,None,None,9,10], [1,3,2,6,None,5,4,None,None,None,None,8,7,10,9]),
        
        # Negative values
        ([-1,-2,-3], [-1,-3,-2]),
        ([1,-2,3], [1,3,-2]),
        
        # Mixed values
        ([0,1,2,3,4,5,6], [0,2,1,6,5,4,3]),
        
        # Large values
        ([100,50,150,25,75,125,175], [100,150,50,175,125,75,25]),
        
        # Incomplete trees
        ([1,2,3,4,None,None,7], [1,3,2,7,None,None,4]),
        ([1,2,3,None,5,6,None], [1,3,2,None,6,5,None]),
    ]
    
    # Test all implementations
    implementations = [
        ("Recursive", invert_tree_recursive),
        ("Iterative BFS", invert_tree_iterative_bfs),
        ("Iterative DFS", invert_tree_iterative_dfs),
        ("Preorder", invert_tree_preorder),
        ("Postorder", invert_tree_postorder),
        ("Inorder", invert_tree_inorder),
        ("Level Order", invert_tree_level_order),
        ("Morris", invert_tree_morris_traversal),
        ("Recursive Swap First", invert_tree_recursive_swap_first),
        ("Functional", invert_tree_functional),
    ]
    
    print("Testing Invert Binary Tree implementations:")
    print("=" * 60)
    
    for i, (original, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {original}")
        print(f"Expected: {expected}")
        
        for name, func in implementations:
            try:
                # Create fresh tree for each implementation
                tree = create_binary_tree(original)
                result_tree = func(tree)
                result = tree_to_list(result_tree)
                
                # Check if result matches expected or is a valid inversion
                is_correct = (result == expected or 
                             is_inverted(original, result))
                
                status = "✓" if is_correct else "✗"
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
        ("Medium tree", [i for i in range(1, 64)]),  # 63 nodes
        ("Large tree", [i for i in range(1, 256)]),  # 255 nodes
        ("Skewed tree", [1] + [None, i] for i in range(2, 101)),  # Skewed
        ("Perfect tree", [i for i in range(1, 128)]),  # 127 nodes
    ]
    
    for scenario_name, values in test_scenarios:
        print(f"\n{scenario_name}:")
        
        for name, func in implementations:
            try:
                tree = create_binary_tree(values)
                
                start_time = time.time()
                result_tree = func(tree)
                end_time = time.time()
                
                result_size = len(tree_to_list(result_tree))
                print(f"  {name}: {result_size} nodes inverted in {end_time - start_time:.4f} seconds")
            except Exception as e:
                print(f"  {name}: Error - {e}")


if __name__ == "__main__":
    test_invert_tree() 