"""
101. Symmetric Tree

Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).

Example 1:
Input: root = [1,2,2,3,4,4,3]
Output: true

Example 2:
Input: root = [1,2,2,null,3,null,3]
Output: false

Constraints:
- The number of nodes in the tree is in the range [1, 1000]
- -100 <= Node.val <= 100

Follow up: Could you solve it both recursively and iteratively?
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


def is_symmetric_recursive(root: Optional[TreeNode]) -> bool:
    """
    Recursive approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree (recursion stack)
    
    Algorithm:
    1. Define helper function to check if two subtrees are mirrors
    2. Compare left subtree with right subtree
    3. For mirror check: left.left == right.right and left.right == right.left
    """
    def is_mirror(left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        
        return (left.val == right.val and 
                is_mirror(left.left, right.right) and 
                is_mirror(left.right, right.left))
    
    if not root:
        return True
    
    return is_mirror(root.left, root.right)


def is_symmetric_iterative_bfs(root: Optional[TreeNode]) -> bool:
    """
    Iterative BFS approach using queue.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use queue to store pairs of nodes to compare
    2. For each pair, check if they are mirrors
    3. Add their children in mirror order to queue
    """
    if not root:
        return True
    
    queue = deque([(root.left, root.right)])
    
    while queue:
        left, right = queue.popleft()
        
        if not left and not right:
            continue
        if not left or not right:
            return False
        if left.val != right.val:
            return False
        
        # Add children in mirror order
        queue.append((left.left, right.right))
        queue.append((left.right, right.left))
    
    return True


def is_symmetric_iterative_dfs(root: Optional[TreeNode]) -> bool:
    """
    Iterative DFS approach using stack.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use stack to store pairs of nodes to compare
    2. For each pair, check if they are mirrors
    3. Add their children in mirror order to stack
    """
    if not root:
        return True
    
    stack = [(root.left, root.right)]
    
    while stack:
        left, right = stack.pop()
        
        if not left and not right:
            continue
        if not left or not right:
            return False
        if left.val != right.val:
            return False
        
        # Add children in mirror order
        stack.append((left.left, right.right))
        stack.append((left.right, right.left))
    
    return True


def is_symmetric_level_order(root: Optional[TreeNode]) -> bool:
    """
    Level-order traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Perform level-order traversal
    2. For each level, check if it's a palindrome
    3. Handle None values properly
    """
    if not root:
        return True
    
    current_level = [root]
    
    while current_level:
        next_level = []
        level_values = []
        
        for node in current_level:
            if node:
                level_values.append(node.val)
                next_level.extend([node.left, node.right])
            else:
                level_values.append(None)
                next_level.extend([None, None])
        
        # Check if level is palindrome
        if level_values != level_values[::-1]:
            return False
        
        # Filter out None values for next iteration
        current_level = [node for node in next_level if node is not None]
        
        # If all nodes are None, we're done
        if not current_level:
            break
    
    return True


def is_symmetric_inorder(root: Optional[TreeNode]) -> bool:
    """
    Inorder traversal approach with structure tracking.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) for storing traversal result
    
    Algorithm:
    1. Perform inorder traversal on left subtree
    2. Perform reverse inorder traversal on right subtree
    3. Compare the two traversals
    """
    if not root:
        return True
    
    def inorder_left(node, result):
        if not node:
            result.append(None)
            return
        
        inorder_left(node.left, result)
        result.append(node.val)
        inorder_left(node.right, result)
    
    def inorder_right(node, result):
        if not node:
            result.append(None)
            return
        
        inorder_right(node.right, result)
        result.append(node.val)
        inorder_right(node.left, result)
    
    left_traversal = []
    right_traversal = []
    
    inorder_left(root.left, left_traversal)
    inorder_right(root.right, right_traversal)
    
    return left_traversal == right_traversal


def is_symmetric_preorder(root: Optional[TreeNode]) -> bool:
    """
    Preorder traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) for storing traversal result
    
    Algorithm:
    1. Perform preorder traversal on left subtree
    2. Perform mirror preorder traversal on right subtree
    3. Compare the two traversals
    """
    if not root:
        return True
    
    def preorder_left(node, result):
        if not node:
            result.append(None)
            return
        
        result.append(node.val)
        preorder_left(node.left, result)
        preorder_left(node.right, result)
    
    def preorder_right(node, result):
        if not node:
            result.append(None)
            return
        
        result.append(node.val)
        preorder_right(node.right, result)
        preorder_right(node.left, result)
    
    left_traversal = []
    right_traversal = []
    
    preorder_left(root.left, left_traversal)
    preorder_right(root.right, right_traversal)
    
    return left_traversal == right_traversal


def is_symmetric_postorder(root: Optional[TreeNode]) -> bool:
    """
    Postorder traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) for storing traversal result
    
    Algorithm:
    1. Perform postorder traversal on left subtree
    2. Perform mirror postorder traversal on right subtree
    3. Compare the two traversals
    """
    if not root:
        return True
    
    def postorder_left(node, result):
        if not node:
            result.append(None)
            return
        
        postorder_left(node.left, result)
        postorder_left(node.right, result)
        result.append(node.val)
    
    def postorder_right(node, result):
        if not node:
            result.append(None)
            return
        
        postorder_right(node.right, result)
        postorder_right(node.left, result)
        result.append(node.val)
    
    left_traversal = []
    right_traversal = []
    
    postorder_left(root.left, left_traversal)
    postorder_right(root.right, right_traversal)
    
    return left_traversal == right_traversal


def is_symmetric_serialization(root: Optional[TreeNode]) -> bool:
    """
    Serialization approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) for serialization strings
    
    Algorithm:
    1. Serialize left subtree in normal order
    2. Serialize right subtree in mirror order
    3. Compare the serializations
    """
    def serialize_normal(node):
        if not node:
            return "None"
        return f"{node.val},{serialize_normal(node.left)},{serialize_normal(node.right)}"
    
    def serialize_mirror(node):
        if not node:
            return "None"
        return f"{node.val},{serialize_mirror(node.right)},{serialize_mirror(node.left)}"
    
    if not root:
        return True
    
    left_serial = serialize_normal(root.left)
    right_serial = serialize_mirror(root.right)
    
    return left_serial == right_serial


def is_symmetric_morris_traversal(root: Optional[TreeNode]) -> bool:
    """
    Morris traversal approach for constant space.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - constant space
    
    Algorithm:
    1. Use Morris traversal for both subtrees
    2. Compare the traversals
    3. Handle the mirror property
    """
    if not root:
        return True
    
    def morris_inorder(node):
        result = []
        current = node
        
        while current:
            if not current.left:
                result.append(current.val)
                current = current.right
            else:
                predecessor = current.left
                while predecessor.right and predecessor.right != current:
                    predecessor = predecessor.right
                
                if not predecessor.right:
                    predecessor.right = current
                    current = current.left
                else:
                    predecessor.right = None
                    result.append(current.val)
                    current = current.right
        
        return result
    
    def morris_reverse_inorder(node):
        result = []
        current = node
        
        while current:
            if not current.right:
                result.append(current.val)
                current = current.left
            else:
                successor = current.right
                while successor.left and successor.left != current:
                    successor = successor.left
                
                if not successor.left:
                    successor.left = current
                    current = current.right
                else:
                    successor.left = None
                    result.append(current.val)
                    current = current.left
        
        return result
    
    left_traversal = morris_inorder(root.left)
    right_traversal = morris_reverse_inorder(root.right)
    
    return left_traversal == right_traversal


def is_symmetric_two_stacks(root: Optional[TreeNode]) -> bool:
    """
    Two stacks approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use two stacks for left and right subtrees
    2. Traverse in mirror order
    3. Compare nodes at each step
    """
    if not root:
        return True
    
    left_stack = [root.left]
    right_stack = [root.right]
    
    while left_stack and right_stack:
        left = left_stack.pop()
        right = right_stack.pop()
        
        if not left and not right:
            continue
        if not left or not right:
            return False
        if left.val != right.val:
            return False
        
        # Push children in mirror order
        left_stack.extend([left.right, left.left])
        right_stack.extend([right.left, right.right])
    
    return len(left_stack) == len(right_stack)


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


def test_is_symmetric():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic symmetric trees
        ([1,2,2,3,4,4,3], True),
        ([1,2,2,None,3,None,3], False),
        ([1], True),
        
        # Simple symmetric
        ([1,2,2], True),
        ([1,2,3], False),
        
        # Empty tree
        ([], True),
        
        # Two levels
        ([1,2,2,3,3,3,3], True),
        ([1,2,2,3,3,3,4], False),
        
        # Asymmetric structures
        ([1,2,2,None,3,3,None], True),
        ([1,2,2,3,None,None,3], True),
        ([1,2,2,3,None,3,None], False),
        
        # Deep symmetric trees
        ([1,2,2,3,4,4,3,5,6,7,8,8,7,6,5], True),
        ([1,2,2,3,4,4,3,5,6,7,8,8,7,6,6], False),
        
        # Skewed trees
        ([1,2,None,3,None,None,3], False),
        ([1,None,2,None,3,3,None], False),
        
        # Perfect symmetry
        ([1,2,2,3,3,3,3,4,4,4,4,4,4,4,4], True),
        ([1,2,2,3,3,3,3,4,4,4,4,4,4,4,5], False),
        
        # Negative values
        ([-1,-2,-2], True),
        ([-1,-2,-3], False),
        ([1,-2,-2,3,-3,-3,3], True),
        ([1,-2,-2,3,-3,-3,-3], False),
        
        # Large values
        ([100,50,50,25,75,75,25], True),
        ([100,50,50,25,75,75,26], False),
        
        # Complex asymmetry
        ([1,2,2,3,4,4,3,None,5,6,None,None,6,5,None], True),
        ([1,2,2,3,4,4,3,None,5,6,None,None,6,None,5], False),
        
        # Single child cases
        ([1,2,None,3,None], False),
        ([1,None,2,None,3], False),
        ([1,2,2,3,None,None,3], True),
        ([1,2,2,None,3,3,None], True),
    ]
    
    # Test all implementations
    implementations = [
        ("Recursive", is_symmetric_recursive),
        ("Iterative BFS", is_symmetric_iterative_bfs),
        ("Iterative DFS", is_symmetric_iterative_dfs),
        ("Level Order", is_symmetric_level_order),
        ("Inorder", is_symmetric_inorder),
        ("Preorder", is_symmetric_preorder),
        ("Postorder", is_symmetric_postorder),
        ("Serialization", is_symmetric_serialization),
        ("Morris", is_symmetric_morris_traversal),
        ("Two Stacks", is_symmetric_two_stacks),
    ]
    
    print("Testing Symmetric Tree implementations:")
    print("=" * 60)
    
    for i, (tree_values, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {tree_values}")
        print(f"Expected: {expected}")
        
        for name, func in implementations:
            try:
                # Create fresh tree for each implementation
                tree = create_binary_tree(tree_values)
                result = func(tree)
                
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
    def generate_symmetric_tree(depth):
        """Generate a symmetric tree of given depth."""
        if depth == 0:
            return []
        
        size = 2 ** depth - 1
        values = [1] * size
        
        # Make it symmetric by construction
        for i in range(size):
            level = 0
            temp = i + 1
            while temp > 2 ** level:
                temp -= 2 ** level
                level += 1
            
            pos_in_level = temp - 1
            level_size = 2 ** level
            mirror_pos = level_size - 1 - pos_in_level
            
            values[i] = i + 1  # Use position as value for testing
        
        return values
    
    test_scenarios = [
        ("Small symmetric", generate_symmetric_tree(4)),
        ("Medium symmetric", generate_symmetric_tree(6)),
        ("Large symmetric", generate_symmetric_tree(8)),
        ("Asymmetric", [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
        ("Single path", [1,2,None,3,None,4,None,5]),
    ]
    
    for scenario_name, values in test_scenarios:
        print(f"\n{scenario_name}:")
        
        for name, func in implementations:
            try:
                tree = create_binary_tree(values)
                
                start_time = time.time()
                result = func(tree)
                end_time = time.time()
                
                print(f"  {name}: {result} in {end_time - start_time:.4f} seconds")
            except Exception as e:
                print(f"  {name}: Error - {e}")


if __name__ == "__main__":
    test_is_symmetric() 