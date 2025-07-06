"""
098. Validate Binary Search Tree

Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:
- The left subtree of a node contains only nodes with keys less than the node's key.
- The right subtree of a node contains only nodes with keys greater than the node's key.
- Both the left and right subtrees must also be binary search trees.

Example 1:
Input: root = [2,1,3]
Output: true

Example 2:
Input: root = [5,1,4,null,null,3,6]
Output: false
Explanation: The root node's value is 5 but its right child's value is 4.

Constraints:
- The number of nodes in the tree is in the range [1, 10^4]
- -2^31 <= Node.val <= 2^31 - 1
"""

from typing import Optional, List
from collections import deque
import sys


class TreeNode:
    """Definition for a binary tree node."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        """String representation for debugging."""
        return f"TreeNode({self.val})"


def is_valid_bst_bounds(root: Optional[TreeNode]) -> bool:
    """
    Bounds checking approach (optimal solution).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree (recursion stack)
    
    Algorithm:
    1. Use bounds (min, max) to validate each node
    2. For left child: max bound becomes current node's value
    3. For right child: min bound becomes current node's value
    4. Recursively validate with updated bounds
    """
    def validate(node, min_val, max_val):
        if not node:
            return True
        
        # Check if current node violates BST property
        if node.val <= min_val or node.val >= max_val:
            return False
        
        # Recursively validate left and right subtrees with updated bounds
        return (validate(node.left, min_val, node.val) and 
                validate(node.right, node.val, max_val))
    
    return validate(root, float('-inf'), float('inf'))


def is_valid_bst_inorder(root: Optional[TreeNode]) -> bool:
    """
    Inorder traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Perform inorder traversal
    2. Check if the sequence is strictly increasing
    3. Use a single previous value to track
    """
    def inorder(node):
        if not node:
            return True
        
        # Traverse left subtree
        if not inorder(node.left):
            return False
        
        # Check current node
        if node.val <= inorder.prev:
            return False
        inorder.prev = node.val
        
        # Traverse right subtree
        return inorder(node.right)
    
    inorder.prev = float('-inf')
    return inorder(root)


def is_valid_bst_inorder_iterative(root: Optional[TreeNode]) -> bool:
    """
    Iterative inorder traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use stack for iterative inorder traversal
    2. Keep track of previous node value
    3. Ensure values are strictly increasing
    """
    if not root:
        return True
    
    stack = []
    prev = float('-inf')
    current = root
    
    while stack or current:
        # Go to the leftmost node
        while current:
            stack.append(current)
            current = current.left
        
        # Process current node
        current = stack.pop()
        
        # Check BST property
        if current.val <= prev:
            return False
        prev = current.val
        
        # Move to right subtree
        current = current.right
    
    return True


def is_valid_bst_preorder(root: Optional[TreeNode]) -> bool:
    """
    Preorder traversal with bounds approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use preorder traversal with bounds
    2. Check each node against its bounds
    3. Update bounds for children
    """
    def preorder(node, min_val, max_val):
        if not node:
            return True
        
        if node.val <= min_val or node.val >= max_val:
            return False
        
        return (preorder(node.left, min_val, node.val) and 
                preorder(node.right, node.val, max_val))
    
    return preorder(root, float('-inf'), float('inf'))


def is_valid_bst_postorder(root: Optional[TreeNode]) -> bool:
    """
    Postorder traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use postorder traversal
    2. Return (is_valid, min_val, max_val) for each subtree
    3. Validate current node against children's bounds
    """
    def postorder(node):
        if not node:
            return True, float('inf'), float('-inf')
        
        left_valid, left_min, left_max = postorder(node.left)
        right_valid, right_min, right_max = postorder(node.right)
        
        # Check if current node is valid
        if not left_valid or not right_valid:
            return False, 0, 0
        
        if (node.left and left_max >= node.val) or (node.right and right_min <= node.val):
            return False, 0, 0
        
        # Update bounds
        current_min = left_min if node.left else node.val
        current_max = right_max if node.right else node.val
        
        return True, current_min, current_max
    
    valid, _, _ = postorder(root)
    return valid


def is_valid_bst_morris(root: Optional[TreeNode]) -> bool:
    """
    Morris traversal approach (O(1) extra space).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - constant extra space
    
    Algorithm:
    1. Use Morris inorder traversal
    2. Maintain previous node value
    3. Check if values are strictly increasing
    """
    if not root:
        return True
    
    prev = float('-inf')
    current = root
    
    while current:
        if not current.left:
            # Process current node
            if current.val <= prev:
                return False
            prev = current.val
            current = current.right
        else:
            # Find inorder predecessor
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                predecessor = predecessor.right
            
            if not predecessor.right:
                # Create threading
                predecessor.right = current
                current = current.left
            else:
                # Remove threading and process current node
                predecessor.right = None
                if current.val <= prev:
                    return False
                prev = current.val
                current = current.right
    
    return True


def is_valid_bst_level_order(root: Optional[TreeNode]) -> bool:
    """
    Level order traversal with bounds.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use level order traversal
    2. Store bounds for each node
    3. Check bounds and update for children
    """
    if not root:
        return True
    
    queue = deque([(root, float('-inf'), float('inf'))])
    
    while queue:
        node, min_val, max_val = queue.popleft()
        
        if node.val <= min_val or node.val >= max_val:
            return False
        
        if node.left:
            queue.append((node.left, min_val, node.val))
        if node.right:
            queue.append((node.right, node.val, max_val))
    
    return True


def is_valid_bst_stack_bounds(root: Optional[TreeNode]) -> bool:
    """
    Stack-based approach with bounds.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use stack to store (node, min_bound, max_bound)
    2. Check each node against its bounds
    3. Push children with updated bounds
    """
    if not root:
        return True
    
    stack = [(root, float('-inf'), float('inf'))]
    
    while stack:
        node, min_val, max_val = stack.pop()
        
        if node.val <= min_val or node.val >= max_val:
            return False
        
        if node.right:
            stack.append((node.right, node.val, max_val))
        if node.left:
            stack.append((node.left, min_val, node.val))
    
    return True


def is_valid_bst_divide_conquer(root: Optional[TreeNode]) -> bool:
    """
    Divide and conquer approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Divide: check left and right subtrees
    2. Conquer: combine results with current node validation
    3. Return validity with min/max bounds
    """
    def divide_conquer(node):
        if not node:
            return True, None, None
        
        left_valid, left_min, left_max = divide_conquer(node.left)
        right_valid, right_min, right_max = divide_conquer(node.right)
        
        if not left_valid or not right_valid:
            return False, None, None
        
        # Check BST property
        if left_max is not None and left_max >= node.val:
            return False, None, None
        if right_min is not None and right_min <= node.val:
            return False, None, None
        
        # Update bounds
        current_min = left_min if left_min is not None else node.val
        current_max = right_max if right_max is not None else node.val
        
        return True, current_min, current_max
    
    valid, _, _ = divide_conquer(root)
    return valid


def is_valid_bst_recursive_simple(root: Optional[TreeNode]) -> bool:
    """
    Simple recursive approach with global previous.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use inorder traversal
    2. Maintain global previous value
    3. Check if sequence is strictly increasing
    """
    def inorder(node):
        if not node:
            return True
        
        if not inorder(node.left):
            return False
        
        if inorder.prev >= node.val:
            return False
        inorder.prev = node.val
        
        return inorder(node.right)
    
    inorder.prev = float('-inf')
    return inorder(root)


# Test cases
def test_is_valid_bst():
    """Test all BST validation approaches."""
    
    def create_test_tree_1():
        """Create test tree: [2,1,3]"""
        root = TreeNode(2)
        root.left = TreeNode(1)
        root.right = TreeNode(3)
        return root
    
    def create_test_tree_2():
        """Create test tree: [5,1,4,null,null,3,6]"""
        root = TreeNode(5)
        root.left = TreeNode(1)
        root.right = TreeNode(4)
        root.right.left = TreeNode(3)
        root.right.right = TreeNode(6)
        return root
    
    def create_test_tree_3():
        """Create test tree: [1]"""
        return TreeNode(1)
    
    def create_test_tree_4():
        """Create test tree: [10,5,15,null,null,6,20]"""
        root = TreeNode(10)
        root.left = TreeNode(5)
        root.right = TreeNode(15)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(20)
        return root
    
    def create_test_tree_5():
        """Create test tree: [3,2,5,1,4]"""
        root = TreeNode(3)
        root.left = TreeNode(2)
        root.right = TreeNode(5)
        root.left.left = TreeNode(1)
        root.right.left = TreeNode(4)
        return root
    
    # Test cases
    test_cases = [
        (create_test_tree_1(), True, "Valid BST [2,1,3]"),
        (create_test_tree_2(), False, "Invalid BST [5,1,4,null,null,3,6]"),
        (create_test_tree_3(), True, "Single node BST [1]"),
        (create_test_tree_4(), False, "Invalid BST [10,5,15,null,null,6,20]"),
        (create_test_tree_5(), True, "Valid BST [3,2,5,1,4]"),
        (None, True, "Empty tree"),
    ]
    
    # Test all approaches
    approaches = [
        (is_valid_bst_bounds, "Bounds checking"),
        (is_valid_bst_inorder, "Inorder traversal"),
        (is_valid_bst_inorder_iterative, "Iterative inorder"),
        (is_valid_bst_preorder, "Preorder with bounds"),
        (is_valid_bst_postorder, "Postorder traversal"),
        (is_valid_bst_morris, "Morris traversal"),
        (is_valid_bst_level_order, "Level order with bounds"),
        (is_valid_bst_stack_bounds, "Stack with bounds"),
        (is_valid_bst_divide_conquer, "Divide and conquer"),
        (is_valid_bst_recursive_simple, "Simple recursive"),
    ]
    
    print("Testing BST validation approaches:")
    print("=" * 50)
    
    for i, (root, expected, description) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: {description}")
        print(f"Expected: {expected}")
        
        for func, name in approaches:
            try:
                result = func(root)
                status = "✓" if result == expected else "✗"
                print(f"{status} {name}: {result}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 50)
    print("Performance Analysis:")
    print("=" * 50)
    
    # Create larger test tree for performance testing
    def create_large_bst(n):
        """Create a balanced BST with n nodes."""
        if n <= 0:
            return None
        
        def build_bst(start, end):
            if start > end:
                return None
            
            mid = (start + end) // 2
            node = TreeNode(mid)
            node.left = build_bst(start, mid - 1)
            node.right = build_bst(mid + 1, end)
            return node
        
        return build_bst(1, n)
    
    large_tree = create_large_bst(1000)
    
    import time
    
    print(f"Testing with large BST (1000 nodes):")
    for func, name in approaches:
        try:
            start_time = time.time()
            result = func(large_tree)
            end_time = time.time()
            print(f"{name}: {result} (Time: {end_time - start_time:.6f}s)")
        except Exception as e:
            print(f"{name}: Error - {e}")


if __name__ == "__main__":
    test_is_valid_bst() 