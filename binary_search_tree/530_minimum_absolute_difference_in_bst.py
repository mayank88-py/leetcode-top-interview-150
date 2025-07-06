"""
530. Minimum Absolute Difference in BST

Given the root of a Binary Search Tree (BST), return the minimum absolute difference between the values of any two different nodes in the tree.

Example 1:
Input: root = [4,2,6,1,3]
Output: 1

Example 2:
Input: root = [1,0,48,null,null,12,49]
Output: 1

Constraints:
- The number of nodes in the tree is in the range [2, 10^4].
- 0 <= Node.val <= 10^5

Note: This question is the same as 783: https://leetcode.com/problems/minimum-distance-between-bst-nodes/
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


def get_minimum_difference_inorder(root: Optional[TreeNode]) -> int:
    """
    Inorder traversal approach (optimal solution).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Perform inorder traversal (gives sorted order)
    2. Track previous node value
    3. Update minimum difference on each node
    """
    def inorder(node):
        if not node:
            return
        
        inorder(node.left)
        
        # Process current node
        if inorder.prev is not None:
            inorder.min_diff = min(inorder.min_diff, node.val - inorder.prev)
        inorder.prev = node.val
        
        inorder(node.right)
    
    inorder.prev = None
    inorder.min_diff = float('inf')
    inorder(root)
    return inorder.min_diff


def get_minimum_difference_iterative(root: Optional[TreeNode]) -> int:
    """
    Iterative inorder traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use stack for iterative inorder traversal
    2. Track previous value
    3. Update minimum difference as we go
    """
    stack = []
    current = root
    prev = None
    min_diff = float('inf')
    
    while stack or current:
        # Go to leftmost node
        while current:
            stack.append(current)
            current = current.left
        
        # Process current node
        current = stack.pop()
        
        if prev is not None:
            min_diff = min(min_diff, current.val - prev)
        prev = current.val
        
        # Move to right subtree
        current = current.right
    
    return min_diff


def get_minimum_difference_morris(root: Optional[TreeNode]) -> int:
    """
    Morris inorder traversal approach (O(1) space).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - constant extra space
    
    Algorithm:
    1. Use Morris threading technique
    2. Process nodes in inorder without using stack
    3. Track minimum difference
    """
    current = root
    prev = None
    min_diff = float('inf')
    
    while current:
        if not current.left:
            # Process current node
            if prev is not None:
                min_diff = min(min_diff, current.val - prev)
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
                if prev is not None:
                    min_diff = min(min_diff, current.val - prev)
                prev = current.val
                current = current.right
    
    return min_diff


def get_minimum_difference_collect_all(root: Optional[TreeNode]) -> int:
    """
    Collect all values and compare approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - storing all values
    
    Algorithm:
    1. Collect all node values using inorder traversal
    2. Compare adjacent values to find minimum difference
    3. Return minimum difference
    """
    def inorder(node):
        if not node:
            return []
        
        result = []
        result.extend(inorder(node.left))
        result.append(node.val)
        result.extend(inorder(node.right))
        return result
    
    values = inorder(root)
    min_diff = float('inf')
    
    for i in range(1, len(values)):
        min_diff = min(min_diff, values[i] - values[i-1])
    
    return min_diff


def get_minimum_difference_level_order(root: Optional[TreeNode]) -> int:
    """
    Level order traversal with sorting approach.
    
    Time Complexity: O(n log n) where n is the number of nodes
    Space Complexity: O(n) - queue and values storage
    
    Algorithm:
    1. Use level order traversal to collect all values
    2. Sort the values
    3. Find minimum difference between adjacent values
    """
    if not root:
        return 0
    
    values = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        values.append(node.val)
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    values.sort()
    min_diff = float('inf')
    
    for i in range(1, len(values)):
        min_diff = min(min_diff, values[i] - values[i-1])
    
    return min_diff


def get_minimum_difference_preorder(root: Optional[TreeNode]) -> int:
    """
    Preorder traversal with all values approach.
    
    Time Complexity: O(n log n) where n is the number of nodes
    Space Complexity: O(n) - storing all values
    
    Algorithm:
    1. Use preorder traversal to collect all values
    2. Sort the values
    3. Find minimum difference between adjacent values
    """
    def preorder(node):
        if not node:
            return
        
        values.append(node.val)
        preorder(node.left)
        preorder(node.right)
    
    values = []
    preorder(root)
    values.sort()
    
    min_diff = float('inf')
    for i in range(1, len(values)):
        min_diff = min(min_diff, values[i] - values[i-1])
    
    return min_diff


def get_minimum_difference_postorder(root: Optional[TreeNode]) -> int:
    """
    Postorder traversal with all values approach.
    
    Time Complexity: O(n log n) where n is the number of nodes
    Space Complexity: O(n) - storing all values
    
    Algorithm:
    1. Use postorder traversal to collect all values
    2. Sort the values
    3. Find minimum difference between adjacent values
    """
    def postorder(node):
        if not node:
            return
        
        postorder(node.left)
        postorder(node.right)
        values.append(node.val)
    
    values = []
    postorder(root)
    values.sort()
    
    min_diff = float('inf')
    for i in range(1, len(values)):
        min_diff = min(min_diff, values[i] - values[i-1])
    
    return min_diff


def get_minimum_difference_dfs_global(root: Optional[TreeNode]) -> int:
    """
    DFS with global minimum tracking approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use DFS to traverse in inorder
    2. Maintain global minimum difference
    3. Update minimum on each comparison
    """
    def dfs(node):
        if not node:
            return
        
        dfs(node.left)
        
        if dfs.prev is not None:
            dfs.min_diff = min(dfs.min_diff, abs(node.val - dfs.prev))
        dfs.prev = node.val
        
        dfs(node.right)
    
    dfs.prev = None
    dfs.min_diff = float('inf')
    dfs(root)
    return dfs.min_diff


def get_minimum_difference_two_pass(root: Optional[TreeNode]) -> int:
    """
    Two-pass approach: first collect, then compare.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - storing all values
    
    Algorithm:
    1. First pass: collect all values using inorder
    2. Second pass: find minimum difference
    3. Return minimum difference
    """
    # First pass: collect values
    def collect_inorder(node, values):
        if not node:
            return
        
        collect_inorder(node.left, values)
        values.append(node.val)
        collect_inorder(node.right, values)
    
    values = []
    collect_inorder(root, values)
    
    # Second pass: find minimum difference
    min_diff = float('inf')
    for i in range(1, len(values)):
        min_diff = min(min_diff, values[i] - values[i-1])
    
    return min_diff


def get_minimum_difference_early_termination(root: Optional[TreeNode]) -> int:
    """
    Early termination approach when minimum possible difference is found.
    
    Time Complexity: O(n) worst case, O(k) average (k < n)
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Perform inorder traversal
    2. Track minimum difference found so far
    3. Early termination when difference of 1 is found
    """
    def inorder(node):
        if not node or inorder.min_diff == 1:
            return
        
        inorder(node.left)
        
        if inorder.min_diff == 1:
            return
        
        if inorder.prev is not None:
            diff = node.val - inorder.prev
            inorder.min_diff = min(inorder.min_diff, diff)
            if inorder.min_diff == 1:
                return
        inorder.prev = node.val
        
        inorder(node.right)
    
    inorder.prev = None
    inorder.min_diff = float('inf')
    inorder(root)
    return inorder.min_diff


def get_minimum_difference_stack_optimized(root: Optional[TreeNode]) -> int:
    """
    Stack-based approach with memory optimization.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use stack for inorder traversal
    2. Process nodes efficiently
    3. Track minimum difference with early termination
    """
    stack = []
    current = root
    prev = None
    min_diff = float('inf')
    
    while stack or current:
        # Go to leftmost node
        while current:
            stack.append(current)
            current = current.left
        
        # Process current node
        current = stack.pop()
        
        if prev is not None:
            diff = current.val - prev
            min_diff = min(min_diff, diff)
            # Early termination for optimal case
            if min_diff == 1:
                return min_diff
        prev = current.val
        
        # Move to right subtree
        current = current.right
    
    return min_diff


def get_minimum_difference_recursive_bounds(root: Optional[TreeNode]) -> int:
    """
    Recursive approach with bounds checking.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use recursive traversal with bounds
    2. Pass minimum and maximum values from subtrees
    3. Calculate minimum difference at each node
    """
    def find_min_diff(node):
        if not node:
            return float('inf'), None, None
        
        left_diff, left_min, left_max = find_min_diff(node.left)
        right_diff, right_min, right_max = find_min_diff(node.right)
        
        # Current node bounds
        current_min = left_min if left_min is not None else node.val
        current_max = right_max if right_max is not None else node.val
        
        # Calculate minimum difference
        min_diff = min(left_diff, right_diff)
        
        if left_max is not None:
            min_diff = min(min_diff, node.val - left_max)
        if right_min is not None:
            min_diff = min(min_diff, right_min - node.val)
        
        return min_diff, current_min, current_max
    
    min_diff, _, _ = find_min_diff(root)
    return min_diff


# Test cases
def test_get_minimum_difference():
    """Test all minimum difference approaches."""
    
    def create_test_tree_1():
        """Create test tree: [4,2,6,1,3]"""
        root = TreeNode(4)
        root.left = TreeNode(2)
        root.right = TreeNode(6)
        root.left.left = TreeNode(1)
        root.left.right = TreeNode(3)
        return root
    
    def create_test_tree_2():
        """Create test tree: [1,0,48,null,null,12,49]"""
        root = TreeNode(1)
        root.left = TreeNode(0)
        root.right = TreeNode(48)
        root.right.left = TreeNode(12)
        root.right.right = TreeNode(49)
        return root
    
    def create_test_tree_3():
        """Create test tree: [90,69,null,49,89,null,52]"""
        root = TreeNode(90)
        root.left = TreeNode(69)
        root.left.left = TreeNode(49)
        root.left.right = TreeNode(89)
        root.left.left.right = TreeNode(52)
        return root
    
    def create_test_tree_4():
        """Create test tree: [27,null,34,null,58,50,null,44]"""
        root = TreeNode(27)
        root.right = TreeNode(34)
        root.right.right = TreeNode(58)
        root.right.right.left = TreeNode(50)
        root.right.right.left.left = TreeNode(44)
        return root
    
    def create_test_tree_5():
        """Create test tree: [1,null,3,2]"""
        root = TreeNode(1)
        root.right = TreeNode(3)
        root.right.left = TreeNode(2)
        return root
    
    # Test cases
    test_cases = [
        (create_test_tree_1(), 1, "[4,2,6,1,3]"),
        (create_test_tree_2(), 1, "[1,0,48,null,null,12,49]"),
        (create_test_tree_3(), 1, "[90,69,null,49,89,null,52]"),
        (create_test_tree_4(), 6, "[27,null,34,null,58,50,null,44]"),
        (create_test_tree_5(), 1, "[1,null,3,2]"),
    ]
    
    # Test all approaches
    approaches = [
        (get_minimum_difference_inorder, "Inorder traversal"),
        (get_minimum_difference_iterative, "Iterative inorder"),
        (get_minimum_difference_morris, "Morris traversal"),
        (get_minimum_difference_collect_all, "Collect all values"),
        (get_minimum_difference_level_order, "Level order"),
        (get_minimum_difference_preorder, "Preorder traversal"),
        (get_minimum_difference_postorder, "Postorder traversal"),
        (get_minimum_difference_dfs_global, "DFS with global min"),
        (get_minimum_difference_two_pass, "Two-pass approach"),
        (get_minimum_difference_early_termination, "Early termination"),
        (get_minimum_difference_stack_optimized, "Stack optimized"),
        (get_minimum_difference_recursive_bounds, "Recursive bounds"),
    ]
    
    print("Testing minimum absolute difference approaches:")
    print("=" * 60)
    
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
    print("\n" + "=" * 60)
    print("Performance Analysis:")
    print("=" * 60)
    
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
    
    # Edge case testing
    print("\n" + "=" * 60)
    print("Edge Case Testing:")
    print("=" * 60)
    
    # Test with minimum tree (2 nodes)
    def create_min_tree():
        """Create minimum tree: [1,null,2]"""
        root = TreeNode(1)
        root.right = TreeNode(2)
        return root
    
    min_tree = create_min_tree()
    print("Testing with minimum tree [1,null,2]:")
    for func, name in approaches:
        try:
            result = func(min_tree)
            print(f"{name}: {result}")
        except Exception as e:
            print(f"{name}: Error - {e}")


if __name__ == "__main__":
    test_get_minimum_difference() 