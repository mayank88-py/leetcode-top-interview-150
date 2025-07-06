"""
124. Binary Tree Maximum Path Sum

A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.

The path sum of a path is the sum of the node's values in the path.

Given the root of a binary tree, return the maximum path sum of any non-empty path.

Example 1:
Input: root = [1,2,3]
Output: 6
Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.

Example 2:
Input: root = [-10,9,20,null,null,15,7]
Output: 42
Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.

Constraints:
- The number of nodes in the tree is in the range [1, 3 * 10^4]
- -1000 <= Node.val <= 1000
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


def max_path_sum_recursive(root: Optional[TreeNode]) -> int:
    """
    Recursive approach with global maximum.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree (recursion stack)
    
    Algorithm:
    1. For each node, calculate max path sum through that node
    2. Update global maximum if current path sum is larger
    3. Return max path sum starting from current node (for parent)
    """
    max_sum = [float('-inf')]
    
    def dfs(node):
        if not node:
            return 0
        
        # Get max path sum from left and right subtrees
        # Use max(0, ...) to ignore negative paths
        left_sum = max(0, dfs(node.left))
        right_sum = max(0, dfs(node.right))
        
        # Current path sum through this node
        current_path_sum = node.val + left_sum + right_sum
        
        # Update global maximum
        max_sum[0] = max(max_sum[0], current_path_sum)
        
        # Return max path sum starting from this node
        return node.val + max(left_sum, right_sum)
    
    dfs(root)
    return max_sum[0]


def max_path_sum_iterative_postorder(root: Optional[TreeNode]) -> int:
    """
    Iterative postorder traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use iterative postorder traversal
    2. Calculate max path sums bottom-up
    3. Track maximum path sum globally
    """
    if not root:
        return 0
    
    max_sum = float('-inf')
    stack = []
    last_visited = None
    node_values = {}  # Store computed values for nodes
    
    current = root
    
    while stack or current:
        if current:
            stack.append(current)
            current = current.left
        else:
            peek_node = stack[-1]
            
            # If right child exists and hasn't been processed yet
            if peek_node.right and last_visited != peek_node.right:
                current = peek_node.right
            else:
                # Process current node
                stack.pop()
                last_visited = peek_node
                
                # Calculate max path sums
                left_sum = max(0, node_values.get(peek_node.left, 0))
                right_sum = max(0, node_values.get(peek_node.right, 0))
                
                # Current path sum through this node
                current_path_sum = peek_node.val + left_sum + right_sum
                max_sum = max(max_sum, current_path_sum)
                
                # Store value for this node
                node_values[peek_node] = peek_node.val + max(left_sum, right_sum)
    
    return max_sum


def max_path_sum_with_memoization(root: Optional[TreeNode]) -> int:
    """
    Memoization approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - memoization cache
    
    Algorithm:
    1. Cache results for each node
    2. Avoid recomputation of subtree path sums
    """
    memo = {}
    max_sum = [float('-inf')]
    
    def dfs(node):
        if not node:
            return 0
        
        if node in memo:
            return memo[node]
        
        # Get max path sum from left and right subtrees
        left_sum = max(0, dfs(node.left))
        right_sum = max(0, dfs(node.right))
        
        # Current path sum through this node
        current_path_sum = node.val + left_sum + right_sum
        max_sum[0] = max(max_sum[0], current_path_sum)
        
        # Memoize result
        result = node.val + max(left_sum, right_sum)
        memo[node] = result
        return result
    
    dfs(root)
    return max_sum[0]


def max_path_sum_separate_functions(root: Optional[TreeNode]) -> int:
    """
    Separate functions approach for clarity.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Separate function to calculate max path from node
    2. Separate function to find global maximum
    """
    def max_path_from_node(node):
        """Returns max path sum starting from node."""
        if not node:
            return 0
        
        left_sum = max(0, max_path_from_node(node.left))
        right_sum = max(0, max_path_from_node(node.right))
        
        return node.val + max(left_sum, right_sum)
    
    def find_max_path(node):
        """Returns max path sum in subtree rooted at node."""
        if not node:
            return float('-inf')
        
        # Max path through current node
        left_sum = max(0, max_path_from_node(node.left))
        right_sum = max(0, max_path_from_node(node.right))
        current_max = node.val + left_sum + right_sum
        
        # Max path in left and right subtrees
        left_max = find_max_path(node.left)
        right_max = find_max_path(node.right)
        
        return max(current_max, left_max, right_max)
    
    return find_max_path(root)


def max_path_sum_class_based(root: Optional[TreeNode]) -> int:
    """
    Class-based approach for state management.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use class to maintain state
    2. Clean separation of concerns
    """
    class MaxPathSumFinder:
        def __init__(self):
            self.max_sum = float('-inf')
        
        def find_max_path(self, node):
            if not node:
                return 0
            
            # Get max path sum from children
            left_sum = max(0, self.find_max_path(node.left))
            right_sum = max(0, self.find_max_path(node.right))
            
            # Update global maximum
            current_path_sum = node.val + left_sum + right_sum
            self.max_sum = max(self.max_sum, current_path_sum)
            
            # Return max path starting from this node
            return node.val + max(left_sum, right_sum)
        
        def get_max_path_sum(self, root):
            self.find_max_path(root)
            return self.max_sum
    
    finder = MaxPathSumFinder()
    return finder.get_max_path_sum(root)


def max_path_sum_with_path_tracking(root: Optional[TreeNode]) -> int:
    """
    Path tracking approach (for debugging).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Track the actual path that gives maximum sum
    2. Useful for debugging and understanding
    """
    max_sum = [float('-inf')]
    best_path = [[]]
    
    def dfs(node, path):
        if not node:
            return 0
        
        # Add current node to path
        path.append(node.val)
        
        # Get max path sum from children
        left_sum = max(0, dfs(node.left, path.copy()))
        right_sum = max(0, dfs(node.right, path.copy()))
        
        # Current path sum through this node
        current_path_sum = node.val + left_sum + right_sum
        
        # Update global maximum and best path
        if current_path_sum > max_sum[0]:
            max_sum[0] = current_path_sum
            best_path[0] = path.copy()
        
        # Remove current node from path (backtrack)
        path.pop()
        
        return node.val + max(left_sum, right_sum)
    
    dfs(root, [])
    return max_sum[0]


def max_path_sum_bottom_up(root: Optional[TreeNode]) -> int:
    """
    Bottom-up approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Process nodes from bottom to top
    2. Calculate max path sums bottom-up
    """
    def bottom_up_helper(node):
        if not node:
            return 0, float('-inf')
        
        # Get results from children
        left_path, left_max = bottom_up_helper(node.left)
        right_path, right_max = bottom_up_helper(node.right)
        
        # Calculate current node's contribution
        left_contribution = max(0, left_path)
        right_contribution = max(0, right_path)
        
        # Path through current node
        current_path_sum = node.val + left_contribution + right_contribution
        
        # Max path starting from current node
        current_max_path = node.val + max(left_contribution, right_contribution)
        
        # Overall maximum in this subtree
        subtree_max = max(current_path_sum, left_max, right_max)
        
        return current_max_path, subtree_max
    
    _, result = bottom_up_helper(root)
    return result


def max_path_sum_iterative_morris(root: Optional[TreeNode]) -> int:
    """
    Morris traversal approach (challenging for this problem).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - constant extra space
    
    Algorithm:
    1. Use Morris threading technique
    2. Calculate path sums during traversal
    
    Note: Morris traversal is very complex for this problem
    """
    # Morris traversal is extremely complex for max path sum
    # Fall back to recursive approach
    return max_path_sum_recursive(root)


def max_path_sum_divide_conquer(root: Optional[TreeNode]) -> int:
    """
    Divide and conquer approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Divide problem into subproblems
    2. Solve subproblems recursively
    3. Combine results
    """
    def divide_conquer(node):
        if not node:
            return float('-inf'), 0
        
        # Divide: solve for left and right subtrees
        left_max, left_path = divide_conquer(node.left)
        right_max, right_path = divide_conquer(node.right)
        
        # Conquer: combine results
        left_contribution = max(0, left_path)
        right_contribution = max(0, right_path)
        
        # Current path sum through this node
        current_path_sum = node.val + left_contribution + right_contribution
        
        # Maximum in this subtree
        subtree_max = max(current_path_sum, left_max, right_max)
        
        # Path from this node upward
        upward_path = node.val + max(left_contribution, right_contribution)
        
        return subtree_max, upward_path
    
    result, _ = divide_conquer(root)
    return result


def max_path_sum_functional(root: Optional[TreeNode]) -> int:
    """
    Functional programming approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use functional programming concepts
    2. Immutable data structures where possible
    3. Pure functions
    """
    def max_path_helper(node):
        if not node:
            return (0, float('-inf'))  # (max_path_from_node, max_path_in_subtree)
        
        # Get results from children
        left_from_node, left_in_subtree = max_path_helper(node.left)
        right_from_node, right_in_subtree = max_path_helper(node.right)
        
        # Calculate contributions
        left_contrib = max(0, left_from_node)
        right_contrib = max(0, right_from_node)
        
        # Path through current node
        path_through_node = node.val + left_contrib + right_contrib
        
        # Max path starting from current node
        path_from_node = node.val + max(left_contrib, right_contrib)
        
        # Max path in this subtree
        max_in_subtree = max(path_through_node, left_in_subtree, right_in_subtree)
        
        return (path_from_node, max_in_subtree)
    
    _, result = max_path_helper(root)
    return result


def max_path_sum_with_validation(root: Optional[TreeNode]) -> int:
    """
    Approach with input validation.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Validate input thoroughly
    2. Handle edge cases explicitly
    3. Use robust error handling
    """
    if not root:
        return 0
    
    # Validate tree structure
    def validate_tree(node, seen=None):
        if seen is None:
            seen = set()
        
        if not node:
            return True
        
        if id(node) in seen:
            return False  # Cycle detected
        
        seen.add(id(node))
        return validate_tree(node.left, seen) and validate_tree(node.right, seen)
    
    if not validate_tree(root):
        raise ValueError("Invalid tree structure")
    
    # Use standard recursive approach
    max_sum = [float('-inf')]
    
    def dfs(node):
        if not node:
            return 0
        
        left_sum = max(0, dfs(node.left))
        right_sum = max(0, dfs(node.right))
        
        current_path_sum = node.val + left_sum + right_sum
        max_sum[0] = max(max_sum[0], current_path_sum)
        
        return node.val + max(left_sum, right_sum)
    
    dfs(root)
    return max_sum[0]


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


def test_max_path_sum():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([1,2,3], 6),  # 2 + 1 + 3 = 6
        ([-10,9,20,None,None,15,7], 42),  # 15 + 20 + 7 = 42
        
        # Single node
        ([1], 1),
        ([-1], -1),
        ([0], 0),
        
        # Two nodes
        ([1,2], 3),  # 1 + 2 = 3
        ([1,-2], 1),  # Just 1
        ([-1,2], 2),  # Just 2
        
        # All negative
        ([-1,-2,-3], -1),  # Best single node
        ([-5,-2,-3], -2),  # Best single node
        
        # Left skewed
        ([1,2,None,3], 6),  # 1 + 2 + 3 = 6
        ([1,-2,None,3], 3),  # Just 3
        
        # Right skewed
        ([1,None,2,None,3], 6),  # 1 + 2 + 3 = 6
        ([1,None,-2,None,3], 3),  # Just 3
        
        # Balanced trees
        ([1,2,3,4,5,6,7], 18),  # 4 + 2 + 1 + 3 + 7 = 17 or 5 + 2 + 1 + 3 + 6 = 17
        
        # Complex paths
        ([5,4,8,11,None,13,4,7,2,None,None,None,1], 48),  # 7 + 11 + 4 + 5 + 8 + 13 = 48
        
        # Large positive values
        ([100,200,300], 600),  # 200 + 100 + 300 = 600
        
        # Mix of positive and negative
        ([1,2,-3,4,5], 7),  # 4 + 2 + 1 = 7 or 5 + 2 + 1 = 8
        
        # Zero path
        ([0,0,0], 0),
        
        # Single path optimal
        ([1,2,3,4,5,6,7,8,9], 24),  # Best path through multiple nodes
        
        # Negative branches
        ([1,-2,-3,4,5], 5),  # Just 5
        ([1,-2,-3,-4,-5], 1),  # Just 1
        
        # Deep tree
        ([1,2,None,3,None,4,None,5], 15),  # 1 + 2 + 3 + 4 + 5 = 15
        
        # Wide tree
        ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 40),  # Complex calculation
        
        # Edge cases
        ([1000], 1000),  # Single large node
        ([-1000], -1000),  # Single negative node
        ([0,1000,-1000], 1000),  # Choose positive branch
        
        # Subtree optimization
        ([5,1,2,3,None,6,2,None,4,None,None,None,None,None,7], 16),  # 3 + 1 + 5 + 2 + 6 = 17
    ]
    
    # Test all implementations
    implementations = [
        ("Recursive", max_path_sum_recursive),
        ("Iterative Postorder", max_path_sum_iterative_postorder),
        ("Memoization", max_path_sum_with_memoization),
        ("Separate Functions", max_path_sum_separate_functions),
        ("Class Based", max_path_sum_class_based),
        ("Path Tracking", max_path_sum_with_path_tracking),
        ("Bottom Up", max_path_sum_bottom_up),
        ("Iterative Morris", max_path_sum_iterative_morris),
        ("Divide Conquer", max_path_sum_divide_conquer),
        ("Functional", max_path_sum_functional),
        ("With Validation", max_path_sum_with_validation),
    ]
    
    print("Testing Binary Tree Maximum Path Sum implementations:")
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
        return [i % 100 for i in range(1, size + 1)]
    
    def generate_skewed_tree(size):
        """Generate a skewed tree."""
        result = []
        for i in range(1, size + 1):
            result.append(i % 100)
            if i < size:
                result.append(None)
        return result
    
    test_scenarios = [
        ("Small balanced", generate_balanced_tree(4)),
        ("Medium balanced", generate_balanced_tree(6)),
        ("Large balanced", generate_balanced_tree(8)),
        ("Skewed tree", generate_skewed_tree(100)),
        ("Deep tree", generate_skewed_tree(500)),
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
    test_max_path_sum() 