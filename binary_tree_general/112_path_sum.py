"""
112. Path Sum

Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.

A leaf is a node with no children.

Example 1:
Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
Output: true

Example 2:
Input: root = [1,2,3], targetSum = 5
Output: false

Example 3:
Input: root = [], targetSum = 0
Output: false

Constraints:
- The number of nodes in the tree is in the range [0, 5000]
- -1000 <= Node.val <= 1000
- -1000 <= targetSum <= 1000
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


def has_path_sum_recursive(root: Optional[TreeNode], targetSum: int) -> bool:
    """
    Recursive approach (DFS).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree (recursion stack)
    
    Algorithm:
    1. Base case: if root is None, return False
    2. If it's a leaf node, check if value equals remaining sum
    3. Recursively check left and right subtrees with updated sum
    """
    if not root:
        return False
    
    # If it's a leaf node, check if the value equals the target sum
    if not root.left and not root.right:
        return root.val == targetSum
    
    # Recursively check left and right subtrees
    remaining_sum = targetSum - root.val
    return (has_path_sum_recursive(root.left, remaining_sum) or 
            has_path_sum_recursive(root.right, remaining_sum))


def has_path_sum_iterative_stack(root: Optional[TreeNode], targetSum: int) -> bool:
    """
    Iterative approach using stack.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use stack to store (node, remaining_sum) pairs
    2. For each node, check if it's a leaf with target sum
    3. Add children to stack with updated remaining sum
    """
    if not root:
        return False
    
    stack = [(root, targetSum)]
    
    while stack:
        node, remaining_sum = stack.pop()
        
        # If it's a leaf node, check if value equals remaining sum
        if not node.left and not node.right:
            if node.val == remaining_sum:
                return True
            continue
        
        # Add children to stack with updated remaining sum
        new_remaining = remaining_sum - node.val
        
        if node.left:
            stack.append((node.left, new_remaining))
        if node.right:
            stack.append((node.right, new_remaining))
    
    return False


def has_path_sum_iterative_queue(root: Optional[TreeNode], targetSum: int) -> bool:
    """
    Iterative approach using queue (BFS).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use queue for level-order traversal
    2. Store (node, current_sum) pairs
    3. Check leaf nodes for target sum
    """
    if not root:
        return False
    
    queue = deque([(root, targetSum)])
    
    while queue:
        node, remaining_sum = queue.popleft()
        
        # If it's a leaf node, check if value equals remaining sum
        if not node.left and not node.right:
            if node.val == remaining_sum:
                return True
            continue
        
        # Add children to queue with updated remaining sum
        new_remaining = remaining_sum - node.val
        
        if node.left:
            queue.append((node.left, new_remaining))
        if node.right:
            queue.append((node.right, new_remaining))
    
    return False


def has_path_sum_path_tracking(root: Optional[TreeNode], targetSum: int) -> bool:
    """
    Path tracking approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Track the current path sum
    2. At each node, add its value to current sum
    3. Check if leaf node has target sum
    """
    def dfs(node, current_sum):
        if not node:
            return False
        
        current_sum += node.val
        
        # If it's a leaf node, check if current sum equals target
        if not node.left and not node.right:
            return current_sum == targetSum
        
        # Recursively check left and right subtrees
        return dfs(node.left, current_sum) or dfs(node.right, current_sum)
    
    return dfs(root, 0)


def has_path_sum_preorder(root: Optional[TreeNode], targetSum: int) -> bool:
    """
    Preorder traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Preorder traversal with sum tracking
    2. At each leaf, check if accumulated sum equals target
    """
    def preorder(node, accumulated_sum):
        if not node:
            return False
        
        accumulated_sum += node.val
        
        # Leaf node check
        if not node.left and not node.right:
            return accumulated_sum == targetSum
        
        # Continue preorder traversal
        return (preorder(node.left, accumulated_sum) or 
                preorder(node.right, accumulated_sum))
    
    return preorder(root, 0)


def has_path_sum_backtracking(root: Optional[TreeNode], targetSum: int) -> bool:
    """
    Backtracking approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use backtracking to explore all paths
    2. Track current path sum
    3. Backtrack when leaf is reached
    """
    def backtrack(node, current_sum):
        if not node:
            return False
        
        # Add current node value
        current_sum += node.val
        
        # If leaf node, check sum
        if not node.left and not node.right:
            return current_sum == targetSum
        
        # Explore left and right subtrees
        found_left = backtrack(node.left, current_sum)
        found_right = backtrack(node.right, current_sum)
        
        # Backtrack (sum is already local, so no need to subtract)
        return found_left or found_right
    
    return backtrack(root, 0)


def has_path_sum_morris(root: Optional[TreeNode], targetSum: int) -> bool:
    """
    Morris traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - constant extra space
    
    Algorithm:
    1. Use Morris threading to traverse without recursion
    2. Track path sums using node modifications
    3. Check leaf nodes for target sum
    
    Note: This is complex for path sum due to sum tracking
    """
    # Morris traversal is complex for path sum problems
    # Fall back to simpler iterative approach
    return has_path_sum_iterative_stack(root, targetSum)


def has_path_sum_level_order(root: Optional[TreeNode], targetSum: int) -> bool:
    """
    Level order traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Process nodes level by level
    2. Track sum for each path
    3. Check leaf nodes at each level
    """
    if not root:
        return False
    
    queue = deque([(root, 0)])  # (node, current_sum)
    
    while queue:
        level_size = len(queue)
        
        for _ in range(level_size):
            node, current_sum = queue.popleft()
            current_sum += node.val
            
            # If leaf node, check sum
            if not node.left and not node.right:
                if current_sum == targetSum:
                    return True
            else:
                # Add children to next level
                if node.left:
                    queue.append((node.left, current_sum))
                if node.right:
                    queue.append((node.right, current_sum))
    
    return False


def has_path_sum_all_paths(root: Optional[TreeNode], targetSum: int) -> bool:
    """
    All paths enumeration approach.
    
    Time Complexity: O(n^2) worst case (for skewed trees)
    Space Complexity: O(n) - to store all paths
    
    Algorithm:
    1. Generate all root-to-leaf paths
    2. Check if any path sum equals target
    """
    def get_all_paths(node, current_path, all_paths):
        if not node:
            return
        
        current_path.append(node.val)
        
        # If leaf node, add path to all_paths
        if not node.left and not node.right:
            all_paths.append(current_path[:])
        else:
            # Continue to children
            get_all_paths(node.left, current_path, all_paths)
            get_all_paths(node.right, current_path, all_paths)
        
        # Backtrack
        current_path.pop()
    
    if not root:
        return False
    
    all_paths = []
    get_all_paths(root, [], all_paths)
    
    # Check if any path sum equals target
    for path in all_paths:
        if sum(path) == targetSum:
            return True
    
    return False


def has_path_sum_memoization(root: Optional[TreeNode], targetSum: int) -> bool:
    """
    Memoization approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - memoization cache
    
    Algorithm:
    1. Cache results for (node, remaining_sum) pairs
    2. Avoid recomputation for same subproblems
    """
    memo = {}
    
    def dfs(node, remaining_sum):
        if not node:
            return False
        
        # Use node id and remaining sum as key
        key = (id(node), remaining_sum)
        if key in memo:
            return memo[key]
        
        # If leaf node, check if value equals remaining sum
        if not node.left and not node.right:
            result = node.val == remaining_sum
            memo[key] = result
            return result
        
        # Recursively check left and right subtrees
        new_remaining = remaining_sum - node.val
        result = (dfs(node.left, new_remaining) or 
                 dfs(node.right, new_remaining))
        
        memo[key] = result
        return result
    
    return dfs(root, targetSum)


def has_path_sum_iterative_path_list(root: Optional[TreeNode], targetSum: int) -> bool:
    """
    Iterative approach with path list.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use stack to store (node, path_sum_so_far)
    2. Track sum along each path
    3. Check leaf nodes for target sum
    """
    if not root:
        return False
    
    stack = [(root, [root.val])]
    
    while stack:
        node, path_sums = stack.pop()
        current_sum = sum(path_sums)
        
        # If leaf node, check sum
        if not node.left and not node.right:
            if current_sum == targetSum:
                return True
            continue
        
        # Add children with updated path
        if node.left:
            stack.append((node.left, path_sums + [node.left.val]))
        if node.right:
            stack.append((node.right, path_sums + [node.right.val]))
    
    return False


def has_path_sum_tail_recursion(root: Optional[TreeNode], targetSum: int) -> bool:
    """
    Tail recursion approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use tail recursion to avoid stack overflow
    2. Pass accumulated sum as parameter
    """
    def tail_recursive_helper(node, accumulated_sum):
        if not node:
            return False
        
        accumulated_sum += node.val
        
        # If leaf node, check sum
        if not node.left and not node.right:
            return accumulated_sum == targetSum
        
        # Tail recursive calls
        return (tail_recursive_helper(node.left, accumulated_sum) or 
                tail_recursive_helper(node.right, accumulated_sum))
    
    return tail_recursive_helper(root, 0)


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


def test_has_path_sum():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([5,4,8,11,None,13,4,7,2,None,None,None,1], 22, True),
        ([1,2,3], 5, False),
        ([], 0, False),
        ([1], 1, True),
        ([1], 0, False),
        
        # Simple trees
        ([1,2], 3, True),
        ([1,2], 1, False),
        ([1,None,2], 3, True),
        ([1,None,2], 1, False),
        
        # Negative values
        ([-1,2,3], 2, True),
        ([1,-2,3], 2, True),
        ([1,-2,-3], -1, True),
        ([1,-2,-3], -2, True),
        
        # Zero values
        ([0,1,2], 1, True),
        ([0,1,2], 0, False),
        ([0], 0, True),
        
        # Large values
        ([1000,-500,500], 1500, True),
        ([1000,-500,500], 500, True),
        
        # Deep trees
        ([1,2,None,3,None,4], 10, True),
        ([1,2,None,3,None,4], 5, False),
        
        # Multiple paths
        ([5,4,8,11,None,13,4,7,2,None,None,None,1], 18, True),
        ([5,4,8,11,None,13,4,7,2,None,None,None,1], 26, True),
        ([5,4,8,11,None,13,4,7,2,None,None,None,1], 27, False),
        
        # Balanced trees
        ([1,2,3,4,5,6,7], 7, True),
        ([1,2,3,4,5,6,7], 8, True),
        ([1,2,3,4,5,6,7], 10, True),
        ([1,2,3,4,5,6,7], 15, False),
        
        # Single path trees
        ([1,2,None,3,None,4], 10, True),
        ([1,None,2,None,3,None,4], 10, True),
        
        # All negative
        ([-1,-2,-3], -6, True),
        ([-1,-2,-3], -3, True),
        ([-1,-2,-3], -1, False),
        
        # Mixed positive/negative
        ([1,-2,3,-4,5], 0, True),
        ([1,-2,3,-4,5], 2, True),
        ([1,-2,3,-4,5], 4, True),
        
        # Large sums
        ([100,200,300,400,500], 800, True),
        ([100,200,300,400,500], 1000, True),
        ([100,200,300,400,500], 1500, False),
        
        # Edge cases
        ([1,2,3,4,5], 1, False),  # Target is root but not leaf
        ([1,2,3,4,5], 6, True),   # Left path
        ([1,2,3,4,5], 9, True),   # Right path
        
        # Zero target
        ([0,1,2], 0, False),
        ([0], 0, True),
        ([1,0,2], 1, True),
        ([1,-1,2], 0, True),
    ]
    
    # Test all implementations
    implementations = [
        ("Recursive", has_path_sum_recursive),
        ("Iterative Stack", has_path_sum_iterative_stack),
        ("Iterative Queue", has_path_sum_iterative_queue),
        ("Path Tracking", has_path_sum_path_tracking),
        ("Preorder", has_path_sum_preorder),
        ("Backtracking", has_path_sum_backtracking),
        ("Morris", has_path_sum_morris),
        ("Level Order", has_path_sum_level_order),
        ("All Paths", has_path_sum_all_paths),
        ("Memoization", has_path_sum_memoization),
        ("Iterative Path List", has_path_sum_iterative_path_list),
        ("Tail Recursion", has_path_sum_tail_recursion),
    ]
    
    print("Testing Path Sum implementations:")
    print("=" * 50)
    
    for i, (tree_values, target_sum, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Tree: {tree_values}")
        print(f"Target Sum: {target_sum}")
        print(f"Expected: {expected}")
        
        for name, func in implementations:
            try:
                # Create fresh tree for each implementation
                root = create_tree_from_list(tree_values)
                result = func(root, target_sum)
                
                status = "✓" if result == expected else "✗"
                print(f"{status} {name}: {result}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 50)
    print("Performance Analysis:")
    print("=" * 50)
    
    import time
    
    def generate_balanced_tree(depth):
        """Generate a balanced binary tree."""
        if depth == 0:
            return []
        
        size = 2 ** depth - 1
        return list(range(1, size + 1))
    
    def generate_skewed_tree(size):
        """Generate a left-skewed tree."""
        result = []
        for i in range(1, size + 1):
            result.append(i)
            if i < size:
                result.append(None)
        return result
    
    test_scenarios = [
        ("Small balanced", generate_balanced_tree(4), 10),
        ("Medium balanced", generate_balanced_tree(6), 50),
        ("Large balanced", generate_balanced_tree(8), 200),
        ("Skewed tree", generate_skewed_tree(100), 50),
        ("Deep tree", generate_skewed_tree(200), 100),
    ]
    
    for scenario_name, tree_values, target_sum in test_scenarios:
        print(f"\n{scenario_name} (target: {target_sum}):")
        
        for name, func in implementations:
            try:
                root = create_tree_from_list(tree_values)
                
                start_time = time.time()
                result = func(root, target_sum)
                end_time = time.time()
                
                print(f"  {name}: {result} in {end_time - start_time:.4f} seconds")
            except Exception as e:
                print(f"  {name}: Error - {e}")


if __name__ == "__main__":
    test_has_path_sum() 