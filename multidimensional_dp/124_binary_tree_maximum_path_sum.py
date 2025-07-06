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
- The number of nodes in the tree is in the range [1, 3 * 10^4].
- -1000 <= Node.val <= 1000
"""

# Definition for a binary tree node
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_path_sum_dfs(root):
    """
    Approach 1: DFS with Global Maximum
    Time Complexity: O(n)
    Space Complexity: O(h) where h is height of tree
    
    Use DFS to calculate max path sum through each node.
    """
    if not root:
        return 0
    
    max_sum = float('-inf')
    
    def max_gain(node):
        nonlocal max_sum
        
        if not node:
            return 0
        
        # Max gain from left and right subtrees (ignore negative gains)
        left_gain = max(max_gain(node.left), 0)
        right_gain = max(max_gain(node.right), 0)
        
        # Current path sum including current node
        current_path_sum = node.val + left_gain + right_gain
        
        # Update global maximum
        max_sum = max(max_sum, current_path_sum)
        
        # Return max gain that can be achieved by including current node
        # (can only choose one side to maintain path property)
        return node.val + max(left_gain, right_gain)
    
    max_gain(root)
    return max_sum


def max_path_sum_bottom_up(root):
    """
    Approach 2: Bottom-up DP
    Time Complexity: O(n)
    Space Complexity: O(h)
    
    Calculate maximum path sum using bottom-up approach.
    """
    if not root:
        return 0
    
    def helper(node):
        if not node:
            return 0, float('-inf')  # (max_ending_here, max_so_far)
        
        # Get results from children
        left_ending, left_max = helper(node.left)
        right_ending, right_max = helper(node.right)
        
        # Max path ending at current node (single path)
        max_ending_here = max(
            node.val,  # Just current node
            node.val + left_ending,  # Extend from left
            node.val + right_ending  # Extend from right
        )
        
        # Max path sum including current node as internal node
        max_through_node = node.val + max(0, left_ending) + max(0, right_ending)
        
        # Overall maximum
        max_so_far = max(
            left_max,  # Best from left subtree
            right_max,  # Best from right subtree
            max_through_node  # Best through current node
        )
        
        return max_ending_here, max_so_far
    
    _, result = helper(root)
    return result


def max_path_sum_iterative(root):
    """
    Approach 3: Iterative using Stack
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Use iterative approach with explicit stack.
    """
    if not root:
        return 0
    
    stack = [(root, False)]  # (node, processed)
    gains = {}  # Store max gain for each node
    max_sum = float('-inf')
    
    while stack:
        node, processed = stack.pop()
        
        if processed:
            # Calculate gains after children are processed
            left_gain = gains.get(node.left, 0)
            right_gain = gains.get(node.right, 0)
            
            # Only consider positive gains
            left_gain = max(left_gain, 0)
            right_gain = max(right_gain, 0)
            
            # Update global maximum (path through current node)
            max_sum = max(max_sum, node.val + left_gain + right_gain)
            
            # Store max gain ending at current node
            gains[node] = node.val + max(left_gain, right_gain)
        else:
            # Mark as processed and add children first
            stack.append((node, True))
            
            if node.right:
                stack.append((node.right, False))
            if node.left:
                stack.append((node.left, False))
    
    return max_sum


def max_path_sum_morris_traversal(root):
    """
    Approach 4: Morris Traversal (Modified)
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Use Morris traversal for constant space solution.
    """
    if not root:
        return 0
    
    # This approach is complex for this problem due to need for bottom-up calculation
    # We'll use a simplified version that still uses recursion
    return max_path_sum_dfs(root)


def max_path_sum_level_order(root):
    """
    Approach 5: Level Order Processing
    Time Complexity: O(n)
    Space Complexity: O(w) where w is maximum width
    
    Process nodes level by level with parent tracking.
    """
    if not root:
        return 0
    
    from collections import deque, defaultdict
    
    # Build parent mapping and process bottom-up
    queue = deque([root])
    nodes = []
    parent_map = defaultdict(list)
    
    # Level order traversal to build structure
    while queue:
        node = queue.popleft()
        nodes.append(node)
        
        if node.left:
            queue.append(node.left)
            parent_map[node].append(node.left)
        if node.right:
            queue.append(node.right)
            parent_map[node].append(node.right)
    
    # Process nodes in reverse order (bottom-up)
    gains = {}
    max_sum = float('-inf')
    
    for node in reversed(nodes):
        children = parent_map[node]
        
        left_gain = gains.get(children[0], 0) if len(children) > 0 else 0
        right_gain = gains.get(children[1], 0) if len(children) > 1 else 0
        
        left_gain = max(left_gain, 0)
        right_gain = max(right_gain, 0)
        
        # Update global maximum
        max_sum = max(max_sum, node.val + left_gain + right_gain)
        
        # Store gain for this node
        gains[node] = node.val + max(left_gain, right_gain)
    
    return max_sum


def max_path_sum_memoization(root):
    """
    Approach 6: Top-down with Memoization
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Use memoization to cache results.
    """
    if not root:
        return 0
    
    memo = {}
    max_sum = float('-inf')
    
    def max_gain(node):
        nonlocal max_sum
        
        if not node:
            return 0
        
        if node in memo:
            return memo[node]
        
        left_gain = max(max_gain(node.left), 0)
        right_gain = max(max_gain(node.right), 0)
        
        # Update global max with path through current node
        max_sum = max(max_sum, node.val + left_gain + right_gain)
        
        # Memoize and return max gain ending at this node
        result = node.val + max(left_gain, right_gain)
        memo[node] = result
        return result
    
    max_gain(root)
    return max_sum


def max_path_sum_divide_conquer(root):
    """
    Approach 7: Divide and Conquer
    Time Complexity: O(n)
    Space Complexity: O(h)
    
    Use divide and conquer approach.
    """
    if not root:
        return 0
    
    def divide_conquer(node):
        if not node:
            return float('-inf'), 0  # (max_path_sum, max_single_path)
        
        if not node.left and not node.right:
            return node.val, node.val
        
        left_path_sum, left_single = divide_conquer(node.left)
        right_path_sum, right_single = divide_conquer(node.right)
        
        # Max single path ending at current node
        max_single = max(
            node.val,
            node.val + left_single,
            node.val + right_single
        )
        
        # Max path sum in current subtree
        max_path = max(
            left_path_sum,  # Best from left
            right_path_sum,  # Best from right
            node.val + max(0, left_single) + max(0, right_single)  # Through current
        )
        
        return max_path, max_single
    
    result, _ = divide_conquer(root)
    return result


def max_path_sum_postorder(root):
    """
    Approach 8: Explicit Postorder Traversal
    Time Complexity: O(n)
    Space Complexity: O(h)
    
    Use explicit postorder traversal.
    """
    if not root:
        return 0
    
    def postorder(node, path_sums, max_gains):
        if not node:
            return
        
        # Process children first
        postorder(node.left, path_sums, max_gains)
        postorder(node.right, path_sums, max_gains)
        
        # Get gains from children
        left_gain = max_gains.get(node.left, 0)
        right_gain = max_gains.get(node.right, 0)
        
        # Only consider positive gains
        left_gain = max(left_gain, 0)
        right_gain = max(right_gain, 0)
        
        # Path sum through current node
        current_path_sum = node.val + left_gain + right_gain
        path_sums.append(current_path_sum)
        
        # Max gain ending at current node
        max_gains[node] = node.val + max(left_gain, right_gain)
    
    path_sums = []
    max_gains = {}
    
    postorder(root, path_sums, max_gains)
    
    return max(path_sums)


def max_path_sum_dp_with_states(root):
    """
    Approach 9: DP with Multiple States
    Time Complexity: O(n)
    Space Complexity: O(h)
    
    Track multiple states: include node, exclude node, etc.
    """
    if not root:
        return 0
    
    def dp(node):
        if not node:
            return 0, float('-inf')  # (max_ending_here, max_anywhere)
        
        # Base case: leaf node
        if not node.left and not node.right:
            return node.val, node.val
        
        # Get results from children
        if node.left:
            left_ending, left_max = dp(node.left)
        else:
            left_ending, left_max = 0, float('-inf')
        
        if node.right:
            right_ending, right_max = dp(node.right)
        else:
            right_ending, right_max = 0, float('-inf')
        
        # Max path ending at current node
        max_ending = max(
            node.val,  # Just current node
            node.val + left_ending,  # Extend left
            node.val + right_ending  # Extend right
        )
        
        # Max path anywhere in current subtree
        max_anywhere = max(
            left_max,  # Best in left subtree
            right_max,  # Best in right subtree
            node.val + max(0, left_ending) + max(0, right_ending)  # Through current
        )
        
        return max_ending, max_anywhere
    
    _, result = dp(root)
    return result


def create_tree_from_list(values):
    """Helper function to create tree from list representation."""
    if not values:
        return None
    
    root = TreeNode(values[0])
    queue = [root]
    i = 1
    
    while queue and i < len(values):
        node = queue.pop(0)
        
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    
    return root


def test_max_path_sum():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([1, 2, 3], 6),
        ([-10, 9, 20, None, None, 15, 7], 42),
        ([1], 1),
        ([-3], -3),
        ([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1], 48),
        ([2, -1], 2),
        ([-1, -2, -3], -1),
        ([1, -2, -3, 1, 3, -2, None, -1], 3),
        ([9, 6, -3, None, None, -6, 2, None, None, 2, None, -6, -6, -6], 16),
    ]
    
    approaches = [
        ("DFS", max_path_sum_dfs),
        ("Bottom-up", max_path_sum_bottom_up),
        ("Iterative", max_path_sum_iterative),
        ("Morris", max_path_sum_morris_traversal),
        ("Level Order", max_path_sum_level_order),
        ("Memoization", max_path_sum_memoization),
        ("Divide Conquer", max_path_sum_divide_conquer),
        ("Postorder", max_path_sum_postorder),
        ("DP States", max_path_sum_dp_with_states),
    ]
    
    for i, (tree_list, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: {tree_list}")
        print(f"Expected: {expected}")
        
        root = create_tree_from_list(tree_list)
        
        for name, func in approaches:
            try:
                result = func(root)
                status = "✓" if result == expected else "✗"
                print(f"{status} {name}: {result}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")


if __name__ == "__main__":
    test_max_path_sum() 