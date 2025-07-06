"""
LeetCode 437: Path Sum III

Given the root of a binary tree and an integer targetSum, return the number of paths 
where the sum of the values along the path equals targetSum.

The path does not need to start or end at the root or a leaf, but it must go downward 
(i.e., traveling only from parent nodes to child nodes).

Example 1:
Input: root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
Output: 3

Example 2:
Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
Output: 3

Constraints:
- The number of nodes in the tree is in the range [0, 1000].
- -10^9 <= Node.val <= 10^9
- -1000 <= targetSum <= 1000
"""

from typing import Optional, List, Dict
from collections import defaultdict, deque


class TreeNode:
    """Binary tree node definition."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def path_sum_brute_force(root: Optional[TreeNode], targetSum: int) -> int:
    """
    Brute force approach - check all possible paths.
    
    Time Complexity: O(n²) in worst case
    Space Complexity: O(h) where h is tree height
    
    Algorithm:
    1. For each node, calculate all paths starting from that node
    2. Count paths that sum to target
    3. Recursively check all nodes
    """
    if not root:
        return 0
    
    def count_paths_from_node(node, current_sum):
        """Count paths starting from current node."""
        if not node:
            return 0
        
        current_sum += node.val
        count = 1 if current_sum == targetSum else 0
        
        # Continue to children
        count += count_paths_from_node(node.left, current_sum)
        count += count_paths_from_node(node.right, current_sum)
        
        return count
    
    # Count paths starting from current node
    result = count_paths_from_node(root, 0)
    
    # Count paths starting from left and right subtrees
    result += path_sum_brute_force(root.left, targetSum)
    result += path_sum_brute_force(root.right, targetSum)
    
    return result


def path_sum_prefix_sum(root: Optional[TreeNode], targetSum: int) -> int:
    """
    Prefix sum approach with memoization.
    
    Time Complexity: O(n)
    Space Complexity: O(h)
    
    Algorithm:
    1. Use prefix sum to track cumulative sum from root
    2. Use hashmap to store frequency of prefix sums
    3. For each node, check if (currentSum - targetSum) exists
    4. Use backtracking to maintain hashmap state
    """
    if not root:
        return 0
    
    prefix_count = defaultdict(int)
    prefix_count[0] = 1  # Empty path has sum 0
    
    def dfs(node, current_sum):
        if not node:
            return 0
        
        current_sum += node.val
        
        # Check if there's a path ending at current node
        count = prefix_count[current_sum - targetSum]
        
        # Add current sum to prefix count
        prefix_count[current_sum] += 1
        
        # Recursively check children
        count += dfs(node.left, current_sum)
        count += dfs(node.right, current_sum)
        
        # Backtrack - remove current sum
        prefix_count[current_sum] -= 1
        
        return count
    
    return dfs(root, 0)


def path_sum_two_pass(root: Optional[TreeNode], targetSum: int) -> int:
    """
    Two-pass approach.
    
    Time Complexity: O(n²)
    Space Complexity: O(h)
    
    Algorithm:
    1. First pass: collect all nodes
    2. Second pass: for each node, count paths starting from it
    3. More organized than brute force
    """
    if not root:
        return 0
    
    def get_all_nodes(node):
        """Get all nodes in the tree."""
        if not node:
            return []
        
        result = [node]
        result.extend(get_all_nodes(node.left))
        result.extend(get_all_nodes(node.right))
        return result
    
    def count_paths_from(node, target, current_sum=0):
        """Count paths starting from given node."""
        if not node:
            return 0
        
        current_sum += node.val
        count = 1 if current_sum == target else 0
        
        count += count_paths_from(node.left, target, current_sum)
        count += count_paths_from(node.right, target, current_sum)
        
        return count
    
    all_nodes = get_all_nodes(root)
    total_count = 0
    
    for node in all_nodes:
        total_count += count_paths_from(node, targetSum)
    
    return total_count


def path_sum_iterative(root: Optional[TreeNode], targetSum: int) -> int:
    """
    Iterative approach using stack.
    
    Time Complexity: O(n²) in worst case
    Space Complexity: O(h)
    
    Algorithm:
    1. Use stack to traverse all nodes
    2. For each node, use another stack to find all paths
    3. Count valid paths
    """
    if not root:
        return 0
    
    def count_paths_iterative(start_node):
        """Count paths starting from start_node iteratively."""
        if not start_node:
            return 0
        
        stack = [(start_node, start_node.val)]
        count = 0
        
        while stack:
            node, current_sum = stack.pop()
            
            if current_sum == targetSum:
                count += 1
            
            if node.left:
                stack.append((node.left, current_sum + node.left.val))
            if node.right:
                stack.append((node.right, current_sum + node.right.val))
        
        return count
    
    # Get all nodes using iterative traversal
    all_nodes = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        all_nodes.append(node)
        
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    # Count paths starting from each node
    total_count = 0
    for node in all_nodes:
        total_count += count_paths_iterative(node)
    
    return total_count


def path_sum_level_order(root: Optional[TreeNode], targetSum: int) -> int:
    """
    Level order traversal approach.
    
    Time Complexity: O(n²)
    Space Complexity: O(w) where w is max width
    
    Algorithm:
    1. Use BFS to traverse nodes level by level
    2. For each node, calculate paths starting from it
    3. Aggregate results
    """
    if not root:
        return 0
    
    def count_paths_bfs(start_node):
        """Count paths starting from start_node using BFS."""
        if not start_node:
            return 0
        
        queue = deque([(start_node, start_node.val)])
        count = 0
        
        while queue:
            node, current_sum = queue.popleft()
            
            if current_sum == targetSum:
                count += 1
            
            if node.left:
                queue.append((node.left, current_sum + node.left.val))
            if node.right:
                queue.append((node.right, current_sum + node.right.val))
        
        return count
    
    # BFS to get all nodes
    queue = deque([root])
    all_nodes = []
    
    while queue:
        node = queue.popleft()
        all_nodes.append(node)
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    # Count paths from each node
    total_count = 0
    for node in all_nodes:
        total_count += count_paths_bfs(node)
    
    return total_count


def path_sum_memoization(root: Optional[TreeNode], targetSum: int) -> int:
    """
    Memoization approach to optimize repeated calculations.
    
    Time Complexity: O(n²) worst case, better average case
    Space Complexity: O(n) for memoization
    
    Algorithm:
    1. Memoize path counts from each node
    2. Avoid recalculating same subproblems
    3. Use node and remaining sum as key
    """
    if not root:
        return 0
    
    memo = {}
    
    def count_paths_memo(node, remaining):
        """Count paths with memoization."""
        if not node:
            return 0
        
        key = (id(node), remaining)
        if key in memo:
            return memo[key]
        
        count = 0
        remaining -= node.val
        
        if remaining == 0:
            count += 1
        
        count += count_paths_memo(node.left, remaining)
        count += count_paths_memo(node.right, remaining)
        
        memo[key] = count
        return count
    
    def dfs(node):
        """DFS to start path counting from each node."""
        if not node:
            return 0
        
        # Count paths starting from current node
        count = count_paths_memo(node, targetSum)
        
        # Add paths from children
        count += dfs(node.left)
        count += dfs(node.right)
        
        return count
    
    return dfs(root)


def path_sum_dynamic_programming(root: Optional[TreeNode], targetSum: int) -> int:
    """
    Dynamic programming approach.
    
    Time Complexity: O(n)
    Space Complexity: O(h)
    
    Algorithm:
    1. Use DP with path tracking
    2. Maintain all possible sums ending at current node
    3. Check if any sum equals target
    """
    if not root:
        return 0
    
    def dfs(node, path_sums):
        """DFS with dynamic path sum tracking."""
        if not node:
            return 0
        
        # Update all existing path sums
        new_path_sums = [s + node.val for s in path_sums]
        new_path_sums.append(node.val)  # New path starting from current node
        
        # Count how many sums equal target
        count = new_path_sums.count(targetSum)
        
        # Recursively check children
        count += dfs(node.left, new_path_sums)
        count += dfs(node.right, new_path_sums)
        
        return count
    
    return dfs(root, [])


def path_sum_preorder_iterative(root: Optional[TreeNode], targetSum: int) -> int:
    """
    Preorder iterative approach with path tracking.
    
    Time Complexity: O(n²)
    Space Complexity: O(h)
    
    Algorithm:
    1. Use preorder traversal
    2. Maintain current path sums
    3. Count valid paths at each node
    """
    if not root:
        return 0
    
    total_count = 0
    stack = [(root, [])]  # (node, path_sums_so_far)
    
    while stack:
        node, path_sums = stack.pop()
        
        # Update path sums with current node
        new_path_sums = [s + node.val for s in path_sums]
        new_path_sums.append(node.val)
        
        # Count paths ending at current node
        total_count += new_path_sums.count(targetSum)
        
        # Add children to stack
        if node.right:
            stack.append((node.right, new_path_sums))
        if node.left:
            stack.append((node.left, new_path_sums))
    
    return total_count


def path_sum_segment_tree(root: Optional[TreeNode], targetSum: int) -> int:
    """
    Segment tree inspired approach.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Algorithm:
    1. Use segment tree concepts for range sum queries
    2. Build path segments efficiently
    3. Query for target sums
    """
    if not root:
        return 0
    
    def build_path_segments(node, current_path):
        """Build all path segments from current path."""
        if not node:
            return 0
        
        current_path.append(node.val)
        count = 0
        
        # Check all segments ending at current node
        current_sum = 0
        for i in range(len(current_path) - 1, -1, -1):
            current_sum += current_path[i]
            if current_sum == targetSum:
                count += 1
        
        # Recursively check children
        count += build_path_segments(node.left, current_path.copy())
        count += build_path_segments(node.right, current_path.copy())
        
        return count
    
    return build_path_segments(root, [])


def path_sum_rolling_hash(root: Optional[TreeNode], targetSum: int) -> int:
    """
    Rolling hash approach for path sum calculation.
    
    Time Complexity: O(n)
    Space Complexity: O(h)
    
    Algorithm:
    1. Use rolling hash technique
    2. Maintain hash of path sums
    3. Quick lookup for target differences
    """
    if not root:
        return 0
    
    from collections import Counter
    
    def dfs(node, current_sum, sum_count):
        if not node:
            return 0
        
        current_sum += node.val
        
        # Count paths ending at current node
        count = sum_count[current_sum - targetSum]
        
        # Add current sum to counter
        sum_count[current_sum] += 1
        
        # Recursively process children
        count += dfs(node.left, current_sum, sum_count)
        count += dfs(node.right, current_sum, sum_count)
        
        # Backtrack - remove current sum
        sum_count[current_sum] -= 1
        
        return count
    
    sum_counter = Counter([0])  # Initialize with 0 for empty path
    return dfs(root, 0, sum_counter)


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


def test_path_sum_iii():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([], 0, 0),
        ([1], 1, 1),
        ([1], 2, 0),
        
        # Simple cases
        ([1, 2], 3, 1),
        ([1, 2], 1, 1),
        ([1, 2], 2, 1),
        
        # Given examples
        ([10, 5, -3, 3, 2, None, 11, 3, -2, None, 1], 8, 3),
        ([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, 5, 1], 22, 3),
        
        # Edge cases
        ([1, -2, -3], -1, 1),
        ([1, -2, -3], -2, 1),
        ([1, -2, -3], -3, 1),
        
        # Negative values
        ([-1, -2, -3], -3, 2),  # -3 alone and -1 + -2
        ([1, -1, 0], 0, 2),     # 0 alone and 1 + (-1)
        
        # Longer paths
        ([1, 2, 3, 4, 5], 6, 1),  # 1 + 2 + 3
        ([1, 2, 3, 4, 5], 7, 1),  # 2 + 3 + 4 - not valid (not continuous)
        ([1, 2, 3, 4, 5], 9, 1),  # 4 + 5
        
        # Multiple valid paths
        ([1, 1, 1, 1, 1], 2, 4),  # Multiple pairs summing to 2
        ([2, 1, 3, 1, 1, 1, 1], 3, 4),  # Multiple paths
        
        # Large values
        ([1000, -500, 200, -100, 50], 500, 1),
        
        # Complex trees
        ([10, 5, -3, 3, 2, None, 11, 3, -2, None, 1], 18, 1),  # 5 + 2 + 11
        ([1, 2, 3, 4, 5, 6, 7], 7, 3),  # Multiple paths to 7
        
        # Zero target
        ([0, 1, 0, 1, 0], 0, 3),
        ([1, -1, 0], 0, 2),
        
        # Single large path
        ([1, 2, 3, 4, 5, 6, 7, 8], 36, 1),  # 1+2+3+4+5+6+7+8
    ]
    
    # Test all implementations
    implementations = [
        ("Brute Force", path_sum_brute_force),
        ("Prefix Sum", path_sum_prefix_sum),
        ("Two Pass", path_sum_two_pass),
        ("Iterative", path_sum_iterative),
        ("Level Order", path_sum_level_order),
        ("Memoization", path_sum_memoization),
        ("Dynamic Programming", path_sum_dynamic_programming),
        ("Preorder Iterative", path_sum_preorder_iterative),
        ("Segment Tree", path_sum_segment_tree),
        ("Rolling Hash", path_sum_rolling_hash),
    ]
    
    print("Testing Path Sum III implementations:")
    print("=" * 75)
    
    for i, (nodes, target, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Tree:     {nodes}")
        print(f"Target:   {target}")
        print(f"Expected: {expected}")
        
        tree = build_test_tree(nodes)
        
        for name, func in implementations:
            try:
                result = func(tree, target)
                status = "✓" if result == expected else "✗"
                print(f"{status} {name}: {result}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 75)
    print("Performance Analysis:")
    print("=" * 75)
    
    import time
    
    # Generate large test cases
    def generate_test_tree(size, target_sum):
        """Generate test tree for performance testing."""
        # Create a tree with values that might create interesting paths
        nodes = []
        for i in range(size):
            if i % 3 == 0:
                nodes.append(target_sum // 3)
            elif i % 3 == 1:
                nodes.append(target_sum // 4)
            else:
                nodes.append(1)
        return nodes, target_sum
    
    test_scenarios = [
        ("Small tree", 15, 10),
        ("Medium tree", 63, 20),
        ("Large tree", 255, 30),
        ("Very large", 511, 40),
    ]
    
    for scenario_name, size, target in test_scenarios:
        print(f"\n{scenario_name} ({size} nodes, target={target}):")
        nodes, target_sum = generate_test_tree(size, target)
        tree = build_test_tree(nodes)
        
        # Test only efficient implementations for large cases
        test_implementations = implementations[:6] if size <= 63 else implementations[1:3]
        
        for name, func in test_implementations:
            try:
                start_time = time.time()
                result = func(tree, target_sum)
                end_time = time.time()
                
                elapsed = (end_time - start_time) * 1000  # Convert to milliseconds
                print(f"  {name}: {elapsed:.2f} ms (result: {result})")
            except Exception as e:
                print(f"  {name}: Error - {e}")
    
    print(f"\n{'='*75}")
    print("Algorithm Complexity Analysis:")
    print("="*75)
    print("1. Brute Force:         O(n²) time, O(h) space")
    print("2. Prefix Sum:          O(n) time, O(h) space")
    print("3. Two Pass:            O(n²) time, O(h) space")
    print("4. Iterative:           O(n²) time, O(h) space")
    print("5. Level Order:         O(n²) time, O(w) space")
    print("6. Memoization:         O(n²) time, O(n) space")
    print("7. Dynamic Programming: O(n²) time, O(h) space")
    print("8. Preorder Iterative:  O(n²) time, O(h) space")
    print("9. Segment Tree:        O(n log n) time, O(n) space")
    print("10. Rolling Hash:       O(n) time, O(h) space")
    
    print(f"\n{'='*75}")
    print("Key Insights:")
    print("="*75)
    print("• Prefix sum with HashMap gives optimal O(n) solution")
    print("• Path doesn't need to start/end at root or leaf")
    print("• Backtracking is crucial for correct HashMap state")
    print("• Multiple approaches exist but prefix sum is most efficient")
    print("• DFS with memoization can optimize brute force approach")


if __name__ == "__main__":
    test_path_sum_iii() 