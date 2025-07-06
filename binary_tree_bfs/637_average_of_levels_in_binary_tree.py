"""
637. Average of Levels in Binary Tree

Given the root of a binary tree, return the average value of the nodes on each level in the form of an array.

Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: [3.00000,14.50000,11.00000]
Explanation: The average value of nodes on level 0 is 3, on level 1 is 14.5, and on level 2 is 11.
Hence return [3, 14.5, 11].

Example 2:
Input: root = [3,9,20,15,7]
Output: [3.00000,14.50000,11.00000]

Constraints:
- The number of nodes in the tree is in the range [1, 10^4]
- -2^31 <= Node.val <= 2^31 - 1
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


def average_of_levels_bfs(root: Optional[TreeNode]) -> List[float]:
    """
    BFS level order traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use BFS to traverse level by level
    2. Calculate sum and count for each level
    3. Compute average for each level
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level_sum = 0
        
        for _ in range(level_size):
            node = queue.popleft()
            level_sum += node.val
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        # Calculate average for this level
        average = level_sum / level_size
        result.append(average)
    
    return result


def average_of_levels_dfs_recursive(root: Optional[TreeNode]) -> List[float]:
    """
    DFS recursive approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use DFS to collect sums and counts for each level
    2. Calculate averages from collected data
    """
    if not root:
        return []
    
    level_sums = []
    level_counts = []
    
    def dfs(node, level):
        if not node:
            return
        
        # Extend lists if this is a new level
        if level >= len(level_sums):
            level_sums.append(0)
            level_counts.append(0)
        
        # Add current node's value to its level
        level_sums[level] += node.val
        level_counts[level] += 1
        
        # Recursively process children
        dfs(node.left, level + 1)
        dfs(node.right, level + 1)
    
    dfs(root, 0)
    
    # Calculate averages
    return [level_sums[i] / level_counts[i] for i in range(len(level_sums))]


def average_of_levels_iterative_stack(root: Optional[TreeNode]) -> List[float]:
    """
    Iterative approach using stack with level tracking.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use stack to store (node, level) pairs
    2. Collect sums and counts for each level
    3. Calculate averages
    """
    if not root:
        return []
    
    level_sums = {}
    level_counts = {}
    stack = [(root, 0)]
    
    while stack:
        node, level = stack.pop()
        
        # Update level statistics
        level_sums[level] = level_sums.get(level, 0) + node.val
        level_counts[level] = level_counts.get(level, 0) + 1
        
        # Add children to stack
        if node.right:
            stack.append((node.right, level + 1))
        if node.left:
            stack.append((node.left, level + 1))
    
    # Calculate averages in level order
    max_level = max(level_sums.keys())
    return [level_sums[i] / level_counts[i] for i in range(max_level + 1)]


def average_of_levels_bfs_precise(root: Optional[TreeNode]) -> List[float]:
    """
    BFS approach with precise floating point arithmetic.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use BFS for level order traversal
    2. Handle large numbers carefully to avoid overflow
    3. Use precise division for accurate averages
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level_sum = 0
        
        for _ in range(level_size):
            node = queue.popleft()
            level_sum += node.val
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        # Use float division for precise result
        average = float(level_sum) / float(level_size)
        result.append(average)
    
    return result


def average_of_levels_list_based(root: Optional[TreeNode]) -> List[float]:
    """
    List-based level processing approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use list to store current level nodes
    2. Process entire level at once
    3. Calculate average for each level
    """
    if not root:
        return []
    
    result = []
    current_level = [root]
    
    while current_level:
        # Calculate average for current level
        level_sum = sum(node.val for node in current_level)
        level_count = len(current_level)
        result.append(level_sum / level_count)
        
        # Build next level
        next_level = []
        for node in current_level:
            if node.left:
                next_level.append(node.left)
            if node.right:
                next_level.append(node.right)
        
        current_level = next_level
    
    return result


def average_of_levels_generator(root: Optional[TreeNode]) -> List[float]:
    """
    Generator-based approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use generator to yield level values
    2. Calculate averages lazily
    3. Memory efficient for large trees
    """
    def generate_level_averages(node):
        if not node:
            return
        
        current_level = [node]
        
        while current_level:
            # Calculate average for current level
            level_sum = sum(n.val for n in current_level)
            level_count = len(current_level)
            yield level_sum / level_count
            
            # Build next level
            next_level = []
            for n in current_level:
                if n.left:
                    next_level.append(n.left)
                if n.right:
                    next_level.append(n.right)
            
            current_level = next_level
    
    return list(generate_level_averages(root))


def average_of_levels_two_pass(root: Optional[TreeNode]) -> List[float]:
    """
    Two-pass approach: collect all nodes, then calculate averages.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) to store all nodes with levels
    
    Algorithm:
    1. First pass: collect all nodes with their levels
    2. Second pass: group by level and calculate averages
    """
    if not root:
        return []
    
    # First pass: collect all nodes with levels
    nodes_with_levels = []
    
    def collect_nodes(node, level):
        if not node:
            return
        
        nodes_with_levels.append((node.val, level))
        collect_nodes(node.left, level + 1)
        collect_nodes(node.right, level + 1)
    
    collect_nodes(root, 0)
    
    # Second pass: group by level and calculate averages
    level_groups = {}
    for val, level in nodes_with_levels:
        if level not in level_groups:
            level_groups[level] = []
        level_groups[level].append(val)
    
    # Calculate averages in level order
    max_level = max(level_groups.keys())
    result = []
    for i in range(max_level + 1):
        values = level_groups[i]
        average = sum(values) / len(values)
        result.append(average)
    
    return result


def average_of_levels_deque_rotation(root: Optional[TreeNode]) -> List[float]:
    """
    Deque with rotation approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use deque for efficient level processing
    2. Rotate deque for level boundaries
    3. Calculate averages efficiently
    """
    if not root:
        return []
    
    result = []
    dq = deque([root])
    
    while dq:
        level_size = len(dq)
        level_sum = 0
        
        # Process current level
        for _ in range(level_size):
            node = dq.popleft()
            level_sum += node.val
            
            # Add children
            if node.left:
                dq.append(node.left)
            if node.right:
                dq.append(node.right)
        
        # Calculate and store average
        result.append(level_sum / level_size)
    
    return result


def average_of_levels_accumulator(root: Optional[TreeNode]) -> List[float]:
    """
    Accumulator-based approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use accumulator pattern for level statistics
    2. Maintain running sums and counts
    3. Calculate averages from accumulators
    """
    if not root:
        return []
    
    def accumulate_levels(node, level, acc_sums, acc_counts):
        if not node:
            return
        
        # Extend accumulators if needed
        while len(acc_sums) <= level:
            acc_sums.append(0)
            acc_counts.append(0)
        
        # Update accumulators
        acc_sums[level] += node.val
        acc_counts[level] += 1
        
        # Recursively process children
        accumulate_levels(node.left, level + 1, acc_sums, acc_counts)
        accumulate_levels(node.right, level + 1, acc_sums, acc_counts)
    
    sums = []
    counts = []
    accumulate_levels(root, 0, sums, counts)
    
    # Calculate averages
    return [sums[i] / counts[i] for i in range(len(sums))]


def average_of_levels_functional(root: Optional[TreeNode]) -> List[float]:
    """
    Functional programming approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use functional programming concepts
    2. Higher-order functions for level processing
    3. Immutable data structures where possible
    """
    if not root:
        return []
    
    def get_level_values(nodes):
        return [node.val for node in nodes if node]
    
    def get_next_level(nodes):
        next_level = []
        for node in nodes:
            if node:
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
        return next_level
    
    def calculate_average(values):
        return sum(values) / len(values) if values else 0.0
    
    result = []
    current_level = [root]
    
    while current_level:
        values = get_level_values(current_level)
        if values:
            result.append(calculate_average(values))
        current_level = get_next_level(current_level)
    
    return result


def average_of_levels_class_based(root: Optional[TreeNode]) -> List[float]:
    """
    Class-based approach for state management.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use class to encapsulate logic
    2. Maintain state within class
    3. Clean separation of concerns
    """
    class AverageCalculator:
        def __init__(self):
            self.result = []
        
        def calculate_averages(self, root):
            if not root:
                return []
            
            queue = deque([root])
            
            while queue:
                level_size = len(queue)
                level_sum = 0
                
                for _ in range(level_size):
                    node = queue.popleft()
                    level_sum += node.val
                    
                    if node.left:
                        queue.append(node.left)
                    if node.right:
                        queue.append(node.right)
                
                average = level_sum / level_size
                self.result.append(average)
            
            return self.result
    
    calculator = AverageCalculator()
    return calculator.calculate_averages(root)


def average_of_levels_with_statistics(root: Optional[TreeNode]) -> List[float]:
    """
    Approach with detailed statistics collection.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Collect detailed statistics for each level
    2. Include min, max, sum, count for debugging
    3. Calculate averages from statistics
    """
    if not root:
        return []
    
    level_stats = {}
    
    def collect_stats(node, level):
        if not node:
            return
        
        if level not in level_stats:
            level_stats[level] = {
                'sum': 0,
                'count': 0,
                'min': float('inf'),
                'max': float('-inf'),
                'values': []
            }
        
        stats = level_stats[level]
        stats['sum'] += node.val
        stats['count'] += 1
        stats['min'] = min(stats['min'], node.val)
        stats['max'] = max(stats['max'], node.val)
        stats['values'].append(node.val)
        
        collect_stats(node.left, level + 1)
        collect_stats(node.right, level + 1)
    
    collect_stats(root, 0)
    
    # Calculate averages from statistics
    max_level = max(level_stats.keys())
    result = []
    for i in range(max_level + 1):
        stats = level_stats[i]
        average = stats['sum'] / stats['count']
        result.append(average)
    
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


def test_average_of_levels():
    """Test all implementations with various test cases."""
    
    def are_close(a, b, tolerance=1e-5):
        """Check if two float lists are approximately equal."""
        if len(a) != len(b):
            return False
        return all(abs(x - y) < tolerance for x, y in zip(a, b))
    
    test_cases = [
        # Basic cases
        ([3,9,20,None,None,15,7], [3.0, 14.5, 11.0]),
        ([3,9,20,15,7], [3.0, 14.5, 11.0]),
        ([1], [1.0]),
        
        # Two nodes
        ([1,2], [1.0, 2.0]),
        ([1,None,2], [1.0, 2.0]),
        
        # Three nodes
        ([1,2,3], [1.0, 2.5]),
        ([1,None,2,None,3], [1.0, 2.0, 3.0]),
        
        # Complete binary tree
        ([1,2,3,4,5,6,7], [1.0, 2.5, 5.5]),
        
        # Left skewed
        ([1,2,None,3,None,4], [1.0, 2.0, 3.0, 4.0]),
        
        # Right skewed
        ([1,None,2,None,3,None,4], [1.0, 2.0, 3.0, 4.0]),
        
        # Negative values
        ([-1,2,-3], [-1.0, -0.5]),
        ([1,-2,3], [1.0, 0.5]),
        ([-1,-2,-3], [-1.0, -2.5]),
        
        # Zero values
        ([0,1,2], [0.0, 1.5]),
        ([1,0,2], [1.0, 1.0]),
        ([0], [0.0]),
        
        # Large values
        ([1000,500,1500], [1000.0, 1000.0]),
        ([100,200,300,400,500], [100.0, 250.0, 450.0]),
        
        # Mixed positive/negative
        ([1,2,-3,4,-5], [1.0, -0.5, -0.5]),
        ([10,-5,15,-10,5], [10.0, 5.0, -2.5]),
        
        # Same values
        ([5,5,5,5,5], [5.0, 5.0, 5.0]),
        ([1,1,1], [1.0, 1.0]),
        
        # Large tree
        ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
         [1.0, 2.5, 5.5, 11.5]),
        
        # Unbalanced
        ([1,2,3,4,None,None,5], [1.0, 2.5, 4.5]),
        ([1,2,3,None,4,5,None], [1.0, 2.5, 4.5]),
        
        # Deep tree
        ([1,2,None,3,None,4,None,5,None,6], 
         [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        
        # Wide tree at bottom
        ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], 
         [1.0, 2.5, 5.5, 11.5, 16.0]),
        
        # Fractional averages
        ([2,1,3], [2.0, 2.0]),
        ([10,5,15,2,7,12,20], [10.0, 10.0, 10.25]),
        
        # Edge cases with precision
        ([1,1000000000,-1000000000], [1.0, 0.0]),
        ([2147483647,-2147483648,0], [2147483647.0, -715827882.5]),
    ]
    
    # Test all implementations
    implementations = [
        ("BFS", average_of_levels_bfs),
        ("DFS Recursive", average_of_levels_dfs_recursive),
        ("Iterative Stack", average_of_levels_iterative_stack),
        ("BFS Precise", average_of_levels_bfs_precise),
        ("List Based", average_of_levels_list_based),
        ("Generator", average_of_levels_generator),
        ("Two Pass", average_of_levels_two_pass),
        ("Deque Rotation", average_of_levels_deque_rotation),
        ("Accumulator", average_of_levels_accumulator),
        ("Functional", average_of_levels_functional),
        ("Class Based", average_of_levels_class_based),
        ("With Statistics", average_of_levels_with_statistics),
    ]
    
    print("Testing Average of Levels in Binary Tree implementations:")
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
                
                status = "✓" if are_close(result, expected) else "✗"
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
    
    def generate_skewed_tree(size):
        """Generate a skewed tree."""
        result = []
        for i in range(1, size + 1):
            result.append(i)
            if i < size:
                result.append(None)
        return result
    
    test_scenarios = [
        ("Small balanced", generate_balanced_tree(4)),
        ("Medium balanced", generate_balanced_tree(6)),
        ("Large balanced", generate_balanced_tree(8)),
        ("Very large balanced", generate_balanced_tree(10)),
        ("Skewed tree", generate_skewed_tree(100)),
    ]
    
    for scenario_name, tree_values in test_scenarios:
        print(f"\n{scenario_name}:")
        
        for name, func in implementations:
            try:
                root = create_tree_from_list(tree_values)
                
                start_time = time.time()
                result = func(root)
                end_time = time.time()
                
                levels = len(result)
                print(f"  {name}: {levels} levels in {end_time - start_time:.4f} seconds")
            except Exception as e:
                print(f"  {name}: Error - {e}")


if __name__ == "__main__":
    test_average_of_levels() 