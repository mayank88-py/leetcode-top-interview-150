"""
103. Binary Tree Zigzag Level Order Traversal

Given the root of a binary tree, return the zigzag level order traversal of its nodes' values. (i.e., from left to right, then right to left for the next level and alternate between).

Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[20,9],[15,7]]

Example 2:
Input: root = [1]
Output: [[1]]

Example 3:
Input: root = []
Output: []

Constraints:
- The number of nodes in the tree is in the range [0, 2000]
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


def zigzag_level_order_bfs_reverse(root: Optional[TreeNode]) -> List[List[int]]:
    """
    BFS approach with alternating reverse.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use standard BFS for level order traversal
    2. Reverse odd-numbered levels (0-indexed)
    3. Maintain left-to-right and right-to-left alternation
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    left_to_right = True
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            # Add children for next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        # Reverse if needed
        if not left_to_right:
            current_level.reverse()
        
        result.append(current_level)
        left_to_right = not left_to_right  # Toggle direction
    
    return result


def zigzag_level_order_deque_alternating(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Deque approach with alternating insertion.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use deque for O(1) insertion at both ends
    2. Alternate between appendleft and append
    3. Add children in appropriate order based on direction
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    left_to_right = True
    
    while queue:
        level_size = len(queue)
        current_level = deque()
        
        for _ in range(level_size):
            node = queue.popleft()
            
            # Add to current level based on direction
            if left_to_right:
                current_level.append(node.val)
            else:
                current_level.appendleft(node.val)
            
            # Add children for next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(list(current_level))
        left_to_right = not left_to_right
    
    return result


def zigzag_level_order_two_stacks(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Two stacks approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use two stacks for alternating levels
    2. One stack for left-to-right, another for right-to-left
    3. Add children in different orders for each stack
    """
    if not root:
        return []
    
    result = []
    current_level = [root]
    left_to_right = True
    
    while current_level:
        level_values = []
        next_level = []
        
        if left_to_right:
            # Process left to right, add children right to left
            for node in current_level:
                level_values.append(node.val)
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
        else:
            # Process right to left, add children left to right
            for node in reversed(current_level):
                level_values.append(node.val)
            
            # Add children in reverse order
            for node in current_level:
                if node.right:
                    next_level.append(node.right)
                if node.left:
                    next_level.append(node.left)
        
        result.append(level_values)
        current_level = next_level
        left_to_right = not left_to_right
    
    return result


def zigzag_level_order_dfs_recursive(root: Optional[TreeNode]) -> List[List[int]]:
    """
    DFS recursive approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use DFS with level tracking
    2. Insert nodes at appropriate positions based on level
    3. Use list insertion for zigzag pattern
    """
    if not root:
        return []
    
    result = []
    
    def dfs(node, level):
        if not node:
            return
        
        # Extend result if needed
        if level >= len(result):
            result.append([])
        
        # Add node based on level parity
        if level % 2 == 0:  # Even levels: left to right
            result[level].append(node.val)
        else:  # Odd levels: right to left
            result[level].insert(0, node.val)
        
        # Recursively process children
        dfs(node.left, level + 1)
        dfs(node.right, level + 1)
    
    dfs(root, 0)
    return result


def zigzag_level_order_iterative_stack(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Iterative approach using stack with level tracking.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use stack to store (node, level) pairs
    2. Build levels and apply zigzag pattern
    3. Handle insertion order based on level
    """
    if not root:
        return []
    
    stack = [(root, 0)]
    level_map = {}
    
    while stack:
        node, level = stack.pop()
        
        # Initialize level if needed
        if level not in level_map:
            level_map[level] = []
        
        # Add node based on level parity
        if level % 2 == 0:  # Even levels: append (left to right)
            level_map[level].append(node.val)
        else:  # Odd levels: prepend (right to left)
            level_map[level].insert(0, node.val)
        
        # Add children to stack (right first for DFS order)
        if node.right:
            stack.append((node.right, level + 1))
        if node.left:
            stack.append((node.left, level + 1))
    
    # Convert to result format
    max_level = max(level_map.keys())
    return [level_map[i] for i in range(max_level + 1)]


def zigzag_level_order_bfs_with_direction(root: Optional[TreeNode]) -> List[List[int]]:
    """
    BFS approach storing direction explicitly.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Store (node, direction) pairs in queue
    2. Direction indicates how to process current level
    3. Alternate direction for each level
    """
    if not root:
        return []
    
    result = []
    queue = deque([(root, True)])  # (node, left_to_right)
    
    while queue:
        level_size = len(queue)
        current_level = []
        next_direction = None
        
        for i in range(level_size):
            node, direction = queue.popleft()
            
            if i == 0:  # Set direction for this level
                next_direction = not direction
            
            current_level.append(node.val)
            
            # Add children with next direction
            if node.left:
                queue.append((node.left, next_direction))
            if node.right:
                queue.append((node.right, next_direction))
        
        # Apply current level's direction
        if not queue or queue[0][1]:  # If next level is left_to_right, current was right_to_left
            current_level.reverse()
        
        result.append(current_level)
    
    return result


def zigzag_level_order_list_reversal(root: Optional[TreeNode]) -> List[List[int]]:
    """
    List-based approach with post-processing reversal.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Perform normal level order traversal
    2. Reverse odd-indexed levels in post-processing
    3. Simple and clear implementation
    """
    if not root:
        return []
    
    # Standard level order traversal
    result = []
    current_level = [root]
    
    while current_level:
        level_values = [node.val for node in current_level]
        result.append(level_values)
        
        # Build next level
        next_level = []
        for node in current_level:
            if node.left:
                next_level.append(node.left)
            if node.right:
                next_level.append(node.right)
        
        current_level = next_level
    
    # Reverse odd-indexed levels (1, 3, 5, ...)
    for i in range(1, len(result), 2):
        result[i].reverse()
    
    return result


def zigzag_level_order_generator(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Generator-based approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use generator to yield zigzag levels
    2. Memory efficient for large trees
    3. Lazy evaluation of levels
    """
    def generate_zigzag_levels(node):
        if not node:
            return
        
        current_level = [node]
        left_to_right = True
        
        while current_level:
            # Extract values
            level_values = [n.val for n in current_level]
            
            # Apply zigzag pattern
            if not left_to_right:
                level_values.reverse()
            
            yield level_values
            
            # Build next level
            next_level = []
            for n in current_level:
                if n.left:
                    next_level.append(n.left)
                if n.right:
                    next_level.append(n.right)
            
            current_level = next_level
            left_to_right = not left_to_right
    
    return list(generate_zigzag_levels(root))


def zigzag_level_order_manual_queue(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Manual queue implementation.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Implement queue manually using list
    2. Handle zigzag pattern during traversal
    3. Educational implementation
    """
    if not root:
        return []
    
    result = []
    queue = [root]
    left_to_right = True
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        # Process current level
        for _ in range(level_size):
            node = queue.pop(0)  # Manual dequeue
            current_level.append(node.val)
            
            # Add children
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        # Apply zigzag pattern
        if not left_to_right:
            current_level.reverse()
        
        result.append(current_level)
        left_to_right = not left_to_right
    
    return result


def zigzag_level_order_functional(root: Optional[TreeNode]) -> List[List[int]]:
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
    
    def get_level_values(nodes, left_to_right):
        values = [node.val for node in nodes if node]
        return values if left_to_right else values[::-1]
    
    def get_next_level(nodes):
        next_level = []
        for node in nodes:
            if node:
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
        return next_level
    
    result = []
    current_level = [root]
    left_to_right = True
    
    while current_level:
        level_values = get_level_values(current_level, left_to_right)
        if level_values:
            result.append(level_values)
        
        current_level = get_next_level(current_level)
        left_to_right = not left_to_right
    
    return result


def zigzag_level_order_class_based(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Class-based approach for state management.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use class to encapsulate zigzag logic
    2. Maintain state within class
    3. Clean separation of concerns
    """
    class ZigzagTraverser:
        def __init__(self):
            self.result = []
            self.left_to_right = True
        
        def traverse(self, root):
            if not root:
                return []
            
            queue = deque([root])
            
            while queue:
                level_size = len(queue)
                current_level = []
                
                for _ in range(level_size):
                    node = queue.popleft()
                    current_level.append(node.val)
                    
                    if node.left:
                        queue.append(node.left)
                    if node.right:
                        queue.append(node.right)
                
                if not self.left_to_right:
                    current_level.reverse()
                
                self.result.append(current_level)
                self.left_to_right = not self.left_to_right
            
            return self.result
    
    traverser = ZigzagTraverser()
    return traverser.traverse(root)


def zigzag_level_order_optimized(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Optimized approach for large trees.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Minimize memory allocations
    2. Optimize for cache locality
    3. Reduce function call overhead
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    level = 0
    
    while queue:
        level_size = len(queue)
        
        if level % 2 == 0:  # Even level: left to right
            current_level = []
            for _ in range(level_size):
                node = queue.popleft()
                current_level.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        else:  # Odd level: right to left
            current_level = [0] * level_size  # Pre-allocate
            for i in range(level_size):
                node = queue.popleft()
                current_level[level_size - 1 - i] = node.val  # Insert in reverse
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        
        result.append(current_level)
        level += 1
    
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


def test_zigzag_level_order():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([3,9,20,None,None,15,7], [[3],[20,9],[15,7]]),
        ([1], [[1]]),
        ([], []),
        
        # Two nodes
        ([1,2], [[1],[2]]),
        ([1,None,2], [[1],[2]]),
        
        # Three nodes
        ([1,2,3], [[1],[3,2]]),
        ([1,None,2,None,3], [[1],[2],[3]]),
        
        # Complete binary tree
        ([1,2,3,4,5,6,7], [[1],[3,2],[4,5,6,7]]),
        
        # Left skewed
        ([1,2,None,3,None,4], [[1],[2],[3],[4]]),
        
        # Right skewed
        ([1,None,2,None,3,None,4], [[1],[2],[3],[4]]),
        
        # Large balanced tree
        ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
         [[1],[3,2],[4,5,6,7],[15,14,13,12,11,10,9,8]]),
        
        # Unbalanced tree
        ([1,2,3,4,None,None,5], [[1],[3,2],[4,5]]),
        ([1,2,3,None,4,5,None], [[1],[3,2],[4,5]]),
        
        # Single path
        ([1,2,None,3,None,4,None,5], [[1],[2],[3],[4],[5]]),
        
        # Negative values
        ([-1,2,-3], [[-1],[-3,2]]),
        ([1,-2,3], [[1],[3,-2]]),
        ([-1,-2,-3], [[-1],[-3,-2]]),
        
        # Zero values
        ([0,1,2], [[0],[2,1]]),
        ([1,0,2], [[1],[2,0]]),
        ([0], [[0]]),
        
        # Mixed values
        ([5,4,8,11,None,13,4,7,2,None,None,None,1], 
         [[5],[8,4],[11,13,4],[1,2,7]]),
        
        # Wide at certain levels
        ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], 
         [[1],[3,2],[4,5,6,7],[15,14,13,12,11,10,9,8],[16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]]),
        
        # Deep tree
        ([1,2,None,3,None,4,None,5,None,6,None,7], 
         [[1],[2],[3],[4],[5],[6],[7]]),
        
        # Complex structure
        ([1,2,3,4,5,None,6,None,None,7,8,None,9], 
         [[1],[3,2],[4,5,6],[9,8,7]]),
        
        # All same values
        ([1,1,1,1,1,1,1], [[1],[1,1],[1,1,1,1]]),
        
        # Large values
        ([100,50,150,25,75,125,175], [[100],[150,50],[25,75,125,175]]),
        
        # Edge cases
        ([1,2,3,None,None,4,5], [[1],[3,2],[4,5]]),
        ([1,None,2,3,4], [[1],[2],[4,3]]),
        
        # Sparse tree
        ([1,None,2,3,None,4,None,5], [[1],[2],[3],[4],[5]]),
        
        # Missing right subtree
        ([1,2,None,3,4], [[1],[2],[4,3]]),
        
        # Missing left subtree
        ([1,None,2,None,3,4], [[1],[2],[3,4]]),
    ]
    
    # Test all implementations
    implementations = [
        ("BFS Reverse", zigzag_level_order_bfs_reverse),
        ("Deque Alternating", zigzag_level_order_deque_alternating),
        ("Two Stacks", zigzag_level_order_two_stacks),
        ("DFS Recursive", zigzag_level_order_dfs_recursive),
        ("Iterative Stack", zigzag_level_order_iterative_stack),
        ("BFS with Direction", zigzag_level_order_bfs_with_direction),
        ("List Reversal", zigzag_level_order_list_reversal),
        ("Generator", zigzag_level_order_generator),
        ("Manual Queue", zigzag_level_order_manual_queue),
        ("Functional", zigzag_level_order_functional),
        ("Class Based", zigzag_level_order_class_based),
        ("Optimized", zigzag_level_order_optimized),
    ]
    
    print("Testing Binary Tree Zigzag Level Order Traversal implementations:")
    print("=" * 70)
    
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
    print("\n" + "=" * 70)
    print("Performance Analysis:")
    print("=" * 70)
    
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
                nodes = sum(len(level) for level in result)
                print(f"  {name}: {levels} levels, {nodes} nodes in {end_time - start_time:.4f} seconds")
            except Exception as e:
                print(f"  {name}: Error - {e}")


if __name__ == "__main__":
    test_zigzag_level_order() 