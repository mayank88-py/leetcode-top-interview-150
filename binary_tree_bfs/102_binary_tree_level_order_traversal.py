"""
102. Binary Tree Level Order Traversal

Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]

Example 2:
Input: root = [1]
Output: [[1]]

Example 3:
Input: root = []
Output: []

Constraints:
- The number of nodes in the tree is in the range [0, 2000]
- -1000 <= Node.val <= 1000
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


def level_order_bfs_queue(root: Optional[TreeNode]) -> List[List[int]]:
    """
    BFS approach using queue.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use queue to store nodes level by level
    2. Process all nodes at current level before moving to next
    3. Track level boundaries using queue size
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
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
        
        result.append(current_level)
    
    return result


def level_order_bfs_two_queues(root: Optional[TreeNode]) -> List[List[int]]:
    """
    BFS approach using two queues.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use two queues: current level and next level
    2. Alternate between queues for each level
    3. Clear distinction between levels
    """
    if not root:
        return []
    
    result = []
    current_level = deque([root])
    
    while current_level:
        next_level = deque()
        level_values = []
        
        while current_level:
            node = current_level.popleft()
            level_values.append(node.val)
            
            # Add children to next level
            if node.left:
                next_level.append(node.left)
            if node.right:
                next_level.append(node.right)
        
        result.append(level_values)
        current_level = next_level
    
    return result


def level_order_dfs_recursive(root: Optional[TreeNode]) -> List[List[int]]:
    """
    DFS recursive approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use DFS with level tracking
    2. Add nodes to appropriate level in result
    3. Ensure levels are processed in order
    """
    if not root:
        return []
    
    result = []
    
    def dfs(node, level):
        if not node:
            return
        
        # Ensure we have enough levels in result
        if len(result) <= level:
            result.append([])
        
        # Add current node to its level
        result[level].append(node.val)
        
        # Recursively process children at next level
        dfs(node.left, level + 1)
        dfs(node.right, level + 1)
    
    dfs(root, 0)
    return result


def level_order_iterative_stack(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Iterative approach using stack with level tracking.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use stack to store (node, level) pairs
    2. Process nodes and track their levels
    3. Build result by level
    """
    if not root:
        return []
    
    result = []
    stack = [(root, 0)]
    
    while stack:
        node, level = stack.pop()
        
        # Ensure we have enough levels in result
        if len(result) <= level:
            result.append([])
        
        # Add current node to its level
        result[level].append(node.val)
        
        # Add children to stack (right first for left-to-right order)
        if node.right:
            stack.append((node.right, level + 1))
        if node.left:
            stack.append((node.left, level + 1))
    
    # Reverse each level to get correct left-to-right order
    for level in result:
        level.reverse()
    
    return result


def level_order_bfs_with_sentinel(root: Optional[TreeNode]) -> List[List[int]]:
    """
    BFS approach using sentinel node to mark level boundaries.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use None as sentinel to mark end of level
    2. When sentinel is encountered, start new level
    3. Continue until queue is empty
    """
    if not root:
        return []
    
    result = []
    queue = deque([root, None])  # None as level separator
    current_level = []
    
    while queue:
        node = queue.popleft()
        
        if node is None:
            # End of current level
            if current_level:
                result.append(current_level)
                current_level = []
                
                # Add sentinel for next level if queue not empty
                if queue:
                    queue.append(None)
        else:
            current_level.append(node.val)
            
            # Add children for next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return result


def level_order_morris_like(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Morris-like approach (educational).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Attempt to use Morris-like threading
    2. Track levels during traversal
    3. Handle level order requirements
    
    Note: True Morris traversal is complex for level order
    """
    # For level order, BFS is more natural than Morris
    # Fall back to BFS queue approach
    return level_order_bfs_queue(root)


def level_order_list_based(root: Optional[TreeNode]) -> List[List[int]]:
    """
    List-based approach for level tracking.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use list to store current level nodes
    2. Process entire level at once
    3. Generate next level from current level
    """
    if not root:
        return []
    
    result = []
    current_level = [root]
    
    while current_level:
        level_values = []
        next_level = []
        
        for node in current_level:
            level_values.append(node.val)
            
            # Collect children for next level
            if node.left:
                next_level.append(node.left)
            if node.right:
                next_level.append(node.right)
        
        result.append(level_values)
        current_level = next_level
    
    return result


def level_order_generator(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Generator-based approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use generator to yield levels one by one
    2. Memory efficient for large trees
    3. Lazy evaluation of levels
    """
    def generate_levels(node):
        if not node:
            return
        
        current_level = [node]
        
        while current_level:
            yield [n.val for n in current_level]
            
            next_level = []
            for n in current_level:
                if n.left:
                    next_level.append(n.left)
                if n.right:
                    next_level.append(n.right)
            
            current_level = next_level
    
    return list(generate_levels(root))


def level_order_deque_rotate(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Deque with rotation approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use deque rotation for level processing
    2. Rotate to maintain level boundaries
    3. Extract levels systematically
    """
    if not root:
        return []
    
    result = []
    dq = deque([root])
    
    while dq:
        level_size = len(dq)
        current_level = []
        
        # Process current level
        for _ in range(level_size):
            node = dq.popleft()
            current_level.append(node.val)
            
            # Add children
            if node.left:
                dq.append(node.left)
            if node.right:
                dq.append(node.right)
        
        result.append(current_level)
    
    return result


def level_order_breadth_first_search(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Classic BFS implementation.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Implement classic BFS algorithm
    2. Track level information
    3. Process nodes level by level
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        # Process all nodes at current level
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            # Enqueue children for next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result


def level_order_functional(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Functional programming approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use functional programming concepts
    2. Immutable data structures where possible
    3. Higher-order functions
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
    
    result = []
    current_level = [root]
    
    while current_level:
        result.append(get_level_values(current_level))
        current_level = get_next_level(current_level)
    
    return result


def level_order_class_based(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Class-based approach for state management.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use class to encapsulate traversal logic
    2. Maintain state within class
    3. Clean separation of concerns
    """
    class LevelOrderTraverser:
        def __init__(self):
            self.result = []
        
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
                
                self.result.append(current_level)
            
            return self.result
    
    traverser = LevelOrderTraverser()
    return traverser.traverse(root)


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


def test_level_order():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([3,9,20,None,None,15,7], [[3],[9,20],[15,7]]),
        ([1], [[1]]),
        ([], []),
        
        # Two nodes
        ([1,2], [[1],[2]]),
        ([1,None,2], [[1],[2]]),
        
        # Three nodes
        ([1,2,3], [[1],[2,3]]),
        ([1,None,2,None,3], [[1],[2],[3]]),
        
        # Complete binary tree
        ([1,2,3,4,5,6,7], [[1],[2,3],[4,5,6,7]]),
        
        # Left skewed
        ([1,2,None,3,None,4], [[1],[2],[3],[4]]),
        
        # Right skewed
        ([1,None,2,None,3,None,4], [[1],[2],[3],[4]]),
        
        # Unbalanced tree
        ([1,2,3,4,None,None,5], [[1],[2,3],[4,5]]),
        
        # Large tree
        ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
         [[1],[2,3],[4,5,6,7],[8,9,10,11,12,13,14,15]]),
        
        # Single path
        ([1,2,None,3,None,4,None,5], [[1],[2],[3],[4],[5]]),
        
        # Negative values
        ([-1,2,-3], [[-1],[2,-3]]),
        ([1,-2,3], [[1],[-2,3]]),
        
        # Zero values
        ([0,1,2], [[0],[1,2]]),
        ([1,0,2], [[1],[0,2]]),
        
        # Mixed values
        ([5,4,8,11,None,13,4,7,2,None,None,None,1], 
         [[5],[4,8],[11,13,4],[7,2,1]]),
        
        # Wide tree
        ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], 
         [[1],[2,3],[4,5,6,7],[8,9,10,11,12,13,14,15],[16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]]),
        
        # Deep tree
        ([1,2,None,3,None,4,None,5,None,6,None,7], 
         [[1],[2],[3],[4],[5],[6],[7]]),
        
        # Complex structure
        ([1,2,3,4,5,None,6,None,None,7,8,None,9], 
         [[1],[2,3],[4,5,6],[7,8,9]]),
        
        # All same values
        ([1,1,1,1,1,1,1], [[1],[1,1],[1,1,1,1]]),
        
        # Large values
        ([1000,500,1500,250,750,1250,1750], 
         [[1000],[500,1500],[250,750,1250,1750]]),
        
        # Edge cases
        ([1,2,3,None,None,4,5], [[1],[2,3],[4,5]]),
        ([1,None,2,3,4], [[1],[2],[3,4]]),
    ]
    
    # Test all implementations
    implementations = [
        ("BFS Queue", level_order_bfs_queue),
        ("BFS Two Queues", level_order_bfs_two_queues),
        ("DFS Recursive", level_order_dfs_recursive),
        ("Iterative Stack", level_order_iterative_stack),
        ("BFS with Sentinel", level_order_bfs_with_sentinel),
        ("Morris-like", level_order_morris_like),
        ("List Based", level_order_list_based),
        ("Generator", level_order_generator),
        ("Deque Rotate", level_order_deque_rotate),
        ("Breadth First Search", level_order_breadth_first_search),
        ("Functional", level_order_functional),
        ("Class Based", level_order_class_based),
    ]
    
    print("Testing Binary Tree Level Order Traversal implementations:")
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
    test_level_order() 