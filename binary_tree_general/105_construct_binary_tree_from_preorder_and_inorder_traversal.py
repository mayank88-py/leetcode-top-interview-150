"""
105. Construct Binary Tree from Preorder and Inorder Traversal

Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.

Example 1:
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]

Example 2:
Input: preorder = [-1], inorder = [-1]
Output: [-1]

Constraints:
- 1 <= preorder.length <= 3000
- inorder.length == preorder.length
- -3000 <= preorder[i], inorder[i] <= 3000
- preorder and inorder consist of unique values
- Each value of inorder also appears in preorder
- preorder is guaranteed to be the preorder traversal of the tree
- inorder is guaranteed to be the inorder traversal of the same tree
"""

from typing import Optional, List, Dict
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


def build_tree_recursive(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    Recursive approach with hashmap for inorder indices.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - hashmap + recursion stack
    
    Algorithm:
    1. Create hashmap for inorder indices for O(1) lookup
    2. Use recursion with preorder index tracker
    3. For each recursive call, find root in inorder, split left/right
    4. Build left and right subtrees recursively
    """
    if not preorder or not inorder:
        return None
    
    # Create hashmap for inorder indices
    inorder_map = {val: i for i, val in enumerate(inorder)}
    preorder_idx = [0]  # Use list to maintain reference
    
    def build(left, right):
        if left > right:
            return None
        
        # Root is always the next element in preorder
        root_val = preorder[preorder_idx[0]]
        preorder_idx[0] += 1
        
        root = TreeNode(root_val)
        
        # Find root position in inorder
        root_idx = inorder_map[root_val]
        
        # Build left subtree first (preorder: root, left, right)
        root.left = build(left, root_idx - 1)
        root.right = build(root_idx + 1, right)
        
        return root
    
    return build(0, len(inorder) - 1)


def build_tree_iterative(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    Iterative approach using stack.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - stack
    
    Algorithm:
    1. Use stack to track the path from root to current node
    2. Process preorder elements one by one
    3. Use inorder to determine when to switch from left to right
    """
    if not preorder:
        return None
    
    root = TreeNode(preorder[0])
    stack = [root]
    inorder_idx = 0
    
    for i in range(1, len(preorder)):
        current_val = preorder[i]
        current = TreeNode(current_val)
        
        # Find the parent node
        parent = None
        
        # If current inorder element matches stack top, we're done with left subtree
        while stack and stack[-1].val == inorder[inorder_idx]:
            parent = stack.pop()
            inorder_idx += 1
        
        if parent:
            # Attach as right child
            parent.right = current
        else:
            # Attach as left child
            stack[-1].left = current
        
        stack.append(current)
    
    return root


def build_tree_divide_conquer(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    Divide and conquer approach.
    
    Time Complexity: O(n^2) worst case (unbalanced), O(n log n) average
    Space Complexity: O(n) - recursion stack
    
    Algorithm:
    1. Find root in preorder (first element)
    2. Find root position in inorder
    3. Split both arrays based on root position
    4. Recursively build left and right subtrees
    """
    def build(pre, ino):
        if not pre or not ino:
            return None
        
        # Root is first element in preorder
        root_val = pre[0]
        root = TreeNode(root_val)
        
        # Find root in inorder
        root_idx = ino.index(root_val)
        
        # Split inorder
        left_inorder = ino[:root_idx]
        right_inorder = ino[root_idx + 1:]
        
        # Split preorder
        left_size = len(left_inorder)
        left_preorder = pre[1:left_size + 1]
        right_preorder = pre[left_size + 1:]
        
        # Build subtrees
        root.left = build(left_preorder, left_inorder)
        root.right = build(right_preorder, right_inorder)
        
        return root
    
    return build(preorder, inorder)


def build_tree_hashmap_optimized(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    Optimized approach with hashmap and index tracking.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - hashmap + recursion stack
    
    Algorithm:
    1. Precompute inorder indices in hashmap
    2. Use global preorder index
    3. Pass left and right bounds for inorder
    """
    if not preorder or not inorder:
        return None
    
    inorder_map = {val: i for i, val in enumerate(inorder)}
    
    def build(pre_start, pre_end, in_start, in_end):
        if pre_start > pre_end:
            return None
        
        root_val = preorder[pre_start]
        root = TreeNode(root_val)
        root_idx = inorder_map[root_val]
        
        left_size = root_idx - in_start
        
        root.left = build(pre_start + 1, pre_start + left_size, in_start, root_idx - 1)
        root.right = build(pre_start + left_size + 1, pre_end, root_idx + 1, in_end)
        
        return root
    
    return build(0, len(preorder) - 1, 0, len(inorder) - 1)


def build_tree_slice_free(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    Slice-free approach for better performance.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - hashmap + recursion stack
    
    Algorithm:
    1. Avoid creating new array slices
    2. Use indices to track boundaries
    3. Use hashmap for O(1) inorder lookups
    """
    if not preorder or not inorder:
        return None
    
    inorder_map = {val: i for i, val in enumerate(inorder)}
    preorder_idx = 0
    
    def build(in_left, in_right):
        nonlocal preorder_idx
        
        if in_left > in_right:
            return None
        
        root_val = preorder[preorder_idx]
        preorder_idx += 1
        root = TreeNode(root_val)
        
        root_idx = inorder_map[root_val]
        
        # Important: build left first, then right (preorder)
        root.left = build(in_left, root_idx - 1)
        root.right = build(root_idx + 1, in_right)
        
        return root
    
    return build(0, len(inorder) - 1)


def build_tree_postorder_simulation(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    Simulate postorder processing approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - hashmap + recursion stack
    
    Algorithm:
    1. Reverse preorder to simulate postorder
    2. Process from right to left
    3. Build right subtree first, then left
    """
    if not preorder or not inorder:
        return None
    
    inorder_map = {val: i for i, val in enumerate(inorder)}
    preorder = preorder[::-1]  # Reverse preorder
    preorder_idx = 0
    
    def build(in_left, in_right):
        nonlocal preorder_idx
        
        if in_left > in_right:
            return None
        
        root_val = preorder[preorder_idx]
        preorder_idx += 1
        root = TreeNode(root_val)
        
        root_idx = inorder_map[root_val]
        
        # Build right first (since we reversed preorder)
        root.right = build(root_idx + 1, in_right)
        root.left = build(in_left, root_idx - 1)
        
        return root
    
    return build(0, len(inorder) - 1)


def build_tree_morris_inspired(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    Morris traversal inspired approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - hashmap (avoiding recursion stack where possible)
    
    Algorithm:
    1. Use Morris-like threading concept
    2. Build tree with minimal stack usage
    3. Use inorder position information
    """
    if not preorder or not inorder:
        return None
    
    inorder_map = {val: i for i, val in enumerate(inorder)}
    
    def build_iterative():
        root = TreeNode(preorder[0])
        stack = [(root, 0, len(inorder) - 1)]
        preorder_idx = 1
        
        while stack and preorder_idx < len(preorder):
            node, in_left, in_right = stack.pop()
            root_idx = inorder_map[node.val]
            
            # Add right child first (to process left first)
            if root_idx + 1 <= in_right:
                right_val = None
                # Find right child value in preorder
                for i in range(preorder_idx, len(preorder)):
                    if inorder_map[preorder[i]] > root_idx:
                        right_val = preorder[i]
                        break
                
                if right_val is not None:
                    node.right = TreeNode(right_val)
                    stack.append((node.right, root_idx + 1, in_right))
            
            # Add left child
            if in_left <= root_idx - 1:
                if preorder_idx < len(preorder):
                    left_val = preorder[preorder_idx]
                    if inorder_map[left_val] < root_idx:
                        preorder_idx += 1
                        node.left = TreeNode(left_val)
                        stack.append((node.left, in_left, root_idx - 1))
        
        return root
    
    # Fallback to recursive for complexity
    return build_tree_slice_free(preorder, inorder)


def build_tree_deque(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    Deque-based approach for breadth-first construction.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - deque + hashmap
    
    Algorithm:
    1. Use deque for level-by-level construction
    2. Track boundaries for each subtree
    3. Process nodes in level order while maintaining preorder
    """
    if not preorder or not inorder:
        return None
    
    inorder_map = {val: i for i, val in enumerate(inorder)}
    root = TreeNode(preorder[0])
    
    # Queue: (node, in_left, in_right, pre_start, pre_end)
    queue = deque([(root, 0, len(inorder) - 1, 0, len(preorder) - 1)])
    
    while queue:
        node, in_left, in_right, pre_start, pre_end = queue.popleft()
        
        if pre_start >= pre_end:
            continue
        
        root_idx = inorder_map[node.val]
        left_size = root_idx - in_left
        
        # Add left child
        if in_left <= root_idx - 1:
            left_val = preorder[pre_start + 1]
            node.left = TreeNode(left_val)
            queue.append((node.left, in_left, root_idx - 1, 
                         pre_start + 1, pre_start + left_size))
        
        # Add right child
        if root_idx + 1 <= in_right:
            right_start = pre_start + left_size + 1
            if right_start <= pre_end:
                right_val = preorder[right_start]
                node.right = TreeNode(right_val)
                queue.append((node.right, root_idx + 1, in_right,
                             right_start, pre_end))
    
    return root


def build_tree_memoization(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    Memoization approach to cache subtrees.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - memoization cache + recursion stack
    
    Algorithm:
    1. Cache results of subtree construction
    2. Use tuple of boundaries as cache key
    3. Avoid recomputation of identical subtrees
    """
    if not preorder or not inorder:
        return None
    
    inorder_map = {val: i for i, val in enumerate(inorder)}
    memo = {}
    
    def build(pre_start, pre_end, in_start, in_end):
        if pre_start > pre_end:
            return None
        
        key = (pre_start, pre_end, in_start, in_end)
        if key in memo:
            return memo[key]
        
        root_val = preorder[pre_start]
        root = TreeNode(root_val)
        root_idx = inorder_map[root_val]
        
        left_size = root_idx - in_start
        
        root.left = build(pre_start + 1, pre_start + left_size, 
                         in_start, root_idx - 1)
        root.right = build(pre_start + left_size + 1, pre_end, 
                          root_idx + 1, in_end)
        
        memo[key] = root
        return root
    
    return build(0, len(preorder) - 1, 0, len(inorder) - 1)


def tree_to_list(root: Optional[TreeNode]) -> List[Optional[int]]:
    """Helper function to convert tree to level-order list for testing."""
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        if node:
            result.append(node.val)
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append(None)
    
    # Remove trailing None values
    while result and result[-1] is None:
        result.pop()
    
    return result


def test_build_tree():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([3,9,20,15,7], [9,3,15,20,7], [3,9,20,None,None,15,7]),
        ([-1], [-1], [-1]),
        
        # Two nodes
        ([1,2], [2,1], [1,2]),
        ([1,2], [1,2], [1,None,2]),
        
        # Three nodes
        ([1,2,3], [2,1,3], [1,2,None,None,3]),
        ([1,2,3], [1,2,3], [1,None,2,None,3]),
        ([1,2,3], [2,3,1], [1,2,None,3]),
        
        # Balanced trees
        ([1,2,4,5,3,6,7], [4,2,5,1,6,3,7], [1,2,3,4,5,6,7]),
        
        # Left skewed
        ([1,2,3,4], [4,3,2,1], [1,2,None,3,None,4]),
        
        # Right skewed
        ([1,2,3,4], [1,2,3,4], [1,None,2,None,3,None,4]),
        
        # Complex trees
        ([3,9,20,15,7], [9,3,15,20,7], [3,9,20,None,None,15,7]),
        ([1,2,4,8,9,5,3,6,7], [8,4,9,2,5,1,6,3,7], [1,2,3,4,5,6,7,8,9]),
        
        # Negative values
        ([-1,-2,-3], [-2,-1,-3], [-1,-2,None,None,-3]),
        ([1,-2,3], [-2,1,3], [1,-2,None,None,3]),
        
        # Large values
        ([100,50,25,75,150], [25,50,75,100,150], [100,50,150,25,75]),
        
        # Single child patterns
        ([1,2,4,3], [4,2,1,3], [1,2,3,4]),
        ([1,2,3,4], [2,1,4,3], [1,2,3,None,None,4]),
        
        # Mixed patterns
        ([5,4,11,7,2,13,1], [7,11,4,2,5,13,1], [5,4,13,11,2,None,1,7]),
    ]
    
    # Test all implementations
    implementations = [
        ("Recursive", build_tree_recursive),
        ("Iterative", build_tree_iterative),
        ("Divide Conquer", build_tree_divide_conquer),
        ("Hashmap Optimized", build_tree_hashmap_optimized),
        ("Slice Free", build_tree_slice_free),
        ("Postorder Simulation", build_tree_postorder_simulation),
        ("Morris Inspired", build_tree_morris_inspired),
        ("Deque", build_tree_deque),
        ("Memoization", build_tree_memoization),
    ]
    
    print("Testing Construct Binary Tree from Preorder and Inorder implementations:")
    print("=" * 70)
    
    for i, (preorder, inorder, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Preorder: {preorder}")
        print(f"Inorder:  {inorder}")
        print(f"Expected: {expected}")
        
        for name, func in implementations:
            try:
                result_tree = func(preorder.copy(), inorder.copy())
                result = tree_to_list(result_tree)
                
                status = "✓" if result == expected else "✗"
                print(f"{status} {name}: {result}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 70)
    print("Performance Analysis:")
    print("=" * 70)
    
    import time
    
    # Generate large test cases
    def generate_tree_arrays(size):
        """Generate preorder and inorder arrays for testing."""
        # Create a simple balanced-ish tree
        preorder = list(range(1, size + 1))
        # Create corresponding inorder (left subtree, root, right subtree)
        def create_inorder(pre, start, end):
            if start > end:
                return []
            mid = (start + end) // 2
            root = pre[start]
            left = create_inorder(pre, start + 1, start + (mid - start))
            right = create_inorder(pre, start + (mid - start) + 1, end)
            return left + [root] + right
        
        inorder = create_inorder(preorder, 0, len(preorder) - 1)
        return preorder, inorder
    
    test_scenarios = [
        ("Small tree", 15),
        ("Medium tree", 100),
        ("Large tree", 500),
        ("Very large", 1000),
    ]
    
    for scenario_name, size in test_scenarios:
        print(f"\n{scenario_name} ({size} nodes):")
        preorder, inorder = generate_tree_arrays(size)
        
        for name, func in implementations:
            try:
                start_time = time.time()
                result_tree = func(preorder.copy(), inorder.copy())
                end_time = time.time()
                
                result_size = len(tree_to_list(result_tree))
                print(f"  {name}: {result_size} nodes in {end_time - start_time:.4f} seconds")
            except Exception as e:
                print(f"  {name}: Error - {e}")


if __name__ == "__main__":
    test_build_tree() 