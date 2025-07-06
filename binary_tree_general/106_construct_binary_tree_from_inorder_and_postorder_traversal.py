"""
LeetCode 106: Construct Binary Tree from Inorder and Postorder Traversal

Given two integer arrays inorder and postorder where inorder is the inorder traversal 
of a binary tree and postorder is the postorder traversal of the same tree, 
construct and return the binary tree.

Example 1:
Input: inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
Output: [3,9,20,null,null,15,7]

Example 2:
Input: inorder = [-1], postorder = [-1]
Output: [-1]

Constraints:
- 1 <= inorder.length <= 3000
- postorder.length == inorder.length
- -3000 <= inorder[i], postorder[i] <= 3000
- inorder and postorder consist of unique values.
- Each value of postorder also appears in inorder.
- inorder is guaranteed to be the inorder traversal of the tree.
- postorder is guaranteed to be the postorder traversal of the same tree.
"""

from typing import Optional, List
from collections import deque


class TreeNode:
    """Binary tree node definition."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree_recursive(inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    """
    Recursive approach.
    
    Time Complexity: O(n^2) worst case (skewed tree), O(n log n) average case
    Space Complexity: O(n) for recursion stack and slicing
    
    Algorithm:
    1. Last element in postorder is always root
    2. Find root in inorder to split left and right subtrees
    3. Recursively build left and right subtrees
    4. Right subtree is built first (postorder: left, right, root)
    """
    if not inorder or not postorder:
        return None
    
    # Last element in postorder is the root
    root_val = postorder[-1]
    root = TreeNode(root_val)
    
    # Find root position in inorder
    root_idx = inorder.index(root_val)
    
    # Split inorder array
    left_inorder = inorder[:root_idx]
    right_inorder = inorder[root_idx + 1:]
    
    # Split postorder array
    # Left subtree has same length as left_inorder
    left_postorder = postorder[:len(left_inorder)]
    right_postorder = postorder[len(left_inorder):-1]
    
    # Build subtrees (order matters: right first in postorder)
    root.left = build_tree_recursive(left_inorder, left_postorder)
    root.right = build_tree_recursive(right_inorder, right_postorder)
    
    return root


def build_tree_hashmap_optimized(inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    """
    Optimized recursive with hashmap for O(1) root lookup.
    
    Time Complexity: O(n)
    Space Complexity: O(n) for hashmap and recursion stack
    
    Algorithm:
    1. Use hashmap to store inorder indices for O(1) lookup
    2. Use global index to traverse postorder from right to left
    3. Build right subtree first, then left subtree
    """
    if not inorder or not postorder:
        return None
    
    # Create hashmap for O(1) index lookup
    inorder_map = {val: i for i, val in enumerate(inorder)}
    postorder_idx = [len(postorder) - 1]  # Use list for mutable reference
    
    def build(left: int, right: int) -> Optional[TreeNode]:
        if left > right:
            return None
        
        # Get root value from postorder (right to left)
        root_val = postorder[postorder_idx[0]]
        postorder_idx[0] -= 1
        
        root = TreeNode(root_val)
        root_idx = inorder_map[root_val]
        
        # Build right subtree first (postorder: left, right, root)
        root.right = build(root_idx + 1, right)
        root.left = build(left, root_idx - 1)
        
        return root
    
    return build(0, len(inorder) - 1)


def build_tree_iterative(inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    """
    Iterative approach using stack.
    
    Time Complexity: O(n)
    Space Complexity: O(n) for stack and hashmap
    
    Algorithm:
    1. Process postorder from right to left
    2. Use stack to track current path
    3. Use inorder to determine when to switch to left subtree
    """
    if not inorder or not postorder:
        return None
    
    inorder_map = {val: i for i, val in enumerate(inorder)}
    stack = []
    root = None
    inorder_idx = len(inorder) - 1
    
    # Process postorder from right to left
    for i in range(len(postorder) - 1, -1, -1):
        val = postorder[i]
        node = TreeNode(val)
        
        if not root:
            root = node
        
        parent = None
        # Find correct parent for current node
        while stack and inorder_map[stack[-1].val] > inorder_map[val]:
            parent = stack.pop()
        
        if parent:
            parent.left = node
        elif stack:
            stack[-1].right = node
        
        stack.append(node)
    
    return root


def build_tree_divide_conquer(inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    """
    Divide and conquer approach with optimizations.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Algorithm:
    1. Use indices instead of array slicing
    2. Build tree using divide and conquer strategy
    3. Process postorder from right to left
    """
    if not inorder or not postorder:
        return None
    
    inorder_map = {val: i for i, val in enumerate(inorder)}
    post_idx = [len(postorder) - 1]
    
    def build(in_left: int, in_right: int) -> Optional[TreeNode]:
        if in_left > in_right:
            return None
        
        root_val = postorder[post_idx[0]]
        post_idx[0] -= 1
        
        root = TreeNode(root_val)
        root_idx = inorder_map[root_val]
        
        # Build right first, then left (postorder pattern)
        root.right = build(root_idx + 1, in_right)
        root.left = build(in_left, root_idx - 1)
        
        return root
    
    return build(0, len(inorder) - 1)


def build_tree_morris_inspired(inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    """
    Morris traversal inspired approach.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Algorithm:
    1. Use the structure of Morris traversal logic
    2. Build tree by understanding the postorder pattern
    3. Optimize space usage where possible
    """
    if not inorder or not postorder:
        return None
    
    inorder_map = {val: i for i, val in enumerate(inorder)}
    
    def build_with_bounds(post_start: int, post_end: int, 
                         in_start: int, in_end: int) -> Optional[TreeNode]:
        if post_start > post_end or in_start > in_end:
            return None
        
        root_val = postorder[post_end]
        root = TreeNode(root_val)
        root_idx = inorder_map[root_val]
        
        left_size = root_idx - in_start
        
        root.left = build_with_bounds(post_start, post_start + left_size - 1,
                                    in_start, root_idx - 1)
        root.right = build_with_bounds(post_start + left_size, post_end - 1,
                                     root_idx + 1, in_end)
        
        return root
    
    return build_with_bounds(0, len(postorder) - 1, 0, len(inorder) - 1)


def build_tree_deque(inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    """
    Deque-based approach.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Algorithm:
    1. Use deque for efficient operations
    2. Process elements in postorder fashion
    3. Build tree efficiently
    """
    if not inorder or not postorder:
        return None
    
    inorder_map = {val: i for i, val in enumerate(inorder)}
    postorder_deque = deque(postorder)
    
    def build(left: int, right: int) -> Optional[TreeNode]:
        if left > right:
            return None
        
        root_val = postorder_deque.pop()
        root = TreeNode(root_val)
        root_idx = inorder_map[root_val]
        
        # Build right first, then left
        root.right = build(root_idx + 1, right)
        root.left = build(left, root_idx - 1)
        
        return root
    
    return build(0, len(inorder) - 1)


def build_tree_memoization(inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    """
    Memoization approach.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Algorithm:
    1. Use memoization to cache computed subtrees
    2. Avoid redundant calculations
    3. Optimize recursive calls
    """
    if not inorder or not postorder:
        return None
    
    inorder_map = {val: i for i, val in enumerate(inorder)}
    memo = {}
    post_idx = [len(postorder) - 1]
    
    def build(left: int, right: int) -> Optional[TreeNode]:
        if left > right:
            return None
        
        key = (left, right, post_idx[0])
        if key in memo:
            return memo[key]
        
        root_val = postorder[post_idx[0]]
        post_idx[0] -= 1
        
        root = TreeNode(root_val)
        root_idx = inorder_map[root_val]
        
        # Build right first, then left
        root.right = build(root_idx + 1, right)
        root.left = build(left, root_idx - 1)
        
        memo[key] = root
        return root
    
    return build(0, len(inorder) - 1)


def build_tree_bottom_up(inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    """
    Bottom-up construction approach.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Algorithm:
    1. Build tree from bottom up
    2. Use postorder characteristics
    3. Maintain tree structure efficiently
    """
    if not inorder or not postorder:
        return None
    
    inorder_map = {val: i for i, val in enumerate(inorder)}
    
    def build_range(in_left: int, in_right: int, 
                   post_left: int, post_right: int) -> Optional[TreeNode]:
        if in_left > in_right or post_left > post_right:
            return None
        
        root_val = postorder[post_right]
        root = TreeNode(root_val)
        root_idx = inorder_map[root_val]
        
        left_size = root_idx - in_left
        
        root.left = build_range(in_left, root_idx - 1,
                              post_left, post_left + left_size - 1)
        root.right = build_range(root_idx + 1, in_right,
                               post_left + left_size, post_right - 1)
        
        return root
    
    return build_range(0, len(inorder) - 1, 0, len(postorder) - 1)


def tree_to_list(root: Optional[TreeNode]) -> List[Optional[int]]:
    """Convert binary tree to list representation."""
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
        ([9,3,15,20,7], [9,15,7,20,3], [3,9,20,None,None,15,7]),
        ([-1], [-1], [-1]),
        
        # Two nodes
        ([2,1], [2,1], [1,2]),
        ([1,2], [1,2], [2,1]),
        
        # Three nodes
        ([2,1,3], [2,3,1], [1,2,3]),
        ([1,2,3], [1,3,2], [2,1,3]),
        ([2,3,1], [3,1,2], [2,3,1]),
        
        # Balanced trees
        ([4,2,5,1,6,3,7], [4,5,2,6,7,3,1], [1,2,3,4,5,6,7]),
        
        # Left skewed
        ([4,3,2,1], [4,3,2,1], [1,2,3,4]),
        
        # Right skewed
        ([1,2,3,4], [4,3,2,1], [1,None,2,None,3,None,4]),
        
        # Complex trees
        ([9,3,15,20,7], [9,15,7,20,3], [3,9,20,None,None,15,7]),
        ([8,4,9,2,5,1,6,3,7], [8,9,4,5,2,6,7,3,1], [1,2,3,4,5,6,7,8,9]),
        
        # Negative values
        ([-2,-1,-3], [-2,-3,-1], [-1,-2,-3]),
        ([1,-2,3], [1,3,-2], [-2,1,3]),
        
        # Large values
        ([25,50,75,100,150], [25,75,50,150,100], [100,50,150,25,75]),
        
        # Single child patterns
        ([4,2,1,3], [4,2,3,1], [1,2,3,4]),
        ([2,1,4,3], [2,4,3,1], [1,2,3,None,None,4]),
        
        # Mixed patterns
        ([7,11,4,2,5,13,1], [7,11,4,2,5,13,1], [5,4,13,11,2,None,1,7]),
    ]
    
    # Test all implementations
    implementations = [
        ("Recursive", build_tree_recursive),
        ("Hashmap Optimized", build_tree_hashmap_optimized),
        ("Iterative", build_tree_iterative),
        ("Divide Conquer", build_tree_divide_conquer),
        ("Morris Inspired", build_tree_morris_inspired),
        ("Deque", build_tree_deque),
        ("Memoization", build_tree_memoization),
        ("Bottom Up", build_tree_bottom_up),
    ]
    
    print("Testing Construct Binary Tree from Inorder and Postorder implementations:")
    print("=" * 75)
    
    for i, (inorder, postorder, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Inorder:   {inorder}")
        print(f"Postorder: {postorder}")
        print(f"Expected:  {expected}")
        
        for name, func in implementations:
            try:
                result_tree = func(inorder.copy(), postorder.copy())
                result = tree_to_list(result_tree)
                
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
    def generate_tree_arrays(size):
        """Generate inorder and postorder arrays for testing."""
        # Create a simple balanced-ish tree
        inorder = list(range(1, size + 1))
        # Create corresponding postorder
        def create_postorder(start, end):
            if start > end:
                return []
            mid = (start + end) // 2
            left = create_postorder(start, mid - 1)
            right = create_postorder(mid + 1, end)
            return left + right + [mid]
        
        postorder = create_postorder(1, size)
        return inorder, postorder
    
    test_scenarios = [
        ("Small tree", 15),
        ("Medium tree", 100),
        ("Large tree", 500),
        ("Very large", 1000),
    ]
    
    for scenario_name, size in test_scenarios:
        print(f"\n{scenario_name} ({size} nodes):")
        inorder_test, postorder_test = generate_tree_arrays(size)
        
        for name, func in implementations[:4]:  # Test first 4 implementations
            try:
                start_time = time.time()
                result = func(inorder_test.copy(), postorder_test.copy())
                end_time = time.time()
                
                elapsed = (end_time - start_time) * 1000  # Convert to milliseconds
                print(f"  {name}: {elapsed:.2f} ms")
            except Exception as e:
                print(f"  {name}: Error - {e}")
    
    print(f"\n{'='*75}")
    print("Algorithm Complexity Analysis:")
    print("="*75)
    print("1. Recursive:           O(n²) time, O(n) space")
    print("2. Hashmap Optimized:   O(n) time, O(n) space")
    print("3. Iterative:           O(n) time, O(n) space")  
    print("4. Divide Conquer:      O(n) time, O(n) space")
    print("5. Morris Inspired:     O(n) time, O(n) space")
    print("6. Deque:               O(n) time, O(n) space")
    print("7. Memoization:         O(n) time, O(n) space")
    print("8. Bottom Up:           O(n) time, O(n) space")


if __name__ == "__main__":
    test_build_tree() 