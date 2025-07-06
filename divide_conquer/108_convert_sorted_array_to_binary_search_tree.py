"""
108. Convert Sorted Array to Binary Search Tree

Problem:
Given an integer array nums where the elements are sorted in ascending order, 
convert it to a height-balanced binary search tree.

A height-balanced binary tree is a binary tree in which the depth of the two subtrees 
of every node never differs by more than one.

Example 1:
Input: nums = [-10,-3,0,5,9]
Output: [0,-3,9,-10,null,5]

Example 2:
Input: nums = [1,3]
Output: [3,1] or [1,null,3]

Time Complexity: O(n) where n is the number of elements
Space Complexity: O(log n) for recursion stack in balanced tree
"""


class TreeNode:
    """Binary tree node definition."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def sorted_array_to_bst_recursive(nums):
    """
    Recursive divide and conquer approach - optimal solution.
    
    Time Complexity: O(n)
    Space Complexity: O(log n) for recursion stack
    
    Algorithm:
    1. Choose middle element as root (ensures balance)
    2. Recursively build left subtree from left half
    3. Recursively build right subtree from right half
    4. This maintains BST property and height balance
    """
    if not nums:
        return None
    
    def helper(left, right):
        if left > right:
            return None
        
        # Choose middle element as root
        mid = (left + right) // 2
        root = TreeNode(nums[mid])
        
        # Recursively build left and right subtrees
        root.left = helper(left, mid - 1)
        root.right = helper(mid + 1, right)
        
        return root
    
    return helper(0, len(nums) - 1)


def sorted_array_to_bst_iterative(nums):
    """
    Iterative approach using stack.
    
    Time Complexity: O(n)
    Space Complexity: O(n) for stack
    
    Algorithm:
    1. Use stack to simulate recursion
    2. Store (node, left_bound, right_bound) in stack
    3. Process each node and create children if bounds allow
    """
    if not nums:
        return None
    
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    
    # Stack stores (node, left_bound, right_bound)
    stack = [(root, 0, len(nums) - 1)]
    
    while stack:
        node, left, right = stack.pop()
        
        # Calculate current middle
        mid = (left + right) // 2
        
        # Create left child if possible
        if left <= mid - 1:
            left_mid = (left + mid - 1) // 2
            node.left = TreeNode(nums[left_mid])
            stack.append((node.left, left, mid - 1))
        
        # Create right child if possible
        if mid + 1 <= right:
            right_mid = (mid + 1 + right) // 2
            node.right = TreeNode(nums[right_mid])
            stack.append((node.right, mid + 1, right))
    
    return root


def sorted_array_to_bst_preorder(nums):
    """
    Build BST by choosing root differently (leftmost middle for ties).
    
    Time Complexity: O(n)
    Space Complexity: O(log n)
    
    Algorithm:
    1. For even length arrays, choose left middle as root
    2. This creates a specific tree structure (left-leaning when possible)
    3. Still maintains height balance property
    """
    if not nums:
        return None
    
    def helper(left, right):
        if left > right:
            return None
        
        # Choose left middle for even-length subarrays
        mid = (left + right) // 2
        root = TreeNode(nums[mid])
        
        root.left = helper(left, mid - 1)
        root.right = helper(mid + 1, right)
        
        return root
    
    return helper(0, len(nums) - 1)


def sorted_array_to_bst_right_middle(nums):
    """
    Build BST by choosing right middle for even-length arrays.
    
    Time Complexity: O(n)
    Space Complexity: O(log n)
    
    Algorithm:
    1. For even length arrays, choose right middle as root
    2. This creates a different tree structure (right-leaning when possible)
    3. Still maintains height balance property
    """
    if not nums:
        return None
    
    def helper(left, right):
        if left > right:
            return None
        
        # Choose right middle for even-length subarrays
        mid = (left + right + 1) // 2
        root = TreeNode(nums[mid])
        
        root.left = helper(left, mid - 1)
        root.right = helper(mid + 1, right)
        
        return root
    
    return helper(0, len(nums) - 1)


def sorted_array_to_bst_random(nums):
    """
    Build BST by randomly choosing middle element for variety.
    
    Time Complexity: O(n)
    Space Complexity: O(log n)
    
    Algorithm:
    1. For even length arrays, randomly choose left or right middle
    2. This creates different balanced trees on different runs
    3. Useful for creating varied tree structures while maintaining balance
    """
    if not nums:
        return None
    
    import random
    
    def helper(left, right):
        if left > right:
            return None
        
        # Randomly choose between left and right middle for even lengths
        if (right - left + 1) % 2 == 0:
            mid = (left + right) // 2 if random.choice([True, False]) else (left + right + 1) // 2
        else:
            mid = (left + right) // 2
        
        root = TreeNode(nums[mid])
        root.left = helper(left, mid - 1)
        root.right = helper(mid + 1, right)
        
        return root
    
    return helper(0, len(nums) - 1)


def tree_to_list_level_order(root):
    """Helper function to convert tree to level-order list for testing."""
    if not root:
        return []
    
    from collections import deque
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


def is_valid_bst(root, min_val=float('-inf'), max_val=float('inf')):
    """Helper function to validate if tree is a valid BST."""
    if not root:
        return True
    
    if root.val <= min_val or root.val >= max_val:
        return False
    
    return (is_valid_bst(root.left, min_val, root.val) and 
            is_valid_bst(root.right, root.val, max_val))


def is_height_balanced(root):
    """Helper function to check if tree is height balanced."""
    def get_height(node):
        if not node:
            return 0
        
        left_height = get_height(node.left)
        right_height = get_height(node.right)
        
        if left_height == -1 or right_height == -1:
            return -1
        
        if abs(left_height - right_height) > 1:
            return -1
        
        return max(left_height, right_height) + 1
    
    return get_height(root) != -1


def test_sorted_array_to_bst():
    """Test all implementations with various test cases."""
    
    test_cases = [
        [-10, -3, 0, 5, 9],
        [1, 3],
        [1],
        [],
        [1, 2, 3, 4, 5, 6, 7],
        [-1, 0, 1, 2]
    ]
    
    implementations = [
        ("Recursive (left middle)", sorted_array_to_bst_recursive),
        ("Iterative", sorted_array_to_bst_iterative),
        ("Preorder style", sorted_array_to_bst_preorder),
        ("Right middle", sorted_array_to_bst_right_middle),
        ("Random middle", sorted_array_to_bst_random)
    ]
    
    print("Testing Convert Sorted Array to Binary Search Tree...")
    
    for i, nums in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: nums = {nums}")
        
        for impl_name, impl_func in implementations:
            result_tree = impl_func(nums[:])  # Make copy since some might modify
            
            # Validate the tree
            is_bst = is_valid_bst(result_tree)
            is_balanced = is_height_balanced(result_tree)
            
            # Convert to list for display (if small enough)
            tree_list = tree_to_list_level_order(result_tree)
            
            status = "✓" if is_bst and is_balanced else "✗"
            print(f"{impl_name:20} | BST: {'✓' if is_bst else '✗'} | Balanced: {'✓' if is_balanced else '✗'} | {status}")
            
            if len(tree_list) <= 15:
                print(f"                       Tree: {tree_list}")


if __name__ == "__main__":
    test_sorted_array_to_bst() 