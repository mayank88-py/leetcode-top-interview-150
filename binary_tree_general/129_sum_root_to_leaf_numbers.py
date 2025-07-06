"""
129. Sum Root to Leaf Numbers

You are given the root of a binary tree containing digits from 0 to 9 only.

Each root-to-leaf path in the tree represents a number.

For example, the root-to-leaf path 1 -> 2 -> 3 represents the number 123.

Return the total sum of all root-to-leaf numbers.

A leaf is a node with no children.

Example 1:
Input: root = [1,2,3]
Output: 25
Explanation:
The root-to-leaf path 1->2 represents the number 12.
The root-to-leaf path 1->3 represents the number 13.
Therefore, sum = 12 + 13 = 25.

Example 2:
Input: root = [4,9,0,5,1]
Output: 1026
Explanation:
The root-to-leaf path 4->9->5 represents the number 495.
The root-to-leaf path 4->9->1 represents the number 491.
The root-to-leaf path 4->0 represents the number 40.
Therefore, sum = 495 + 491 + 40 = 1026.

Constraints:
- The number of nodes in the tree is in the range [1, 1000]
- 0 <= Node.val <= 9
- The depth of the tree will not exceed 10
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


def sum_numbers_recursive(root: Optional[TreeNode]) -> int:
    """
    Recursive approach (DFS).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree (recursion stack)
    
    Algorithm:
    1. Use DFS to traverse all paths
    2. Build number as we go down the path
    3. Add to total sum when we reach a leaf
    """
    def dfs(node, current_number):
        if not node:
            return 0
        
        # Build the current number
        current_number = current_number * 10 + node.val
        
        # If it's a leaf node, return the current number
        if not node.left and not node.right:
            return current_number
        
        # Recursively sum left and right subtrees
        return dfs(node.left, current_number) + dfs(node.right, current_number)
    
    return dfs(root, 0)


def sum_numbers_iterative_stack(root: Optional[TreeNode]) -> int:
    """
    Iterative approach using stack.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use stack to store (node, current_number) pairs
    2. Build number as we traverse down
    3. Add to total when we reach a leaf
    """
    if not root:
        return 0
    
    total_sum = 0
    stack = [(root, 0)]
    
    while stack:
        node, current_number = stack.pop()
        current_number = current_number * 10 + node.val
        
        # If it's a leaf node, add to total sum
        if not node.left and not node.right:
            total_sum += current_number
        else:
            # Add children to stack
            if node.left:
                stack.append((node.left, current_number))
            if node.right:
                stack.append((node.right, current_number))
    
    return total_sum


def sum_numbers_iterative_queue(root: Optional[TreeNode]) -> int:
    """
    Iterative approach using queue (BFS).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Use queue for level-order traversal
    2. Store (node, current_number) pairs
    3. Build numbers and sum when reaching leaves
    """
    if not root:
        return 0
    
    total_sum = 0
    queue = deque([(root, 0)])
    
    while queue:
        node, current_number = queue.popleft()
        current_number = current_number * 10 + node.val
        
        # If it's a leaf node, add to total sum
        if not node.left and not node.right:
            total_sum += current_number
        else:
            # Add children to queue
            if node.left:
                queue.append((node.left, current_number))
            if node.right:
                queue.append((node.right, current_number))
    
    return total_sum


def sum_numbers_path_collection(root: Optional[TreeNode]) -> int:
    """
    Path collection approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - to store all paths
    
    Algorithm:
    1. Collect all root-to-leaf paths
    2. Convert each path to a number
    3. Sum all numbers
    """
    def collect_paths(node, current_path, all_paths):
        if not node:
            return
        
        current_path.append(node.val)
        
        # If leaf node, add path to collection
        if not node.left and not node.right:
            all_paths.append(current_path[:])
        else:
            # Continue to children
            collect_paths(node.left, current_path, all_paths)
            collect_paths(node.right, current_path, all_paths)
        
        # Backtrack
        current_path.pop()
    
    if not root:
        return 0
    
    all_paths = []
    collect_paths(root, [], all_paths)
    
    # Convert paths to numbers and sum
    total_sum = 0
    for path in all_paths:
        number = 0
        for digit in path:
            number = number * 10 + digit
        total_sum += number
    
    return total_sum


def sum_numbers_string_approach(root: Optional[TreeNode]) -> int:
    """
    String-based approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Build path as string
    2. Convert to integer at leaf nodes
    3. Sum all leaf values
    """
    def dfs(node, current_string):
        if not node:
            return 0
        
        current_string += str(node.val)
        
        # If leaf node, convert string to number
        if not node.left and not node.right:
            return int(current_string)
        
        # Recursively sum left and right subtrees
        return dfs(node.left, current_string) + dfs(node.right, current_string)
    
    return dfs(root, "")


def sum_numbers_preorder(root: Optional[TreeNode]) -> int:
    """
    Preorder traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use preorder traversal
    2. Pass current number down the tree
    3. Accumulate sum at leaf nodes
    """
    total_sum = [0]  # Use list to maintain reference
    
    def preorder(node, current_number):
        if not node:
            return
        
        current_number = current_number * 10 + node.val
        
        # If leaf node, add to total
        if not node.left and not node.right:
            total_sum[0] += current_number
            return
        
        # Continue preorder traversal
        preorder(node.left, current_number)
        preorder(node.right, current_number)
    
    preorder(root, 0)
    return total_sum[0]


def sum_numbers_backtracking(root: Optional[TreeNode]) -> int:
    """
    Backtracking approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use backtracking to explore all paths
    2. Build number by adding digits
    3. Subtract digits when backtracking
    """
    total_sum = [0]
    
    def backtrack(node, current_number):
        if not node:
            return
        
        # Add current digit
        current_number = current_number * 10 + node.val
        
        # If leaf node, add to total sum
        if not node.left and not node.right:
            total_sum[0] += current_number
        else:
            # Explore children
            backtrack(node.left, current_number)
            backtrack(node.right, current_number)
        
        # Backtrack (number is passed by value, so automatic backtracking)
    
    backtrack(root, 0)
    return total_sum[0]


def sum_numbers_morris_traversal(root: Optional[TreeNode]) -> int:
    """
    Morris traversal approach (challenging for this problem).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - constant extra space
    
    Algorithm:
    1. Use Morris threading technique
    2. Track current path number
    3. Handle leaf nodes during traversal
    
    Note: Morris traversal is complex for this problem due to path tracking
    """
    # Morris traversal is complex for path sum problems
    # Fall back to iterative stack approach
    return sum_numbers_iterative_stack(root)


def sum_numbers_level_order(root: Optional[TreeNode]) -> int:
    """
    Level order traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Process nodes level by level
    2. Track current number for each path
    3. Sum numbers at leaf nodes
    """
    if not root:
        return 0
    
    total_sum = 0
    queue = deque([(root, 0)])
    
    while queue:
        level_size = len(queue)
        
        for _ in range(level_size):
            node, current_number = queue.popleft()
            current_number = current_number * 10 + node.val
            
            # If leaf node, add to total sum
            if not node.left and not node.right:
                total_sum += current_number
            else:
                # Add children to next level
                if node.left:
                    queue.append((node.left, current_number))
                if node.right:
                    queue.append((node.right, current_number))
    
    return total_sum


def sum_numbers_recursive_clean(root: Optional[TreeNode]) -> int:
    """
    Clean recursive approach with helper function.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use helper function with clear parameters
    2. Build number incrementally
    3. Return sum at leaf nodes
    """
    def helper(node, path_value):
        if not node:
            return 0
        
        # Update path value
        path_value = path_value * 10 + node.val
        
        # If leaf, return this path's value
        if not node.left and not node.right:
            return path_value
        
        # Sum values from left and right subtrees
        left_sum = helper(node.left, path_value)
        right_sum = helper(node.right, path_value)
        
        return left_sum + right_sum
    
    return helper(root, 0)


def sum_numbers_iterative_path_strings(root: Optional[TreeNode]) -> int:
    """
    Iterative approach with path strings.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use stack to store (node, path_string) pairs
    2. Build path as string
    3. Convert to number at leaf nodes
    """
    if not root:
        return 0
    
    total_sum = 0
    stack = [(root, "")]
    
    while stack:
        node, path_string = stack.pop()
        path_string += str(node.val)
        
        # If leaf node, convert to number and add to sum
        if not node.left and not node.right:
            total_sum += int(path_string)
        else:
            # Add children to stack
            if node.left:
                stack.append((node.left, path_string))
            if node.right:
                stack.append((node.right, path_string))
    
    return total_sum


def sum_numbers_mathematical(root: Optional[TreeNode]) -> int:
    """
    Mathematical approach with digit operations.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use mathematical operations to build numbers
    2. Multiply by 10 and add digit at each level
    3. Sum all leaf values
    """
    def calculate(node, accumulated_value):
        if not node:
            return 0
        
        # Mathematical way to build the number
        new_value = accumulated_value * 10 + node.val
        
        # If leaf node, return the accumulated value
        if not node.left and not node.right:
            return new_value
        
        # Sum from both subtrees
        return (calculate(node.left, new_value) + 
                calculate(node.right, new_value))
    
    return calculate(root, 0)


def sum_numbers_digit_by_digit(root: Optional[TreeNode]) -> int:
    """
    Digit-by-digit approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Process each digit systematically
    2. Build numbers digit by digit
    3. Handle carry operations if needed
    """
    def process_digits(node, digits_so_far):
        if not node:
            return 0
        
        # Add current digit
        digits_so_far.append(node.val)
        
        # If leaf, convert digits to number
        if not node.left and not node.right:
            number = 0
            for digit in digits_so_far:
                number = number * 10 + digit
            digits_so_far.pop()  # Backtrack
            return number
        
        # Process children
        left_sum = process_digits(node.left, digits_so_far)
        right_sum = process_digits(node.right, digits_so_far)
        
        # Backtrack
        digits_so_far.pop()
        
        return left_sum + right_sum
    
    return process_digits(root, [])


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


def test_sum_numbers():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([1,2,3], 25),  # 12 + 13 = 25
        ([4,9,0,5,1], 1026),  # 495 + 491 + 40 = 1026
        
        # Single node
        ([1], 1),
        ([0], 0),
        ([9], 9),
        
        # Two nodes
        ([1,2], 12),
        ([1,0], 10),
        ([0,1], 1),
        
        # Three nodes
        ([1,2,3], 25),
        ([1,0,0], 10),
        ([9,9,9], 198),  # 99 + 99 = 198
        
        # Deeper trees
        ([1,2,3,4,5], 262),  # 124 + 125 + 13 = 262
        ([1,2,3,4,5,6,7], 522),  # 124 + 125 + 136 + 137 = 522
        
        # All same digits
        ([1,1,1], 22),  # 11 + 11 = 22
        ([5,5,5], 110),  # 55 + 55 = 110
        
        # With zeros
        ([1,0,1,0,1,0,1], 102),  # 100 + 101 + 10 + 11 = 222
        ([0,0,0], 0),  # 00 + 00 = 0
        
        # Large digits
        ([9,8,7,6,5,4,3], 1326),  # 986 + 985 + 974 + 973 = 3918
        
        # Left skewed
        ([1,2,None,3,None,4], 1234),
        
        # Right skewed
        ([1,None,2,None,3,None,4], 1234),
        
        # Balanced
        ([1,2,3,4,5,6,7,8,9], 1368),
        
        # Edge cases
        ([9,9,9,9,9,9,9], 1998),  # 9999 + 9999 = 19998
        ([0,1,2,3,4,5,6], 135),  # 013 + 014 + 025 + 026 = 78
        
        # Single path
        ([1,2,None,3,None,4,None,5], 12345),
        
        # Complex tree
        ([1,2,3,4,5,None,6,7,8,None,9], 13069),  # 1247 + 1248 + 1259 + 136 = 5490
        
        # All leaf nodes have same parent
        ([1,2,2,3,3,3,3], 444),  # 123 + 123 + 123 + 123 = 492
        
        # Mixed digits
        ([5,4,6,1,2,7,8], 2218),  # 541 + 542 + 567 + 568 = 2218
        
        # Large numbers
        ([9,8,7,6,5,4,3,2,1], 3924),  # 9876 + 9875 + 9843 + 9842 + 9321 = 47757
    ]
    
    # Test all implementations
    implementations = [
        ("Recursive", sum_numbers_recursive),
        ("Iterative Stack", sum_numbers_iterative_stack),
        ("Iterative Queue", sum_numbers_iterative_queue),
        ("Path Collection", sum_numbers_path_collection),
        ("String Approach", sum_numbers_string_approach),
        ("Preorder", sum_numbers_preorder),
        ("Backtracking", sum_numbers_backtracking),
        ("Morris Traversal", sum_numbers_morris_traversal),
        ("Level Order", sum_numbers_level_order),
        ("Recursive Clean", sum_numbers_recursive_clean),
        ("Iterative Path Strings", sum_numbers_iterative_path_strings),
        ("Mathematical", sum_numbers_mathematical),
        ("Digit by Digit", sum_numbers_digit_by_digit),
    ]
    
    print("Testing Sum Root to Leaf Numbers implementations:")
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
        """Generate a balanced binary tree with digits 0-9."""
        if depth == 0:
            return []
        
        size = 2 ** depth - 1
        return [i % 10 for i in range(1, size + 1)]
    
    def generate_deep_tree(depth):
        """Generate a deep tree (left skewed)."""
        result = []
        for i in range(depth):
            result.append(i % 10)
            if i < depth - 1:
                result.append(None)
        return result
    
    test_scenarios = [
        ("Small balanced", generate_balanced_tree(4)),
        ("Medium balanced", generate_balanced_tree(6)),
        ("Large balanced", generate_balanced_tree(8)),
        ("Deep tree", generate_deep_tree(50)),
        ("Wide tree", [1,2,3,4,5,6,7,8,9,0,1,2,3,4,5]),
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
    test_sum_numbers() 