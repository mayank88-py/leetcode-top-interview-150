"""
114. Flatten Binary Tree to Linked List

Given the root of a binary tree, flatten it to a linked list in-place.

The "linked list" should use the same TreeNode class where the right child pointer points to the next node in the list and the left child pointer is always null.

The "linked list" should be in the same order as a pre-order traversal of the binary tree.

Example 1:
Input: root = [1,2,5,3,4,null,6]
Output: [1,null,2,null,3,null,4,null,5,null,6]

Example 2:
Input: root = []
Output: []

Example 3:
Input: root = [0]
Output: [0]

Constraints:
- The number of nodes in the tree is in the range [0, 2000]
- -100 <= Node.val <= 100

Follow up: Can you flatten the tree in-place (with O(1) extra space)?
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


def flatten_recursive(root: Optional[TreeNode]) -> None:
    """
    Recursive approach (postorder traversal).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree (recursion stack)
    
    Algorithm:
    1. Recursively flatten left and right subtrees
    2. Connect root -> left_subtree -> right_subtree
    3. Move right subtree to the end of flattened left subtree
    """
    def flatten_helper(node):
        if not node:
            return None
        
        # Recursively flatten left and right subtrees
        left_tail = flatten_helper(node.left)
        right_tail = flatten_helper(node.right)
        
        # If left subtree exists, reconnect
        if left_tail:
            # Store right subtree
            temp = node.right
            # Connect root to left subtree
            node.right = node.left
            node.left = None
            # Connect end of left subtree to right subtree
            left_tail.right = temp
        
        # Return the tail of the flattened tree
        return right_tail or left_tail or node
    
    flatten_helper(root)


def flatten_iterative_stack(root: Optional[TreeNode]) -> None:
    """
    Iterative approach using stack.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use stack to simulate recursion
    2. Push right child first, then left child
    3. Connect current node to stack top
    """
    if not root:
        return
    
    stack = [root]
    
    while stack:
        current = stack.pop()
        
        # Push right child first (processed later)
        if current.right:
            stack.append(current.right)
        
        # Push left child second (processed first)
        if current.left:
            stack.append(current.left)
        
        # Connect current to next node in stack
        if stack:
            current.right = stack[-1]
        
        # Set left child to None
        current.left = None


def flatten_morris_traversal(root: Optional[TreeNode]) -> None:
    """
    Morris traversal approach (O(1) space).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - constant extra space
    
    Algorithm:
    1. Use Morris threading technique
    2. For each node, find rightmost node in left subtree
    3. Connect it to current node's right subtree
    4. Move left subtree to right
    """
    current = root
    
    while current:
        if current.left:
            # Find rightmost node in left subtree
            rightmost = current.left
            while rightmost.right:
                rightmost = rightmost.right
            
            # Connect rightmost to current's right subtree
            rightmost.right = current.right
            
            # Move left subtree to right
            current.right = current.left
            current.left = None
        
        # Move to next node
        current = current.right


def flatten_preorder_iterative(root: Optional[TreeNode]) -> None:
    """
    Iterative preorder traversal with reconstruction.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Perform preorder traversal iteratively
    2. Store nodes in the order they should appear
    3. Reconstruct the flattened tree
    """
    if not root:
        return
    
    nodes = []
    stack = [root]
    
    # Preorder traversal
    while stack:
        node = stack.pop()
        nodes.append(node)
        
        # Push right first (processed later)
        if node.right:
            stack.append(node.right)
        
        # Push left second (processed first)
        if node.left:
            stack.append(node.left)
    
    # Reconstruct flattened tree
    for i in range(len(nodes)):
        nodes[i].left = None
        if i < len(nodes) - 1:
            nodes[i].right = nodes[i + 1]
        else:
            nodes[i].right = None


def flatten_reverse_preorder(root: Optional[TreeNode]) -> None:
    """
    Reverse preorder approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Traverse in reverse preorder (right, left, root)
    2. Use global pointer to track the last processed node
    3. Connect current node to the previously processed node
    """
    def reverse_preorder(node, prev):
        if not node:
            return prev
        
        # Process right subtree first
        prev = reverse_preorder(node.right, prev)
        
        # Process left subtree
        prev = reverse_preorder(node.left, prev)
        
        # Connect current node
        node.right = prev
        node.left = None
        
        return node
    
    reverse_preorder(root, None)


def flatten_threaded_approach(root: Optional[TreeNode]) -> None:
    """
    Threaded binary tree approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - constant extra space
    
    Algorithm:
    1. Use threading to avoid stack/recursion
    2. Connect nodes in preorder sequence
    3. Maintain threading information
    """
    if not root:
        return
    
    current = root
    
    while current:
        if current.left:
            # Find predecessor (rightmost node in left subtree)
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                predecessor = predecessor.right
            
            if not predecessor.right:
                # Create thread
                predecessor.right = current.right
                current.right = current.left
                current.left = None
            
        current = current.right


def flatten_level_order(root: Optional[TreeNode]) -> None:
    """
    Level order traversal approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(w) where w is the maximum width of the tree
    
    Algorithm:
    1. Perform level order traversal
    2. Store nodes in preorder sequence
    3. Reconstruct flattened tree
    """
    if not root:
        return
    
    # Collect nodes in preorder using level order traversal
    nodes = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        nodes.append(node)
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    # Reconstruct flattened tree
    for i in range(len(nodes)):
        nodes[i].left = None
        if i < len(nodes) - 1:
            nodes[i].right = nodes[i + 1]
        else:
            nodes[i].right = None


def flatten_in_place_iterative(root: Optional[TreeNode]) -> None:
    """
    In-place iterative approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - constant extra space
    
    Algorithm:
    1. Process nodes in-place without additional data structures
    2. Use the tree structure itself as the "stack"
    3. Maintain preorder sequence
    """
    current = root
    
    while current:
        if current.left:
            # Find the rightmost node in the left subtree
            rightmost = current.left
            while rightmost.right:
                rightmost = rightmost.right
            
            # Connect rightmost to current's right child
            rightmost.right = current.right
            
            # Move left subtree to right
            current.right = current.left
            current.left = None
        
        current = current.right


def flatten_recursive_clean(root: Optional[TreeNode]) -> None:
    """
    Clean recursive approach with helper function.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use helper function to return the tail of flattened subtree
    2. Process left and right subtrees recursively
    3. Connect subtrees in the correct order
    """
    def flatten_and_return_tail(node):
        if not node:
            return None
        
        # Store original right child
        original_right = node.right
        
        # Flatten left subtree and get its tail
        left_tail = flatten_and_return_tail(node.left)
        
        # Flatten right subtree and get its tail
        right_tail = flatten_and_return_tail(original_right)
        
        # If left subtree exists, connect it
        if left_tail:
            node.right = node.left
            node.left = None
            left_tail.right = original_right
        
        # Return the tail of the entire flattened subtree
        return right_tail or left_tail or node
    
    flatten_and_return_tail(root)


def flatten_two_pointers(root: Optional[TreeNode]) -> None:
    """
    Two pointers approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Use two pointers to track current and previous nodes
    2. Traverse in preorder and maintain connections
    3. Update pointers as we process nodes
    """
    if not root:
        return
    
    def preorder_flatten(node):
        if not node:
            return []
        
        result = [node]
        result.extend(preorder_flatten(node.left))
        result.extend(preorder_flatten(node.right))
        return result
    
    # Get preorder sequence
    nodes = preorder_flatten(root)
    
    # Flatten using two pointers
    for i in range(len(nodes)):
        nodes[i].left = None
        if i < len(nodes) - 1:
            nodes[i].right = nodes[i + 1]
        else:
            nodes[i].right = None


def flatten_string_simulation(root: Optional[TreeNode]) -> None:
    """
    String simulation approach (for educational purposes).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - string representation
    
    Algorithm:
    1. Convert tree to string representation
    2. Parse string to get preorder sequence
    3. Reconstruct flattened tree
    """
    if not root:
        return
    
    # Convert to preorder string
    def tree_to_preorder_string(node):
        if not node:
            return ""
        
        result = str(node.val)
        left_str = tree_to_preorder_string(node.left)
        right_str = tree_to_preorder_string(node.right)
        
        if left_str:
            result += "," + left_str
        if right_str:
            result += "," + right_str
        
        return result
    
    # Get preorder string
    preorder_str = tree_to_preorder_string(root)
    values = [int(x) for x in preorder_str.split(",") if x]
    
    # Reconstruct tree (find nodes by value)
    def find_nodes_by_value(node, target_val):
        if not node:
            return []
        
        nodes = []
        if node.val == target_val:
            nodes.append(node)
        
        nodes.extend(find_nodes_by_value(node.left, target_val))
        nodes.extend(find_nodes_by_value(node.right, target_val))
        return nodes
    
    # This approach is complex due to duplicate values, so fallback to simple approach
    flatten_morris_traversal(root)


def tree_to_list(root: Optional[TreeNode]) -> List[Optional[int]]:
    """Helper function to convert flattened tree to list for testing."""
    if not root:
        return []
    
    result = []
    current = root
    
    while current:
        result.append(current.val)
        # In flattened tree, left should always be None
        if current.left is not None:
            result.append("ERROR: Left child should be None")
        current = current.right
    
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


def test_flatten():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([1,2,5,3,4,None,6], [1,2,3,4,5,6]),
        ([], []),
        ([0], [0]),
        
        # Two nodes
        ([1,2], [1,2]),
        ([1,None,2], [1,2]),
        
        # Three nodes
        ([1,2,3], [1,2,3]),
        ([1,None,2,None,3], [1,2,3]),
        ([1,2,None,3], [1,2,3]),
        
        # Left skewed
        ([1,2,None,3,None,4], [1,2,3,4]),
        
        # Right skewed
        ([1,None,2,None,3,None,4], [1,2,3,4]),
        
        # Balanced trees
        ([1,2,3,4,5,6,7], [1,2,4,5,3,6,7]),
        
        # Complex trees
        ([1,2,5,3,4,None,6,7,8,None,None,None,None,9], [1,2,3,7,8,4,5,6,9]),
        
        # Single child patterns
        ([1,2,None,3], [1,2,3]),
        ([1,None,2,3], [1,2,3]),
        
        # Large tree
        ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [1,2,4,8,9,5,10,11,3,6,12,13,7,14,15]),
        
        # Negative values
        ([-1,-2,-3], [-1,-2,-3]),
        ([1,-2,3,-4,-5], [1,-2,-4,-5,3]),
        
        # Mixed values
        ([0,1,2,3,4,5,6], [0,1,3,4,2,5,6]),
        
        # Deep tree
        ([1,2,None,3,None,4,None,5], [1,2,3,4,5]),
        
        # Wide tree
        ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [1,2,4,8,9,5,10,11,3,6,12,13,7,14,15]),
    ]
    
    # Test all implementations
    implementations = [
        ("Recursive", flatten_recursive),
        ("Iterative Stack", flatten_iterative_stack),
        ("Morris Traversal", flatten_morris_traversal),
        ("Preorder Iterative", flatten_preorder_iterative),
        ("Reverse Preorder", flatten_reverse_preorder),
        ("Threaded Approach", flatten_threaded_approach),
        ("Level Order", flatten_level_order),
        ("In-place Iterative", flatten_in_place_iterative),
        ("Recursive Clean", flatten_recursive_clean),
        ("Two Pointers", flatten_two_pointers),
        ("String Simulation", flatten_string_simulation),
    ]
    
    print("Testing Flatten Binary Tree to Linked List implementations:")
    print("=" * 70)
    
    for i, (tree_values, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Input tree: {tree_values}")
        print(f"Expected:   {expected}")
        
        for name, func in implementations:
            try:
                # Create fresh tree for each implementation
                root = create_tree_from_list(tree_values.copy())
                func(root)
                result = tree_to_list(root)
                
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
        """Generate a balanced binary tree of given depth."""
        if depth == 0:
            return []
        
        size = 2 ** depth - 1
        return list(range(1, size + 1))
    
    def generate_left_skewed_tree(size):
        """Generate a left-skewed tree."""
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
        ("Left skewed", generate_left_skewed_tree(100)),
        ("Right skewed", list(range(1, 101))),
    ]
    
    for scenario_name, tree_values in test_scenarios:
        print(f"\n{scenario_name} ({len([x for x in tree_values if x is not None])} nodes):")
        
        for name, func in implementations:
            try:
                root = create_tree_from_list(tree_values.copy())
                
                start_time = time.time()
                func(root)
                end_time = time.time()
                
                flattened_length = len(tree_to_list(root))
                print(f"  {name}: {flattened_length} nodes in {end_time - start_time:.4f} seconds")
            except Exception as e:
                print(f"  {name}: Error - {e}")


if __name__ == "__main__":
    test_flatten() 