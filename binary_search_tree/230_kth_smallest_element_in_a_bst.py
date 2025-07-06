"""
230. Kth Smallest Element in a BST

Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.

Example 1:
Input: root = [3,1,4,null,2], k = 1
Output: 1

Example 2:
Input: root = [5,3,6,2,4,null,null,1], k = 3
Output: 3

Constraints:
- The number of nodes in the tree is n.
- 1 <= k <= n <= 10^4
- 0 <= Node.val <= 10^4

Follow up: If the BST is modified often (i.e., we can do insert and delete operations) and you need to find the kth smallest frequently, how would you optimize?
"""

from typing import Optional, List
from collections import deque
import heapq


class TreeNode:
    """Definition for a binary tree node."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        """String representation for debugging."""
        return f"TreeNode({self.val})"


def kth_smallest_inorder_recursive(root: Optional[TreeNode], k: int) -> int:
    """
    Recursive inorder traversal approach.
    
    Time Complexity: O(H + k) where H is the height of the tree
    Space Complexity: O(H) - recursion stack
    
    Algorithm:
    1. Perform inorder traversal (gives sorted order)
    2. Stop when we reach the kth element
    3. Use early termination for efficiency
    """
    def inorder(node):
        if not node:
            return None
        
        # Try left subtree
        left_result = inorder(node.left)
        if left_result is not None:
            return left_result
        
        # Process current node
        inorder.count += 1
        if inorder.count == k:
            return node.val
        
        # Try right subtree
        return inorder(node.right)
    
    inorder.count = 0
    return inorder(root)


def kth_smallest_inorder_iterative(root: Optional[TreeNode], k: int) -> int:
    """
    Iterative inorder traversal approach.
    
    Time Complexity: O(H + k) where H is the height of the tree
    Space Complexity: O(H) - stack space
    
    Algorithm:
    1. Use stack for iterative inorder traversal
    2. Count elements as we process them
    3. Return when we reach the kth element
    """
    stack = []
    count = 0
    current = root
    
    while stack or current:
        # Go to leftmost node
        while current:
            stack.append(current)
            current = current.left
        
        # Process current node
        current = stack.pop()
        count += 1
        
        if count == k:
            return current.val
        
        # Move to right subtree
        current = current.right
    
    return -1  # Should never reach here with valid input


def kth_smallest_morris_traversal(root: Optional[TreeNode], k: int) -> int:
    """
    Morris inorder traversal approach (O(1) space).
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - constant extra space
    
    Algorithm:
    1. Use Morris threading technique
    2. Process nodes in inorder without using stack
    3. Count elements and return kth
    """
    count = 0
    current = root
    
    while current:
        if not current.left:
            # Process current node
            count += 1
            if count == k:
                return current.val
            current = current.right
        else:
            # Find inorder predecessor
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                predecessor = predecessor.right
            
            if not predecessor.right:
                # Create threading
                predecessor.right = current
                current = current.left
            else:
                # Remove threading and process current node
                predecessor.right = None
                count += 1
                if count == k:
                    return current.val
                current = current.right
    
    return -1  # Should never reach here with valid input


def kth_smallest_collect_all(root: Optional[TreeNode], k: int) -> int:
    """
    Collect all values and sort approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - storing all values
    
    Algorithm:
    1. Collect all node values
    2. Sort them (though BST inorder is already sorted)
    3. Return kth element
    """
    def collect_values(node):
        if not node:
            return []
        
        values = []
        values.extend(collect_values(node.left))
        values.append(node.val)
        values.extend(collect_values(node.right))
        return values
    
    values = collect_values(root)
    return values[k - 1]


def kth_smallest_binary_search(root: Optional[TreeNode], k: int) -> int:
    """
    Binary search approach using subtree size.
    
    Time Complexity: O(H^2) where H is the height of the tree
    Space Complexity: O(H) - recursion stack
    
    Algorithm:
    1. Count nodes in left subtree
    2. Use binary search logic to decide which subtree to explore
    3. Adjust k based on subtree sizes
    """
    def count_nodes(node):
        if not node:
            return 0
        return 1 + count_nodes(node.left) + count_nodes(node.right)
    
    def binary_search(node, k):
        if not node:
            return -1
        
        left_count = count_nodes(node.left)
        
        if k <= left_count:
            # kth element is in left subtree
            return binary_search(node.left, k)
        elif k == left_count + 1:
            # Current node is the kth element
            return node.val
        else:
            # kth element is in right subtree
            return binary_search(node.right, k - left_count - 1)
    
    return binary_search(root, k)


def kth_smallest_priority_queue(root: Optional[TreeNode], k: int) -> int:
    """
    Priority queue (min-heap) approach.
    
    Time Complexity: O(n log n) where n is the number of nodes
    Space Complexity: O(n) - heap space
    
    Algorithm:
    1. Add all node values to min-heap
    2. Extract minimum k times
    3. Return the kth minimum
    """
    def collect_values(node):
        if not node:
            return
        
        heapq.heappush(heap, node.val)
        collect_values(node.left)
        collect_values(node.right)
    
    heap = []
    collect_values(root)
    
    for _ in range(k - 1):
        heapq.heappop(heap)
    
    return heapq.heappop(heap)


def kth_smallest_level_order(root: Optional[TreeNode], k: int) -> int:
    """
    Level order traversal with sorting.
    
    Time Complexity: O(n log n) where n is the number of nodes
    Space Complexity: O(n) - queue and values storage
    
    Algorithm:
    1. Use level order traversal to collect all values
    2. Sort the values
    3. Return kth element
    """
    if not root:
        return -1
    
    values = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        values.append(node.val)
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    values.sort()
    return values[k - 1]


def kth_smallest_divide_conquer(root: Optional[TreeNode], k: int) -> int:
    """
    Divide and conquer approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(h) where h is the height of the tree
    
    Algorithm:
    1. Divide: get sorted values from left and right subtrees
    2. Conquer: merge results with current node
    3. Return kth element from merged result
    """
    def divide_conquer(node):
        if not node:
            return []
        
        left_values = divide_conquer(node.left)
        right_values = divide_conquer(node.right)
        
        # Merge in sorted order (inorder traversal)
        result = left_values + [node.val] + right_values
        return result
    
    values = divide_conquer(root)
    return values[k - 1]


def kth_smallest_early_termination(root: Optional[TreeNode], k: int) -> int:
    """
    Early termination with counter approach.
    
    Time Complexity: O(H + k) where H is the height of the tree
    Space Complexity: O(H) - recursion stack
    
    Algorithm:
    1. Use inorder traversal with early termination
    2. Maintain a global counter
    3. Return as soon as we reach the kth element
    """
    def inorder(node):
        if not node or inorder.found:
            return
        
        inorder(node.left)
        
        if not inorder.found:
            inorder.counter += 1
            if inorder.counter == k:
                inorder.result = node.val
                inorder.found = True
                return
        
        inorder(node.right)
    
    inorder.counter = 0
    inorder.result = -1
    inorder.found = False
    
    inorder(root)
    return inorder.result


def kth_smallest_stack_with_size(root: Optional[TreeNode], k: int) -> int:
    """
    Stack approach with subtree size optimization.
    
    Time Complexity: O(H + k) where H is the height of the tree
    Space Complexity: O(H) - stack space
    
    Algorithm:
    1. Precompute subtree sizes
    2. Use stack with smart navigation
    3. Skip subtrees when possible
    """
    # Precompute subtree sizes
    def compute_sizes(node):
        if not node:
            return 0
        
        left_size = compute_sizes(node.left)
        right_size = compute_sizes(node.right)
        sizes[node] = left_size + right_size + 1
        return sizes[node]
    
    sizes = {}
    compute_sizes(root)
    
    # Navigate with size information
    current = root
    while current:
        left_size = sizes.get(current.left, 0)
        
        if k <= left_size:
            current = current.left
        elif k == left_size + 1:
            return current.val
        else:
            k -= left_size + 1
            current = current.right
    
    return -1


# Enhanced BST with size tracking for frequent queries
class BSTWithSize:
    """
    Enhanced BST that tracks subtree sizes for O(H) kth element queries.
    """
    
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        self.size = 1  # Size of subtree rooted at this node
    
    def update_size(self):
        """Update size based on children."""
        left_size = self.left.size if self.left else 0
        right_size = self.right.size if self.right else 0
        self.size = left_size + right_size + 1
    
    def insert(self, val):
        """Insert value and update sizes."""
        if val < self.val:
            if self.left:
                self.left.insert(val)
            else:
                self.left = BSTWithSize(val)
        else:
            if self.right:
                self.right.insert(val)
            else:
                self.right = BSTWithSize(val)
        self.update_size()
    
    def kth_smallest(self, k):
        """Find kth smallest in O(H) time."""
        left_size = self.left.size if self.left else 0
        
        if k <= left_size:
            return self.left.kth_smallest(k)
        elif k == left_size + 1:
            return self.val
        else:
            return self.right.kth_smallest(k - left_size - 1)


def kth_smallest_optimized_bst(root: Optional[TreeNode], k: int) -> int:
    """
    Optimized approach for frequent queries.
    
    Time Complexity: O(H) where H is the height of the tree
    Space Complexity: O(n) - storing size information
    
    Algorithm:
    1. Convert to enhanced BST with size tracking
    2. Use binary search with size information
    3. Optimal for frequent kth queries
    """
    def convert_to_enhanced(node):
        if not node:
            return None
        
        enhanced = BSTWithSize(node.val)
        enhanced.left = convert_to_enhanced(node.left)
        enhanced.right = convert_to_enhanced(node.right)
        enhanced.update_size()
        return enhanced
    
    enhanced_root = convert_to_enhanced(root)
    return enhanced_root.kth_smallest(k)


# Test cases
def test_kth_smallest():
    """Test all kth smallest approaches."""
    
    def create_test_tree_1():
        """Create test tree: [3,1,4,null,2]"""
        root = TreeNode(3)
        root.left = TreeNode(1)
        root.right = TreeNode(4)
        root.left.right = TreeNode(2)
        return root
    
    def create_test_tree_2():
        """Create test tree: [5,3,6,2,4,null,null,1]"""
        root = TreeNode(5)
        root.left = TreeNode(3)
        root.right = TreeNode(6)
        root.left.left = TreeNode(2)
        root.left.right = TreeNode(4)
        root.left.left.left = TreeNode(1)
        return root
    
    def create_test_tree_3():
        """Create test tree: [1]"""
        return TreeNode(1)
    
    def create_test_tree_4():
        """Create test tree: [4,2,6,1,3,5,7]"""
        root = TreeNode(4)
        root.left = TreeNode(2)
        root.right = TreeNode(6)
        root.left.left = TreeNode(1)
        root.left.right = TreeNode(3)
        root.right.left = TreeNode(5)
        root.right.right = TreeNode(7)
        return root
    
    # Test cases
    test_cases = [
        (create_test_tree_1(), 1, 1, "[3,1,4,null,2], k=1"),
        (create_test_tree_2(), 3, 3, "[5,3,6,2,4,null,null,1], k=3"),
        (create_test_tree_3(), 1, 1, "[1], k=1"),
        (create_test_tree_4(), 1, 1, "[4,2,6,1,3,5,7], k=1"),
        (create_test_tree_4(), 4, 4, "[4,2,6,1,3,5,7], k=4"),
        (create_test_tree_4(), 7, 7, "[4,2,6,1,3,5,7], k=7"),
    ]
    
    # Test all approaches
    approaches = [
        (kth_smallest_inorder_recursive, "Recursive inorder"),
        (kth_smallest_inorder_iterative, "Iterative inorder"),
        (kth_smallest_morris_traversal, "Morris traversal"),
        (kth_smallest_collect_all, "Collect all values"),
        (kth_smallest_binary_search, "Binary search"),
        (kth_smallest_priority_queue, "Priority queue"),
        (kth_smallest_level_order, "Level order"),
        (kth_smallest_divide_conquer, "Divide and conquer"),
        (kth_smallest_early_termination, "Early termination"),
        (kth_smallest_stack_with_size, "Stack with size"),
        (kth_smallest_optimized_bst, "Optimized BST"),
    ]
    
    print("Testing kth smallest approaches:")
    print("=" * 50)
    
    for i, (root, k, expected, description) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: {description}")
        print(f"Expected: {expected}")
        
        for func, name in approaches:
            try:
                result = func(root, k)
                status = "✓" if result == expected else "✗"
                print(f"{status} {name}: {result}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 50)
    print("Performance Analysis:")
    print("=" * 50)
    
    # Create larger test tree for performance testing
    def create_large_bst(n):
        """Create a balanced BST with n nodes."""
        if n <= 0:
            return None
        
        def build_bst(start, end):
            if start > end:
                return None
            
            mid = (start + end) // 2
            node = TreeNode(mid)
            node.left = build_bst(start, mid - 1)
            node.right = build_bst(mid + 1, end)
            return node
        
        return build_bst(1, n)
    
    large_tree = create_large_bst(1000)
    k_test = 500
    
    import time
    
    print(f"Testing with large BST (1000 nodes), k={k_test}:")
    for func, name in approaches:
        try:
            start_time = time.time()
            result = func(large_tree, k_test)
            end_time = time.time()
            print(f"{name}: {result} (Time: {end_time - start_time:.6f}s)")
        except Exception as e:
            print(f"{name}: Error - {e}")


if __name__ == "__main__":
    test_kth_smallest() 