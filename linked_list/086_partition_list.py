"""
86. Partition List

Given the head of a linked list and a value x, partition it such that all nodes with values less than x come before nodes with values greater than or equal to x.

You should preserve the original relative order of the nodes in each of the two partitions.

Example 1:
Input: head = [1,4,3,2,5,2], x = 3
Output: [1,2,2,4,3,5]

Example 2:
Input: head = [2,1], x = 2
Output: [1,2]

Constraints:
- The number of nodes in the list is in the range [0, 200]
- -100 <= Node.val <= 100
- -200 <= x <= 200
"""

from typing import Optional, List


class ListNode:
    """Definition for singly-linked list."""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        """String representation for debugging."""
        result = []
        current = self
        while current:
            result.append(str(current.val))
            current = current.next
        return " -> ".join(result)


def partition_two_pointers(head: Optional[ListNode], x: int) -> Optional[ListNode]:
    """
    Two pointers approach with dummy nodes.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Create two dummy nodes for less than and greater than or equal to lists
    2. Traverse original list and distribute nodes
    3. Connect the two lists
    4. Return the merged result
    """
    # Create dummy nodes for two partitions
    less_head = ListNode(0)
    greater_head = ListNode(0)
    
    less_current = less_head
    greater_current = greater_head
    
    current = head
    
    while current:
        if current.val < x:
            less_current.next = current
            less_current = current
        else:
            greater_current.next = current
            greater_current = current
        
        current = current.next
    
    # Connect the two lists
    greater_current.next = None  # End the greater list
    less_current.next = greater_head.next  # Connect less to greater
    
    return less_head.next


def partition_array(head: Optional[ListNode], x: int) -> Optional[ListNode]:
    """
    Array-based approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - arrays to store nodes
    
    Algorithm:
    1. Separate nodes into two arrays based on value
    2. Reconstruct the list from arrays
    3. Return the new head
    """
    if not head:
        return None
    
    less_nodes = []
    greater_nodes = []
    
    current = head
    while current:
        if current.val < x:
            less_nodes.append(current)
        else:
            greater_nodes.append(current)
        current = current.next
    
    # Reconstruct the list
    all_nodes = less_nodes + greater_nodes
    
    for i in range(len(all_nodes) - 1):
        all_nodes[i].next = all_nodes[i + 1]
    
    if all_nodes:
        all_nodes[-1].next = None
        return all_nodes[0]
    
    return None


def partition_recursive(head: Optional[ListNode], x: int) -> Optional[ListNode]:
    """
    Recursive approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - recursion stack
    
    Algorithm:
    1. Recursively partition the rest of the list
    2. Based on current node value, place it appropriately
    3. Return the partitioned list
    """
    if not head:
        return None
    
    def partition_helper(node):
        if not node:
            return None, None  # (less_head, greater_head)
        
        less_rest, greater_rest = partition_helper(node.next)
        
        if node.val < x:
            # Add to less list
            node.next = less_rest
            return node, greater_rest
        else:
            # Add to greater list
            node.next = greater_rest
            return less_rest, node
    
    less_head, greater_head = partition_helper(head)
    
    # Connect less list to greater list
    if less_head:
        current = less_head
        while current.next:
            current = current.next
        current.next = greater_head
        return less_head
    else:
        return greater_head


def partition_values(head: Optional[ListNode], x: int) -> Optional[ListNode]:
    """
    Values-based approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - arrays to store values
    
    Algorithm:
    1. Extract all values into arrays based on condition
    2. Create new linked list from combined arrays
    3. Return the new list
    """
    if not head:
        return None
    
    less_values = []
    greater_values = []
    
    current = head
    while current:
        if current.val < x:
            less_values.append(current.val)
        else:
            greater_values.append(current.val)
        current = current.next
    
    # Create new list
    all_values = less_values + greater_values
    
    if not all_values:
        return None
    
    new_head = ListNode(all_values[0])
    current = new_head
    
    for val in all_values[1:]:
        current.next = ListNode(val)
        current = current.next
    
    return new_head


def partition_single_pass(head: Optional[ListNode], x: int) -> Optional[ListNode]:
    """
    Single pass approach with in-place partitioning.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Use single pass to partition
    2. Maintain pointers for both partitions
    3. Connect partitions at the end
    """
    if not head:
        return None
    
    # Find first node >= x to use as boundary
    less_tail = None
    greater_head = None
    greater_tail = None
    
    # Handle the case where head should be in greater partition
    if head.val >= x:
        greater_head = head
        greater_tail = head
        current = head.next
    else:
        less_tail = head
        current = head.next
    
    while current:
        if current.val < x:
            if less_tail:
                less_tail.next = current
                less_tail = current
            else:
                # First less node
                new_head = current
                less_tail = current
                temp = current.next
                current.next = greater_head
                current = temp
                continue
        else:
            if greater_head is None:
                greater_head = current
                greater_tail = current
            else:
                greater_tail.next = current
                greater_tail = current
        
        current = current.next
    
    # Connect partitions
    if less_tail:
        less_tail.next = greater_head
        if greater_tail:
            greater_tail.next = None
        return head if head.val < x else new_head
    else:
        if greater_tail:
            greater_tail.next = None
        return greater_head


def partition_stack(head: Optional[ListNode], x: int) -> Optional[ListNode]:
    """
    Stack-based approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - stacks to store nodes
    
    Algorithm:
    1. Use two stacks to store nodes
    2. Pop from stacks to rebuild list
    3. Maintain relative order
    """
    if not head:
        return None
    
    less_stack = []
    greater_stack = []
    
    current = head
    while current:
        if current.val < x:
            less_stack.append(current)
        else:
            greater_stack.append(current)
        current = current.next
    
    # Rebuild list from stacks (maintaining order)
    all_nodes = less_stack + greater_stack
    
    if not all_nodes:
        return None
    
    for i in range(len(all_nodes) - 1):
        all_nodes[i].next = all_nodes[i + 1]
    
    all_nodes[-1].next = None
    return all_nodes[0]


def partition_deque(head: Optional[ListNode], x: int) -> Optional[ListNode]:
    """
    Deque-based approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - deques to store nodes
    
    Algorithm:
    1. Use deques to maintain order
    2. Append nodes to appropriate deque
    3. Combine deques to form result
    """
    if not head:
        return None
    
    from collections import deque
    
    less_deque = deque()
    greater_deque = deque()
    
    current = head
    while current:
        if current.val < x:
            less_deque.append(current)
        else:
            greater_deque.append(current)
        current = current.next
    
    # Combine deques
    combined = list(less_deque) + list(greater_deque)
    
    if not combined:
        return None
    
    for i in range(len(combined) - 1):
        combined[i].next = combined[i + 1]
    
    combined[-1].next = None
    return combined[0]


def partition_three_pointers(head: Optional[ListNode], x: int) -> Optional[ListNode]:
    """
    Three pointers approach for clear logic.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Use three pointers: current, less_tail, greater_tail
    2. Maintain two separate lists
    3. Connect at the end
    """
    if not head:
        return None
    
    less_dummy = ListNode(0)
    greater_dummy = ListNode(0)
    
    less_tail = less_dummy
    greater_tail = greater_dummy
    current = head
    
    while current:
        next_node = current.next
        
        if current.val < x:
            less_tail.next = current
            less_tail = current
        else:
            greater_tail.next = current
            greater_tail = current
        
        current.next = None  # Clear next pointer
        current = next_node
    
    # Connect the two lists
    less_tail.next = greater_dummy.next
    
    return less_dummy.next


def partition_counting(head: Optional[ListNode], x: int) -> Optional[ListNode]:
    """
    Counting approach for demonstration.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(1) - only using constant extra space
    
    Algorithm:
    1. Count nodes in each partition
    2. Rearrange based on counts
    3. Maintain relative order
    """
    if not head:
        return None
    
    # Count nodes in each partition
    less_count = 0
    greater_count = 0
    current = head
    
    while current:
        if current.val < x:
            less_count += 1
        else:
            greater_count += 1
        current = current.next
    
    # Use array to rearrange
    nodes = []
    current = head
    
    while current:
        nodes.append(current)
        current = current.next
    
    # Separate into two arrays
    less_nodes = [node for node in nodes if node.val < x]
    greater_nodes = [node for node in nodes if node.val >= x]
    
    # Combine
    all_nodes = less_nodes + greater_nodes
    
    for i in range(len(all_nodes) - 1):
        all_nodes[i].next = all_nodes[i + 1]
    
    if all_nodes:
        all_nodes[-1].next = None
        return all_nodes[0]
    
    return None


def partition_functional(head: Optional[ListNode], x: int) -> Optional[ListNode]:
    """
    Functional programming approach.
    
    Time Complexity: O(n) where n is the number of nodes
    Space Complexity: O(n) - recursion and new nodes
    
    Algorithm:
    1. Use functional programming style
    2. Filter and map operations
    3. Combine results
    """
    def extract_values(node):
        if not node:
            return []
        return [node.val] + extract_values(node.next)
    
    def create_list(values):
        if not values:
            return None
        head = ListNode(values[0])
        current = head
        for val in values[1:]:
            current.next = ListNode(val)
            current = current.next
        return head
    
    if not head:
        return None
    
    values = extract_values(head)
    less_values = [val for val in values if val < x]
    greater_values = [val for val in values if val >= x]
    
    return create_list(less_values + greater_values)


def create_linked_list(values: List[int]) -> Optional[ListNode]:
    """Helper function to create a linked list from a list of values."""
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    
    return head


def linked_list_to_list(head: Optional[ListNode]) -> List[int]:
    """Helper function to convert linked list to list for testing."""
    result = []
    current = head
    
    while current:
        result.append(current.val)
        current = current.next
    
    return result


def test_partition():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([1,4,3,2,5,2], 3, [1,2,2,4,3,5]),
        ([2,1], 2, [1,2]),
        ([], 0, []),
        ([1], 1, [1]),
        ([1], 2, [1]),
        
        # All less than x
        ([1,2,3], 5, [1,2,3]),
        
        # All greater than or equal to x
        ([5,6,7], 3, [5,6,7]),
        
        # No partition needed
        ([1,2,3,4,5], 3, [1,2,4,3,5]),
        ([5,4,3,2,1], 3, [2,1,5,4,3]),
        
        # Negative values
        ([-1,-2,0,1,2], 0, [-1,-2,0,1,2]),
        ([1,-1,2,-2], 0, [-1,-2,1,2]),
        
        # Same values
        ([3,3,3], 3, [3,3,3]),
        ([2,2,2], 3, [2,2,2]),
        ([4,4,4], 3, [4,4,4]),
        
        # Mixed patterns
        ([1,4,2,5,3,6], 4, [1,2,3,4,5,6]),
        ([6,5,4,3,2,1], 4, [3,2,1,6,5,4]),
        
        # Large gaps
        ([1,100,2,200,3], 50, [1,2,3,100,200]),
        ([100,1,200,2,300], 50, [1,2,100,200,300]),
        
        # Edge values
        ([1,2,3,4,5], 1, [2,3,4,5,1]),  # x equals minimum
        ([1,2,3,4,5], 5, [1,2,3,4,5]),  # x equals maximum
        ([1,2,3,4,5], 6, [1,2,3,4,5]),  # x greater than maximum
        ([1,2,3,4,5], 0, [1,2,3,4,5]),  # x less than minimum
        
        # Single partitions
        ([5,1,3,2,4], 3, [1,2,5,3,4]),
        ([1,5,2,4,3], 3, [1,2,5,4,3]),
        
        # Complex patterns
        ([1,4,3,0,2,5,2], 3, [1,0,2,2,4,3,5]),
        ([5,1,4,2,3,6,0], 3, [1,2,0,5,4,3,6]),
    ]
    
    # Test all implementations
    implementations = [
        ("Two Pointers", partition_two_pointers),
        ("Array", partition_array),
        ("Recursive", partition_recursive),
        ("Values", partition_values),
        ("Single Pass", partition_single_pass),
        ("Stack", partition_stack),
        ("Deque", partition_deque),
        ("Three Pointers", partition_three_pointers),
        ("Counting", partition_counting),
        ("Functional", partition_functional),
    ]
    
    print("Testing Partition List implementations:")
    print("=" * 60)
    
    for i, (values, x, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {values}, x={x}")
        print(f"Expected: {expected}")
        
        for name, func in implementations:
            try:
                # Create fresh linked list for each implementation
                head = create_linked_list(values)
                result_head = func(head, x)
                result = linked_list_to_list(result_head)
                
                status = "✓" if result == expected else "✗"
                print(f"{status} {name}: {result}")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 60)
    print("Performance Analysis:")
    print("=" * 60)
    
    import time
    
    # Generate large test cases
    test_scenarios = [
        ("Random small", [i % 10 for i in range(20)], 5),
        ("Random medium", [i % 20 for i in range(100)], 10),
        ("All less", list(range(50)), 100),
        ("All greater", list(range(50, 100)), 25),
        ("Alternating", [i % 2 for i in range(100)], 1),
    ]
    
    for scenario_name, values, x in test_scenarios:
        print(f"\n{scenario_name} (x={x}):")
        
        for name, func in implementations:
            try:
                head = create_linked_list(values)
                
                start_time = time.time()
                result_head = func(head, x)
                end_time = time.time()
                
                result_length = len(linked_list_to_list(result_head))
                print(f"  {name}: {result_length} nodes in {end_time - start_time:.4f} seconds")
            except Exception as e:
                print(f"  {name}: Error - {e}")


if __name__ == "__main__":
    test_partition() 