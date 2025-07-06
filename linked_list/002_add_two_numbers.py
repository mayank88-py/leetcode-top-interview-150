"""
2. Add Two Numbers

You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Example 1:
Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.

Example 2:
Input: l1 = [0], l2 = [0]
Output: [0]

Example 3:
Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
Output: [8,9,9,9,0,0,0,1]

Constraints:
- The number of nodes in each linked list is in the range [1, 100]
- 0 <= Node.val <= 9
- It is guaranteed that the list represents a number that does not have leading zeros
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


def add_two_numbers_iterative(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Iterative approach with carry handling.
    
    Time Complexity: O(max(n, m)) where n and m are lengths of the lists
    Space Complexity: O(max(n, m)) - for the result list
    
    Algorithm:
    1. Use dummy node to simplify result construction
    2. Iterate through both lists simultaneously
    3. Handle carry from previous addition
    4. Create new nodes for each digit of the result
    """
    dummy = ListNode(0)
    current = dummy
    carry = 0
    
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        
        total = val1 + val2 + carry
        carry = total // 10
        digit = total % 10
        
        current.next = ListNode(digit)
        current = current.next
        
        if l1:
            l1 = l1.next
        if l2:
            l2 = l2.next
    
    return dummy.next


def add_two_numbers_recursive(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Recursive approach with carry handling.
    
    Time Complexity: O(max(n, m)) where n and m are lengths of the lists
    Space Complexity: O(max(n, m)) - recursion stack + result list
    
    Algorithm:
    1. Use recursion with carry parameter
    2. Handle base cases (empty lists)
    3. Compute sum and carry for current position
    4. Recursively process remaining nodes
    """
    def helper(l1, l2, carry):
        if not l1 and not l2 and carry == 0:
            return None
        
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        
        total = val1 + val2 + carry
        new_carry = total // 10
        digit = total % 10
        
        next_l1 = l1.next if l1 else None
        next_l2 = l2.next if l2 else None
        
        result = ListNode(digit)
        result.next = helper(next_l1, next_l2, new_carry)
        
        return result
    
    return helper(l1, l2, 0)


def add_two_numbers_in_place(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    """
    In-place approach modifying the longer list.
    
    Time Complexity: O(max(n, m)) where n and m are lengths of the lists
    Space Complexity: O(1) - only using constant extra space (excluding result)
    
    Algorithm:
    1. Determine which list is longer
    2. Use the longer list as the result
    3. Modify nodes in place
    4. Handle carry propagation
    """
    # Determine lengths
    def get_length(node):
        length = 0
        while node:
            length += 1
            node = node.next
        return length
    
    len1 = get_length(l1)
    len2 = get_length(l2)
    
    # Ensure l1 is the longer list
    if len1 < len2:
        l1, l2 = l2, l1
    
    head = l1
    carry = 0
    
    while l1:
        val1 = l1.val
        val2 = l2.val if l2 else 0
        
        total = val1 + val2 + carry
        carry = total // 10
        l1.val = total % 10
        
        if l1.next is None and carry > 0:
            l1.next = ListNode(carry)
            break
        
        l1 = l1.next
        if l2:
            l2 = l2.next
    
    return head


def add_two_numbers_string_conversion(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    """
    String conversion approach (alternative method).
    
    Time Complexity: O(max(n, m)) where n and m are lengths of the lists
    Space Complexity: O(max(n, m)) - for string representations
    
    Algorithm:
    1. Convert linked lists to numbers
    2. Add the numbers
    3. Convert result back to linked list
    """
    def list_to_number(head):
        number = 0
        multiplier = 1
        
        while head:
            number += head.val * multiplier
            multiplier *= 10
            head = head.next
        
        return number
    
    def number_to_list(number):
        if number == 0:
            return ListNode(0)
        
        dummy = ListNode(0)
        current = dummy
        
        while number > 0:
            digit = number % 10
            current.next = ListNode(digit)
            current = current.next
            number //= 10
        
        return dummy.next
    
    num1 = list_to_number(l1)
    num2 = list_to_number(l2)
    result = num1 + num2
    
    return number_to_list(result)


def add_two_numbers_stack(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Stack-based approach for demonstration.
    
    Time Complexity: O(max(n, m)) where n and m are lengths of the lists
    Space Complexity: O(max(n, m)) - for stacks
    
    Algorithm:
    1. Push all digits to stacks
    2. Pop and add with carry handling
    3. Build result list
    """
    stack1 = []
    stack2 = []
    
    # Push all digits to stacks
    while l1:
        stack1.append(l1.val)
        l1 = l1.next
    
    while l2:
        stack2.append(l2.val)
        l2 = l2.next
    
    # Add from least significant digit
    dummy = ListNode(0)
    current = dummy
    carry = 0
    
    while stack1 or stack2 or carry:
        val1 = stack1.pop() if stack1 else 0
        val2 = stack2.pop() if stack2 else 0
        
        total = val1 + val2 + carry
        carry = total // 10
        digit = total % 10
        
        current.next = ListNode(digit)
        current = current.next
    
    return dummy.next


def add_two_numbers_array(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Array-based approach for comparison.
    
    Time Complexity: O(max(n, m)) where n and m are lengths of the lists
    Space Complexity: O(max(n, m)) - for arrays
    
    Algorithm:
    1. Convert lists to arrays
    2. Add arrays with carry handling
    3. Convert result array back to linked list
    """
    def list_to_array(head):
        arr = []
        while head:
            arr.append(head.val)
            head = head.next
        return arr
    
    def array_to_list(arr):
        if not arr:
            return ListNode(0)
        
        dummy = ListNode(0)
        current = dummy
        
        for digit in arr:
            current.next = ListNode(digit)
            current = current.next
        
        return dummy.next
    
    arr1 = list_to_array(l1)
    arr2 = list_to_array(l2)
    
    max_len = max(len(arr1), len(arr2))
    result = []
    carry = 0
    
    for i in range(max_len):
        val1 = arr1[i] if i < len(arr1) else 0
        val2 = arr2[i] if i < len(arr2) else 0
        
        total = val1 + val2 + carry
        carry = total // 10
        digit = total % 10
        
        result.append(digit)
    
    if carry:
        result.append(carry)
    
    return array_to_list(result)


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


def test_add_two_numbers():
    """Test all implementations with various test cases."""
    
    test_cases = [
        # Basic cases
        ([2,4,3], [5,6,4], [7,0,8]),
        ([0], [0], [0]),
        ([9,9,9,9,9,9,9], [9,9,9,9], [8,9,9,9,0,0,0,1]),
        
        # Single digits
        ([1], [2], [3]),
        ([9], [1], [0,1]),
        ([5], [5], [0,1]),
        
        # Different lengths
        ([1,2,3], [4,5,6,7,8], [5,7,9,7,8]),
        ([9,9], [1], [0,0,1]),
        
        # Carry propagation
        ([9,9,9], [1], [0,0,0,1]),
        ([1], [9,9,9], [0,0,0,1]),
        
        # Multiple carries
        ([9,9,9,9,9], [9,9,9,9,9], [8,9,9,9,9,1]),
        
        # Edge cases
        ([1,8], [0], [1,8]),
        ([0], [1,8], [1,8]),
        ([5,5], [5,5], [0,1,1]),
        
        # Large numbers
        ([1,2,3,4,5], [6,7,8,9,0], [7,9,1,4,5]),
        ([9,8,7,6,5], [1,2,3,4,5], [0,1,1,1,1,1]),
        
        # All zeros except last
        ([0,0,0,1], [0,0,0,2], [0,0,0,3]),
        ([0,0,0,9], [0,0,0,9], [0,0,0,8,1]),
    ]
    
    # Test all implementations
    implementations = [
        ("Iterative", add_two_numbers_iterative),
        ("Recursive", add_two_numbers_recursive),
        ("In-place", add_two_numbers_in_place),
        ("String Conversion", add_two_numbers_string_conversion),
        ("Stack", add_two_numbers_stack),
        ("Array", add_two_numbers_array),
    ]
    
    print("Testing Add Two Numbers implementations:")
    print("=" * 60)
    
    for i, (l1_vals, l2_vals, expected) in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {l1_vals} + {l2_vals}")
        print(f"Expected: {expected}")
        
        for name, func in implementations:
            try:
                # Create fresh linked lists for each implementation
                l1 = create_linked_list(l1_vals)
                l2 = create_linked_list(l2_vals)
                
                result_head = func(l1, l2)
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
        ("Small numbers", [1,2,3], [4,5,6]),
        ("Large equal numbers", [9]*50, [9]*50),
        ("Different lengths", [1,2,3,4,5,6,7,8,9], [1]),
        ("Maximum carry", [9]*100, [1]),
    ]
    
    for scenario_name, l1_vals, l2_vals in test_scenarios:
        print(f"\n{scenario_name}:")
        
        for name, func in implementations:
            try:
                l1 = create_linked_list(l1_vals)
                l2 = create_linked_list(l2_vals)
                
                start_time = time.time()
                result_head = func(l1, l2)
                end_time = time.time()
                
                result_length = len(linked_list_to_list(result_head))
                print(f"  {name}: {result_length} digits in {end_time - start_time:.4f} seconds")
            except Exception as e:
                print(f"  {name}: Error - {e}")


if __name__ == "__main__":
    test_add_two_numbers() 