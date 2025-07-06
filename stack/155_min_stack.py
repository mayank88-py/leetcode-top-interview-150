"""
155. Min Stack

Problem:
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:
- MinStack() initializes the stack object.
- void push(int val) pushes the element val onto the stack.
- void pop() removes the element on the top of the stack.
- int top() gets the top element of the stack.
- int getMin() retrieves the minimum element in the stack.

You must implement a solution with O(1) time complexity for each function.

Example 1:
Input:
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

Output:
[null,null,null,null,-3,null,0,-2]

Explanation:
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2

Time Complexity: O(1) for all operations
Space Complexity: O(n)
"""


class MinStack:
    """
    Stack implementation with constant time minimum retrieval.
    Uses auxiliary stack to track minimum values.
    """
    
    def __init__(self):
        """Initialize the stack."""
        self.stack = []
        self.min_stack = []
    
    def push(self, val):
        """
        Push element onto stack.
        
        Args:
            val: Value to push
        """
        self.stack.append(val)
        
        # Update minimum stack
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        """Remove and return top element."""
        if not self.stack:
            return None
        
        val = self.stack.pop()
        
        # Update minimum stack
        if self.min_stack and val == self.min_stack[-1]:
            self.min_stack.pop()
        
        return val
    
    def top(self):
        """
        Get top element without removing it.
        
        Returns:
            Top element of stack
        """
        if not self.stack:
            return None
        return self.stack[-1]
    
    def get_min(self):
        """
        Get minimum element in stack.
        
        Returns:
            Minimum element
        """
        if not self.min_stack:
            return None
        return self.min_stack[-1]


class MinStackSingleStack:
    """
    Alternative implementation using single stack with tuples.
    Each element stores (value, current_minimum).
    """
    
    def __init__(self):
        """Initialize the stack."""
        self.stack = []
    
    def push(self, val):
        """
        Push element onto stack.
        
        Args:
            val: Value to push
        """
        if not self.stack:
            self.stack.append((val, val))
        else:
            current_min = min(val, self.stack[-1][1])
            self.stack.append((val, current_min))
    
    def pop(self):
        """Remove and return top element."""
        if not self.stack:
            return None
        return self.stack.pop()[0]
    
    def top(self):
        """
        Get top element without removing it.
        
        Returns:
            Top element of stack
        """
        if not self.stack:
            return None
        return self.stack[-1][0]
    
    def get_min(self):
        """
        Get minimum element in stack.
        
        Returns:
            Minimum element
        """
        if not self.stack:
            return None
        return self.stack[-1][1]


class MinStackOptimized:
    """
    Space-optimized implementation that only stores differences.
    """
    
    def __init__(self):
        """Initialize the stack."""
        self.stack = []
        self.min_val = None
    
    def push(self, val):
        """
        Push element onto stack.
        
        Args:
            val: Value to push
        """
        if not self.stack:
            self.stack.append(0)
            self.min_val = val
        else:
            # Store difference from current minimum
            diff = val - self.min_val
            self.stack.append(diff)
            
            # Update minimum if necessary
            if val < self.min_val:
                self.min_val = val
    
    def pop(self):
        """Remove and return top element."""
        if not self.stack:
            return None
        
        diff = self.stack.pop()
        
        if diff < 0:
            # The popped element was the minimum
            val = self.min_val
            self.min_val = val - diff
            return val
        else:
            return self.min_val + diff
    
    def top(self):
        """
        Get top element without removing it.
        
        Returns:
            Top element of stack
        """
        if not self.stack:
            return None
        
        diff = self.stack[-1]
        if diff < 0:
            return self.min_val
        else:
            return self.min_val + diff
    
    def get_min(self):
        """
        Get minimum element in stack.
        
        Returns:
            Minimum element
        """
        return self.min_val


# Test cases
if __name__ == "__main__":
    # Test MinStack
    print("Testing MinStack:")
    min_stack = MinStack()
    
    min_stack.push(-2)
    min_stack.push(0)
    min_stack.push(-3)
    print(f"getMin(): {min_stack.get_min()}")  # Expected: -3
    
    min_stack.pop()
    print(f"top(): {min_stack.top()}")  # Expected: 0
    print(f"getMin(): {min_stack.get_min()}")  # Expected: -2
    
    # Test MinStackSingleStack
    print("\nTesting MinStackSingleStack:")
    min_stack2 = MinStackSingleStack()
    
    min_stack2.push(-2)
    min_stack2.push(0)
    min_stack2.push(-3)
    print(f"getMin(): {min_stack2.get_min()}")  # Expected: -3
    
    min_stack2.pop()
    print(f"top(): {min_stack2.top()}")  # Expected: 0
    print(f"getMin(): {min_stack2.get_min()}")  # Expected: -2
    
    # Test MinStackOptimized
    print("\nTesting MinStackOptimized:")
    min_stack3 = MinStackOptimized()
    
    min_stack3.push(-2)
    min_stack3.push(0)
    min_stack3.push(-3)
    print(f"getMin(): {min_stack3.get_min()}")  # Expected: -3
    
    min_stack3.pop()
    print(f"top(): {min_stack3.top()}")  # Expected: 0
    print(f"getMin(): {min_stack3.get_min()}")  # Expected: -2
    
    # Additional test cases
    print("\nAdditional test cases:")
    stack = MinStack()
    
    stack.push(1)
    stack.push(2)
    print(f"Min: {stack.get_min()}")  # Expected: 1
    
    stack.push(0)
    print(f"Min: {stack.get_min()}")  # Expected: 0
    
    stack.pop()
    print(f"Min: {stack.get_min()}")  # Expected: 1
    
    stack.push(-1)
    print(f"Min: {stack.get_min()}")  # Expected: -1
    print(f"Top: {stack.top()}")  # Expected: -1 