"""
150. Evaluate Reverse Polish Notation

Problem:
You are given an array of strings tokens that represents an arithmetic expression in Reverse Polish Notation.
Evaluate the expression. Return an integer that represents the value of the expression.

Note that:
- The valid operators are '+', '-', '*', and '/'.
- Each operand may be an integer or another expression.
- The division between two integers always truncates toward zero.
- There will not be any division by zero.
- The input represents a valid arithmetic expression in a reverse polish notation.
- The answer and all the intermediate calculations can be represented in a 32-bit integer.

Example 1:
Input: tokens = ["2","1","+","3","*"]
Output: 9
Explanation: ((2 + 1) * 3) = 9

Example 2:
Input: tokens = ["4","13","5","/","+"]
Output: 6
Explanation: (4 + (13 / 5)) = 6

Example 3:
Input: tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
Output: 22
Explanation: ((10 * (6 / ((9 + 3) * -11))) + 17) + 5 = 22

Time Complexity: O(n)
Space Complexity: O(n)
"""


def eval_rpn(tokens):
    """
    Evaluate Reverse Polish Notation using stack.
    
    Args:
        tokens: List of strings representing RPN expression
    
    Returns:
        Integer result of the expression
    """
    if not tokens:
        return 0
    
    stack = []
    operators = {'+', '-', '*', '/'}
    
    for token in tokens:
        if token in operators:
            # Pop two operands (note the order)
            b = stack.pop()
            a = stack.pop()
            
            # Perform operation
            if token == '+':
                result = a + b
            elif token == '-':
                result = a - b
            elif token == '*':
                result = a * b
            elif token == '/':
                # Truncate toward zero
                result = int(a / b)
            
            stack.append(result)
        else:
            # It's a number
            stack.append(int(token))
    
    return stack[0]


def eval_rpn_lambda(tokens):
    """
    Evaluate RPN using lambda functions for operations.
    
    Args:
        tokens: List of strings representing RPN expression
    
    Returns:
        Integer result of the expression
    """
    if not tokens:
        return 0
    
    stack = []
    operations = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: int(a / b)
    }
    
    for token in tokens:
        if token in operations:
            b = stack.pop()
            a = stack.pop()
            stack.append(operations[token](a, b))
        else:
            stack.append(int(token))
    
    return stack[0]


def eval_rpn_recursive(tokens):
    """
    Evaluate RPN using recursion.
    
    Args:
        tokens: List of strings representing RPN expression
    
    Returns:
        Integer result of the expression
    """
    def helper(index):
        """
        Recursively evaluate RPN starting from given index.
        Returns (result, next_index)
        """
        token = tokens[index]
        
        if token in {'+', '-', '*', '/'}:
            # Get right operand first (it's evaluated last)
            right_result, next_idx = helper(index - 1)
            # Get left operand
            left_result, next_idx = helper(next_idx - 1)
            
            if token == '+':
                return left_result + right_result, next_idx
            elif token == '-':
                return left_result - right_result, next_idx
            elif token == '*':
                return left_result * right_result, next_idx
            elif token == '/':
                return int(left_result / right_result), next_idx
        else:
            # It's a number
            return int(token), index
    
    result, _ = helper(len(tokens) - 1)
    return result


def eval_rpn_no_stack(tokens):
    """
    Evaluate RPN without using explicit stack (using recursion instead).
    
    Args:
        tokens: List of strings representing RPN expression
    
    Returns:
        Integer result of the expression
    """
    def evaluate():
        token = tokens.pop()
        
        if token in {'+', '-', '*', '/'}:
            # Pop two operands (note the order is reversed)
            right = evaluate()
            left = evaluate()
            
            if token == '+':
                return left + right
            elif token == '-':
                return left - right
            elif token == '*':
                return left * right
            elif token == '/':
                return int(left / right)
        else:
            return int(token)
    
    # Make a copy to avoid modifying the original
    tokens_copy = tokens[:]
    tokens_copy.reverse()
    return evaluate()


# Test cases
if __name__ == "__main__":
    # Test case 1
    tokens1 = ["2", "1", "+", "3", "*"]
    result1a = eval_rpn(tokens1)
    result1b = eval_rpn_lambda(tokens1)
    result1c = eval_rpn_recursive(tokens1)
    print(f"Test 1 - Expected: 9, Stack: {result1a}, Lambda: {result1b}, Recursive: {result1c}")
    
    # Test case 2
    tokens2 = ["4", "13", "5", "/", "+"]
    result2a = eval_rpn(tokens2)
    result2b = eval_rpn_lambda(tokens2)
    result2c = eval_rpn_recursive(tokens2)
    print(f"Test 2 - Expected: 6, Stack: {result2a}, Lambda: {result2b}, Recursive: {result2c}")
    
    # Test case 3
    tokens3 = ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
    result3a = eval_rpn(tokens3)
    result3b = eval_rpn_lambda(tokens3)
    result3c = eval_rpn_recursive(tokens3)
    print(f"Test 3 - Expected: 22, Stack: {result3a}, Lambda: {result3b}, Recursive: {result3c}")
    
    # Test case 4 - Single number
    tokens4 = ["42"]
    result4a = eval_rpn(tokens4)
    result4b = eval_rpn_lambda(tokens4)
    result4c = eval_rpn_recursive(tokens4)
    print(f"Test 4 - Expected: 42, Stack: {result4a}, Lambda: {result4b}, Recursive: {result4c}")
    
    # Test case 5 - Simple addition
    tokens5 = ["1", "2", "+"]
    result5a = eval_rpn(tokens5)
    result5b = eval_rpn_lambda(tokens5)
    result5c = eval_rpn_recursive(tokens5)
    print(f"Test 5 - Expected: 3, Stack: {result5a}, Lambda: {result5b}, Recursive: {result5c}")
    
    # Test case 6 - Division with truncation
    tokens6 = ["7", "3", "/"]
    result6a = eval_rpn(tokens6)
    result6b = eval_rpn_lambda(tokens6)
    result6c = eval_rpn_recursive(tokens6)
    print(f"Test 6 - Expected: 2, Stack: {result6a}, Lambda: {result6b}, Recursive: {result6c}")
    
    # Test case 7 - Negative division
    tokens7 = ["7", "-3", "/"]
    result7a = eval_rpn(tokens7)
    result7b = eval_rpn_lambda(tokens7)
    result7c = eval_rpn_recursive(tokens7)
    print(f"Test 7 - Expected: -2, Stack: {result7a}, Lambda: {result7b}, Recursive: {result7c}")
    
    # Test case 8 - Complex expression with negative numbers
    tokens8 = ["3", "-4", "+", "2", "*", "7", "/"]
    result8a = eval_rpn(tokens8)
    result8b = eval_rpn_lambda(tokens8)
    result8c = eval_rpn_recursive(tokens8)
    print(f"Test 8 - Expected: 0, Stack: {result8a}, Lambda: {result8b}, Recursive: {result8c}")
    
    # Test case 9 - Subtraction
    tokens9 = ["15", "7", "1", "1", "+", "-", "/", "3", "*", "2", "1", "1", "+", "+", "-"]
    result9a = eval_rpn(tokens9)
    result9b = eval_rpn_lambda(tokens9)
    result9c = eval_rpn_recursive(tokens9)
    print(f"Test 9 - Expected: 5, Stack: {result9a}, Lambda: {result9b}, Recursive: {result9c}")
    
    # Test case 10 - Large numbers
    tokens10 = ["1000", "2000", "+", "3", "*"]
    result10a = eval_rpn(tokens10)
    result10b = eval_rpn_lambda(tokens10)
    result10c = eval_rpn_recursive(tokens10)
    print(f"Test 10 - Expected: 9000, Stack: {result10a}, Lambda: {result10b}, Recursive: {result10c}") 