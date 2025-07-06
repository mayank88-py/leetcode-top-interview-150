"""
22. Generate Parentheses

Problem:
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

Example 1:
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]

Example 2:
Input: n = 1
Output: ["()"]

Time Complexity: O(4^n / sqrt(n)) - Catalan number
Space Complexity: O(4^n / sqrt(n)) for storing all valid combinations
"""


def generate_parentheses_backtrack(n):
    """
    Backtracking approach - optimal solution.
    
    Time Complexity: O(4^n / sqrt(n)) - nth Catalan number
    Space Complexity: O(4^n / sqrt(n)) for result + O(n) for recursion stack
    
    Algorithm:
    1. Use backtracking to build valid parentheses combinations
    2. Track count of open and close parentheses used
    3. Only add '(' if we haven't used all n open parentheses
    4. Only add ')' if it won't make string invalid (close <= open)
    """
    result = []
    
    def backtrack(current, open_count, close_count):
        # Base case: we've used all n pairs
        if len(current) == 2 * n:
            result.append(current)
            return
        
        # Add '(' if we haven't used all n open parentheses
        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)
        
        # Add ')' if it won't make the string invalid
        if close_count < open_count:
            backtrack(current + ')', open_count, close_count + 1)
    
    backtrack("", 0, 0)
    return result


def generate_parentheses_iterative(n):
    """
    Iterative BFS approach using queue.
    
    Time Complexity: O(4^n / sqrt(n))
    Space Complexity: O(4^n / sqrt(n))
    
    Algorithm:
    1. Use queue to store (current_string, open_count, close_count)
    2. Process each state and generate next valid states
    3. Continue until all strings have length 2*n
    """
    if n == 0:
        return [""]
    
    from collections import deque
    
    # Queue stores (current_string, open_count, close_count)
    queue = deque([("", 0, 0)])
    result = []
    
    while queue:
        current, open_count, close_count = queue.popleft()
        
        # If we've built a complete string
        if len(current) == 2 * n:
            result.append(current)
            continue
        
        # Add '(' if possible
        if open_count < n:
            queue.append((current + '(', open_count + 1, close_count))
        
        # Add ')' if possible
        if close_count < open_count:
            queue.append((current + ')', open_count, close_count + 1))
    
    return result


def generate_parentheses_dp(n):
    """
    Dynamic programming approach.
    
    Time Complexity: O(4^n / sqrt(n))
    Space Complexity: O(4^n / sqrt(n))
    
    Algorithm:
    1. Build solutions for i pairs using solutions for j and i-1-j pairs
    2. For each i, try all possible splits: "(" + dp[j] + ")" + dp[i-1-j]
    3. Use memoization to avoid recomputing
    """
    if n == 0:
        return [""]
    
    # dp[i] will store all valid parentheses for i pairs
    dp = [[] for _ in range(n + 1)]
    dp[0] = [""]
    
    for i in range(1, n + 1):
        for j in range(i):
            # Split into j pairs inside parentheses and (i-1-j) pairs after
            for left in dp[j]:
                for right in dp[i - 1 - j]:
                    dp[i].append("(" + left + ")" + right)
    
    return dp[n]


def generate_parentheses_recursive(n):
    """
    Pure recursive approach with memoization.
    
    Time Complexity: O(4^n / sqrt(n))
    Space Complexity: O(4^n / sqrt(n))
    
    Algorithm:
    1. Recursively generate combinations by trying all valid splits
    2. Use memoization to cache results for each n
    3. Base case: n=0 returns [""], n=1 returns ["()"]
    """
    memo = {}
    
    def helper(num_pairs):
        if num_pairs in memo:
            return memo[num_pairs]
        
        if num_pairs == 0:
            memo[num_pairs] = [""]
            return [""]
        
        if num_pairs == 1:
            memo[num_pairs] = ["()"]
            return ["()"]
        
        result = []
        for i in range(num_pairs):
            # i pairs inside, (num_pairs-1-i) pairs after
            left_combinations = helper(i)
            right_combinations = helper(num_pairs - 1 - i)
            
            for left in left_combinations:
                for right in right_combinations:
                    result.append("(" + left + ")" + right)
        
        memo[num_pairs] = result
        return result
    
    return helper(n)


def generate_parentheses_closure(n):
    """
    Closure number approach (mathematical).
    
    Time Complexity: O(4^n / sqrt(n))
    Space Complexity: O(4^n / sqrt(n))
    
    Algorithm:
    1. Every valid parentheses can be written as (A)B
    2. Where A and B are valid parentheses (possibly empty)
    3. Enumerate all possible A and B combinations
    """
    if n == 0:
        return [""]
    
    result = []
    
    # Try all possible closure numbers
    for c in range(n):
        # c pairs inside first parentheses, n-1-c pairs outside
        inside = generate_parentheses_closure(c)
        outside = generate_parentheses_closure(n - 1 - c)
        
        for inside_combo in inside:
            for outside_combo in outside:
                result.append("(" + inside_combo + ")" + outside_combo)
    
    return result


def is_valid_parentheses(s):
    """Helper function to validate parentheses string."""
    count = 0
    for char in s:
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1
            if count < 0:
                return False
    return count == 0


def test_generate_parentheses():
    """Test all implementations with various test cases."""
    
    test_cases = [
        (1, ["()"]),
        (2, ["(())","()()"]),
        (3, ["((()))","(()())","(())()","()(())","()()()"]),
        (0, [""]),
        (4, None)  # We'll just check count for n=4
    ]
    
    implementations = [
        ("Backtracking", generate_parentheses_backtrack),
        ("Iterative", generate_parentheses_iterative),
        ("Dynamic Programming", generate_parentheses_dp),
        ("Recursive + Memo", generate_parentheses_recursive),
        ("Closure Number", generate_parentheses_closure)
    ]
    
    print("Testing Generate Parentheses...")
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: n = {n}")
        
        if expected:
            print(f"Expected: {expected}")
        
        for impl_name, impl_func in implementations:
            result = impl_func(n)
            
            if expected:
                # Sort both for comparison since order may vary
                result_sorted = sorted(result)
                expected_sorted = sorted(expected)
                is_correct = result_sorted == expected_sorted
                
                # Also validate that all results are valid parentheses
                all_valid = all(is_valid_parentheses(s) for s in result)
                
                status = "✓" if is_correct and all_valid else "✗"
                print(f"{impl_name:20} | Count: {len(result):2} | {status}")
                
                if not is_correct:
                    print(f"                       Got: {sorted(result)}")
            else:
                # For larger n, just check count and validity
                all_valid = all(is_valid_parentheses(s) for s in result)
                print(f"{impl_name:20} | Count: {len(result):2} | Valid: {'✓' if all_valid else '✗'}")


if __name__ == "__main__":
    test_generate_parentheses() 