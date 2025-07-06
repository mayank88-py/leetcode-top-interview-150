"""
70. Climbing Stairs

You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Example 1:
Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps

Example 2:
Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step

Constraints:
- 1 <= n <= 45
"""

def climb_stairs_dp_bottom_up(n):
    """
    Approach 1: Dynamic Programming (Bottom-up)
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Build up solutions from the base cases.
    """
    if n <= 2:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]


def climb_stairs_dp_optimized(n):
    """
    Approach 2: Space-Optimized DP
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Only keep track of the last two values since we only need them.
    """
    if n <= 2:
        return n
    
    prev2 = 1  # dp[i-2]
    prev1 = 2  # dp[i-1]
    
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1


def climb_stairs_memoization(n):
    """
    Approach 3: Memoization (Top-down DP)
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Use memoization to avoid recalculating subproblems.
    """
    memo = {}
    
    def climb_helper(steps):
        if steps <= 2:
            return steps
        
        if steps in memo:
            return memo[steps]
        
        memo[steps] = climb_helper(steps - 1) + climb_helper(steps - 2)
        return memo[steps]
    
    return climb_helper(n)


def climb_stairs_fibonacci(n):
    """
    Approach 4: Fibonacci Sequence
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Recognize that this is just the Fibonacci sequence shifted by 1.
    """
    if n <= 2:
        return n
    
    a, b = 1, 2
    for _ in range(2, n):
        a, b = b, a + b
    
    return b


def climb_stairs_matrix_exponentiation(n):
    """
    Approach 5: Matrix Exponentiation
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Use matrix exponentiation for faster computation.
    """
    def matrix_multiply(A, B):
        """Multiply two 2x2 matrices."""
        return [
            [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
            [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]
        ]
    
    def matrix_power(matrix, power):
        """Compute matrix^power using fast exponentiation."""
        if power == 1:
            return matrix
        
        result = [[1, 0], [0, 1]]  # Identity matrix
        base = matrix
        
        while power > 0:
            if power % 2 == 1:
                result = matrix_multiply(result, base)
            base = matrix_multiply(base, base)
            power //= 2
        
        return result
    
    if n <= 2:
        return n
    
    # Transformation matrix for Fibonacci
    # [F(n), F(n-1)] = [F(n-1), F(n-2)] * [[1, 1], [1, 0]]
    transformation = [[1, 1], [1, 0]]
    
    # Compute transformation^(n-2)
    result_matrix = matrix_power(transformation, n - 2)
    
    # [F(n), F(n-1)] = [2, 1] * result_matrix
    return 2 * result_matrix[0][0] + result_matrix[0][1]


def climb_stairs_recursive_naive(n):
    """
    Approach 6: Naive Recursion (Not efficient)
    Time Complexity: O(2^n)
    Space Complexity: O(n) - recursion depth
    
    Simple recursive solution without memoization.
    """
    if n > 35:  # Avoid TLE for large n
        return climb_stairs_dp_optimized(n)
    
    if n <= 2:
        return n
    
    return climb_stairs_recursive_naive(n - 1) + climb_stairs_recursive_naive(n - 2)


def climb_stairs_mathematical(n):
    """
    Approach 7: Mathematical Formula (Binet's Formula)
    Time Complexity: O(1)
    Space Complexity: O(1)
    
    Use Binet's formula for Fibonacci numbers (with precision considerations).
    """
    if n <= 2:
        return n
    
    import math
    
    # Binet's formula for Fibonacci numbers
    sqrt5 = math.sqrt(5)
    phi = (1 + sqrt5) / 2
    psi = (1 - sqrt5) / 2
    
    # F(n+1) = (phi^(n+1) - psi^(n+1)) / sqrt5
    # Since climbing stairs is F(n+1), we compute F(n+1)
    fib_n_plus_1 = (phi**(n + 1) - psi**(n + 1)) / sqrt5
    
    return round(fib_n_plus_1)


def climb_stairs_iterative_variants(n):
    """
    Approach 8: Iterative with Different Starting Points
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Different way to iterate through the sequence.
    """
    if n <= 2:
        return n
    
    ways = [0, 1, 2]  # ways[0] is dummy, ways[1] = 1, ways[2] = 2
    
    for i in range(3, n + 1):
        ways.append(ways[-1] + ways[-2])
        # Keep only last 3 elements to save space
        if len(ways) > 3:
            ways.pop(0)
    
    return ways[-1]


def climb_stairs_combinatorial(n):
    """
    Approach 9: Combinatorial Approach
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Think of it as choosing positions for 2-steps among total steps.
    """
    def combination(n, k):
        """Calculate C(n, k) = n! / (k! * (n-k)!)"""
        if k > n - k:
            k = n - k
        
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        
        return result
    
    if n <= 2:
        return n
    
    total_ways = 0
    
    # Try different numbers of 2-steps
    max_two_steps = n // 2
    
    for two_steps in range(max_two_steps + 1):
        one_steps = n - 2 * two_steps
        total_steps = one_steps + two_steps
        
        # Choose positions for two_steps among total_steps
        ways = combination(total_steps, two_steps)
        total_ways += ways
    
    return total_ways


def test_climb_stairs():
    """Test all approaches with various test cases."""
    
    test_cases = [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 5),
        (5, 8),
        (6, 13),
        (10, 89),
        (15, 987),
        (20, 10946),
        (25, 121393),
        (30, 1346269),
        (35, 14930352),
        (40, 165580141),
        (45, 1836311903),
    ]
    
    approaches = [
        ("DP Bottom-up", climb_stairs_dp_bottom_up),
        ("DP Optimized", climb_stairs_dp_optimized),
        ("Memoization", climb_stairs_memoization),
        ("Fibonacci", climb_stairs_fibonacci),
        ("Matrix Exponentiation", climb_stairs_matrix_exponentiation),
        ("Recursive Naive", climb_stairs_recursive_naive),
        ("Mathematical", climb_stairs_mathematical),
        ("Iterative Variants", climb_stairs_iterative_variants),
        ("Combinatorial", climb_stairs_combinatorial),
    ]
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: n = {n}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            result = func(n)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_climb_stairs() 