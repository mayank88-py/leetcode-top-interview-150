"""
72. Edit Distance

Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.

You have the following three operations permitted on a word:
- Insert a character
- Delete a character
- Replace a character

Example 1:
Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')

Example 2:
Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation: 
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')

Constraints:
- 0 <= word1.length, word2.length <= 500
- word1 and word2 consist of lowercase English letters.
"""

def min_distance_2d_dp(word1, word2):
    """
    Approach 1: 2D Dynamic Programming
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    dp[i][j] = minimum operations to convert word1[0:i] to word2[0:j]
    """
    m, n = len(word1), len(word2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters from word1
    
    for j in range(n + 1):
        dp[0][j] = j  # Insert all characters to get word2
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # No operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Delete from word1
                    dp[i][j-1],    # Insert to word1
                    dp[i-1][j-1]   # Replace in word1
                )
    
    return dp[m][n]


def min_distance_space_optimized(word1, word2):
    """
    Approach 2: Space Optimized DP
    Time Complexity: O(m * n)
    Space Complexity: O(min(m, n))
    
    Use only two rows for DP computation.
    """
    m, n = len(word1), len(word2)
    
    # Choose shorter string for space optimization
    if m < n:
        word1, word2 = word2, word1
        m, n = n, m
    
    # Use only one array
    prev = list(range(n + 1))  # Previous row
    
    for i in range(1, m + 1):
        curr = [i]  # First element is always i
        
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                curr.append(prev[j-1])
            else:
                curr.append(1 + min(prev[j], curr[j-1], prev[j-1]))
        
        prev = curr
    
    return prev[n]


def min_distance_memoization(word1, word2):
    """
    Approach 3: Top-down DP with Memoization
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Use memoization to avoid recalculating subproblems.
    """
    memo = {}
    
    def min_ops(i, j):
        if i == 0:
            return j  # Insert all characters from word2
        if j == 0:
            return i  # Delete all characters from word1
        
        if (i, j) in memo:
            return memo[(i, j)]
        
        if word1[i-1] == word2[j-1]:
            result = min_ops(i-1, j-1)
        else:
            result = 1 + min(
                min_ops(i-1, j),    # Delete
                min_ops(i, j-1),    # Insert
                min_ops(i-1, j-1)   # Replace
            )
        
        memo[(i, j)] = result
        return result
    
    return min_ops(len(word1), len(word2))


def min_distance_iterative_optimized(word1, word2):
    """
    Approach 4: Iterative with Single Array
    Time Complexity: O(m * n)
    Space Complexity: O(n)
    
    Use single array with careful updating.
    """
    m, n = len(word1), len(word2)
    
    # Use single array
    dp = list(range(n + 1))
    
    for i in range(1, m + 1):
        prev_diag = dp[0]
        dp[0] = i
        
        for j in range(1, n + 1):
            temp = dp[j]
            
            if word1[i-1] == word2[j-1]:
                dp[j] = prev_diag
            else:
                dp[j] = 1 + min(dp[j], dp[j-1], prev_diag)
            
            prev_diag = temp
    
    return dp[n]


def min_distance_recursive_naive(word1, word2):
    """
    Approach 5: Naive Recursion
    Time Complexity: O(3^(m+n))
    Space Complexity: O(m + n)
    
    Simple recursive solution without memoization.
    """
    def min_ops(i, j):
        if i == 0:
            return j
        if j == 0:
            return i
        
        if word1[i-1] == word2[j-1]:
            return min_ops(i-1, j-1)
        else:
            return 1 + min(
                min_ops(i-1, j),    # Delete
                min_ops(i, j-1),    # Insert
                min_ops(i-1, j-1)   # Replace
            )
    
    # Avoid TLE for large strings
    if len(word1) * len(word2) > 100:
        return min_distance_space_optimized(word1, word2)
    
    return min_ops(len(word1), len(word2))


def min_distance_with_operations(word1, word2):
    """
    Approach 6: DP with Operation Tracking
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Track the actual operations performed.
    """
    m, n = len(word1), len(word2)
    
    # DP table with operation tracking
    dp = [[(0, "")] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = (i, f"Delete {i} chars")
    
    for j in range(n + 1):
        dp[0][j] = (j, f"Insert {j} chars")
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                delete_cost = dp[i-1][j][0] + 1
                insert_cost = dp[i][j-1][0] + 1
                replace_cost = dp[i-1][j-1][0] + 1
                
                min_cost = min(delete_cost, insert_cost, replace_cost)
                
                if min_cost == delete_cost:
                    dp[i][j] = (min_cost, f"Delete '{word1[i-1]}'")
                elif min_cost == insert_cost:
                    dp[i][j] = (min_cost, f"Insert '{word2[j-1]}'")
                else:
                    dp[i][j] = (min_cost, f"Replace '{word1[i-1]}' with '{word2[j-1]}'")
    
    return dp[m][n][0]


def min_distance_bfs(word1, word2):
    """
    Approach 7: BFS Approach
    Time Complexity: O(3^(m+n)) worst case
    Space Complexity: O(3^(m+n))
    
    Use BFS to find minimum operations.
    """
    from collections import deque
    
    if word1 == word2:
        return 0
    
    # Avoid TLE for large strings
    if len(word1) * len(word2) > 50:
        return min_distance_space_optimized(word1, word2)
    
    queue = deque([(word1, 0)])
    visited = {word1}
    
    while queue:
        current_word, operations = queue.popleft()
        
        if current_word == word2:
            return operations
        
        # Generate all possible next states
        for i in range(len(current_word) + 1):
            # Insert operation
            for c in set(word2):
                new_word = current_word[:i] + c + current_word[i:]
                if new_word not in visited and len(new_word) <= len(word2) + 1:
                    visited.add(new_word)
                    queue.append((new_word, operations + 1))
        
        for i in range(len(current_word)):
            # Delete operation
            new_word = current_word[:i] + current_word[i+1:]
            if new_word not in visited:
                visited.add(new_word)
                queue.append((new_word, operations + 1))
            
            # Replace operation
            for c in set(word2):
                new_word = current_word[:i] + c + current_word[i+1:]
                if new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, operations + 1))
    
    return -1


def min_distance_optimized_early_termination(word1, word2):
    """
    Approach 8: DP with Early Termination
    Time Complexity: O(m * n)
    Space Complexity: O(min(m, n))
    
    Optimize with early termination conditions.
    """
    # Early termination conditions
    if word1 == word2:
        return 0
    if not word1:
        return len(word2)
    if not word2:
        return len(word1)
    
    m, n = len(word1), len(word2)
    
    # If one string is much longer, early estimate
    diff = abs(m - n)
    if diff > min(m, n):
        return diff + min(m, n)
    
    # Use space-optimized DP
    if m < n:
        word1, word2 = word2, word1
        m, n = n, m
    
    prev = list(range(n + 1))
    
    for i in range(1, m + 1):
        curr = [i]
        
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                curr.append(prev[j-1])
            else:
                curr.append(1 + min(prev[j], curr[j-1], prev[j-1]))
        
        prev = curr
    
    return prev[n]


def test_min_distance():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ("horse", "ros", 3),
        ("intention", "execution", 5),
        ("", "", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("abc", "abc", 0),
        ("a", "b", 1),
        ("cat", "cut", 1),
        ("sunday", "saturday", 3),
        ("kitten", "sitting", 3),
        ("flaw", "lawn", 2),
        ("gumbo", "gambol", 2),
        ("ab", "aa", 1),
        ("teacher", "teach", 2),
        ("leetcode", "etco", 4),
    ]
    
    approaches = [
        ("2D DP", min_distance_2d_dp),
        ("Space Optimized", min_distance_space_optimized),
        ("Memoization", min_distance_memoization),
        ("Iterative Optimized", min_distance_iterative_optimized),
        ("Recursive Naive", min_distance_recursive_naive),
        ("With Operations", min_distance_with_operations),
        ("BFS", min_distance_bfs),
        ("Early Termination", min_distance_optimized_early_termination),
    ]
    
    for i, (word1, word2, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: word1 = \"{word1}\", word2 = \"{word2}\"")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            result = func(word1, word2)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_min_distance() 