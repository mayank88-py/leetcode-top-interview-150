"""
97. Interleaving String

Given strings s1, s2, and s3, find whether s3 is formed by an interleaving of s1 and s2.

An interleaving of two strings s and t is a configuration where s and t are divided into n and m substrings respectively, such that:
- s = s1 + s2 + ... + sn
- t = t1 + t2 + ... + tm
- |n - m| <= 1
- The interleaving is s1 + t1 + s2 + t2 + ... or t1 + s1 + t2 + s2 + ...

Note: a + b is the concatenation of strings a and b.

Example 1:
Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
Output: true
Explanation: One way to obtain s3 is:
Split s1 into s1 = "aa" + "bc" + "c", and s2 into s2 = "dbbc" + "a".
Interleaving the two splits, we get "aa" + "dbbc" + "bc" + "a" + "c" = "aadbbcbcac".
Since s3 can be obtained by interleaving s1 and s2, we return true.

Example 2:
Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"
Output: false
Explanation: Notice how it is impossible to interleave s2 with any other string to obtain s3.

Example 3:
Input: s1 = "", s2 = "b", s3 = "b"
Output: true

Constraints:
- 0 <= s1.length, s2.length <= 100
- 1 <= s3.length <= 200
- s1, s2, and s3 consist of lowercase English letters.
"""

def is_interleave_2d_dp(s1, s2, s3):
    """
    Approach 1: 2D Dynamic Programming
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    dp[i][j] = True if s3[0:i+j] can be formed by interleaving s1[0:i] and s2[0:j]
    """
    m, n, k = len(s1), len(s2), len(s3)
    
    if m + n != k:
        return False
    
    # dp[i][j] represents if s3[0:i+j] can be formed by s1[0:i] and s2[0:j]
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    
    # Base case: empty strings
    dp[0][0] = True
    
    # Initialize first row (only using s2)
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
    
    # Initialize first column (only using s1)
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Can we form s3[0:i+j] by taking character from s1 or s2?
            dp[i][j] = ((dp[i-1][j] and s1[i-1] == s3[i+j-1]) or
                       (dp[i][j-1] and s2[j-1] == s3[i+j-1]))
    
    return dp[m][n]


def is_interleave_space_optimized(s1, s2, s3):
    """
    Approach 2: Space Optimized DP
    Time Complexity: O(m * n)
    Space Complexity: O(min(m, n))
    
    Use only one row for DP computation.
    """
    m, n, k = len(s1), len(s2), len(s3)
    
    if m + n != k:
        return False
    
    # Make s1 the shorter string for space optimization
    if m > n:
        s1, s2 = s2, s1
        m, n = n, m
    
    # Use only one array
    dp = [False] * (m + 1)
    dp[0] = True
    
    # Initialize first row
    for i in range(1, m + 1):
        dp[i] = dp[i-1] and s1[i-1] == s3[i-1]
    
    # Process each row
    for j in range(1, n + 1):
        # Update dp[0] (first column)
        dp[0] = dp[0] and s2[j-1] == s3[j-1]
        
        for i in range(1, m + 1):
            dp[i] = ((dp[i] and s2[j-1] == s3[i+j-1]) or
                    (dp[i-1] and s1[i-1] == s3[i+j-1]))
    
    return dp[m]


def is_interleave_memoization(s1, s2, s3):
    """
    Approach 3: Top-down DP with Memoization
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Use memoization to avoid recalculating subproblems.
    """
    m, n, k = len(s1), len(s2), len(s3)
    
    if m + n != k:
        return False
    
    memo = {}
    
    def helper(i, j, idx):
        if idx == k:
            return i == m and j == n
        
        if (i, j, idx) in memo:
            return memo[(i, j, idx)]
        
        result = False
        
        # Try taking character from s1
        if i < m and s1[i] == s3[idx]:
            result = helper(i + 1, j, idx + 1)
        
        # Try taking character from s2
        if not result and j < n and s2[j] == s3[idx]:
            result = helper(i, j + 1, idx + 1)
        
        memo[(i, j, idx)] = result
        return result
    
    return helper(0, 0, 0)


def is_interleave_bfs(s1, s2, s3):
    """
    Approach 4: BFS Approach
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Use BFS to explore all possible interleavings.
    """
    m, n, k = len(s1), len(s2), len(s3)
    
    if m + n != k:
        return False
    
    from collections import deque
    
    queue = deque([(0, 0, 0)])  # (i, j, idx)
    visited = set()
    
    while queue:
        i, j, idx = queue.popleft()
        
        if idx == k:
            return True
        
        if (i, j, idx) in visited:
            continue
        
        visited.add((i, j, idx))
        
        # Try taking character from s1
        if i < m and s1[i] == s3[idx]:
            queue.append((i + 1, j, idx + 1))
        
        # Try taking character from s2
        if j < n and s2[j] == s3[idx]:
            queue.append((i, j + 1, idx + 1))
    
    return False


def is_interleave_rolling_hash(s1, s2, s3):
    """
    Approach 5: Rolling Hash (Alternative approach)
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Use rolling hash for pattern matching.
    """
    m, n, k = len(s1), len(s2), len(s3)
    
    if m + n != k:
        return False
    
    # Simple approach: use DP but with hash optimization for large strings
    dp = {}
    
    def helper(i, j):
        if i == m and j == n:
            return True
        if i + j >= k:
            return False
        
        if (i, j) in dp:
            return dp[(i, j)]
        
        result = False
        
        # Try taking from s1
        if i < m and s1[i] == s3[i + j]:
            result = helper(i + 1, j)
        
        # Try taking from s2
        if not result and j < n and s2[j] == s3[i + j]:
            result = helper(i, j + 1)
        
        dp[(i, j)] = result
        return result
    
    return helper(0, 0)


def is_interleave_iterative_optimized(s1, s2, s3):
    """
    Approach 6: Iterative with Early Termination
    Time Complexity: O(m * n)
    Space Complexity: O(n)
    
    Optimized iterative approach with early termination.
    """
    m, n, k = len(s1), len(s2), len(s3)
    
    if m + n != k:
        return False
    
    # Early termination: check character frequency
    from collections import Counter
    if Counter(s1 + s2) != Counter(s3):
        return False
    
    # DP with space optimization
    dp = [False] * (n + 1)
    dp[0] = True
    
    # Initialize first row
    for j in range(1, n + 1):
        dp[j] = dp[j-1] and s2[j-1] == s3[j-1]
    
    # Process each row
    for i in range(1, m + 1):
        dp[0] = dp[0] and s1[i-1] == s3[i-1]
        
        for j in range(1, n + 1):
            dp[j] = ((dp[j] and s1[i-1] == s3[i+j-1]) or
                    (dp[j-1] and s2[j-1] == s3[i+j-1]))
    
    return dp[n]


def is_interleave_dfs(s1, s2, s3):
    """
    Approach 7: DFS with Pruning
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Use DFS with pruning for exploration.
    """
    m, n, k = len(s1), len(s2), len(s3)
    
    if m + n != k:
        return False
    
    visited = set()
    
    def dfs(i, j, idx):
        if idx == k:
            return i == m and j == n
        
        if (i, j) in visited:
            return False
        
        # Pruning: check if remaining characters can match
        remaining = k - idx
        if (m - i) + (n - j) != remaining:
            return False
        
        visited.add((i, j))
        
        result = False
        
        # Try s1
        if i < m and s1[i] == s3[idx]:
            result = dfs(i + 1, j, idx + 1)
        
        # Try s2
        if not result and j < n and s2[j] == s3[idx]:
            result = dfs(i, j + 1, idx + 1)
        
        return result
    
    return dfs(0, 0, 0)


def is_interleave_bottom_up(s1, s2, s3):
    """
    Approach 8: Bottom-up DP
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Build solution from the end backwards.
    """
    m, n, k = len(s1), len(s2), len(s3)
    
    if m + n != k:
        return False
    
    # DP table
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    
    # Base case: both strings are exhausted
    dp[m][n] = True
    
    # Fill from bottom-right to top-left
    for i in range(m, -1, -1):
        for j in range(n, -1, -1):
            if i == m and j == n:
                continue
            
            idx = i + j
            if idx >= k:
                continue
            
            # Can extend from s1?
            if i < m and s1[i] == s3[idx] and dp[i+1][j]:
                dp[i][j] = True
            
            # Can extend from s2?
            if j < n and s2[j] == s3[idx] and dp[i][j+1]:
                dp[i][j] = True
    
    return dp[0][0]


def is_interleave_trie_based(s1, s2, s3):
    """
    Approach 9: Trie-based Approach
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    
    Use trie structure to track valid paths.
    """
    m, n, k = len(s1), len(s2), len(s3)
    
    if m + n != k:
        return False
    
    # Use DP with path tracking
    paths = {(0, 0): True}
    
    for idx in range(k):
        new_paths = {}
        
        for (i, j), valid in paths.items():
            if not valid:
                continue
            
            # Try extending with s1
            if i < m and s1[i] == s3[idx]:
                new_paths[(i+1, j)] = True
            
            # Try extending with s2
            if j < n and s2[j] == s3[idx]:
                new_paths[(i, j+1)] = True
        
        paths = new_paths
        
        if not paths:  # No valid paths remaining
            return False
    
    return (m, n) in paths


def test_is_interleave():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ("aabcc", "dbbca", "aadbbcbcac", True),
        ("aabcc", "dbbca", "aadbbbaccc", False),
        ("", "b", "b", True),
        ("", "", "", True),
        ("a", "", "a", True),
        ("", "a", "a", True),
        ("a", "b", "ab", True),
        ("a", "b", "ba", True),
        ("a", "b", "abc", False),
        ("abc", "def", "adbecf", True),
        ("abc", "def", "adbefc", False),
        ("ab", "bc", "abc", False),
        ("db", "b", "cbb", False),
    ]
    
    approaches = [
        ("2D DP", is_interleave_2d_dp),
        ("Space Optimized", is_interleave_space_optimized),
        ("Memoization", is_interleave_memoization),
        ("BFS", is_interleave_bfs),
        ("Rolling Hash", is_interleave_rolling_hash),
        ("Iterative Optimized", is_interleave_iterative_optimized),
        ("DFS", is_interleave_dfs),
        ("Bottom-up", is_interleave_bottom_up),
        ("Trie-based", is_interleave_trie_based),
    ]
    
    for i, (s1, s2, s3, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: s1=\"{s1}\", s2=\"{s2}\", s3=\"{s3}\"")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            result = func(s1, s2, s3)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_is_interleave() 