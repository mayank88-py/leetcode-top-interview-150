"""
139. Word Break

Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

Example 1:
Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".

Example 2:
Input: s = "applepenapple", wordDict = ["apple","pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.

Example 3:
Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: false

Constraints:
- 1 <= s.length <= 300
- 1 <= wordDict.length <= 1000
- 1 <= wordDict[i].length <= 20
- s and wordDict[i] consist of only lowercase English letters.
- All the strings of wordDict are unique.
"""

def word_break_dp_bottom_up(s, wordDict):
    """
    Approach 1: Dynamic Programming (Bottom-up)
    Time Complexity: O(n^2 * m) where n = len(s), m = average word length
    Space Complexity: O(n)
    
    dp[i] = True if s[0:i] can be segmented into words from wordDict
    """
    word_set = set(wordDict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True  # Empty string can always be segmented
    
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[n]


def word_break_memoization(s, wordDict):
    """
    Approach 2: Memoization (Top-down DP)
    Time Complexity: O(n^2 * m)
    Space Complexity: O(n)
    
    Use memoization to avoid recalculating subproblems.
    """
    word_set = set(wordDict)
    memo = {}
    
    def can_break(start):
        if start == len(s):
            return True
        
        if start in memo:
            return memo[start]
        
        for end in range(start + 1, len(s) + 1):
            if s[start:end] in word_set and can_break(end):
                memo[start] = True
                return True
        
        memo[start] = False
        return False
    
    return can_break(0)


def word_break_bfs(s, wordDict):
    """
    Approach 3: BFS (Breadth-First Search)
    Time Complexity: O(n^2 * m)
    Space Complexity: O(n)
    
    Use BFS to explore all possible segmentations.
    """
    from collections import deque
    
    word_set = set(wordDict)
    queue = deque([0])  # Start positions to explore
    visited = set()
    
    while queue:
        start = queue.popleft()
        
        if start in visited:
            continue
        visited.add(start)
        
        for end in range(start + 1, len(s) + 1):
            if s[start:end] in word_set:
                if end == len(s):
                    return True
                queue.append(end)
    
    return False


def word_break_trie(s, wordDict):
    """
    Approach 4: Trie + DP
    Time Complexity: O(n^2 + total word length)
    Space Complexity: O(total word length)
    
    Use trie to efficiently check word prefixes.
    """
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_word = False
    
    # Build trie
    root = TrieNode()
    for word in wordDict:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
    
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    
    for i in range(1, n + 1):
        node = root
        for j in range(i - 1, -1, -1):
            if s[j] not in node.children:
                break
            node = node.children[s[j]]
            if node.is_word and dp[j]:
                dp[i] = True
                break
    
    return dp[n]


def word_break_optimized_dp(s, wordDict):
    """
    Approach 5: Optimized DP with Early Termination
    Time Complexity: O(n^2 * m)
    Space Complexity: O(n)
    
    Optimized DP with early termination and length checking.
    """
    word_set = set(wordDict)
    max_word_len = max(len(word) for word in wordDict) if wordDict else 0
    
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    
    for i in range(1, n + 1):
        for j in range(max(0, i - max_word_len), i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[n]


def word_break_recursive_with_pruning(s, wordDict):
    """
    Approach 6: Recursive with Pruning
    Time Complexity: O(n^2 * m) with memoization
    Space Complexity: O(n)
    
    Recursive approach with pruning for invalid prefixes.
    """
    word_set = set(wordDict)
    memo = {}
    
    def can_break(remaining):
        if not remaining:
            return True
        
        if remaining in memo:
            return memo[remaining]
        
        for word in word_set:
            if remaining.startswith(word):
                if can_break(remaining[len(word):]):
                    memo[remaining] = True
                    return True
        
        memo[remaining] = False
        return False
    
    return can_break(s)


def word_break_dp_reverse(s, wordDict):
    """
    Approach 7: Reverse DP (Right to Left)
    Time Complexity: O(n^2 * m)
    Space Complexity: O(n)
    
    DP from right to left instead of left to right.
    """
    word_set = set(wordDict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[n] = True  # Empty suffix can always be segmented
    
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n + 1):
            if s[i:j] in word_set and dp[j]:
                dp[i] = True
                break
    
    return dp[0]


def word_break_segment_lengths(s, wordDict):
    """
    Approach 8: DP with Segment Lengths
    Time Complexity: O(n * total_word_lengths)
    Space Complexity: O(n)
    
    Iterate by word lengths instead of positions.
    """
    word_set = set(wordDict)
    word_lengths = set(len(word) for word in wordDict)
    
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    
    for i in range(1, n + 1):
        for length in word_lengths:
            if length <= i and dp[i - length] and s[i - length:i] in word_set:
                dp[i] = True
                break
    
    return dp[n]


def word_break_iterative_expansion(s, wordDict):
    """
    Approach 9: Iterative Expansion
    Time Complexity: O(n^2 * m)
    Space Complexity: O(n)
    
    Iteratively expand valid positions.
    """
    word_set = set(wordDict)
    valid_positions = {0}  # Positions from which we can continue segmentation
    
    for i in range(len(s)):
        if i in valid_positions:
            for word in word_set:
                if s[i:].startswith(word):
                    valid_positions.add(i + len(word))
    
    return len(s) in valid_positions


def test_word_break():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ("leetcode", ["leet", "code"], True),
        ("applepenapple", ["apple", "pen"], True),
        ("catsandog", ["cats", "dog", "sand", "and", "cat"], False),
        ("", [], True),  # Empty string
        ("a", ["a"], True),
        ("ab", ["a", "b"], True),
        ("aaaaaaa", ["aaaa", "aaa"], True),
        ("aaaaaaa", ["aaaa", "aa"], False),
        ("cars", ["car", "ca", "rs"], True),
        ("raceacar", ["race", "a", "car"], True),
        ("abcd", ["a", "abc", "b", "cd"], True),
        ("goalspecial", ["go", "goal", "goals", "special"], True),
        ("wordbreak", ["word", "break"], True),
        ("wordbreaker", ["word", "break"], False),
        ("bb", ["a", "b", "bbb", "bbbb"], True),
    ]
    
    approaches = [
        ("DP Bottom-up", word_break_dp_bottom_up),
        ("Memoization", word_break_memoization),
        ("BFS", word_break_bfs),
        ("Trie + DP", word_break_trie),
        ("Optimized DP", word_break_optimized_dp),
        ("Recursive with Pruning", word_break_recursive_with_pruning),
        ("DP Reverse", word_break_dp_reverse),
        ("Segment Lengths", word_break_segment_lengths),
        ("Iterative Expansion", word_break_iterative_expansion),
    ]
    
    for i, (s, wordDict, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: s = \"{s}\", wordDict = {wordDict}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            result = func(s, wordDict.copy())
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_word_break() 