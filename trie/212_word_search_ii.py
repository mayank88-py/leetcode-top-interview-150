"""
LeetCode 212: Word Search II

Given an m x n board of characters and a list of strings words, return all words on the board.

Each word must be constructed from letters of sequentially adjacent cells, where adjacent 
cells are horizontally or vertically neighboring. The same letter cell may not be used 
more than once in a word.

Example 1:
Input: board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], 
       words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]

Example 2:
Input: board = [["a","b"],["c","d"]], words = ["abcb"]
Output: []

Constraints:
- m == board.length
- n == board[i].length
- 1 <= m, n <= 12
- board[i][j] is a lowercase English letter.
- 1 <= words.length <= 3 * 10^4
- 1 <= words[i].length <= 10
- words[i] consists of lowercase English letters.
- All the values of words are unique.
"""

from typing import List, Set, Dict, Tuple
from collections import defaultdict, deque


class TrieNode:
    """Trie node for word search."""
    def __init__(self):
        self.children = {}
        self.word = None  # Store complete word when reached end


class WordSearch1_TrieBacktracking:
    """
    Trie + Backtracking approach (most efficient).
    
    Time Complexity: O(M * N * 4^L * W) where M*N is board size, L is max word length, W is words count
    Space Complexity: O(W * L) for trie + O(L) for recursion stack
    
    Algorithm:
    1. Build trie from all words
    2. For each cell, start DFS with trie traversal
    3. Use backtracking to explore all paths
    4. Mark cells as visited during DFS
    """
    
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        """Find all words using Trie + Backtracking."""
        if not board or not board[0] or not words:
            return []
        
        # Build trie
        root = self._build_trie(words)
        
        result = set()
        m, n = len(board), len(board[0])
        
        # Start DFS from each cell
        for i in range(m):
            for j in range(n):
                self._dfs(board, i, j, root, result)
        
        return list(result)
    
    def _build_trie(self, words: List[str]) -> TrieNode:
        """Build trie from words list."""
        root = TrieNode()
        
        for word in words:
            current = root
            for char in word:
                if char not in current.children:
                    current.children[char] = TrieNode()
                current = current.children[char]
            current.word = word
        
        return root
    
    def _dfs(self, board: List[List[str]], i: int, j: int, 
             node: TrieNode, result: Set[str]) -> None:
        """DFS with backtracking."""
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]):
            return
        
        char = board[i][j]
        if char not in node.children:
            return
        
        # Mark as visited
        board[i][j] = '#'
        
        node = node.children[char]
        
        # Found a word
        if node.word:
            result.add(node.word)
            # Don't return here - continue searching for longer words
        
        # Explore all 4 directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for di, dj in directions:
            self._dfs(board, i + di, j + dj, node, result)
        
        # Backtrack
        board[i][j] = char


class WordSearch2_TrieOptimized:
    """
    Optimized Trie with pruning.
    
    Time Complexity: Better than basic trie due to pruning
    Space Complexity: O(W * L) for trie
    
    Optimizations:
    1. Remove words from trie after finding them
    2. Prune empty branches
    3. Early termination
    """
    
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        """Optimized trie search with pruning."""
        if not board or not board[0] or not words:
            return []
        
        root = self._build_trie(words)
        result = []
        m, n = len(board), len(board[0])
        
        for i in range(m):
            for j in range(n):
                self._dfs(board, i, j, root, result)
        
        return result
    
    def _build_trie(self, words: List[str]) -> TrieNode:
        """Build trie with optimization flags."""
        root = TrieNode()
        
        for word in words:
            current = root
            for char in word:
                if char not in current.children:
                    current.children[char] = TrieNode()
                current = current.children[char]
            current.word = word
        
        return root
    
    def _dfs(self, board: List[List[str]], i: int, j: int, 
             node: TrieNode, result: List[str]) -> None:
        """DFS with optimizations."""
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]):
            return
        
        char = board[i][j]
        if char not in node.children:
            return
        
        board[i][j] = '#'  # Mark visited
        node = node.children[char]
        
        # Found word
        if node.word:
            result.append(node.word)
            node.word = None  # Remove to avoid duplicates
        
        # Continue only if there are more characters to explore
        if node.children:
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for di, dj in directions:
                self._dfs(board, i + di, j + dj, node, result)
        
        board[i][j] = char  # Backtrack
        
        # Pruning: remove empty nodes
        if not node.children and not node.word:
            del node.children[char] if hasattr(node, 'parent') else None


class WordSearch3_BruteForce:
    """
    Brute force approach - search each word individually.
    
    Time Complexity: O(W * M * N * 4^L)
    Space Complexity: O(L) for recursion stack
    
    Algorithm:
    1. For each word, search the entire board
    2. Use DFS for each word separately
    3. Less efficient but simpler logic
    """
    
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        """Brute force word search."""
        result = []
        
        for word in words:
            if self._exists(board, word):
                result.append(word)
        
        return result
    
    def _exists(self, board: List[List[str]], word: str) -> bool:
        """Check if word exists in board."""
        m, n = len(board), len(board[0])
        
        for i in range(m):
            for j in range(n):
                if self._dfs(board, i, j, word, 0):
                    return True
        
        return False
    
    def _dfs(self, board: List[List[str]], i: int, j: int, 
             word: str, index: int) -> bool:
        """DFS for single word."""
        if index == len(word):
            return True
        
        if (i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or
            board[i][j] != word[index]):
            return False
        
        # Mark visited
        temp = board[i][j]
        board[i][j] = '#'
        
        # Explore all directions
        found = (self._dfs(board, i + 1, j, word, index + 1) or
                self._dfs(board, i - 1, j, word, index + 1) or
                self._dfs(board, i, j + 1, word, index + 1) or
                self._dfs(board, i, j - 1, word, index + 1))
        
        # Backtrack
        board[i][j] = temp
        
        return found


class WordSearch4_SetBased:
    """
    Set-based approach with prefix checking.
    
    Time Complexity: O(M * N * 4^L * prefix_checks)
    Space Complexity: O(W * L) for prefix sets
    
    Algorithm:
    1. Create sets of all prefixes
    2. Use DFS with prefix validation
    3. Early termination when prefix not in set
    """
    
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        """Set-based word search."""
        if not board or not board[0] or not words:
            return []
        
        # Build prefix sets
        word_set = set(words)
        prefix_set = set()
        
        for word in words:
            for i in range(1, len(word) + 1):
                prefix_set.add(word[:i])
        
        result = set()
        m, n = len(board), len(board[0])
        
        for i in range(m):
            for j in range(n):
                self._dfs(board, i, j, "", word_set, prefix_set, result)
        
        return list(result)
    
    def _dfs(self, board: List[List[str]], i: int, j: int, 
             current: str, word_set: Set[str], prefix_set: Set[str], 
             result: Set[str]) -> None:
        """DFS with prefix checking."""
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]):
            return
        
        char = board[i][j]
        if char == '#':  # Already visited
            return
        
        new_word = current + char
        
        # Early termination if prefix doesn't exist
        if new_word not in prefix_set:
            return
        
        # Found complete word
        if new_word in word_set:
            result.add(new_word)
        
        # Mark visited and continue DFS
        board[i][j] = '#'
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for di, dj in directions:
            self._dfs(board, i + di, j + dj, new_word, word_set, prefix_set, result)
        
        # Backtrack
        board[i][j] = char


class WordSearch5_BFS:
    """
    BFS approach for word search.
    
    Time Complexity: O(M * N * 4^L * W)
    Space Complexity: O(M * N * L) for queue
    
    Algorithm:
    1. Use BFS instead of DFS
    2. Queue stores (position, path, visited_set)
    3. Less common but demonstrates different traversal
    """
    
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        """BFS word search."""
        if not board or not board[0] or not words:
            return []
        
        word_set = set(words)
        result = set()
        m, n = len(board), len(board[0])
        
        # Start BFS from each cell
        for i in range(m):
            for j in range(n):
                if board[i][j] in [word[0] for word in words]:
                    self._bfs(board, i, j, word_set, result)
        
        return list(result)
    
    def _bfs(self, board: List[List[str]], start_i: int, start_j: int,
             word_set: Set[str], result: Set[str]) -> None:
        """BFS implementation."""
        m, n = len(board), len(board[0])
        queue = deque([(start_i, start_j, board[start_i][start_j], {(start_i, start_j)})])
        
        while queue:
            i, j, path, visited = queue.popleft()
            
            # Check if current path is a word
            if path in word_set:
                result.add(path)
            
            # Continue BFS if path could be extended to form a word
            if any(word.startswith(path) for word in word_set):
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    
                    if (0 <= ni < m and 0 <= nj < n and 
                        (ni, nj) not in visited and 
                        len(path) < 10):  # Max word length constraint
                        
                        new_visited = visited | {(ni, nj)}
                        new_path = path + board[ni][nj]
                        queue.append((ni, nj, new_path, new_visited))


class WordSearch6_Iterative:
    """
    Iterative DFS using explicit stack.
    
    Time Complexity: O(M * N * 4^L * W)
    Space Complexity: O(M * N * L) for stack
    
    Algorithm:
    1. Use explicit stack instead of recursion
    2. Stack stores (position, trie_node, visited_set)
    3. Avoid recursion stack limitations
    """
    
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        """Iterative DFS word search."""
        if not board or not board[0] or not words:
            return []
        
        root = self._build_trie(words)
        result = set()
        m, n = len(board), len(board[0])
        
        for i in range(m):
            for j in range(n):
                if board[i][j] in root.children:
                    self._iterative_dfs(board, i, j, root, result)
        
        return list(result)
    
    def _build_trie(self, words: List[str]) -> TrieNode:
        """Build trie from words."""
        root = TrieNode()
        
        for word in words:
            current = root
            for char in word:
                if char not in current.children:
                    current.children[char] = TrieNode()
                current = current.children[char]
            current.word = word
        
        return root
    
    def _iterative_dfs(self, board: List[List[str]], start_i: int, start_j: int,
                      root: TrieNode, result: Set[str]) -> None:
        """Iterative DFS implementation."""
        m, n = len(board), len(board[0])
        stack = [(start_i, start_j, root.children[board[start_i][start_j]], {(start_i, start_j)})]
        
        while stack:
            i, j, node, visited = stack.pop()
            
            # Found word
            if node.word:
                result.add(node.word)
            
            # Explore neighbors
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                if (0 <= ni < m and 0 <= nj < n and 
                    (ni, nj) not in visited and
                    board[ni][nj] in node.children):
                    
                    new_visited = visited | {(ni, nj)}
                    next_node = node.children[board[ni][nj]]
                    stack.append((ni, nj, next_node, new_visited))


class WordSearch7_Memoized:
    """
    Memoized approach to avoid redundant calculations.
    
    Time Complexity: Better than basic due to memoization
    Space Complexity: O(M * N * L * cache_size)
    
    Algorithm:
    1. Cache results of DFS from specific positions
    2. Key: (position, remaining_word)
    3. Avoid recalculating same subproblems
    """
    
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        """Memoized word search."""
        if not board or not board[0] or not words:
            return []
        
        result = []
        memo = {}
        
        for word in words:
            if self._exists_memo(board, word, memo):
                result.append(word)
        
        return result
    
    def _exists_memo(self, board: List[List[str]], word: str, memo: Dict) -> bool:
        """Check if word exists with memoization."""
        m, n = len(board), len(board[0])
        
        for i in range(m):
            for j in range(n):
                if self._dfs_memo(board, i, j, word, 0, frozenset(), memo):
                    return True
        
        return False
    
    def _dfs_memo(self, board: List[List[str]], i: int, j: int, 
                  word: str, index: int, visited: frozenset, memo: Dict) -> bool:
        """Memoized DFS."""
        key = (i, j, word[index:], visited)
        
        if key in memo:
            return memo[key]
        
        if index == len(word):
            memo[key] = True
            return True
        
        if (i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or
            (i, j) in visited or board[i][j] != word[index]):
            memo[key] = False
            return False
        
        new_visited = visited | {(i, j)}
        
        # Try all directions
        result = (self._dfs_memo(board, i + 1, j, word, index + 1, new_visited, memo) or
                 self._dfs_memo(board, i - 1, j, word, index + 1, new_visited, memo) or
                 self._dfs_memo(board, i, j + 1, word, index + 1, new_visited, memo) or
                 self._dfs_memo(board, i, j - 1, word, index + 1, new_visited, memo))
        
        memo[key] = result
        return result


class WordSearch8_PrefixFiltered:
    """
    Prefix-filtered approach with early termination.
    
    Time Complexity: O(M * N * 4^L) with better pruning
    Space Complexity: O(W * L) for prefix trie
    
    Algorithm:
    1. Build prefix trie for efficient prefix checking
    2. Terminate early if no word has current prefix
    3. More aggressive pruning than basic approaches
    """
    
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        """Prefix-filtered search."""
        if not board or not board[0] or not words:
            return []
        
        # Build prefix trie
        prefix_trie = self._build_prefix_trie(words)
        word_set = set(words)
        result = set()
        m, n = len(board), len(board[0])
        
        for i in range(m):
            for j in range(n):
                self._dfs_filtered(board, i, j, "", prefix_trie, word_set, result)
        
        return list(result)
    
    def _build_prefix_trie(self, words: List[str]) -> TrieNode:
        """Build trie for prefix checking."""
        root = TrieNode()
        
        for word in words:
            current = root
            for char in word:
                if char not in current.children:
                    current.children[char] = TrieNode()
                current = current.children[char]
        
        return root
    
    def _dfs_filtered(self, board: List[List[str]], i: int, j: int,
                     current: str, trie_node: TrieNode, word_set: Set[str],
                     result: Set[str]) -> None:
        """DFS with prefix filtering."""
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]):
            return
        
        char = board[i][j]
        if char == '#' or char not in trie_node.children:
            return
        
        new_word = current + char
        next_node = trie_node.children[char]
        
        # Check if complete word
        if new_word in word_set:
            result.add(new_word)
        
        # Continue only if there are possible extensions
        if next_node.children:
            board[i][j] = '#'  # Mark visited
            
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for di, dj in directions:
                self._dfs_filtered(board, i + di, j + dj, new_word, next_node, word_set, result)
            
            board[i][j] = char  # Backtrack


def test_word_search_ii():
    """Test all Word Search II implementations."""
    
    test_cases = [
        # Basic test case from problem
        {
            "board": [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]],
            "words": ["oath","pea","eat","rain"],
            "expected": {"eat", "oath"}
        },
        
        # Empty result
        {
            "board": [["a","b"],["c","d"]],
            "words": ["abcb"],
            "expected": set()
        },
        
        # Single cell board
        {
            "board": [["a"]],
            "words": ["a", "b"],
            "expected": {"a"}
        },
        
        # All words found
        {
            "board": [["a","b"],["c","d"]],
            "words": ["ab", "ac", "bd", "cd", "abdc"],
            "expected": {"ab", "ac", "bd", "cd"}
        },
        
        # Overlapping words
        {
            "board": [["a","b","c"],["a","e","d"],["a","f","g"]],
            "words": ["abcded", "bed", "abed", "ade"],
            "expected": {"bed", "ade"}
        },
        
        # Long word
        {
            "board": [["a","b","c","e"],["s","f","c","s"],["a","d","e","e"]],
            "words": ["abcced", "see", "deed"],
            "expected": {"abcced", "see"}
        },
        
        # Complex case
        {
            "board": [["o","a","b","n"],["o","t","a","e"],["a","h","k","r"],["a","f","l","v"]],
            "words": ["oa", "oaa", "oaaa", "oab", "oaba", "oat", "oath", "oathk", "oathkr"],
            "expected": {"oa", "oaa", "oab", "oat"}
        },
        
        # No valid paths
        {
            "board": [["a","a","a"],["a","a","a"],["a","a","a"]],
            "words": ["aaaaaaaaa", "aaaaaaaaaa"],
            "expected": {"aaaaaaaaa"}
        },
        
        # Single character words
        {
            "board": [["a","b"],["c","d"]],
            "words": ["a", "b", "c", "d", "e"],
            "expected": {"a", "b", "c", "d"}
        },
    ]
    
    implementations = [
        ("Trie + Backtracking", WordSearch1_TrieBacktracking()),
        ("Trie Optimized", WordSearch2_TrieOptimized()),
        ("Brute Force", WordSearch3_BruteForce()),
        ("Set Based", WordSearch4_SetBased()),
        ("BFS", WordSearch5_BFS()),
        ("Iterative DFS", WordSearch6_Iterative()),
        ("Memoized", WordSearch7_Memoized()),
        ("Prefix Filtered", WordSearch8_PrefixFiltered()),
    ]
    
    print("Testing Word Search II implementations:")
    print("=" * 75)
    
    for test_idx, test_case in enumerate(test_cases):
        print(f"\nTest Case {test_idx + 1}:")
        print(f"Board: {test_case['board']}")
        print(f"Words: {test_case['words']}")
        print(f"Expected: {sorted(test_case['expected'])}")
        
        for name, implementation in implementations:
            try:
                # Make a copy of board since some methods modify it
                board_copy = [row[:] for row in test_case['board']]
                result = implementation.findWords(board_copy, test_case['words'])
                result_set = set(result)
                
                status = "✓" if result_set == test_case['expected'] else "✗"
                print(f"{status} {name}: {sorted(result_set)}")
                
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 75)
    print("Performance Analysis:")
    print("=" * 75)
    
    import time
    import random
    import string
    
    def generate_test_case(board_size: int, word_count: int):
        """Generate random test case."""
        # Create random board
        board = []
        for _ in range(board_size):
            row = [random.choice(string.ascii_lowercase[:10]) for _ in range(board_size)]
            board.append(row)
        
        # Generate random words
        words = []
        for _ in range(word_count):
            length = random.randint(3, min(6, board_size * 2))
            word = ''.join(random.choices(string.ascii_lowercase[:10], k=length))
            words.append(word)
        
        return board, words
    
    test_scenarios = [
        ("Small board", 4, 20),
        ("Medium board", 6, 50),
        ("Large board", 8, 100),
    ]
    
    # Test only efficient implementations for larger cases
    efficient_implementations = [
        ("Trie + Backtracking", WordSearch1_TrieBacktracking()),
        ("Trie Optimized", WordSearch2_TrieOptimized()),
        ("Set Based", WordSearch4_SetBased()),
        ("Prefix Filtered", WordSearch8_PrefixFiltered()),
    ]
    
    for scenario_name, board_size, word_count in test_scenarios:
        print(f"\n{scenario_name} ({board_size}x{board_size} board, {word_count} words):")
        board, words = generate_test_case(board_size, word_count)
        
        test_impls = efficient_implementations if board_size > 4 else implementations[:6]
        
        for name, implementation in test_impls:
            try:
                board_copy = [row[:] for row in board]
                
                start_time = time.time()
                result = implementation.findWords(board_copy, words)
                elapsed = (time.time() - start_time) * 1000
                
                print(f"  {name}: {elapsed:.2f}ms ({len(result)} words found)")
                
            except Exception as e:
                print(f"  {name}: Error - {e}")
    
    print(f"\n{'='*75}")
    print("Algorithm Complexity Analysis:")
    print("="*75)
    print("1. Trie + Backtracking:   O(M*N*4^L) time, O(W*L) space")
    print("2. Trie Optimized:        O(M*N*4^L) time with pruning")
    print("3. Brute Force:           O(W*M*N*4^L) time, O(L) space")
    print("4. Set Based:             O(M*N*4^L*prefix_checks) time")
    print("5. BFS:                   O(M*N*4^L) time, O(M*N*L) space")
    print("6. Iterative DFS:         O(M*N*4^L) time, O(M*N*L) space")
    print("7. Memoized:              O(M*N*4^L) time with memoization")
    print("8. Prefix Filtered:       O(M*N*4^L) time with better pruning")
    
    print(f"\n{'='*75}")
    print("Key Insights:")
    print("="*75)
    print("• Trie + DFS backtracking is the most efficient approach")
    print("• Building trie allows sharing prefixes across words")
    print("• Pruning and early termination significantly improve performance")
    print("• Backtracking is essential to explore all possible paths")
    print("• Board modification for visited tracking is more efficient than sets")
    print("• BFS generally less efficient than DFS for this problem")
    print("• Memoization helps but has overhead for key generation")


if __name__ == "__main__":
    test_word_search_ii() 