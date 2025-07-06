"""
79. Word Search

Problem:
Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells 
are horizontally or vertically neighboring. The same letter cell may not be used more than once.

Example 1:
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true

Example 2:
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
Output: true

Example 3:
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
Output: false

Time Complexity: O(M * N * 4^L) where M,N are board dimensions, L is word length
Space Complexity: O(L) for recursion stack
"""


def exist_backtrack(board, word):
    """
    Classic backtracking with visited array.
    
    Time Complexity: O(M * N * 4^L)
    Space Complexity: O(M * N) for visited array + O(L) for recursion stack
    
    Algorithm:
    1. Try starting from each cell in the board
    2. Use DFS with backtracking to explore all paths
    3. Mark cells as visited to avoid reuse
    4. Unmark when backtracking
    """
    if not board or not board[0] or not word:
        return False
    
    rows, cols = len(board), len(board[0])
    
    def dfs(r, c, index, visited):
        # Base case: found complete word
        if index == len(word):
            return True
        
        # Check bounds and constraints
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            visited[r][c] or board[r][c] != word[index]):
            return False
        
        # Mark as visited
        visited[r][c] = True
        
        # Explore all 4 directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dr, dc in directions:
            if dfs(r + dr, c + dc, index + 1, visited):
                return True
        
        # Backtrack
        visited[r][c] = False
        return False
    
    # Try starting from each cell
    for i in range(rows):
        for j in range(cols):
            if board[i][j] == word[0]:
                visited = [[False] * cols for _ in range(rows)]
                if dfs(i, j, 0, visited):
                    return True
    
    return False


def exist_optimized(board, word):
    """
    Optimized backtracking by modifying board in-place.
    
    Time Complexity: O(M * N * 4^L)
    Space Complexity: O(L) for recursion stack only
    
    Algorithm:
    1. Modify board cells in-place to mark as visited
    2. Use a special marker character (e.g., '#')
    3. Restore original character when backtracking
    """
    if not board or not board[0] or not word:
        return False
    
    rows, cols = len(board), len(board[0])
    
    def dfs(r, c, index):
        # Base case: found complete word
        if index == len(word):
            return True
        
        # Check bounds and constraints
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            board[r][c] != word[index]):
            return False
        
        # Mark as visited by replacing with special character
        temp = board[r][c]
        board[r][c] = '#'
        
        # Explore all 4 directions
        found = (dfs(r + 1, c, index + 1) or
                dfs(r - 1, c, index + 1) or
                dfs(r, c + 1, index + 1) or
                dfs(r, c - 1, index + 1))
        
        # Backtrack: restore original character
        board[r][c] = temp
        
        return found
    
    # Try starting from each cell
    for i in range(rows):
        for j in range(cols):
            if board[i][j] == word[0] and dfs(i, j, 0):
                return True
    
    return False


def exist_trie_optimized(board, word):
    """
    Trie-based optimization for multiple word searches.
    
    Time Complexity: O(M * N * 4^L)
    Space Complexity: O(L) for trie + O(L) for recursion stack
    
    Algorithm:
    1. Build trie from the word
    2. Use trie traversal instead of string indexing
    3. Can be extended to search multiple words efficiently
    """
    if not board or not board[0] or not word:
        return False
    
    # Build trie for the word
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_word = False
    
    root = TrieNode()
    node = root
    for char in word:
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
    node.is_word = True
    
    rows, cols = len(board), len(board[0])
    
    def dfs(r, c, node):
        # Check bounds
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False
        
        char = board[r][c]
        if char not in node.children:
            return False
        
        next_node = node.children[char]
        if next_node.is_word:
            return True
        
        # Mark as visited
        board[r][c] = '#'
        
        # Explore all 4 directions
        found = (dfs(r + 1, c, next_node) or
                dfs(r - 1, c, next_node) or
                dfs(r, c + 1, next_node) or
                dfs(r, c - 1, next_node))
        
        # Backtrack
        board[r][c] = char
        
        return found
    
    # Try starting from each cell
    for i in range(rows):
        for j in range(cols):
            if board[i][j] in root.children:
                if dfs(i, j, root):
                    return True
    
    return False


def exist_iterative(board, word):
    """
    Iterative approach using stack.
    
    Time Complexity: O(M * N * 4^L)
    Space Complexity: O(M * N * L) for stack states
    
    Algorithm:
    1. Use stack to store (row, col, index, visited_set)
    2. Process each state and generate next valid states
    3. Return true when complete word is found
    """
    if not board or not board[0] or not word:
        return False
    
    rows, cols = len(board), len(board[0])
    
    # Stack stores (row, col, index, visited_positions)
    for i in range(rows):
        for j in range(cols):
            if board[i][j] == word[0]:
                stack = [(i, j, 0, {(i, j)})]
                
                while stack:
                    r, c, index, visited = stack.pop()
                    
                    # Found complete word
                    if index == len(word) - 1:
                        return True
                    
                    # Explore all 4 directions
                    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        
                        if (0 <= nr < rows and 0 <= nc < cols and
                            (nr, nc) not in visited and
                            index + 1 < len(word) and
                            board[nr][nc] == word[index + 1]):
                            
                            new_visited = visited | {(nr, nc)}
                            stack.append((nr, nc, index + 1, new_visited))
    
    return False


def exist_early_termination(board, word):
    """
    Optimized with early termination checks.
    
    Time Complexity: O(M * N * 4^L) but with better average case
    Space Complexity: O(L)
    
    Algorithm:
    1. Check if all word characters exist in board
    2. Count character frequencies and validate
    3. Try starting from rarer characters first
    """
    if not board or not board[0] or not word:
        return False
    
    # Count characters in board
    board_chars = {}
    for row in board:
        for char in row:
            board_chars[char] = board_chars.get(char, 0) + 1
    
    # Check if all word characters exist in board
    word_chars = {}
    for char in word:
        word_chars[char] = word_chars.get(char, 0) + 1
        if char not in board_chars or word_chars[char] > board_chars[char]:
            return False
    
    # Optimize: if first or last char is rarer, reverse word
    if board_chars[word[0]] > board_chars[word[-1]]:
        word = word[::-1]
    
    rows, cols = len(board), len(board[0])
    
    def dfs(r, c, index):
        if index == len(word):
            return True
        
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            board[r][c] != word[index]):
            return False
        
        temp = board[r][c]
        board[r][c] = '#'
        
        found = (dfs(r + 1, c, index + 1) or
                dfs(r - 1, c, index + 1) or
                dfs(r, c + 1, index + 1) or
                dfs(r, c - 1, index + 1))
        
        board[r][c] = temp
        return found
    
    # Try starting from each occurrence of first character
    for i in range(rows):
        for j in range(cols):
            if board[i][j] == word[0] and dfs(i, j, 0):
                return True
    
    return False


def test_word_search():
    """Test all implementations with various test cases."""
    
    test_cases = [
        ([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCCED", True),
        ([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "SEE", True),
        ([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCB", False),
        ([["A"]], "A", True),
        ([["A"]], "B", False),
        ([["a","b"],["c","d"]], "abcd", False),
        ([["a","b"],["c","d"]], "acdb", True)
    ]
    
    implementations = [
        ("Backtrack (visited)", exist_backtrack),
        ("Optimized (in-place)", exist_optimized),
        ("Trie-based", exist_trie_optimized),
        ("Iterative", exist_iterative),
        ("Early Termination", exist_early_termination)
    ]
    
    print("Testing Word Search...")
    
    for i, (board, word, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: word = '{word}'")
        print(f"Board: {board}")
        print(f"Expected: {expected}")
        
        for impl_name, impl_func in implementations:
            # Make a deep copy since some implementations modify board
            board_copy = [row[:] for row in board]
            result = impl_func(board_copy, word)
            
            is_correct = result == expected
            print(f"{impl_name:20} | Result: {result} | {'✓' if is_correct else '✗'}")


if __name__ == "__main__":
    test_word_search() 