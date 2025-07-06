"""
LeetCode 211: Design Add and Search Words Data Structure

Design a data structure that supports adding new words and finding if a string matches 
any previously added string.

Implement the WordDictionary class:
- WordDictionary() Initializes the object.
- void addWord(word) Adds word to the data structure, it can be matched later.
- bool search(word) Returns true if there is any string in the data structure that 
  matches word or false otherwise. word may contain dots '.' where dots can be matched 
  with any letter.

Example 1:
Input: ["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
       [[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
Output: [null,null,null,null,false,true,true,true]

Constraints:
- 1 <= word.length <= 25
- word in addWord consists of lowercase English letters.
- word in search consists of '.' or lowercase English letters.
- There will be at most 2 dots in word for search queries.
- At most 10^4 calls will be made to addWord and search.
"""

from typing import List, Set, Dict, Optional
import re
from collections import defaultdict, deque
import bisect


class TrieNode:
    """Standard Trie node for wildcard search."""
    def __init__(self):
        self.children = {}
        self.is_word = False


class ArrayTrieNode:
    """Array-based Trie node."""
    def __init__(self):
        self.children = [None] * 26
        self.is_word = False


class WordDictionary1_TrieDFS:
    """
    Trie with DFS for wildcard search.
    
    Time Complexity:
    - addWord: O(m) where m is word length
    - search: O(n * 26^k) where n is number of nodes, k is number of dots
    
    Space Complexity: O(ALPHABET_SIZE * N * M)
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def addWord(self, word: str) -> None:
        """Add word to trie."""
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_word = True
    
    def search(self, word: str) -> bool:
        """Search with wildcard support using DFS."""
        return self._dfs_search(word, 0, self.root)
    
    def _dfs_search(self, word: str, index: int, node: TrieNode) -> bool:
        """DFS helper for wildcard search."""
        if index == len(word):
            return node.is_word
        
        char = word[index]
        
        if char == '.':
            # Try all possible characters
            for child in node.children.values():
                if self._dfs_search(word, index + 1, child):
                    return True
            return False
        else:
            # Regular character
            if char not in node.children:
                return False
            return self._dfs_search(word, index + 1, node.children[char])


class WordDictionary2_TrieBFS:
    """
    Trie with BFS for wildcard search.
    
    Time Complexity: Similar to DFS but different traversal pattern
    Space Complexity: O(ALPHABET_SIZE * N * M) + O(queue size)
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def addWord(self, word: str) -> None:
        """Add word to trie."""
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_word = True
    
    def search(self, word: str) -> bool:
        """Search with wildcard support using BFS."""
        if not word:
            return self.root.is_word
        
        queue = deque([(self.root, 0)])  # (node, word_index)
        
        while queue:
            node, index = queue.popleft()
            
            if index == len(word):
                if node.is_word:
                    return True
                continue
            
            char = word[index]
            
            if char == '.':
                # Add all children to queue
                for child in node.children.values():
                    queue.append((child, index + 1))
            else:
                # Add specific child if exists
                if char in node.children:
                    queue.append((node.children[char], index + 1))
        
        return False


class WordDictionary3_ArrayTrie:
    """
    Array-based Trie implementation.
    
    Time Complexity: Same as HashMap Trie but better constants
    Space Complexity: O(26 * N * M)
    """
    
    def __init__(self):
        self.root = ArrayTrieNode()
    
    def _char_to_index(self, char: str) -> int:
        """Convert character to array index."""
        return ord(char) - ord('a')
    
    def addWord(self, word: str) -> None:
        """Add word to array-based trie."""
        current = self.root
        for char in word:
            index = self._char_to_index(char)
            if current.children[index] is None:
                current.children[index] = ArrayTrieNode()
            current = current.children[index]
        current.is_word = True
    
    def search(self, word: str) -> bool:
        """Search with array-based trie."""
        return self._dfs_search(word, 0, self.root)
    
    def _dfs_search(self, word: str, index: int, node: ArrayTrieNode) -> bool:
        """DFS search with array-based children."""
        if index == len(word):
            return node.is_word
        
        char = word[index]
        
        if char == '.':
            # Try all 26 possible children
            for i in range(26):
                if node.children[i] is not None:
                    if self._dfs_search(word, index + 1, node.children[i]):
                        return True
            return False
        else:
            # Regular character
            char_index = self._char_to_index(char)
            if node.children[char_index] is None:
                return False
            return self._dfs_search(word, index + 1, node.children[char_index])


class WordDictionary4_RegexBased:
    """
    Regex-based implementation.
    
    Time Complexity: O(N * M) for search in worst case
    Space Complexity: O(N * M)
    """
    
    def __init__(self):
        self.words = set()
    
    def addWord(self, word: str) -> None:
        """Add word to set."""
        self.words.add(word)
    
    def search(self, word: str) -> bool:
        """Search using regex pattern matching."""
        pattern = word.replace('.', '[a-z]')
        regex = re.compile(f"^{pattern}$")
        
        for stored_word in self.words:
            if regex.match(stored_word):
                return True
        
        return False


class WordDictionary5_LengthGrouped:
    """
    Group words by length for optimized search.
    
    Time Complexity: 
    - addWord: O(m)
    - search: O(26^k * words_of_same_length) where k is dots count
    
    Space Complexity: O(N * M)
    """
    
    def __init__(self):
        self.words_by_length = defaultdict(set)
    
    def addWord(self, word: str) -> None:
        """Add word grouped by length."""
        self.words_by_length[len(word)].add(word)
    
    def search(self, word: str) -> bool:
        """Search only among words of same length."""
        length = len(word)
        if length not in self.words_by_length:
            return False
        
        for stored_word in self.words_by_length[length]:
            if self._matches(word, stored_word):
                return True
        
        return False
    
    def _matches(self, pattern: str, word: str) -> bool:
        """Check if pattern matches word."""
        if len(pattern) != len(word):
            return False
        
        for p_char, w_char in zip(pattern, word):
            if p_char != '.' and p_char != w_char:
                return False
        
        return True


class WordDictionary6_Backtracking:
    """
    Backtracking approach with explicit state management.
    
    Time Complexity: O(26^k * N) where k is dots, N is nodes
    Space Complexity: O(ALPHABET_SIZE * N * M)
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def addWord(self, word: str) -> None:
        """Add word to trie."""
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_word = True
    
    def search(self, word: str) -> bool:
        """Backtracking search."""
        return self._backtrack(word, 0, self.root, [])
    
    def _backtrack(self, word: str, index: int, node: TrieNode, path: List[str]) -> bool:
        """Backtracking with path tracking."""
        if index == len(word):
            return node.is_word
        
        char = word[index]
        
        if char == '.':
            # Try each possible character
            for child_char, child_node in node.children.items():
                path.append(child_char)
                if self._backtrack(word, index + 1, child_node, path):
                    return True
                path.pop()  # Backtrack
            return False
        else:
            # Regular character
            if char not in node.children:
                return False
            path.append(char)
            result = self._backtrack(word, index + 1, node.children[char], path)
            path.pop()  # Backtrack
            return result


class WordDictionary7_Memoized:
    """
    Memoized Trie search for repeated patterns.
    
    Time Complexity: O(26^k) first time, O(1) for repeated patterns
    Space Complexity: O(ALPHABET_SIZE * N * M) + memoization cache
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.memo = {}  # Cache for search results
    
    def addWord(self, word: str) -> None:
        """Add word and clear relevant memoization."""
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_word = True
        
        # Clear memoization cache as new word might affect results
        self.memo.clear()
    
    def search(self, word: str) -> bool:
        """Memoized search."""
        return self._memoized_search(word, 0, id(self.root))
    
    def _memoized_search(self, word: str, index: int, node_id: int) -> bool:
        """Search with memoization."""
        key = (word[index:], node_id)
        
        if key in self.memo:
            return self.memo[key]
        
        node = self._get_node_by_id(node_id)
        result = self._dfs_search(word, index, node)
        self.memo[key] = result
        return result
    
    def _get_node_by_id(self, node_id: int) -> TrieNode:
        """Get node by ID (simplified for demo)."""
        # In practice, would maintain ID to node mapping
        return self._find_node_by_id(self.root, node_id)
    
    def _find_node_by_id(self, node: TrieNode, target_id: int) -> TrieNode:
        """Find node by ID using DFS."""
        if id(node) == target_id:
            return node
        
        for child in node.children.values():
            result = self._find_node_by_id(child, target_id)
            if result:
                return result
        
        return None
    
    def _dfs_search(self, word: str, index: int, node: TrieNode) -> bool:
        """Standard DFS search."""
        if index == len(word):
            return node.is_word
        
        char = word[index]
        
        if char == '.':
            for child in node.children.values():
                if self._dfs_search(word, index + 1, child):
                    return True
            return False
        else:
            if char not in node.children:
                return False
            return self._dfs_search(word, index + 1, node.children[char])


class WordDictionary8_IterativeStack:
    """
    Iterative implementation using explicit stack.
    
    Time Complexity: O(26^k * N)
    Space Complexity: O(stack size) instead of recursion stack
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def addWord(self, word: str) -> None:
        """Add word to trie."""
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_word = True
    
    def search(self, word: str) -> bool:
        """Iterative search using stack."""
        if not word:
            return self.root.is_word
        
        stack = [(self.root, 0)]  # (node, word_index)
        
        while stack:
            node, index = stack.pop()
            
            if index == len(word):
                if node.is_word:
                    return True
                continue
            
            char = word[index]
            
            if char == '.':
                # Add all children to stack
                for child in node.children.values():
                    stack.append((child, index + 1))
            else:
                # Add specific child if exists
                if char in node.children:
                    stack.append((node.children[char], index + 1))
        
        return False


class WordDictionary9_CompressedTrie:
    """
    Compressed Trie implementation for space efficiency.
    
    Time Complexity: O(m) for add, O(26^k * compressed_nodes) for search
    Space Complexity: Reduced space due to path compression
    """
    
    def __init__(self):
        self.root = {'prefix': '', 'children': {}, 'is_word': False}
    
    def addWord(self, word: str) -> None:
        """Add word with path compression."""
        current = self.root
        i = 0
        
        while i < len(word):
            char = word[i]
            
            if char not in current['children']:
                # Create new compressed node
                remaining = word[i:]
                current['children'][char] = {
                    'prefix': remaining,
                    'children': {},
                    'is_word': True
                }
                return
            
            child = current['children'][char]
            prefix = child['prefix']
            
            # Find common prefix length
            j = 0
            while (j < len(prefix) and 
                   i + j < len(word) and 
                   prefix[j] == word[i + j]):
                j += 1
            
            if j == len(prefix):
                # Full match, continue
                i += j
                current = child
            else:
                # Split node
                # Create new child for remaining part of existing prefix
                split_child = {
                    'prefix': prefix[j:],
                    'children': child['children'],
                    'is_word': child['is_word']
                }
                
                # Update current child
                child['prefix'] = prefix[:j]
                child['children'] = {prefix[j]: split_child}
                child['is_word'] = False
                
                # Add remaining word
                if i + j < len(word):
                    remaining_word = word[i + j:]
                    child['children'][remaining_word[0]] = {
                        'prefix': remaining_word,
                        'children': {},
                        'is_word': True
                    }
                else:
                    child['is_word'] = True
                
                return
        
        current['is_word'] = True
    
    def search(self, word: str) -> bool:
        """Search in compressed trie."""
        return self._compressed_search(word, 0, self.root)
    
    def _compressed_search(self, word: str, index: int, node: dict) -> bool:
        """Search with compressed nodes."""
        if index == len(word):
            return node['is_word']
        
        char = word[index]
        
        if char == '.':
            # Try all children
            for child in node['children'].values():
                prefix = child['prefix']
                if self._matches_prefix(word[index:], prefix):
                    if self._compressed_search(word, index + len(prefix), child):
                        return True
            return False
        else:
            if char not in node['children']:
                return False
            
            child = node['children'][char]
            prefix = child['prefix']
            
            # Check if word matches prefix
            if not self._matches_prefix(word[index:], prefix):
                return False
            
            return self._compressed_search(word, index + len(prefix), child)
    
    def _matches_prefix(self, word_part: str, prefix: str) -> bool:
        """Check if word part matches prefix with wildcards."""
        if len(word_part) < len(prefix):
            return False
        
        for i in range(len(prefix)):
            if word_part[i] != '.' and word_part[i] != prefix[i]:
                return False
        
        return True


class WordDictionary10_OptimizedArray:
    """
    Highly optimized array-based implementation.
    
    Features:
    - Array-based children for better cache locality
    - Optimized character indexing
    - Minimal object creation
    """
    
    def __init__(self):
        self.root = self._create_node()
    
    def _create_node(self) -> List:
        """Create optimized node representation."""
        return [False] + [None] * 26  # [is_word, children...]
    
    def _char_to_index(self, char: str) -> int:
        """Optimized character to index."""
        return ord(char) - 97  # ord('a') = 97
    
    def addWord(self, word: str) -> None:
        """Highly optimized add."""
        current = self.root
        
        for char in word:
            index = self._char_to_index(char) + 1  # +1 for is_word slot
            
            if current[index] is None:
                current[index] = self._create_node()
            
            current = current[index]
        
        current[0] = True  # Set is_word flag
    
    def search(self, word: str) -> bool:
        """Optimized search."""
        return self._fast_search(word, 0, self.root)
    
    def _fast_search(self, word: str, index: int, node: List) -> bool:
        """Fast array-based search."""
        if index == len(word):
            return node[0]  # is_word flag
        
        char = word[index]
        
        if char == '.':
            # Check all 26 children
            for i in range(1, 27):  # Skip is_word slot
                if node[i] is not None:
                    if self._fast_search(word, index + 1, node[i]):
                        return True
            return False
        else:
            char_index = self._char_to_index(char) + 1
            if node[char_index] is None:
                return False
            return self._fast_search(word, index + 1, node[char_index])


def test_word_dictionary():
    """Test all WordDictionary implementations."""
    
    test_cases = [
        # Basic functionality
        {
            "operations": ["addWord", "addWord", "addWord", "search", "search", "search", "search"],
            "args": [["bad"], ["dad"], ["mad"], ["pad"], ["bad"], [".ad"], ["b.."]],
            "expected": [None, None, None, False, True, True, True]
        },
        
        # Single character and dots
        {
            "operations": ["addWord", "search", "search"],
            "args": [["a"], ["a"], ["."]],
            "expected": [None, True, True]
        },
        
        # Empty word handling
        {
            "operations": ["search"],
            "args": [[""]],
            "expected": [False]
        },
        
        # Multiple dots
        {
            "operations": ["addWord", "addWord", "search", "search", "search"],
            "args": [["abc"], ["def"], ["..."], ["a.c"], [".e."]],
            "expected": [None, None, True, True, True]
        },
        
        # Complex patterns
        {
            "operations": ["addWord", "addWord", "addWord", "search", "search", "search", "search"],
            "args": [["word"], ["word"], ["ward"], ["w.rd"], ["w..d"], ["...."], ["w..."]],
            "expected": [None, None, None, True, True, True, True]
        },
        
        # No matches
        {
            "operations": ["addWord", "search", "search", "search"],
            "args": [["test"], ["best"], ["t..t"], ["te.t."]],
            "expected": [None, False, False, False]
        },
        
        # Long words
        {
            "operations": ["addWord", "search", "search"],
            "args": [["supercalifragilisticexpialidocious"], 
                    ["supercalifragilisticexpialidocious"],
                    ["....rcalifragilisticexpialidocious"]],
            "expected": [None, True, True]
        }
    ]
    
    implementations = [
        ("Trie DFS", WordDictionary1_TrieDFS),
        ("Trie BFS", WordDictionary2_TrieBFS),
        ("Array Trie", WordDictionary3_ArrayTrie),
        ("Regex Based", WordDictionary4_RegexBased),
        ("Length Grouped", WordDictionary5_LengthGrouped),
        ("Backtracking", WordDictionary6_Backtracking),
        ("Memoized", WordDictionary7_Memoized),
        ("Iterative Stack", WordDictionary8_IterativeStack),
        ("Compressed Trie", WordDictionary9_CompressedTrie),
        ("Optimized Array", WordDictionary10_OptimizedArray),
    ]
    
    print("Testing Add and Search Words Data Structure:")
    print("=" * 75)
    
    for test_idx, test_case in enumerate(test_cases):
        print(f"\nTest Case {test_idx + 1}:")
        print(f"Operations: {test_case['operations']}")
        print(f"Arguments:  {test_case['args']}")
        print(f"Expected:   {test_case['expected']}")
        
        for name, DictClass in implementations:
            try:
                dictionary = DictClass()
                results = []
                
                for op, args in zip(test_case['operations'], test_case['args']):
                    if op == "addWord":
                        result = dictionary.addWord(args[0])
                        results.append(result)
                    elif op == "search":
                        result = dictionary.search(args[0])
                        results.append(result)
                
                status = "✓" if results == test_case['expected'] else "✗"
                print(f"{status} {name}: {results}")
                
            except Exception as e:
                print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 75)
    print("Performance Analysis:")
    print("=" * 75)
    
    import time
    import random
    import string
    
    def generate_test_data(word_count: int, search_count: int):
        """Generate test data with words and search patterns."""
        words = []
        for _ in range(word_count):
            length = random.randint(3, 8)
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            words.append(word)
        
        # Generate search patterns (mix of exact words and wildcard patterns)
        searches = []
        for _ in range(search_count):
            if random.random() < 0.5:
                # Exact word from dictionary
                if words:
                    searches.append(random.choice(words))
            else:
                # Wildcard pattern
                word = random.choice(words) if words else "test"
                pattern = list(word)
                # Replace some characters with dots
                for i in range(len(pattern)):
                    if random.random() < 0.2:  # 20% chance
                        pattern[i] = '.'
                searches.append(''.join(pattern))
        
        return words, searches
    
    test_scenarios = [
        ("Small dataset", 50, 100),
        ("Medium dataset", 200, 500),
        ("Large dataset", 1000, 2000),
    ]
    
    for scenario_name, word_count, search_count in test_scenarios:
        print(f"\n{scenario_name} ({word_count} words, {search_count} searches):")
        words, searches = generate_test_data(word_count, search_count)
        
        for name, DictClass in implementations:
            try:
                # Test insertion
                dictionary = DictClass()
                start_time = time.time()
                for word in words:
                    dictionary.addWord(word)
                insert_time = (time.time() - start_time) * 1000
                
                # Test search
                start_time = time.time()
                for search_pattern in searches:
                    dictionary.search(search_pattern)
                search_time = (time.time() - start_time) * 1000
                
                print(f"  {name}: Insert {insert_time:.2f}ms, Search {search_time:.2f}ms")
                
            except Exception as e:
                print(f"  {name}: Error - {e}")
    
    print(f"\n{'='*75}")
    print("Algorithm Complexity Analysis:")
    print("="*75)
    print("1. Trie DFS:         O(m) add, O(26^k * N) search")
    print("2. Trie BFS:         O(m) add, O(26^k * N) search")
    print("3. Array Trie:       O(m) add, O(26^k * N) search (faster constants)")
    print("4. Regex Based:      O(m) add, O(N * M) search")
    print("5. Length Grouped:   O(m) add, O(words_same_length * 26^k) search")
    print("6. Backtracking:     O(m) add, O(26^k * N) search")
    print("7. Memoized:         O(m) add, O(26^k) first search, O(1) repeated")
    print("8. Iterative Stack:  O(m) add, O(26^k * N) search")
    print("9. Compressed Trie:  O(m) add, O(26^k * compressed_nodes) search")
    print("10. Optimized Array: O(m) add, O(26^k * N) search (best constants)")
    
    print(f"\n{'='*75}")
    print("Key Insights:")
    print("="*75)
    print("• Wildcard '.' requires exploring multiple branches in Trie")
    print("• DFS vs BFS: similar complexity, different memory usage patterns")
    print("• Array-based children faster than HashMap for small alphabets")
    print("• Grouping by length reduces search space significantly")
    print("• Memoization helps with repeated wildcard patterns")
    print("• Regex approach simple but generally slower for complex patterns")


if __name__ == "__main__":
    test_word_dictionary() 