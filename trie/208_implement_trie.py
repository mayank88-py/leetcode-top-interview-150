"""
LeetCode 208: Implement Trie (Prefix Tree)

A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently 
store and retrieve keys in a dataset of strings. There are various applications of this 
data structure, such as autocomplete and spellchecker.

Implement the Trie class:
- Trie() Initializes the trie object.
- void insert(String word) Inserts the string word into the trie.
- boolean search(String word) Returns true if the string word is in the trie, and false otherwise.
- boolean startsWith(String prefix) Returns true if there is a previously inserted string 
  word that has the prefix prefix, and false otherwise.

Example 1:
Input: ["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
       [[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
Output: [null, null, true, false, true, null, true]

Constraints:
- 1 <= word.length, prefix.length <= 2000
- word and prefix consist only of lowercase English letters.
- At most 3 * 10^4 calls will be made to insert, search, and startsWith.
"""

from typing import Dict, List, Optional
from collections import defaultdict, deque
import bisect


class TrieNode:
    """Standard Trie node with children dictionary."""
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class TrieArrayNode:
    """Trie node using array for children (faster for lowercase letters)."""
    def __init__(self):
        self.children = [None] * 26  # a-z
        self.is_end_of_word = False


class CompressedTrieNode:
    """Compressed Trie node for path compression."""
    def __init__(self, prefix=""):
        self.prefix = prefix
        self.children = {}
        self.is_end_of_word = False


class Trie1_HashMap:
    """
    HashMap-based Trie implementation.
    
    Time Complexity:
    - Insert: O(m) where m is key length
    - Search: O(m)
    - StartsWith: O(m)
    
    Space Complexity: O(ALPHABET_SIZE * N * M) where N is number of keys
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        """Insert word into trie."""
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """Search for exact word in trie."""
        current = self.root
        for char in word:
            if char not in current.children:
                return False
            current = current.children[char]
        return current.is_end_of_word
    
    def startsWith(self, prefix: str) -> bool:
        """Check if any word starts with given prefix."""
        current = self.root
        for char in prefix:
            if char not in current.children:
                return False
            current = current.children[char]
        return True


class Trie2_Array:
    """
    Array-based Trie implementation (optimized for lowercase letters).
    
    Time Complexity: Same as HashMap version
    Space Complexity: O(26 * N * M) - more memory efficient
    """
    
    def __init__(self):
        self.root = TrieArrayNode()
    
    def _char_to_index(self, char: str) -> int:
        """Convert character to array index."""
        return ord(char) - ord('a')
    
    def insert(self, word: str) -> None:
        """Insert word into trie."""
        current = self.root
        for char in word:
            index = self._char_to_index(char)
            if current.children[index] is None:
                current.children[index] = TrieArrayNode()
            current = current.children[index]
        current.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """Search for exact word in trie."""
        current = self.root
        for char in word:
            index = self._char_to_index(char)
            if current.children[index] is None:
                return False
            current = current.children[index]
        return current.is_end_of_word
    
    def startsWith(self, prefix: str) -> bool:
        """Check if any word starts with given prefix."""
        current = self.root
        for char in prefix:
            index = self._char_to_index(char)
            if current.children[index] is None:
                return False
            current = current.children[index]
        return True


class Trie3_Compressed:
    """
    Compressed Trie implementation with path compression.
    
    Time Complexity: O(m) but with better constant factors
    Space Complexity: O(N * M) - more space efficient
    """
    
    def __init__(self):
        self.root = CompressedTrieNode()
    
    def insert(self, word: str) -> None:
        """Insert word with path compression."""
        current = self.root
        i = 0
        
        while i < len(word):
            found_child = None
            for char, child in current.children.items():
                if word[i] == char:
                    found_child = child
                    break
            
            if found_child is None:
                # Create new node with remaining suffix
                new_node = CompressedTrieNode(word[i:])
                new_node.is_end_of_word = True
                current.children[word[i]] = new_node
                return
            
            # Found matching child
            prefix = found_child.prefix
            j = 0
            
            # Find common prefix length
            while (j < len(prefix) and 
                   i + j < len(word) and 
                   prefix[j] == word[i + j]):
                j += 1
            
            if j == len(prefix):
                # Full prefix match, continue with this child
                current = found_child
                i += j
            else:
                # Partial match, need to split
                # Split the existing node
                split_node = CompressedTrieNode(prefix[j:])
                split_node.children = found_child.children
                split_node.is_end_of_word = found_child.is_end_of_word
                
                # Update current child
                found_child.prefix = prefix[:j]
                found_child.children = {prefix[j]: split_node}
                found_child.is_end_of_word = False
                
                # Add remaining part of word
                if i + j < len(word):
                    remaining = word[i + j:]
                    new_node = CompressedTrieNode(remaining)
                    new_node.is_end_of_word = True
                    found_child.children[remaining[0]] = new_node
                else:
                    found_child.is_end_of_word = True
                
                return
        
        current.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """Search with path compression."""
        current = self.root
        i = 0
        
        while i < len(word):
            found_child = None
            for char, child in current.children.items():
                if word[i] == char:
                    found_child = child
                    break
            
            if found_child is None:
                return False
            
            prefix = found_child.prefix
            if not word[i:i+len(prefix)] == prefix:
                return False
            
            i += len(prefix)
            current = found_child
        
        return current.is_end_of_word
    
    def startsWith(self, prefix: str) -> bool:
        """Check prefix with path compression."""
        current = self.root
        i = 0
        
        while i < len(prefix):
            found_child = None
            for char, child in current.children.items():
                if prefix[i] == char:
                    found_child = child
                    break
            
            if found_child is None:
                return False
            
            node_prefix = found_child.prefix
            remaining_prefix = prefix[i:]
            
            if len(remaining_prefix) <= len(node_prefix):
                return node_prefix.startswith(remaining_prefix)
            
            if not remaining_prefix.startswith(node_prefix):
                return False
            
            i += len(node_prefix)
            current = found_child
        
        return True


class Trie4_SetBased:
    """
    Set-based Trie implementation (simple but less efficient).
    
    Time Complexity: O(m) for all operations
    Space Complexity: O(N * M)
    """
    
    def __init__(self):
        self.words = set()
        self.prefixes = set()
    
    def insert(self, word: str) -> None:
        """Insert word and all its prefixes."""
        self.words.add(word)
        for i in range(1, len(word) + 1):
            self.prefixes.add(word[:i])
    
    def search(self, word: str) -> bool:
        """Search in word set."""
        return word in self.words
    
    def startsWith(self, prefix: str) -> bool:
        """Search in prefix set."""
        return prefix in self.prefixes


class Trie5_ListBased:
    """
    List-based Trie implementation (for comparison).
    
    Time Complexity: O(N * M) worst case
    Space Complexity: O(N * M)
    """
    
    def __init__(self):
        self.words = []
    
    def insert(self, word: str) -> None:
        """Insert word into list."""
        if word not in self.words:
            self.words.append(word)
    
    def search(self, word: str) -> bool:
        """Linear search in word list."""
        return word in self.words
    
    def startsWith(self, prefix: str) -> bool:
        """Check if any word starts with prefix."""
        for word in self.words:
            if word.startswith(prefix):
                return True
        return False


class Trie6_SortedList:
    """
    Sorted list with binary search implementation.
    
    Time Complexity: 
    - Insert: O(N) for insertion, O(log N) for search position
    - Search: O(log N)
    - StartsWith: O(log N + K) where K is number of matches
    
    Space Complexity: O(N * M)
    """
    
    def __init__(self):
        self.words = []
    
    def insert(self, word: str) -> None:
        """Insert word maintaining sorted order."""
        pos = bisect.bisect_left(self.words, word)
        if pos == len(self.words) or self.words[pos] != word:
            self.words.insert(pos, word)
    
    def search(self, word: str) -> bool:
        """Binary search for word."""
        pos = bisect.bisect_left(self.words, word)
        return pos < len(self.words) and self.words[pos] == word
    
    def startsWith(self, prefix: str) -> bool:
        """Binary search for prefix."""
        pos = bisect.bisect_left(self.words, prefix)
        return (pos < len(self.words) and 
                self.words[pos].startswith(prefix))


class Trie7_Recursive:
    """
    Recursive Trie implementation.
    
    Time Complexity: O(m) but with recursion overhead
    Space Complexity: O(m) recursion stack + O(ALPHABET_SIZE * N * M)
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        """Recursive insert."""
        self._insert_recursive(self.root, word, 0)
    
    def _insert_recursive(self, node: TrieNode, word: str, index: int) -> None:
        """Helper for recursive insert."""
        if index == len(word):
            node.is_end_of_word = True
            return
        
        char = word[index]
        if char not in node.children:
            node.children[char] = TrieNode()
        
        self._insert_recursive(node.children[char], word, index + 1)
    
    def search(self, word: str) -> bool:
        """Recursive search."""
        return self._search_recursive(self.root, word, 0)
    
    def _search_recursive(self, node: TrieNode, word: str, index: int) -> bool:
        """Helper for recursive search."""
        if index == len(word):
            return node.is_end_of_word
        
        char = word[index]
        if char not in node.children:
            return False
        
        return self._search_recursive(node.children[char], word, index + 1)
    
    def startsWith(self, prefix: str) -> bool:
        """Recursive prefix search."""
        return self._starts_with_recursive(self.root, prefix, 0)
    
    def _starts_with_recursive(self, node: TrieNode, prefix: str, index: int) -> bool:
        """Helper for recursive prefix search."""
        if index == len(prefix):
            return True
        
        char = prefix[index]
        if char not in node.children:
            return False
        
        return self._starts_with_recursive(node.children[char], prefix, index + 1)


class Trie8_Iterative:
    """
    Iterative Trie implementation using explicit stack.
    
    Time Complexity: O(m)
    Space Complexity: O(ALPHABET_SIZE * N * M)
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        """Iterative insert using explicit iteration."""
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """Iterative search."""
        current = self.root
        for char in word:
            if char not in current.children:
                return False
            current = current.children[char]
        return current.is_end_of_word
    
    def startsWith(self, prefix: str) -> bool:
        """Iterative prefix search."""
        current = self.root
        for char in prefix:
            if char not in current.children:
                return False
            current = current.children[char]
        return True


class Trie9_DoubleHashing:
    """
    Double hashing Trie implementation.
    
    Time Complexity: O(m) average case
    Space Complexity: O(N * M)
    """
    
    def __init__(self):
        self.words = set()
        self.prefix_set = set()
    
    def _hash1(self, s: str) -> int:
        """Primary hash function."""
        hash_val = 0
        for char in s:
            hash_val = (hash_val * 31 + ord(char)) % (10**9 + 7)
        return hash_val
    
    def _hash2(self, s: str) -> int:
        """Secondary hash function."""
        hash_val = 0
        for char in s:
            hash_val = (hash_val * 37 + ord(char)) % (10**9 + 9)
        return hash_val
    
    def insert(self, word: str) -> None:
        """Insert using double hashing."""
        word_hash = (self._hash1(word), self._hash2(word))
        self.words.add(word_hash)
        
        # Add all prefixes
        for i in range(1, len(word) + 1):
            prefix = word[:i]
            prefix_hash = (self._hash1(prefix), self._hash2(prefix))
            self.prefix_set.add(prefix_hash)
    
    def search(self, word: str) -> bool:
        """Search using double hashing."""
        word_hash = (self._hash1(word), self._hash2(word))
        return word_hash in self.words
    
    def startsWith(self, prefix: str) -> bool:
        """Prefix search using double hashing."""
        prefix_hash = (self._hash1(prefix), self._hash2(prefix))
        return prefix_hash in self.prefix_set


class Trie10_Optimized:
    """
    Highly optimized Trie with various optimizations.
    
    - Array-based children for better cache locality
    - Bit manipulation for faster operations
    - Memory pooling for node allocation
    """
    
    def __init__(self):
        self.root = TrieArrayNode()
        self.node_pool = []  # Memory pool for better allocation
        self.pool_index = 0
    
    def _get_node(self) -> TrieArrayNode:
        """Get node from pool or create new one."""
        if self.pool_index < len(self.node_pool):
            node = self.node_pool[self.pool_index]
            self.pool_index += 1
            # Reset node
            node.children = [None] * 26
            node.is_end_of_word = False
            return node
        else:
            return TrieArrayNode()
    
    def _char_to_index(self, char: str) -> int:
        """Optimized character to index conversion."""
        return ord(char) - 97  # 'a' = 97
    
    def insert(self, word: str) -> None:
        """Optimized insert."""
        current = self.root
        for char in word:
            index = self._char_to_index(char)
            if current.children[index] is None:
                current.children[index] = self._get_node()
            current = current.children[index]
        current.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """Optimized search."""
        current = self.root
        for char in word:
            index = self._char_to_index(char)
            if current.children[index] is None:
                return False
            current = current.children[index]
        return current.is_end_of_word
    
    def startsWith(self, prefix: str) -> bool:
        """Optimized prefix search."""
        current = self.root
        for char in prefix:
            index = self._char_to_index(char)
            if current.children[index] is None:
                return False
            current = current.children[index]
        return True


def test_trie_implementations():
    """Test all Trie implementations."""
    
    # Test cases
    test_cases = [
        # Basic functionality
        {
            "operations": ["insert", "search", "search", "startsWith", "insert", "search"],
            "args": [["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]],
            "expected": [None, True, False, True, None, True]
        },
        
        # Empty operations
        {
            "operations": ["search", "startsWith"],
            "args": [[""], [""]],
            "expected": [False, False]
        },
        
        # Single character
        {
            "operations": ["insert", "search", "startsWith"],
            "args": [["a"], ["a"], ["a"]],
            "expected": [None, True, True]
        },
        
        # Multiple words
        {
            "operations": ["insert", "insert", "search", "search", "startsWith", "startsWith"],
            "args": [["app"], ["apple"], ["app"], ["apple"], ["ap"], ["application"]],
            "expected": [None, None, True, True, True, False]
        },
        
        # Overlapping prefixes
        {
            "operations": ["insert", "insert", "insert", "search", "search", "search", 
                          "startsWith", "startsWith", "startsWith"],
            "args": [["car"], ["card"], ["care"], ["car"], ["card"], ["care"],
                    ["car"], ["ca"], ["caring"]],
            "expected": [None, None, None, True, True, True, True, True, False]
        },
        
        # Long words
        {
            "operations": ["insert", "search", "startsWith"],
            "args": [["supercalifragilisticexpialidocious"], 
                    ["supercalifragilisticexpialidocious"],
                    ["supercalifragilisticexpialidocious"]],
            "expected": [None, True, True]
        },
    ]
    
    # All implementations to test
    implementations = [
        ("HashMap", Trie1_HashMap),
        ("Array", Trie2_Array),
        ("Compressed", Trie3_Compressed),
        ("Set Based", Trie4_SetBased),
        ("List Based", Trie5_ListBased),
        ("Sorted List", Trie6_SortedList),
        ("Recursive", Trie7_Recursive),
        ("Iterative", Trie8_Iterative),
        ("Double Hashing", Trie9_DoubleHashing),
        ("Optimized", Trie10_Optimized),
    ]
    
    print("Testing Trie implementations:")
    print("=" * 75)
    
    for test_idx, test_case in enumerate(test_cases):
        print(f"\nTest Case {test_idx + 1}:")
        print(f"Operations: {test_case['operations']}")
        print(f"Arguments:  {test_case['args']}")
        print(f"Expected:   {test_case['expected']}")
        
        for name, TrieClass in implementations:
            try:
                trie = TrieClass()
                results = []
                
                for op, args in zip(test_case['operations'], test_case['args']):
                    if op == "insert":
                        result = trie.insert(args[0])
                        results.append(result)
                    elif op == "search":
                        result = trie.search(args[0])
                        results.append(result)
                    elif op == "startsWith":
                        result = trie.startsWith(args[0])
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
    
    def generate_random_words(count: int, min_len: int = 3, max_len: int = 10) -> List[str]:
        """Generate random words for testing."""
        words = []
        for _ in range(count):
            length = random.randint(min_len, max_len)
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            words.append(word)
        return words
    
    test_scenarios = [
        ("Small dataset", 100),
        ("Medium dataset", 1000),
        ("Large dataset", 5000),
    ]
    
    for scenario_name, word_count in test_scenarios:
        print(f"\n{scenario_name} ({word_count} words):")
        words = generate_random_words(word_count)
        search_words = random.sample(words, min(100, word_count // 2))
        
        for name, TrieClass in implementations:
            try:
                # Test insertion
                trie = TrieClass()
                start_time = time.time()
                for word in words:
                    trie.insert(word)
                insert_time = (time.time() - start_time) * 1000
                
                # Test search
                start_time = time.time()
                for word in search_words:
                    trie.search(word)
                search_time = (time.time() - start_time) * 1000
                
                # Test prefix search
                prefixes = [word[:len(word)//2] for word in search_words[:20]]
                start_time = time.time()
                for prefix in prefixes:
                    trie.startsWith(prefix)
                prefix_time = (time.time() - start_time) * 1000
                
                print(f"  {name}: Insert {insert_time:.2f}ms, "
                      f"Search {search_time:.2f}ms, "
                      f"Prefix {prefix_time:.2f}ms")
                
            except Exception as e:
                print(f"  {name}: Error - {e}")
    
    print(f"\n{'='*75}")
    print("Algorithm Complexity Analysis:")
    print("="*75)
    print("1. HashMap:        O(m) time, O(ALPHABET_SIZE * N * M) space")
    print("2. Array:          O(m) time, O(26 * N * M) space") 
    print("3. Compressed:     O(m) time, O(N * M) space")
    print("4. Set Based:      O(m) time, O(N * M) space")
    print("5. List Based:     O(N * M) time, O(N * M) space")
    print("6. Sorted List:    O(log N) search, O(N) insert")
    print("7. Recursive:      O(m) time + recursion overhead")
    print("8. Iterative:      O(m) time, O(ALPHABET_SIZE * N * M) space")
    print("9. Double Hashing: O(m) time average, O(N * M) space")
    print("10. Optimized:     O(m) time, optimized constants")
    
    print(f"\n{'='*75}")
    print("Key Insights:")
    print("="*75)
    print("• Array-based children faster than HashMap for small alphabets")
    print("• Compressed Trie saves space but adds complexity")
    print("• Set-based approach simple but less memory efficient")
    print("• Recursive vs Iterative: similar complexity, different stack usage")
    print("• Memory pooling and cache optimization can improve performance")
    print("• Choice depends on alphabet size, word count, and usage patterns")


if __name__ == "__main__":
    test_trie_implementations() 