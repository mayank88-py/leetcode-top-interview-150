"""
14. Longest Common Prefix

Problem:
Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

Example 1:
Input: strs = ["flower","flow","flight"]
Output: "fl"

Example 2:
Input: strs = ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.

Time Complexity: O(S) where S is the sum of all characters in all strings
Space Complexity: O(1)
"""


def longest_common_prefix(strs):
    """
    Find longest common prefix using vertical scanning.
    
    Args:
        strs: List of strings
    
    Returns:
        Longest common prefix string
    """
    if not strs:
        return ""
    
    # Compare characters column by column
    for i in range(len(strs[0])):
        char = strs[0][i]
        
        # Check if this character is present at same position in all strings
        for j in range(1, len(strs)):
            if i >= len(strs[j]) or strs[j][i] != char:
                return strs[0][:i]
    
    return strs[0]


def longest_common_prefix_horizontal(strs):
    """
    Find longest common prefix using horizontal scanning.
    
    Args:
        strs: List of strings
    
    Returns:
        Longest common prefix string
    """
    if not strs:
        return ""
    
    prefix = strs[0]
    
    for i in range(1, len(strs)):
        while strs[i].find(prefix) != 0:
            prefix = prefix[:-1]
            if not prefix:
                return ""
    
    return prefix


def longest_common_prefix_divide_conquer(strs):
    """
    Find longest common prefix using divide and conquer.
    
    Args:
        strs: List of strings
    
    Returns:
        Longest common prefix string
    """
    if not strs:
        return ""
    
    def common_prefix(left, right):
        """Find common prefix between two strings"""
        min_len = min(len(left), len(right))
        for i in range(min_len):
            if left[i] != right[i]:
                return left[:i]
        return left[:min_len]
    
    def divide_conquer(start, end):
        """Divide and conquer approach"""
        if start == end:
            return strs[start]
        
        mid = (start + end) // 2
        left_prefix = divide_conquer(start, mid)
        right_prefix = divide_conquer(mid + 1, end)
        
        return common_prefix(left_prefix, right_prefix)
    
    return divide_conquer(0, len(strs) - 1)


def longest_common_prefix_binary_search(strs):
    """
    Find longest common prefix using binary search.
    
    Args:
        strs: List of strings
    
    Returns:
        Longest common prefix string
    """
    if not strs:
        return ""
    
    def is_common_prefix(length):
        """Check if prefix of given length is common to all strings"""
        prefix = strs[0][:length]
        for i in range(1, len(strs)):
            if not strs[i].startswith(prefix):
                return False
        return True
    
    min_len = min(len(s) for s in strs)
    low, high = 0, min_len
    
    while low < high:
        mid = (low + high + 1) // 2
        if is_common_prefix(mid):
            low = mid
        else:
            high = mid - 1
    
    return strs[0][:low]


def longest_common_prefix_trie(strs):
    """
    Find longest common prefix using trie data structure.
    
    Args:
        strs: List of strings
    
    Returns:
        Longest common prefix string
    """
    if not strs:
        return ""
    
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end = False
    
    # Build trie
    root = TrieNode()
    for s in strs:
        node = root
        for char in s:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    # Find longest common prefix
    prefix = ""
    node = root
    
    while len(node.children) == 1 and not node.is_end:
        char = next(iter(node.children))
        prefix += char
        node = node.children[char]
    
    return prefix


def longest_common_prefix_zip(strs):
    """
    Find longest common prefix using zip function.
    
    Args:
        strs: List of strings
    
    Returns:
        Longest common prefix string
    """
    if not strs:
        return ""
    
    # Zip all strings together
    prefix = ""
    for chars in zip(*strs):
        if len(set(chars)) == 1:
            prefix += chars[0]
        else:
            break
    
    return prefix


def longest_common_prefix_recursive(strs):
    """
    Find longest common prefix using recursion.
    
    Args:
        strs: List of strings
    
    Returns:
        Longest common prefix string
    """
    if not strs:
        return ""
    
    if len(strs) == 1:
        return strs[0]
    
    # Find common prefix between first two strings
    def find_common(s1, s2):
        i = 0
        while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
            i += 1
        return s1[:i]
    
    # Recursively find prefix
    prefix = find_common(strs[0], strs[1])
    
    for i in range(2, len(strs)):
        prefix = find_common(prefix, strs[i])
        if not prefix:
            break
    
    return prefix


def longest_common_prefix_sort(strs):
    """
    Find longest common prefix using sorting.
    
    Args:
        strs: List of strings
    
    Returns:
        Longest common prefix string
    """
    if not strs:
        return ""
    
    # Sort the strings
    strs.sort()
    
    # Compare first and last strings
    first = strs[0]
    last = strs[-1]
    
    prefix = ""
    for i in range(min(len(first), len(last))):
        if first[i] == last[i]:
            prefix += first[i]
        else:
            break
    
    return prefix


def longest_common_prefix_functional(strs):
    """
    Find longest common prefix using functional programming.
    
    Args:
        strs: List of strings
    
    Returns:
        Longest common prefix string
    """
    if not strs:
        return ""
    
    from functools import reduce
    
    def common_prefix_two(s1, s2):
        """Find common prefix between two strings"""
        i = 0
        while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
            i += 1
        return s1[:i]
    
    return reduce(common_prefix_two, strs)


# Test cases
if __name__ == "__main__":
    # Test case 1
    strs1 = ["flower", "flow", "flight"]
    result1a = longest_common_prefix(strs1)
    result1b = longest_common_prefix_horizontal(strs1)
    result1c = longest_common_prefix_divide_conquer(strs1)
    result1d = longest_common_prefix_binary_search(strs1)
    result1e = longest_common_prefix_trie(strs1)
    result1f = longest_common_prefix_zip(strs1)
    result1g = longest_common_prefix_recursive(strs1)
    result1h = longest_common_prefix_sort(strs1.copy())
    result1i = longest_common_prefix_functional(strs1)
    print(f"Test 1 - Strings: {strs1}, Expected: 'fl'")
    print(f"Vertical: '{result1a}', Horizontal: '{result1b}', DivideConquer: '{result1c}', BinarySearch: '{result1d}', Trie: '{result1e}', Zip: '{result1f}', Recursive: '{result1g}', Sort: '{result1h}', Functional: '{result1i}'")
    print()
    
    # Test case 2
    strs2 = ["dog", "racecar", "car"]
    result2a = longest_common_prefix(strs2)
    result2b = longest_common_prefix_horizontal(strs2)
    result2c = longest_common_prefix_divide_conquer(strs2)
    result2d = longest_common_prefix_binary_search(strs2)
    result2e = longest_common_prefix_trie(strs2)
    result2f = longest_common_prefix_zip(strs2)
    result2g = longest_common_prefix_recursive(strs2)
    result2h = longest_common_prefix_sort(strs2.copy())
    result2i = longest_common_prefix_functional(strs2)
    print(f"Test 2 - Strings: {strs2}, Expected: ''")
    print(f"Vertical: '{result2a}', Horizontal: '{result2b}', DivideConquer: '{result2c}', BinarySearch: '{result2d}', Trie: '{result2e}', Zip: '{result2f}', Recursive: '{result2g}', Sort: '{result2h}', Functional: '{result2i}'")
    print()
    
    # Test case 3 - Empty array
    strs3 = []
    result3a = longest_common_prefix(strs3)
    result3b = longest_common_prefix_horizontal(strs3)
    result3c = longest_common_prefix_divide_conquer(strs3)
    result3d = longest_common_prefix_binary_search(strs3)
    result3e = longest_common_prefix_trie(strs3)
    result3f = longest_common_prefix_zip(strs3)
    result3g = longest_common_prefix_recursive(strs3)
    result3h = longest_common_prefix_sort(strs3.copy())
    result3i = longest_common_prefix_functional(strs3)
    print(f"Test 3 - Strings: {strs3}, Expected: ''")
    print(f"Vertical: '{result3a}', Horizontal: '{result3b}', DivideConquer: '{result3c}', BinarySearch: '{result3d}', Trie: '{result3e}', Zip: '{result3f}', Recursive: '{result3g}', Sort: '{result3h}', Functional: '{result3i}'")
    print()
    
    # Test case 4 - Single string
    strs4 = ["abc"]
    result4a = longest_common_prefix(strs4)
    result4b = longest_common_prefix_horizontal(strs4)
    result4c = longest_common_prefix_divide_conquer(strs4)
    result4d = longest_common_prefix_binary_search(strs4)
    result4e = longest_common_prefix_trie(strs4)
    result4f = longest_common_prefix_zip(strs4)
    result4g = longest_common_prefix_recursive(strs4)
    result4h = longest_common_prefix_sort(strs4.copy())
    result4i = longest_common_prefix_functional(strs4)
    print(f"Test 4 - Strings: {strs4}, Expected: 'abc'")
    print(f"Vertical: '{result4a}', Horizontal: '{result4b}', DivideConquer: '{result4c}', BinarySearch: '{result4d}', Trie: '{result4e}', Zip: '{result4f}', Recursive: '{result4g}', Sort: '{result4h}', Functional: '{result4i}'")
    print()
    
    # Test case 5 - All same strings
    strs5 = ["abc", "abc", "abc"]
    result5a = longest_common_prefix(strs5)
    result5b = longest_common_prefix_horizontal(strs5)
    result5c = longest_common_prefix_divide_conquer(strs5)
    result5d = longest_common_prefix_binary_search(strs5)
    result5e = longest_common_prefix_trie(strs5)
    result5f = longest_common_prefix_zip(strs5)
    result5g = longest_common_prefix_recursive(strs5)
    result5h = longest_common_prefix_sort(strs5.copy())
    result5i = longest_common_prefix_functional(strs5)
    print(f"Test 5 - Strings: {strs5}, Expected: 'abc'")
    print(f"Vertical: '{result5a}', Horizontal: '{result5b}', DivideConquer: '{result5c}', BinarySearch: '{result5d}', Trie: '{result5e}', Zip: '{result5f}', Recursive: '{result5g}', Sort: '{result5h}', Functional: '{result5i}'")
    print()
    
    # Test case 6 - One empty string
    strs6 = ["", "abc", "abcd"]
    result6a = longest_common_prefix(strs6)
    result6b = longest_common_prefix_horizontal(strs6)
    result6c = longest_common_prefix_divide_conquer(strs6)
    result6d = longest_common_prefix_binary_search(strs6)
    result6e = longest_common_prefix_trie(strs6)
    result6f = longest_common_prefix_zip(strs6)
    result6g = longest_common_prefix_recursive(strs6)
    result6h = longest_common_prefix_sort(strs6.copy())
    result6i = longest_common_prefix_functional(strs6)
    print(f"Test 6 - Strings: {strs6}, Expected: ''")
    print(f"Vertical: '{result6a}', Horizontal: '{result6b}', DivideConquer: '{result6c}', BinarySearch: '{result6d}', Trie: '{result6e}', Zip: '{result6f}', Recursive: '{result6g}', Sort: '{result6h}', Functional: '{result6i}'")
    print()
    
    # Test case 7 - Different lengths
    strs7 = ["abcdef", "abc", "abcde"]
    result7a = longest_common_prefix(strs7)
    result7b = longest_common_prefix_horizontal(strs7)
    result7c = longest_common_prefix_divide_conquer(strs7)
    result7d = longest_common_prefix_binary_search(strs7)
    result7e = longest_common_prefix_trie(strs7)
    result7f = longest_common_prefix_zip(strs7)
    result7g = longest_common_prefix_recursive(strs7)
    result7h = longest_common_prefix_sort(strs7.copy())
    result7i = longest_common_prefix_functional(strs7)
    print(f"Test 7 - Strings: {strs7}, Expected: 'abc'")
    print(f"Vertical: '{result7a}', Horizontal: '{result7b}', DivideConquer: '{result7c}', BinarySearch: '{result7d}', Trie: '{result7e}', Zip: '{result7f}', Recursive: '{result7g}', Sort: '{result7h}', Functional: '{result7i}'")
    print()
    
    # Test case 8 - Two strings
    strs8 = ["programming", "program"]
    result8a = longest_common_prefix(strs8)
    result8b = longest_common_prefix_horizontal(strs8)
    result8c = longest_common_prefix_divide_conquer(strs8)
    result8d = longest_common_prefix_binary_search(strs8)
    result8e = longest_common_prefix_trie(strs8)
    result8f = longest_common_prefix_zip(strs8)
    result8g = longest_common_prefix_recursive(strs8)
    result8h = longest_common_prefix_sort(strs8.copy())
    result8i = longest_common_prefix_functional(strs8)
    print(f"Test 8 - Strings: {strs8}, Expected: 'program'")
    print(f"Vertical: '{result8a}', Horizontal: '{result8b}', DivideConquer: '{result8c}', BinarySearch: '{result8d}', Trie: '{result8e}', Zip: '{result8f}', Recursive: '{result8g}', Sort: '{result8h}', Functional: '{result8i}'")
    print()
    
    # Test case 9 - Single characters
    strs9 = ["a", "aa", "aaa"]
    result9a = longest_common_prefix(strs9)
    result9b = longest_common_prefix_horizontal(strs9)
    result9c = longest_common_prefix_divide_conquer(strs9)
    result9d = longest_common_prefix_binary_search(strs9)
    result9e = longest_common_prefix_trie(strs9)
    result9f = longest_common_prefix_zip(strs9)
    result9g = longest_common_prefix_recursive(strs9)
    result9h = longest_common_prefix_sort(strs9.copy())
    result9i = longest_common_prefix_functional(strs9)
    print(f"Test 9 - Strings: {strs9}, Expected: 'a'")
    print(f"Vertical: '{result9a}', Horizontal: '{result9b}', DivideConquer: '{result9c}', BinarySearch: '{result9d}', Trie: '{result9e}', Zip: '{result9f}', Recursive: '{result9g}', Sort: '{result9h}', Functional: '{result9i}'")
    print()
    
    # Test case 10 - Long common prefix
    strs10 = ["international", "internet", "internal"]
    result10a = longest_common_prefix(strs10)
    result10b = longest_common_prefix_horizontal(strs10)
    result10c = longest_common_prefix_divide_conquer(strs10)
    result10d = longest_common_prefix_binary_search(strs10)
    result10e = longest_common_prefix_trie(strs10)
    result10f = longest_common_prefix_zip(strs10)
    result10g = longest_common_prefix_recursive(strs10)
    result10h = longest_common_prefix_sort(strs10.copy())
    result10i = longest_common_prefix_functional(strs10)
    print(f"Test 10 - Strings: {strs10}, Expected: 'inter'")
    print(f"Vertical: '{result10a}', Horizontal: '{result10b}', DivideConquer: '{result10c}', BinarySearch: '{result10d}', Trie: '{result10e}', Zip: '{result10f}', Recursive: '{result10g}', Sort: '{result10h}', Functional: '{result10i}'") 