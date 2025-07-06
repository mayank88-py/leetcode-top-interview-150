"""
28. Find the Index of the First Occurrence in a String

Problem:
Given two strings needle and haystack, return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

Example 1:
Input: haystack = "sadbutsad", needle = "sad"
Output: 0
Explanation: "sad" occurs at index 0 and 6.
The first occurrence is at index 0, so we return 0.

Example 2:
Input: haystack = "leetcode", needle = "leeto"
Output: -1
Explanation: "leeto" did not occur in "leetcode", so we return -1.

Time Complexity: O(n*m) for brute force, O(n+m) for KMP
Space Complexity: O(m) for KMP preprocessing
"""


def str_str(haystack, needle):
    """
    Find first occurrence of needle in haystack using built-in find.
    
    Args:
        haystack: String to search in
        needle: String to search for
    
    Returns:
        Index of first occurrence, or -1 if not found
    """
    return haystack.find(needle)


def str_str_brute_force(haystack, needle):
    """
    Find first occurrence of needle in haystack using brute force.
    
    Args:
        haystack: String to search in
        needle: String to search for
    
    Returns:
        Index of first occurrence, or -1 if not found
    """
    if not needle:
        return 0
    
    if len(needle) > len(haystack):
        return -1
    
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i+len(needle)] == needle:
            return i
    
    return -1


def str_str_two_pointers(haystack, needle):
    """
    Find first occurrence of needle in haystack using two pointers.
    
    Args:
        haystack: String to search in
        needle: String to search for
    
    Returns:
        Index of first occurrence, or -1 if not found
    """
    if not needle:
        return 0
    
    if len(needle) > len(haystack):
        return -1
    
    for i in range(len(haystack) - len(needle) + 1):
        j = 0
        while j < len(needle) and haystack[i + j] == needle[j]:
            j += 1
        
        if j == len(needle):
            return i
    
    return -1


def str_str_kmp(haystack, needle):
    """
    Find first occurrence of needle in haystack using KMP algorithm.
    
    Args:
        haystack: String to search in
        needle: String to search for
    
    Returns:
        Index of first occurrence, or -1 if not found
    """
    if not needle:
        return 0
    
    if len(needle) > len(haystack):
        return -1
    
    # Build failure function (prefix table)
    def build_failure_function(pattern):
        failure = [0] * len(pattern)
        j = 0
        
        for i in range(1, len(pattern)):
            while j > 0 and pattern[i] != pattern[j]:
                j = failure[j - 1]
            
            if pattern[i] == pattern[j]:
                j += 1
            
            failure[i] = j
        
        return failure
    
    failure = build_failure_function(needle)
    
    # Search using KMP
    i = j = 0
    while i < len(haystack):
        if haystack[i] == needle[j]:
            i += 1
            j += 1
        
        if j == len(needle):
            return i - j
        elif i < len(haystack) and haystack[i] != needle[j]:
            if j != 0:
                j = failure[j - 1]
            else:
                i += 1
    
    return -1


def str_str_rabin_karp(haystack, needle):
    """
    Find first occurrence of needle in haystack using Rabin-Karp algorithm.
    
    Args:
        haystack: String to search in
        needle: String to search for
    
    Returns:
        Index of first occurrence, or -1 if not found
    """
    if not needle:
        return 0
    
    if len(needle) > len(haystack):
        return -1
    
    # Rolling hash parameters
    base = 256
    mod = 10**9 + 7
    
    # Calculate hash of needle
    needle_hash = 0
    power = 1
    for char in needle:
        needle_hash = (needle_hash * base + ord(char)) % mod
    
    # Calculate power for rolling hash
    for _ in range(len(needle) - 1):
        power = (power * base) % mod
    
    # Rolling hash for haystack
    window_hash = 0
    
    for i in range(len(haystack)):
        # Add new character
        window_hash = (window_hash * base + ord(haystack[i])) % mod
        
        # Remove old character if window is full
        if i >= len(needle):
            window_hash = (window_hash - ord(haystack[i - len(needle)]) * power) % mod
        
        # Check if hashes match and verify actual string
        if i >= len(needle) - 1 and window_hash == needle_hash:
            if haystack[i - len(needle) + 1:i + 1] == needle:
                return i - len(needle) + 1
    
    return -1


def str_str_boyer_moore(haystack, needle):
    """
    Find first occurrence of needle in haystack using Boyer-Moore algorithm.
    
    Args:
        haystack: String to search in
        needle: String to search for
    
    Returns:
        Index of first occurrence, or -1 if not found
    """
    if not needle:
        return 0
    
    if len(needle) > len(haystack):
        return -1
    
    # Bad character rule
    def build_bad_char_table(pattern):
        table = {}
        for i, char in enumerate(pattern):
            table[char] = i
        return table
    
    bad_char = build_bad_char_table(needle)
    
    # Search
    i = len(needle) - 1
    while i < len(haystack):
        j = len(needle) - 1
        k = i
        
        while j >= 0 and haystack[k] == needle[j]:
            j -= 1
            k -= 1
        
        if j == -1:
            return k + 1
        
        # Bad character rule
        bad_char_shift = j - bad_char.get(haystack[k], -1)
        i += max(1, bad_char_shift)
    
    return -1


def str_str_z_algorithm(haystack, needle):
    """
    Find first occurrence of needle in haystack using Z algorithm.
    
    Args:
        haystack: String to search in
        needle: String to search for
    
    Returns:
        Index of first occurrence, or -1 if not found
    """
    if not needle:
        return 0
    
    if len(needle) > len(haystack):
        return -1
    
    # Create combined string
    combined = needle + "$" + haystack
    
    # Z algorithm
    def z_algorithm(s):
        n = len(s)
        z = [0] * n
        l = r = 0
        
        for i in range(1, n):
            if i > r:
                l = r = i
                while r < n and s[r - l] == s[r]:
                    r += 1
                z[i] = r - l
                r -= 1
            else:
                k = i - l
                if z[k] < r - i + 1:
                    z[i] = z[k]
                else:
                    l = i
                    while r < n and s[r - l] == s[r]:
                        r += 1
                    z[i] = r - l
                    r -= 1
        
        return z
    
    z = z_algorithm(combined)
    
    # Find first occurrence
    for i in range(len(needle) + 1, len(combined)):
        if z[i] == len(needle):
            return i - len(needle) - 1
    
    return -1


def str_str_sliding_window(haystack, needle):
    """
    Find first occurrence of needle in haystack using sliding window.
    
    Args:
        haystack: String to search in
        needle: String to search for
    
    Returns:
        Index of first occurrence, or -1 if not found
    """
    if not needle:
        return 0
    
    if len(needle) > len(haystack):
        return -1
    
    window_size = len(needle)
    
    for i in range(len(haystack) - window_size + 1):
        window = haystack[i:i + window_size]
        if window == needle:
            return i
    
    return -1


def str_str_recursive(haystack, needle):
    """
    Find first occurrence of needle in haystack using recursion.
    
    Args:
        haystack: String to search in
        needle: String to search for
    
    Returns:
        Index of first occurrence, or -1 if not found
    """
    if not needle:
        return 0
    
    def search_recursive(h_idx, n_idx):
        # If we've matched all characters in needle
        if n_idx == len(needle):
            return h_idx - len(needle)
        
        # If we've reached end of haystack
        if h_idx == len(haystack):
            return -1
        
        # If characters match, continue matching
        if haystack[h_idx] == needle[n_idx]:
            return search_recursive(h_idx + 1, n_idx + 1)
        
        # If characters don't match, try next position in haystack
        # But only if we haven't started matching yet
        if n_idx == 0:
            return search_recursive(h_idx + 1, 0)
        else:
            # Reset needle index and try from current haystack position
            return search_recursive(h_idx - n_idx + 1, 0)
    
    return search_recursive(0, 0)


# Test cases
if __name__ == "__main__":
    # Test case 1
    haystack1 = "sadbutsad"
    needle1 = "sad"
    result1a = str_str(haystack1, needle1)
    result1b = str_str_brute_force(haystack1, needle1)
    result1c = str_str_two_pointers(haystack1, needle1)
    result1d = str_str_kmp(haystack1, needle1)
    result1e = str_str_rabin_karp(haystack1, needle1)
    result1f = str_str_boyer_moore(haystack1, needle1)
    result1g = str_str_z_algorithm(haystack1, needle1)
    result1h = str_str_sliding_window(haystack1, needle1)
    result1i = str_str_recursive(haystack1, needle1)
    print(f"Test 1 - Haystack: '{haystack1}', Needle: '{needle1}', Expected: 0")
    print(f"BuiltIn: {result1a}, BruteForce: {result1b}, TwoPointers: {result1c}, KMP: {result1d}, RabinKarp: {result1e}, BoyerMoore: {result1f}, ZAlgorithm: {result1g}, SlidingWindow: {result1h}, Recursive: {result1i}")
    print()
    
    # Test case 2
    haystack2 = "leetcode"
    needle2 = "leeto"
    result2a = str_str(haystack2, needle2)
    result2b = str_str_brute_force(haystack2, needle2)
    result2c = str_str_two_pointers(haystack2, needle2)
    result2d = str_str_kmp(haystack2, needle2)
    result2e = str_str_rabin_karp(haystack2, needle2)
    result2f = str_str_boyer_moore(haystack2, needle2)
    result2g = str_str_z_algorithm(haystack2, needle2)
    result2h = str_str_sliding_window(haystack2, needle2)
    result2i = str_str_recursive(haystack2, needle2)
    print(f"Test 2 - Haystack: '{haystack2}', Needle: '{needle2}', Expected: -1")
    print(f"BuiltIn: {result2a}, BruteForce: {result2b}, TwoPointers: {result2c}, KMP: {result2d}, RabinKarp: {result2e}, BoyerMoore: {result2f}, ZAlgorithm: {result2g}, SlidingWindow: {result2h}, Recursive: {result2i}")
    print()
    
    # Test case 3 - Empty needle
    haystack3 = "hello"
    needle3 = ""
    result3a = str_str(haystack3, needle3)
    result3b = str_str_brute_force(haystack3, needle3)
    result3c = str_str_two_pointers(haystack3, needle3)
    result3d = str_str_kmp(haystack3, needle3)
    result3e = str_str_rabin_karp(haystack3, needle3)
    result3f = str_str_boyer_moore(haystack3, needle3)
    result3g = str_str_z_algorithm(haystack3, needle3)
    result3h = str_str_sliding_window(haystack3, needle3)
    result3i = str_str_recursive(haystack3, needle3)
    print(f"Test 3 - Haystack: '{haystack3}', Needle: '{needle3}', Expected: 0")
    print(f"BuiltIn: {result3a}, BruteForce: {result3b}, TwoPointers: {result3c}, KMP: {result3d}, RabinKarp: {result3e}, BoyerMoore: {result3f}, ZAlgorithm: {result3g}, SlidingWindow: {result3h}, Recursive: {result3i}")
    print()
    
    # Test case 4 - Single character
    haystack4 = "a"
    needle4 = "a"
    result4a = str_str(haystack4, needle4)
    result4b = str_str_brute_force(haystack4, needle4)
    result4c = str_str_two_pointers(haystack4, needle4)
    result4d = str_str_kmp(haystack4, needle4)
    result4e = str_str_rabin_karp(haystack4, needle4)
    result4f = str_str_boyer_moore(haystack4, needle4)
    result4g = str_str_z_algorithm(haystack4, needle4)
    result4h = str_str_sliding_window(haystack4, needle4)
    result4i = str_str_recursive(haystack4, needle4)
    print(f"Test 4 - Haystack: '{haystack4}', Needle: '{needle4}', Expected: 0")
    print(f"BuiltIn: {result4a}, BruteForce: {result4b}, TwoPointers: {result4c}, KMP: {result4d}, RabinKarp: {result4e}, BoyerMoore: {result4f}, ZAlgorithm: {result4g}, SlidingWindow: {result4h}, Recursive: {result4i}")
    print()
    
    # Test case 5 - Needle longer than haystack
    haystack5 = "abc"
    needle5 = "abcdef"
    result5a = str_str(haystack5, needle5)
    result5b = str_str_brute_force(haystack5, needle5)
    result5c = str_str_two_pointers(haystack5, needle5)
    result5d = str_str_kmp(haystack5, needle5)
    result5e = str_str_rabin_karp(haystack5, needle5)
    result5f = str_str_boyer_moore(haystack5, needle5)
    result5g = str_str_z_algorithm(haystack5, needle5)
    result5h = str_str_sliding_window(haystack5, needle5)
    result5i = str_str_recursive(haystack5, needle5)
    print(f"Test 5 - Haystack: '{haystack5}', Needle: '{needle5}', Expected: -1")
    print(f"BuiltIn: {result5a}, BruteForce: {result5b}, TwoPointers: {result5c}, KMP: {result5d}, RabinKarp: {result5e}, BoyerMoore: {result5f}, ZAlgorithm: {result5g}, SlidingWindow: {result5h}, Recursive: {result5i}")
    print()
    
    # Test case 6 - Repeated pattern
    haystack6 = "aaaaaab"
    needle6 = "aaab"
    result6a = str_str(haystack6, needle6)
    result6b = str_str_brute_force(haystack6, needle6)
    result6c = str_str_two_pointers(haystack6, needle6)
    result6d = str_str_kmp(haystack6, needle6)
    result6e = str_str_rabin_karp(haystack6, needle6)
    result6f = str_str_boyer_moore(haystack6, needle6)
    result6g = str_str_z_algorithm(haystack6, needle6)
    result6h = str_str_sliding_window(haystack6, needle6)
    result6i = str_str_recursive(haystack6, needle6)
    print(f"Test 6 - Haystack: '{haystack6}', Needle: '{needle6}', Expected: 3")
    print(f"BuiltIn: {result6a}, BruteForce: {result6b}, TwoPointers: {result6c}, KMP: {result6d}, RabinKarp: {result6e}, BoyerMoore: {result6f}, ZAlgorithm: {result6g}, SlidingWindow: {result6h}, Recursive: {result6i}")
    print()
    
    # Test case 7 - At the end
    haystack7 = "programming"
    needle7 = "ing"
    result7a = str_str(haystack7, needle7)
    result7b = str_str_brute_force(haystack7, needle7)
    result7c = str_str_two_pointers(haystack7, needle7)
    result7d = str_str_kmp(haystack7, needle7)
    result7e = str_str_rabin_karp(haystack7, needle7)
    result7f = str_str_boyer_moore(haystack7, needle7)
    result7g = str_str_z_algorithm(haystack7, needle7)
    result7h = str_str_sliding_window(haystack7, needle7)
    result7i = str_str_recursive(haystack7, needle7)
    print(f"Test 7 - Haystack: '{haystack7}', Needle: '{needle7}', Expected: 8")
    print(f"BuiltIn: {result7a}, BruteForce: {result7b}, TwoPointers: {result7c}, KMP: {result7d}, RabinKarp: {result7e}, BoyerMoore: {result7f}, ZAlgorithm: {result7g}, SlidingWindow: {result7h}, Recursive: {result7i}")
    print()
    
    # Test case 8 - Multiple occurrences
    haystack8 = "ababcababa"
    needle8 = "ababa"
    result8a = str_str(haystack8, needle8)
    result8b = str_str_brute_force(haystack8, needle8)
    result8c = str_str_two_pointers(haystack8, needle8)
    result8d = str_str_kmp(haystack8, needle8)
    result8e = str_str_rabin_karp(haystack8, needle8)
    result8f = str_str_boyer_moore(haystack8, needle8)
    result8g = str_str_z_algorithm(haystack8, needle8)
    result8h = str_str_sliding_window(haystack8, needle8)
    result8i = str_str_recursive(haystack8, needle8)
    print(f"Test 8 - Haystack: '{haystack8}', Needle: '{needle8}', Expected: 5")
    print(f"BuiltIn: {result8a}, BruteForce: {result8b}, TwoPointers: {result8c}, KMP: {result8d}, RabinKarp: {result8e}, BoyerMoore: {result8f}, ZAlgorithm: {result8g}, SlidingWindow: {result8h}, Recursive: {result8i}")
    print()
    
    # Test case 9 - Same string
    haystack9 = "hello"
    needle9 = "hello"
    result9a = str_str(haystack9, needle9)
    result9b = str_str_brute_force(haystack9, needle9)
    result9c = str_str_two_pointers(haystack9, needle9)
    result9d = str_str_kmp(haystack9, needle9)
    result9e = str_str_rabin_karp(haystack9, needle9)
    result9f = str_str_boyer_moore(haystack9, needle9)
    result9g = str_str_z_algorithm(haystack9, needle9)
    result9h = str_str_sliding_window(haystack9, needle9)
    result9i = str_str_recursive(haystack9, needle9)
    print(f"Test 9 - Haystack: '{haystack9}', Needle: '{needle9}', Expected: 0")
    print(f"BuiltIn: {result9a}, BruteForce: {result9b}, TwoPointers: {result9c}, KMP: {result9d}, RabinKarp: {result9e}, BoyerMoore: {result9f}, ZAlgorithm: {result9g}, SlidingWindow: {result9h}, Recursive: {result9i}")
    print()
    
    # Test case 10 - Partial match
    haystack10 = "mississippi"
    needle10 = "issip"
    result10a = str_str(haystack10, needle10)
    result10b = str_str_brute_force(haystack10, needle10)
    result10c = str_str_two_pointers(haystack10, needle10)
    result10d = str_str_kmp(haystack10, needle10)
    result10e = str_str_rabin_karp(haystack10, needle10)
    result10f = str_str_boyer_moore(haystack10, needle10)
    result10g = str_str_z_algorithm(haystack10, needle10)
    result10h = str_str_sliding_window(haystack10, needle10)
    result10i = str_str_recursive(haystack10, needle10)
    print(f"Test 10 - Haystack: '{haystack10}', Needle: '{needle10}', Expected: 4")
    print(f"BuiltIn: {result10a}, BruteForce: {result10b}, TwoPointers: {result10c}, KMP: {result10d}, RabinKarp: {result10e}, BoyerMoore: {result10f}, ZAlgorithm: {result10g}, SlidingWindow: {result10h}, Recursive: {result10i}") 