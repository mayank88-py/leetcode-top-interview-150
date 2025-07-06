"""
392. Is Subsequence

Problem:
Given two strings s and t, return true if s is a subsequence of t, or false otherwise.
A subsequence of a string is a new string that is formed from the original string by deleting some 
(can be zero) of the characters without disturbing the relative positions of the remaining characters.

Example 1:
Input: s = "abc", t = "aebdc"
Output: true

Example 2:
Input: s = "axc", t = "ahbgdc"
Output: false

Follow up: Suppose there are lots of incoming s, say s1, s2, ..., sk where k >= 10^9, and you want to 
check one by one to see if t has its subsequence. In this scenario, how would you design your algorithm?

Time Complexity: O(n) where n is length of t
Space Complexity: O(1)
"""


def is_subsequence(s, t):
    """
    Check if s is a subsequence of t using two pointers.
    
    Args:
        s: Source string (subsequence to find)
        t: Target string (string to search in)
    
    Returns:
        True if s is a subsequence of t, False otherwise
    """
    if not s:  # Empty string is subsequence of any string
        return True
    if not t:  # Non-empty s cannot be subsequence of empty t
        return False
    
    s_ptr = 0  # Pointer for string s
    t_ptr = 0  # Pointer for string t
    
    while t_ptr < len(t) and s_ptr < len(s):
        # If characters match, move both pointers
        if s[s_ptr] == t[t_ptr]:
            s_ptr += 1
        
        # Always move t pointer
        t_ptr += 1
    
    # If we've matched all characters in s
    return s_ptr == len(s)


def is_subsequence_recursive(s, t):
    """
    Check if s is a subsequence of t using recursion.
    
    Args:
        s: Source string (subsequence to find)
        t: Target string (string to search in)
    
    Returns:
        True if s is a subsequence of t, False otherwise
    """
    def helper(s_idx, t_idx):
        # Base cases
        if s_idx == len(s):  # Found all characters of s
            return True
        if t_idx == len(t):  # Reached end of t without finding all of s
            return False
        
        # If characters match, move both pointers
        if s[s_idx] == t[t_idx]:
            return helper(s_idx + 1, t_idx + 1)
        else:
            # Only move t pointer
            return helper(s_idx, t_idx + 1)
    
    return helper(0, 0)


def is_subsequence_dp(s, t):
    """
    Check if s is a subsequence of t using dynamic programming.
    
    Args:
        s: Source string (subsequence to find)
        t: Target string (string to search in)
    
    Returns:
        True if s is a subsequence of t, False otherwise
    """
    if not s:
        return True
    if not t:
        return False
    
    m, n = len(s), len(t)
    # dp[i][j] = True if s[0:i] is subsequence of t[0:j]
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    
    # Empty string is subsequence of any string
    for j in range(n + 1):
        dp[0][j] = True
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = dp[i][j - 1]
    
    return dp[m][n]


def is_subsequence_binary_search(s, t):
    """
    Check if s is a subsequence of t using binary search (optimized for multiple queries).
    
    Args:
        s: Source string (subsequence to find)
        t: Target string (string to search in)
    
    Returns:
        True if s is a subsequence of t, False otherwise
    """
    if not s:
        return True
    if not t:
        return False
    
    # Build index map for each character in t
    char_indices = {}
    for i, char in enumerate(t):
        if char not in char_indices:
            char_indices[char] = []
        char_indices[char].append(i)
    
    def binary_search(arr, target):
        """Find the first index in arr that is > target"""
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] <= target:
                left = mid + 1
            else:
                right = mid
        return left
    
    prev_index = -1
    for char in s:
        if char not in char_indices:
            return False
        
        # Find the first occurrence of char after prev_index
        indices = char_indices[char]
        pos = binary_search(indices, prev_index)
        
        if pos == len(indices):
            return False
        
        prev_index = indices[pos]
    
    return True


# Test cases
if __name__ == "__main__":
    # Test case 1
    s1, t1 = "abc", "aebdc"
    result1a = is_subsequence(s1, t1)
    result1b = is_subsequence_recursive(s1, t1)
    result1c = is_subsequence_dp(s1, t1)
    result1d = is_subsequence_binary_search(s1, t1)
    print(f"Test 1 - Expected: True, Two Pointers: {result1a}, Recursive: {result1b}, DP: {result1c}, Binary Search: {result1d}")
    
    # Test case 2
    s2, t2 = "axc", "ahbgdc"
    result2a = is_subsequence(s2, t2)
    result2b = is_subsequence_recursive(s2, t2)
    result2c = is_subsequence_dp(s2, t2)
    result2d = is_subsequence_binary_search(s2, t2)
    print(f"Test 2 - Expected: False, Two Pointers: {result2a}, Recursive: {result2b}, DP: {result2c}, Binary Search: {result2d}")
    
    # Test case 3 - Empty s
    s3, t3 = "", "abc"
    result3a = is_subsequence(s3, t3)
    result3b = is_subsequence_recursive(s3, t3)
    result3c = is_subsequence_dp(s3, t3)
    result3d = is_subsequence_binary_search(s3, t3)
    print(f"Test 3 - Expected: True, Two Pointers: {result3a}, Recursive: {result3b}, DP: {result3c}, Binary Search: {result3d}")
    
    # Test case 4 - Empty t
    s4, t4 = "abc", ""
    result4a = is_subsequence(s4, t4)
    result4b = is_subsequence_recursive(s4, t4)
    result4c = is_subsequence_dp(s4, t4)
    result4d = is_subsequence_binary_search(s4, t4)
    print(f"Test 4 - Expected: False, Two Pointers: {result4a}, Recursive: {result4b}, DP: {result4c}, Binary Search: {result4d}")
    
    # Test case 5 - Same strings
    s5, t5 = "abc", "abc"
    result5a = is_subsequence(s5, t5)
    result5b = is_subsequence_recursive(s5, t5)
    result5c = is_subsequence_dp(s5, t5)
    result5d = is_subsequence_binary_search(s5, t5)
    print(f"Test 5 - Expected: True, Two Pointers: {result5a}, Recursive: {result5b}, DP: {result5c}, Binary Search: {result5d}")
    
    # Test case 6 - Single character
    s6, t6 = "a", "ba"
    result6a = is_subsequence(s6, t6)
    result6b = is_subsequence_recursive(s6, t6)
    result6c = is_subsequence_dp(s6, t6)
    result6d = is_subsequence_binary_search(s6, t6)
    print(f"Test 6 - Expected: True, Two Pointers: {result6a}, Recursive: {result6b}, DP: {result6c}, Binary Search: {result6d}")
    
    # Test case 7 - No match
    s7, t7 = "xyz", "abc"
    result7a = is_subsequence(s7, t7)
    result7b = is_subsequence_recursive(s7, t7)
    result7c = is_subsequence_dp(s7, t7)
    result7d = is_subsequence_binary_search(s7, t7)
    print(f"Test 7 - Expected: False, Two Pointers: {result7a}, Recursive: {result7b}, DP: {result7c}, Binary Search: {result7d}")
    
    # Test case 8 - Longer example
    s8, t8 = "ace", "abcde"
    result8a = is_subsequence(s8, t8)
    result8b = is_subsequence_recursive(s8, t8)
    result8c = is_subsequence_dp(s8, t8)
    result8d = is_subsequence_binary_search(s8, t8)
    print(f"Test 8 - Expected: True, Two Pointers: {result8a}, Recursive: {result8b}, DP: {result8c}, Binary Search: {result8d}")
    
    # Test case 9 - Repeated characters
    s9, t9 = "aaa", "aaabaa"
    result9a = is_subsequence(s9, t9)
    result9b = is_subsequence_recursive(s9, t9)
    result9c = is_subsequence_dp(s9, t9)
    result9d = is_subsequence_binary_search(s9, t9)
    print(f"Test 9 - Expected: True, Two Pointers: {result9a}, Recursive: {result9b}, DP: {result9c}, Binary Search: {result9d}")
    
    # Test case 10 - Case sensitivity
    s10, t10 = "Ab", "AaAb"
    result10a = is_subsequence(s10, t10)
    result10b = is_subsequence_recursive(s10, t10)
    result10c = is_subsequence_dp(s10, t10)
    result10d = is_subsequence_binary_search(s10, t10)
    print(f"Test 10 - Expected: True, Two Pointers: {result10a}, Recursive: {result10b}, DP: {result10c}, Binary Search: {result10d}") 