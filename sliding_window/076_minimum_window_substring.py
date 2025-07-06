"""
76. Minimum Window Substring

Problem:
Given two strings s and t, return the minimum window substring of s such that every character in t 
(including duplicates) is included in the window. If there is no such substring, return the empty string "".

The testcases will be generated such that the answer is unique.

Example 1:
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.

Example 2:
Input: s = "a", t = "a"
Output: "a"
Explanation: The entire string s is the minimum window.

Example 3:
Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be included in the window.

Time Complexity: O(|s| + |t|)
Space Complexity: O(|s| + |t|)
"""


def min_window(s, t):
    """
    Find minimum window substring using sliding window technique.
    
    Args:
        s: Source string
        t: Target string containing characters to find
    
    Returns:
        Minimum window substring or empty string if not found
    """
    if not s or not t or len(s) < len(t):
        return ""
    
    # Count characters in t
    t_count = {}
    for char in t:
        t_count[char] = t_count.get(char, 0) + 1
    
    # Sliding window variables
    left = 0
    min_len = float('inf')
    min_start = 0
    required = len(t_count)  # Number of unique characters in t
    formed = 0  # Number of unique characters in current window with desired frequency
    
    # Window character counts
    window_count = {}
    
    for right in range(len(s)):
        # Add character from right to window
        char = s[right]
        window_count[char] = window_count.get(char, 0) + 1
        
        # If current character's frequency matches required frequency
        if char in t_count and window_count[char] == t_count[char]:
            formed += 1
        
        # Try to shrink window from left
        while left <= right and formed == required:
            # Update minimum window if current is smaller
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_start = left
            
            # Remove character from left
            left_char = s[left]
            window_count[left_char] -= 1
            if left_char in t_count and window_count[left_char] < t_count[left_char]:
                formed -= 1
            
            left += 1
    
    return "" if min_len == float('inf') else s[min_start:min_start + min_len]


def min_window_optimized(s, t):
    """
    Find minimum window substring with optimized approach.
    
    Args:
        s: Source string
        t: Target string containing characters to find
    
    Returns:
        Minimum window substring or empty string if not found
    """
    if not s or not t:
        return ""
    
    # Count characters in t
    t_count = {}
    for char in t:
        t_count[char] = t_count.get(char, 0) + 1
    
    # Filter s to only include characters that are in t
    filtered_s = []
    for i, char in enumerate(s):
        if char in t_count:
            filtered_s.append((i, char))
    
    left = 0
    min_len = float('inf')
    min_start = 0
    required = len(t_count)
    formed = 0
    window_count = {}
    
    for right in range(len(filtered_s)):
        # Add character from right to window
        char = filtered_s[right][1]
        window_count[char] = window_count.get(char, 0) + 1
        
        if window_count[char] == t_count[char]:
            formed += 1
        
        # Try to shrink window from left
        while left <= right and formed == required:
            # Get actual indices in original string
            start_idx = filtered_s[left][0]
            end_idx = filtered_s[right][0]
            
            # Update minimum window if current is smaller
            if end_idx - start_idx + 1 < min_len:
                min_len = end_idx - start_idx + 1
                min_start = start_idx
            
            # Remove character from left
            left_char = filtered_s[left][1]
            window_count[left_char] -= 1
            if window_count[left_char] < t_count[left_char]:
                formed -= 1
            
            left += 1
    
    return "" if min_len == float('inf') else s[min_start:min_start + min_len]


def min_window_brute_force(s, t):
    """
    Find minimum window substring using brute force approach.
    
    Args:
        s: Source string
        t: Target string containing characters to find
    
    Returns:
        Minimum window substring or empty string if not found
    """
    if not s or not t or len(s) < len(t):
        return ""
    
    def contains_all(window, target):
        """Check if window contains all characters from target"""
        t_count = {}
        for char in target:
            t_count[char] = t_count.get(char, 0) + 1
        
        w_count = {}
        for char in window:
            w_count[char] = w_count.get(char, 0) + 1
        
        for char, count in t_count.items():
            if w_count.get(char, 0) < count:
                return False
        return True
    
    min_len = float('inf')
    min_window = ""
    
    # Try all possible substrings
    for i in range(len(s)):
        for j in range(i + len(t), len(s) + 1):
            window = s[i:j]
            if contains_all(window, t):
                if len(window) < min_len:
                    min_len = len(window)
                    min_window = window
                break  # Found valid window starting at i, no need to check longer ones
    
    return min_window


# Test cases
if __name__ == "__main__":
    # Test case 1
    s1, t1 = "ADOBECODEBANC", "ABC"
    result1a = min_window(s1, t1)
    result1b = min_window_optimized(s1, t1)
    result1c = min_window_brute_force(s1, t1)
    print(f"Test 1 - Expected: BANC, Sliding Window: {result1a}, Optimized: {result1b}, Brute Force: {result1c}")
    
    # Test case 2
    s2, t2 = "a", "a"
    result2a = min_window(s2, t2)
    result2b = min_window_optimized(s2, t2)
    result2c = min_window_brute_force(s2, t2)
    print(f"Test 2 - Expected: a, Sliding Window: {result2a}, Optimized: {result2b}, Brute Force: {result2c}")
    
    # Test case 3
    s3, t3 = "a", "aa"
    result3a = min_window(s3, t3)
    result3b = min_window_optimized(s3, t3)
    result3c = min_window_brute_force(s3, t3)
    print(f"Test 3 - Expected: '', Sliding Window: {result3a}, Optimized: {result3b}, Brute Force: {result3c}")
    
    # Test case 4 - No valid window
    s4, t4 = "abc", "de"
    result4a = min_window(s4, t4)
    result4b = min_window_optimized(s4, t4)
    result4c = min_window_brute_force(s4, t4)
    print(f"Test 4 - Expected: '', Sliding Window: {result4a}, Optimized: {result4b}, Brute Force: {result4c}")
    
    # Test case 5 - Entire string is minimum window
    s5, t5 = "abc", "abc"
    result5a = min_window(s5, t5)
    result5b = min_window_optimized(s5, t5)
    result5c = min_window_brute_force(s5, t5)
    print(f"Test 5 - Expected: abc, Sliding Window: {result5a}, Optimized: {result5b}, Brute Force: {result5c}")
    
    # Test case 6 - Repeated characters
    s6, t6 = "ADOBECODEBANC", "AABC"
    result6a = min_window(s6, t6)
    result6b = min_window_optimized(s6, t6)
    result6c = min_window_brute_force(s6, t6)
    print(f"Test 6 - Expected: ADOBEC, Sliding Window: {result6a}, Optimized: {result6b}, Brute Force: {result6c}")
    
    # Test case 7 - Single character
    s7, t7 = "bba", "ab"
    result7a = min_window(s7, t7)
    result7b = min_window_optimized(s7, t7)
    result7c = min_window_brute_force(s7, t7)
    print(f"Test 7 - Expected: ba, Sliding Window: {result7a}, Optimized: {result7b}, Brute Force: {result7c}")
    
    # Test case 8 - Empty strings
    s8, t8 = "", "a"
    result8a = min_window(s8, t8)
    result8b = min_window_optimized(s8, t8)
    result8c = min_window_brute_force(s8, t8)
    print(f"Test 8 - Expected: '', Sliding Window: {result8a}, Optimized: {result8b}, Brute Force: {result8c}")
    
    # Test case 9 - Case sensitivity
    s9, t9 = "Ab", "A"
    result9a = min_window(s9, t9)
    result9b = min_window_optimized(s9, t9)
    result9c = min_window_brute_force(s9, t9)
    print(f"Test 9 - Expected: A, Sliding Window: {result9a}, Optimized: {result9b}, Brute Force: {result9c}")
    
    # Test case 10 - More complex example
    s10, t10 = "abcdef", "cf"
    result10a = min_window(s10, t10)
    result10b = min_window_optimized(s10, t10)
    result10c = min_window_brute_force(s10, t10)
    print(f"Test 10 - Expected: cdef, Sliding Window: {result10a}, Optimized: {result10b}, Brute Force: {result10c}")