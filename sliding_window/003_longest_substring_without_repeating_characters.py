"""
3. Longest Substring Without Repeating Characters

Problem:
Given a string s, find the length of the longest substring without repeating characters.

Example 1:
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

Example 2:
Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.

Example 3:
Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.

Example 4:
Input: s = ""
Output: 0

Time Complexity: O(n)
Space Complexity: O(min(m, n)) where m is the size of character set
"""


def length_of_longest_substring(s):
    """
    Find longest substring without repeating characters using sliding window.
    
    Args:
        s: Input string
    
    Returns:
        Length of longest substring without repeating characters
    """
    if not s:
        return 0
    
    char_index = {}
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        char = s[right]
        
        # If character is already in current window, move left pointer
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1
        
        # Update character's latest index
        char_index[char] = right
        
        # Update maximum length
        max_length = max(max_length, right - left + 1)
    
    return max_length


def length_of_longest_substring_set(s):
    """
    Find longest substring using sliding window with set.
    
    Args:
        s: Input string
    
    Returns:
        Length of longest substring without repeating characters
    """
    if not s:
        return 0
    
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Remove characters from left until no duplicates
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length


def length_of_longest_substring_brute_force(s):
    """
    Find longest substring using brute force approach.
    
    Args:
        s: Input string
    
    Returns:
        Length of longest substring without repeating characters
    """
    if not s:
        return 0
    
    max_length = 0
    
    for i in range(len(s)):
        seen = set()
        for j in range(i, len(s)):
            if s[j] in seen:
                break
            seen.add(s[j])
            max_length = max(max_length, j - i + 1)
    
    return max_length


def length_of_longest_substring_ascii(s):
    """
    Find longest substring optimized for ASCII characters.
    
    Args:
        s: Input string
    
    Returns:
        Length of longest substring without repeating characters
    """
    if not s:
        return 0
    
    # Array to store last seen index of each ASCII character
    char_index = [-1] * 128
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        char_code = ord(s[right])
        
        # If character is in current window, move left pointer
        if char_index[char_code] >= left:
            left = char_index[char_code] + 1
        
        char_index[char_code] = right
        max_length = max(max_length, right - left + 1)
    
    return max_length


# Test cases
if __name__ == "__main__":
    # Test case 1
    s1 = "abcabcbb"
    result1a = length_of_longest_substring(s1)
    result1b = length_of_longest_substring_set(s1)
    result1c = length_of_longest_substring_brute_force(s1)
    result1d = length_of_longest_substring_ascii(s1)
    print(f"Test 1 - Expected: 3, HashMap: {result1a}, Set: {result1b}, Brute: {result1c}, ASCII: {result1d}")
    
    # Test case 2
    s2 = "bbbbb"
    result2a = length_of_longest_substring(s2)
    result2b = length_of_longest_substring_set(s2)
    result2c = length_of_longest_substring_brute_force(s2)
    result2d = length_of_longest_substring_ascii(s2)
    print(f"Test 2 - Expected: 1, HashMap: {result2a}, Set: {result2b}, Brute: {result2c}, ASCII: {result2d}")
    
    # Test case 3
    s3 = "pwwkew"
    result3a = length_of_longest_substring(s3)
    result3b = length_of_longest_substring_set(s3)
    result3c = length_of_longest_substring_brute_force(s3)
    result3d = length_of_longest_substring_ascii(s3)
    print(f"Test 3 - Expected: 3, HashMap: {result3a}, Set: {result3b}, Brute: {result3c}, ASCII: {result3d}")
    
    # Test case 4
    s4 = ""
    result4a = length_of_longest_substring(s4)
    result4b = length_of_longest_substring_set(s4)
    result4c = length_of_longest_substring_brute_force(s4)
    result4d = length_of_longest_substring_ascii(s4)
    print(f"Test 4 - Expected: 0, HashMap: {result4a}, Set: {result4b}, Brute: {result4c}, ASCII: {result4d}")
    
    # Test case 5 - Single character
    s5 = "a"
    result5a = length_of_longest_substring(s5)
    result5b = length_of_longest_substring_set(s5)
    result5c = length_of_longest_substring_brute_force(s5)
    result5d = length_of_longest_substring_ascii(s5)
    print(f"Test 5 - Expected: 1, HashMap: {result5a}, Set: {result5b}, Brute: {result5c}, ASCII: {result5d}")
    
    # Test case 6 - All unique characters
    s6 = "abcdef"
    result6a = length_of_longest_substring(s6)
    result6b = length_of_longest_substring_set(s6)
    result6c = length_of_longest_substring_brute_force(s6)
    result6d = length_of_longest_substring_ascii(s6)
    print(f"Test 6 - Expected: 6, HashMap: {result6a}, Set: {result6b}, Brute: {result6c}, ASCII: {result6d}")
    
    # Test case 7 - Complex case
    s7 = "abba"
    result7a = length_of_longest_substring(s7)
    result7b = length_of_longest_substring_set(s7)
    result7c = length_of_longest_substring_brute_force(s7)
    result7d = length_of_longest_substring_ascii(s7)
    print(f"Test 7 - Expected: 2, HashMap: {result7a}, Set: {result7b}, Brute: {result7c}, ASCII: {result7d}")
    
    # Test case 8 - Numbers and special characters
    s8 = "123!@#123"
    result8a = length_of_longest_substring(s8)
    result8b = length_of_longest_substring_set(s8)
    result8c = length_of_longest_substring_brute_force(s8)
    result8d = length_of_longest_substring_ascii(s8)
    print(f"Test 8 - Expected: 6, HashMap: {result8a}, Set: {result8b}, Brute: {result8c}, ASCII: {result8d}") 