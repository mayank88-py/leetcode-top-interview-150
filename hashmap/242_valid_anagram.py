"""
242. Valid Anagram

Problem:
Given two strings s and t, return true if t is an anagram of s, and false otherwise.
An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, 
typically using all the original letters exactly once.

Example 1:
Input: s = "anagram", t = "nagaram"
Output: true

Example 2:
Input: s = "rat", t = "car"
Output: false

Time Complexity: O(n) where n is the length of the strings
Space Complexity: O(1) for constant character set or O(k) for k unique characters
"""


def is_anagram(s, t):
    """
    Check if two strings are anagrams using character count comparison.
    
    Args:
        s: First string
        t: Second string
    
    Returns:
        True if strings are anagrams, False otherwise
    """
    if len(s) != len(t):
        return False
    
    # Count characters in both strings
    char_count = {}
    
    # Add counts from first string
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    # Subtract counts from second string
    for char in t:
        if char not in char_count:
            return False
        char_count[char] -= 1
        if char_count[char] == 0:
            del char_count[char]
    
    # If all characters cancelled out, it's an anagram
    return len(char_count) == 0


def is_anagram_sorting(s, t):
    """
    Check if two strings are anagrams by sorting both strings.
    
    Args:
        s: First string
        t: Second string
    
    Returns:
        True if strings are anagrams, False otherwise
    """
    if len(s) != len(t):
        return False
    
    return sorted(s) == sorted(t)


def is_anagram_counter(s, t):
    """
    Check if two strings are anagrams using separate counters.
    
    Args:
        s: First string
        t: Second string
    
    Returns:
        True if strings are anagrams, False otherwise
    """
    if len(s) != len(t):
        return False
    
    def count_chars(string):
        """Count characters in string"""
        counts = {}
        for char in string:
            counts[char] = counts.get(char, 0) + 1
        return counts
    
    return count_chars(s) == count_chars(t)


def is_anagram_array(s, t):
    """
    Check if two strings are anagrams using character array (for lowercase letters).
    
    Args:
        s: First string
        t: Second string
    
    Returns:
        True if strings are anagrams, False otherwise
    """
    if len(s) != len(t):
        return False
    
    # Use array for counting (assuming lowercase English letters)
    char_count = [0] * 26
    
    # Count characters
    for i in range(len(s)):
        char_count[ord(s[i]) - ord('a')] += 1
        char_count[ord(t[i]) - ord('a')] -= 1
    
    # Check if all counts are zero
    return all(count == 0 for count in char_count)


def is_anagram_xor(s, t):
    """
    Check if two strings are anagrams using XOR (works only for single occurrence).
    Note: This approach has limitations and doesn't work for general anagrams.
    
    Args:
        s: First string
        t: Second string
    
    Returns:
        True if strings are anagrams, False otherwise
    """
    if len(s) != len(t):
        return False
    
    # XOR all characters - this only works if each character appears exactly once
    xor_result = 0
    for char in s:
        xor_result ^= ord(char)
    for char in t:
        xor_result ^= ord(char)
    
    # This approach is flawed for general anagrams but included for educational purposes
    # It would return True for "aab" and "abb" which are not anagrams
    return xor_result == 0


def is_anagram_unicode(s, t):
    """
    Check if two strings are anagrams handling Unicode characters.
    
    Args:
        s: First string (may contain Unicode)
        t: Second string (may contain Unicode)
    
    Returns:
        True if strings are anagrams, False otherwise
    """
    if len(s) != len(t):
        return False
    
    # Use dictionary to handle any Unicode character
    char_count = {}
    
    # Process both strings simultaneously
    for i in range(len(s)):
        # Increment count for character in s
        char_s = s[i]
        char_count[char_s] = char_count.get(char_s, 0) + 1
        
        # Decrement count for character in t
        char_t = t[i]
        char_count[char_t] = char_count.get(char_t, 0) - 1
        
        # Remove zero counts to save space
        if char_count[char_s] == 0:
            del char_count[char_s]
        if char_t in char_count and char_count[char_t] == 0:
            del char_count[char_t]
    
    return len(char_count) == 0


# Test cases
if __name__ == "__main__":
    # Test case 1
    s1, t1 = "anagram", "nagaram"
    result1a = is_anagram(s1, t1)
    result1b = is_anagram_sorting(s1, t1)
    result1c = is_anagram_counter(s1, t1)
    result1d = is_anagram_array(s1, t1)
    result1e = is_anagram_xor(s1, t1)
    result1f = is_anagram_unicode(s1, t1)
    print(f"Test 1 - Expected: True, HashMap: {result1a}, Sorting: {result1b}, Counter: {result1c}, Array: {result1d}, XOR: {result1e}, Unicode: {result1f}")
    
    # Test case 2
    s2, t2 = "rat", "car"
    result2a = is_anagram(s2, t2)
    result2b = is_anagram_sorting(s2, t2)
    result2c = is_anagram_counter(s2, t2)
    result2d = is_anagram_array(s2, t2)
    result2e = is_anagram_xor(s2, t2)
    result2f = is_anagram_unicode(s2, t2)
    print(f"Test 2 - Expected: False, HashMap: {result2a}, Sorting: {result2b}, Counter: {result2c}, Array: {result2d}, XOR: {result2e}, Unicode: {result2f}")
    
    # Test case 3 - Different lengths
    s3, t3 = "hello", "world!"
    result3a = is_anagram(s3, t3)
    result3b = is_anagram_sorting(s3, t3)
    result3c = is_anagram_counter(s3, t3)
    result3d = is_anagram_array(s3, t3)
    result3e = is_anagram_xor(s3, t3)
    result3f = is_anagram_unicode(s3, t3)
    print(f"Test 3 - Expected: False, HashMap: {result3a}, Sorting: {result3b}, Counter: {result3c}, Array: {result3d}, XOR: {result3e}, Unicode: {result3f}")
    
    # Test case 4 - Empty strings
    s4, t4 = "", ""
    result4a = is_anagram(s4, t4)
    result4b = is_anagram_sorting(s4, t4)
    result4c = is_anagram_counter(s4, t4)
    result4d = is_anagram_array(s4, t4)
    result4e = is_anagram_xor(s4, t4)
    result4f = is_anagram_unicode(s4, t4)
    print(f"Test 4 - Expected: True, HashMap: {result4a}, Sorting: {result4b}, Counter: {result4c}, Array: {result4d}, XOR: {result4e}, Unicode: {result4f}")
    
    # Test case 5 - Single character
    s5, t5 = "a", "a"
    result5a = is_anagram(s5, t5)
    result5b = is_anagram_sorting(s5, t5)
    result5c = is_anagram_counter(s5, t5)
    result5d = is_anagram_array(s5, t5)
    result5e = is_anagram_xor(s5, t5)
    result5f = is_anagram_unicode(s5, t5)
    print(f"Test 5 - Expected: True, HashMap: {result5a}, Sorting: {result5b}, Counter: {result5c}, Array: {result5d}, XOR: {result5e}, Unicode: {result5f}")
    
    # Test case 6 - Repeated characters
    s6, t6 = "aab", "aba"
    result6a = is_anagram(s6, t6)
    result6b = is_anagram_sorting(s6, t6)
    result6c = is_anagram_counter(s6, t6)
    result6d = is_anagram_array(s6, t6)
    result6e = is_anagram_xor(s6, t6)
    result6f = is_anagram_unicode(s6, t6)
    print(f"Test 6 - Expected: True, HashMap: {result6a}, Sorting: {result6b}, Counter: {result6c}, Array: {result6d}, XOR: {result6e}, Unicode: {result6f}")
    
    # Test case 7 - Case sensitivity
    s7, t7 = "Listen", "Silent"
    result7a = is_anagram(s7.lower(), t7.lower())
    result7b = is_anagram_sorting(s7.lower(), t7.lower())
    result7c = is_anagram_counter(s7.lower(), t7.lower())
    result7d = is_anagram_array(s7.lower(), t7.lower())
    result7e = is_anagram_xor(s7.lower(), t7.lower())
    result7f = is_anagram_unicode(s7.lower(), t7.lower())
    print(f"Test 7 - Expected: True, HashMap: {result7a}, Sorting: {result7b}, Counter: {result7c}, Array: {result7d}, XOR: {result7e}, Unicode: {result7f}")
    
    # Test case 8 - All same character
    s8, t8 = "aaaa", "aaaa"
    result8a = is_anagram(s8, t8)
    result8b = is_anagram_sorting(s8, t8)
    result8c = is_anagram_counter(s8, t8)
    result8d = is_anagram_array(s8, t8)
    result8e = is_anagram_xor(s8, t8)
    result8f = is_anagram_unicode(s8, t8)
    print(f"Test 8 - Expected: True, HashMap: {result8a}, Sorting: {result8b}, Counter: {result8c}, Array: {result8d}, XOR: {result8e}, Unicode: {result8f}")
    
    # Test case 9 - XOR limitation example (educational)
    s9, t9 = "aab", "abb"
    result9a = is_anagram(s9, t9)
    result9b = is_anagram_sorting(s9, t9)
    result9c = is_anagram_counter(s9, t9)
    result9d = is_anagram_array(s9, t9)
    result9e = is_anagram_xor(s9, t9)  # This will incorrectly return True
    result9f = is_anagram_unicode(s9, t9)
    print(f"Test 9 - Expected: False, HashMap: {result9a}, Sorting: {result9b}, Counter: {result9c}, Array: {result9d}, XOR: {result9e} (incorrect!), Unicode: {result9f}")
    
    # Test case 10 - Long anagrams
    s10, t10 = "conversation", "voices rant on"
    s10_clean = s10.replace(" ", "").lower()
    t10_clean = t10.replace(" ", "").lower()
    result10a = is_anagram(s10_clean, t10_clean)
    result10b = is_anagram_sorting(s10_clean, t10_clean)
    result10c = is_anagram_counter(s10_clean, t10_clean)
    result10d = is_anagram_array(s10_clean, t10_clean)
    result10e = is_anagram_xor(s10_clean, t10_clean)
    result10f = is_anagram_unicode(s10_clean, t10_clean)
    print(f"Test 10 - Expected: True, HashMap: {result10a}, Sorting: {result10b}, Counter: {result10c}, Array: {result10d}, XOR: {result10e}, Unicode: {result10f}") 