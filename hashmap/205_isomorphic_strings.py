"""
205. Isomorphic Strings

Problem:
Given two strings s and t, determine if they are isomorphic.
Two strings s and t are isomorphic if the characters in s can be replaced to get t.
All occurrences of a character must be replaced with the same character while preserving the order of characters.
No two characters may map to the same character, but a character may map to itself.

Example 1:
Input: s = "egg", t = "add"
Output: true

Example 2:
Input: s = "foo", t = "bar"
Output: false

Example 3:
Input: s = "paper", t = "title"
Output: true

Time Complexity: O(n) where n is the length of the strings
Space Complexity: O(1) for ASCII characters or O(k) where k is the number of unique characters
"""


def is_isomorphic(s, t):
    """
    Check if two strings are isomorphic using two hashmaps.
    
    Args:
        s: First string
        t: Second string
    
    Returns:
        True if strings are isomorphic, False otherwise
    """
    if len(s) != len(t):
        return False
    
    # Mappings from s to t and t to s
    s_to_t = {}
    t_to_s = {}
    
    for i in range(len(s)):
        char_s = s[i]
        char_t = t[i]
        
        # Check if mapping exists and is consistent
        if char_s in s_to_t:
            if s_to_t[char_s] != char_t:
                return False
        else:
            s_to_t[char_s] = char_t
        
        if char_t in t_to_s:
            if t_to_s[char_t] != char_s:
                return False
        else:
            t_to_s[char_t] = char_s
    
    return True


def is_isomorphic_one_pass(s, t):
    """
    Check isomorphic strings using single hashmap approach.
    
    Args:
        s: First string
        t: Second string
    
    Returns:
        True if strings are isomorphic, False otherwise
    """
    if len(s) != len(t):
        return False
    
    mapping = {}
    used = set()
    
    for i in range(len(s)):
        char_s = s[i]
        char_t = t[i]
        
        if char_s in mapping:
            if mapping[char_s] != char_t:
                return False
        else:
            if char_t in used:
                return False
            mapping[char_s] = char_t
            used.add(char_t)
    
    return True


def is_isomorphic_array(s, t):
    """
    Check isomorphic strings using character arrays (for ASCII).
    
    Args:
        s: First string
        t: Second string
    
    Returns:
        True if strings are isomorphic, False otherwise
    """
    if len(s) != len(t):
        return False
    
    # Arrays to store mappings (assuming ASCII characters)
    s_to_t = [0] * 256
    t_to_s = [0] * 256
    
    for i in range(len(s)):
        char_s = ord(s[i])
        char_t = ord(t[i])
        
        # Check if mapping exists and is consistent
        if s_to_t[char_s] != 0:
            if s_to_t[char_s] != char_t:
                return False
        else:
            s_to_t[char_s] = char_t
        
        if t_to_s[char_t] != 0:
            if t_to_s[char_t] != char_s:
                return False
        else:
            t_to_s[char_t] = char_s
    
    return True


def is_isomorphic_normalize(s, t):
    """
    Check isomorphic strings by normalizing both strings.
    
    Args:
        s: First string
        t: Second string
    
    Returns:
        True if strings are isomorphic, False otherwise
    """
    if len(s) != len(t):
        return False
    
    def normalize(string):
        """Convert string to normalized form"""
        mapping = {}
        result = []
        next_char = 0
        
        for char in string:
            if char not in mapping:
                mapping[char] = next_char
                next_char += 1
            result.append(mapping[char])
        
        return result
    
    return normalize(s) == normalize(t)


def is_isomorphic_zip(s, t):
    """
    Check isomorphic strings using zip and set properties.
    
    Args:
        s: First string
        t: Second string
    
    Returns:
        True if strings are isomorphic, False otherwise
    """
    if len(s) != len(t):
        return False
    
    # Check if the number of unique characters and unique pairs match
    return len(set(s)) == len(set(t)) == len(set(zip(s, t)))


# Test cases
if __name__ == "__main__":
    # Test case 1
    s1, t1 = "egg", "add"
    result1a = is_isomorphic(s1, t1)
    result1b = is_isomorphic_one_pass(s1, t1)
    result1c = is_isomorphic_array(s1, t1)
    result1d = is_isomorphic_normalize(s1, t1)
    result1e = is_isomorphic_zip(s1, t1)
    print(f"Test 1 - Expected: True, Two Maps: {result1a}, One Map: {result1b}, Array: {result1c}, Normalize: {result1d}, Zip: {result1e}")
    
    # Test case 2
    s2, t2 = "foo", "bar"
    result2a = is_isomorphic(s2, t2)
    result2b = is_isomorphic_one_pass(s2, t2)
    result2c = is_isomorphic_array(s2, t2)
    result2d = is_isomorphic_normalize(s2, t2)
    result2e = is_isomorphic_zip(s2, t2)
    print(f"Test 2 - Expected: False, Two Maps: {result2a}, One Map: {result2b}, Array: {result2c}, Normalize: {result2d}, Zip: {result2e}")
    
    # Test case 3
    s3, t3 = "paper", "title"
    result3a = is_isomorphic(s3, t3)
    result3b = is_isomorphic_one_pass(s3, t3)
    result3c = is_isomorphic_array(s3, t3)
    result3d = is_isomorphic_normalize(s3, t3)
    result3e = is_isomorphic_zip(s3, t3)
    print(f"Test 3 - Expected: True, Two Maps: {result3a}, One Map: {result3b}, Array: {result3c}, Normalize: {result3d}, Zip: {result3e}")
    
    # Test case 4 - Same strings
    s4, t4 = "abc", "abc"
    result4a = is_isomorphic(s4, t4)
    result4b = is_isomorphic_one_pass(s4, t4)
    result4c = is_isomorphic_array(s4, t4)
    result4d = is_isomorphic_normalize(s4, t4)
    result4e = is_isomorphic_zip(s4, t4)
    print(f"Test 4 - Expected: True, Two Maps: {result4a}, One Map: {result4b}, Array: {result4c}, Normalize: {result4d}, Zip: {result4e}")
    
    # Test case 5 - Different lengths
    s5, t5 = "ab", "abc"
    result5a = is_isomorphic(s5, t5)
    result5b = is_isomorphic_one_pass(s5, t5)
    result5c = is_isomorphic_array(s5, t5)
    result5d = is_isomorphic_normalize(s5, t5)
    result5e = is_isomorphic_zip(s5, t5)
    print(f"Test 5 - Expected: False, Two Maps: {result5a}, One Map: {result5b}, Array: {result5c}, Normalize: {result5d}, Zip: {result5e}")
    
    # Test case 6 - Single character
    s6, t6 = "a", "a"
    result6a = is_isomorphic(s6, t6)
    result6b = is_isomorphic_one_pass(s6, t6)
    result6c = is_isomorphic_array(s6, t6)
    result6d = is_isomorphic_normalize(s6, t6)
    result6e = is_isomorphic_zip(s6, t6)
    print(f"Test 6 - Expected: True, Two Maps: {result6a}, One Map: {result6b}, Array: {result6c}, Normalize: {result6d}, Zip: {result6e}")
    
    # Test case 7 - Empty strings
    s7, t7 = "", ""
    result7a = is_isomorphic(s7, t7)
    result7b = is_isomorphic_one_pass(s7, t7)
    result7c = is_isomorphic_array(s7, t7)
    result7d = is_isomorphic_normalize(s7, t7)
    result7e = is_isomorphic_zip(s7, t7)
    print(f"Test 7 - Expected: True, Two Maps: {result7a}, One Map: {result7b}, Array: {result7c}, Normalize: {result7d}, Zip: {result7e}")
    
    # Test case 8 - Tricky case
    s8, t8 = "abab", "baba"
    result8a = is_isomorphic(s8, t8)
    result8b = is_isomorphic_one_pass(s8, t8)
    result8c = is_isomorphic_array(s8, t8)
    result8d = is_isomorphic_normalize(s8, t8)
    result8e = is_isomorphic_zip(s8, t8)
    print(f"Test 8 - Expected: True, Two Maps: {result8a}, One Map: {result8b}, Array: {result8c}, Normalize: {result8d}, Zip: {result8e}")
    
    # Test case 9 - False case
    s9, t9 = "badc", "baba"
    result9a = is_isomorphic(s9, t9)
    result9b = is_isomorphic_one_pass(s9, t9)
    result9c = is_isomorphic_array(s9, t9)
    result9d = is_isomorphic_normalize(s9, t9)
    result9e = is_isomorphic_zip(s9, t9)
    print(f"Test 9 - Expected: False, Two Maps: {result9a}, One Map: {result9b}, Array: {result9c}, Normalize: {result9d}, Zip: {result9e}")
    
    # Test case 10 - Numbers as strings
    s10, t10 = "13", "42"
    result10a = is_isomorphic(s10, t10)
    result10b = is_isomorphic_one_pass(s10, t10)
    result10c = is_isomorphic_array(s10, t10)
    result10d = is_isomorphic_normalize(s10, t10)
    result10e = is_isomorphic_zip(s10, t10)
    print(f"Test 10 - Expected: True, Two Maps: {result10a}, One Map: {result10b}, Array: {result10c}, Normalize: {result10d}, Zip: {result10e}") 