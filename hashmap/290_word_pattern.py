"""
290. Word Pattern

Problem:
Given a pattern and a string s, find if s follows the same pattern.

Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty word in s.

Example 1:
Input: pattern = "abba", s = "dog cat cat dog"
Output: true

Example 2:
Input: pattern = "abba", s = "dog cat cat fish"
Output: false

Example 3:
Input: pattern = "aaaa", s = "dog cat cat dog"
Output: false

Time Complexity: O(n + m) where n = len(pattern), m = len(s)
Space Complexity: O(w) where w is the number of unique words
"""


def word_pattern(pattern, s):
    """
    Check if string follows the pattern using bidirectional mapping.
    
    Args:
        pattern: String representing the pattern
        s: String containing words separated by spaces
    
    Returns:
        True if s follows the pattern, False otherwise
    """
    words = s.split()
    
    # Different lengths means no valid mapping
    if len(pattern) != len(words):
        return False
    
    # Bidirectional mapping to ensure one-to-one correspondence
    char_to_word = {}
    word_to_char = {}
    
    for char, word in zip(pattern, words):
        # Check if char already mapped to a different word
        if char in char_to_word:
            if char_to_word[char] != word:
                return False
        else:
            char_to_word[char] = word
        
        # Check if word already mapped to a different char
        if word in word_to_char:
            if word_to_char[word] != char:
                return False
        else:
            word_to_char[word] = char
    
    return True


def word_pattern_single_dict(pattern, s):
    """
    Check if string follows pattern using single dictionary with tuple keys.
    
    Args:
        pattern: String representing the pattern
        s: String containing words separated by spaces
    
    Returns:
        True if s follows the pattern, False otherwise
    """
    words = s.split()
    
    if len(pattern) != len(words):
        return False
    
    mapping = {}
    
    for char, word in zip(pattern, words):
        # Use tuple as key to store both directions
        if (char, word) not in mapping:
            # Check if char or word already exists with different pair
            for existing_char, existing_word in mapping:
                if existing_char == char and existing_word != word:
                    return False
                if existing_char != char and existing_word == word:
                    return False
            mapping[(char, word)] = True
    
    return True


def word_pattern_set_comparison(pattern, s):
    """
    Check pattern using set comparison approach.
    
    Args:
        pattern: String representing the pattern
        s: String containing words separated by spaces
    
    Returns:
        True if s follows the pattern, False otherwise
    """
    words = s.split()
    
    if len(pattern) != len(words):
        return False
    
    # If pattern and words follow same structure, 
    # number of unique chars should equal number of unique words
    # and number of unique (char, word) pairs should also be the same
    return (len(set(pattern)) == len(set(words)) == 
            len(set(zip(pattern, words))))


def word_pattern_index_mapping(pattern, s):
    """
    Check pattern by comparing first occurrence indices.
    
    Args:
        pattern: String representing the pattern
        s: String containing words separated by spaces
    
    Returns:
        True if s follows the pattern, False otherwise
    """
    words = s.split()
    
    if len(pattern) != len(words):
        return False
    
    # Map each character and word to their first occurrence index
    def first_occurrence_pattern(items):
        seen = {}
        result = []
        for i, item in enumerate(items):
            if item not in seen:
                seen[item] = i
            result.append(seen[item])
        return result
    
    return (first_occurrence_pattern(pattern) == 
            first_occurrence_pattern(words))


def word_pattern_normalize(pattern, s):
    """
    Check pattern by normalizing both to same format.
    
    Args:
        pattern: String representing the pattern
        s: String containing words separated by spaces
    
    Returns:
        True if s follows the pattern, False otherwise
    """
    words = s.split()
    
    if len(pattern) != len(words):
        return False
    
    def normalize(items):
        """Convert sequence to normalized pattern"""
        mapping = {}
        result = []
        next_id = 0
        
        for item in items:
            if item not in mapping:
                mapping[item] = next_id
                next_id += 1
            result.append(mapping[item])
        
        return result
    
    return normalize(pattern) == normalize(words)


# Test cases
if __name__ == "__main__":
    # Test case 1
    pattern1, s1 = "abba", "dog cat cat dog"
    result1a = word_pattern(pattern1, s1)
    result1b = word_pattern_single_dict(pattern1, s1)
    result1c = word_pattern_set_comparison(pattern1, s1)
    result1d = word_pattern_index_mapping(pattern1, s1)
    result1e = word_pattern_normalize(pattern1, s1)
    print(f"Test 1 - Expected: True, BiDict: {result1a}, SingleDict: {result1b}, SetComp: {result1c}, IndexMap: {result1d}, Normalize: {result1e}")
    
    # Test case 2
    pattern2, s2 = "abba", "dog cat cat fish"
    result2a = word_pattern(pattern2, s2)
    result2b = word_pattern_single_dict(pattern2, s2)
    result2c = word_pattern_set_comparison(pattern2, s2)
    result2d = word_pattern_index_mapping(pattern2, s2)
    result2e = word_pattern_normalize(pattern2, s2)
    print(f"Test 2 - Expected: False, BiDict: {result2a}, SingleDict: {result2b}, SetComp: {result2c}, IndexMap: {result2d}, Normalize: {result2e}")
    
    # Test case 3
    pattern3, s3 = "aaaa", "dog cat cat dog"
    result3a = word_pattern(pattern3, s3)
    result3b = word_pattern_single_dict(pattern3, s3)
    result3c = word_pattern_set_comparison(pattern3, s3)
    result3d = word_pattern_index_mapping(pattern3, s3)
    result3e = word_pattern_normalize(pattern3, s3)
    print(f"Test 3 - Expected: False, BiDict: {result3a}, SingleDict: {result3b}, SetComp: {result3c}, IndexMap: {result3d}, Normalize: {result3e}")
    
    # Test case 4 - Single character
    pattern4, s4 = "a", "dog"
    result4a = word_pattern(pattern4, s4)
    result4b = word_pattern_single_dict(pattern4, s4)
    result4c = word_pattern_set_comparison(pattern4, s4)
    result4d = word_pattern_index_mapping(pattern4, s4)
    result4e = word_pattern_normalize(pattern4, s4)
    print(f"Test 4 - Expected: True, BiDict: {result4a}, SingleDict: {result4b}, SetComp: {result4c}, IndexMap: {result4d}, Normalize: {result4e}")
    
    # Test case 5 - Different lengths
    pattern5, s5 = "abc", "dog cat"
    result5a = word_pattern(pattern5, s5)
    result5b = word_pattern_single_dict(pattern5, s5)
    result5c = word_pattern_set_comparison(pattern5, s5)
    result5d = word_pattern_index_mapping(pattern5, s5)
    result5e = word_pattern_normalize(pattern5, s5)
    print(f"Test 5 - Expected: False, BiDict: {result5a}, SingleDict: {result5b}, SetComp: {result5c}, IndexMap: {result5d}, Normalize: {result5e}")
    
    # Test case 6 - Empty pattern
    pattern6, s6 = "", ""
    result6a = word_pattern(pattern6, s6)
    result6b = word_pattern_single_dict(pattern6, s6)
    result6c = word_pattern_set_comparison(pattern6, s6)
    result6d = word_pattern_index_mapping(pattern6, s6)
    result6e = word_pattern_normalize(pattern6, s6)
    print(f"Test 6 - Expected: True, BiDict: {result6a}, SingleDict: {result6b}, SetComp: {result6c}, IndexMap: {result6d}, Normalize: {result6e}")
    
    # Test case 7 - Complex pattern
    pattern7, s7 = "abcabc", "dog cat fish dog cat fish"
    result7a = word_pattern(pattern7, s7)
    result7b = word_pattern_single_dict(pattern7, s7)
    result7c = word_pattern_set_comparison(pattern7, s7)
    result7d = word_pattern_index_mapping(pattern7, s7)
    result7e = word_pattern_normalize(pattern7, s7)
    print(f"Test 7 - Expected: True, BiDict: {result7a}, SingleDict: {result7b}, SetComp: {result7c}, IndexMap: {result7d}, Normalize: {result7e}")
    
    # Test case 8 - One word maps to multiple patterns
    pattern8, s8 = "aaa", "dog dog dog"
    result8a = word_pattern(pattern8, s8)
    result8b = word_pattern_single_dict(pattern8, s8)
    result8c = word_pattern_set_comparison(pattern8, s8)
    result8d = word_pattern_index_mapping(pattern8, s8)
    result8e = word_pattern_normalize(pattern8, s8)
    print(f"Test 8 - Expected: True, BiDict: {result8a}, SingleDict: {result8b}, SetComp: {result8c}, IndexMap: {result8d}, Normalize: {result8e}")
    
    # Test case 9 - Edge case with multiple spaces
    pattern9, s9 = "ab", "dog  cat"
    result9a = word_pattern(pattern9, s9)
    result9b = word_pattern_single_dict(pattern9, s9)
    result9c = word_pattern_set_comparison(pattern9, s9)
    result9d = word_pattern_index_mapping(pattern9, s9)
    result9e = word_pattern_normalize(pattern9, s9)
    print(f"Test 9 - Expected: False, BiDict: {result9a}, SingleDict: {result9b}, SetComp: {result9c}, IndexMap: {result9d}, Normalize: {result9e}")
    
    # Test case 10 - Same word different pattern chars
    pattern10, s10 = "ab", "dog dog"
    result10a = word_pattern(pattern10, s10)
    result10b = word_pattern_single_dict(pattern10, s10)
    result10c = word_pattern_set_comparison(pattern10, s10)
    result10d = word_pattern_index_mapping(pattern10, s10)
    result10e = word_pattern_normalize(pattern10, s10)
    print(f"Test 10 - Expected: False, BiDict: {result10a}, SingleDict: {result10b}, SetComp: {result10c}, IndexMap: {result10d}, Normalize: {result10e}") 