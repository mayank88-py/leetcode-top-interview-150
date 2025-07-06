"""
383. Ransom Note

Problem:
Given two strings ransomNote and magazine, return true if ransomNote can be constructed by using the 
letters from magazine and false otherwise.

Each letter in magazine can only be used once in ransomNote.

Example 1:
Input: ransomNote = "a", magazine = "b"
Output: false

Example 2:
Input: ransomNote = "aa", magazine = "ab"
Output: false

Example 3:
Input: ransomNote = "aa", magazine = "aab"
Output: true

Time Complexity: O(m + n) where m = len(ransomNote), n = len(magazine)
Space Complexity: O(1) for counting array or O(k) for hashmap where k is number of unique characters
"""


def can_construct(ransom_note, magazine):
    """
    Check if ransom note can be constructed from magazine using hashmap.
    
    Args:
        ransom_note: String representing the ransom note
        magazine: String representing the magazine
    
    Returns:
        True if ransom note can be constructed, False otherwise
    """
    if not ransom_note:
        return True
    if not magazine or len(ransom_note) > len(magazine):
        return False
    
    # Count characters in magazine
    magazine_count = {}
    for char in magazine:
        magazine_count[char] = magazine_count.get(char, 0) + 1
    
    # Check if we can construct ransom note
    for char in ransom_note:
        if char not in magazine_count or magazine_count[char] == 0:
            return False
        magazine_count[char] -= 1
    
    return True


def can_construct_counter(ransom_note, magazine):
    """
    Check using Counter-like approach.
    
    Args:
        ransom_note: String representing the ransom note
        magazine: String representing the magazine
    
    Returns:
        True if ransom note can be constructed, False otherwise
    """
    if not ransom_note:
        return True
    if not magazine or len(ransom_note) > len(magazine):
        return False
    
    # Count characters in both strings
    ransom_count = {}
    magazine_count = {}
    
    for char in ransom_note:
        ransom_count[char] = ransom_count.get(char, 0) + 1
    
    for char in magazine:
        magazine_count[char] = magazine_count.get(char, 0) + 1
    
    # Check if magazine has enough characters
    for char, count in ransom_count.items():
        if magazine_count.get(char, 0) < count:
            return False
    
    return True


def can_construct_sorting(ransom_note, magazine):
    """
    Check using sorting approach.
    
    Args:
        ransom_note: String representing the ransom note
        magazine: String representing the magazine
    
    Returns:
        True if ransom note can be constructed, False otherwise
    """
    if not ransom_note:
        return True
    if not magazine or len(ransom_note) > len(magazine):
        return False
    
    # Sort both strings
    ransom_sorted = sorted(ransom_note)
    magazine_sorted = sorted(magazine)
    
    # Use two pointers to check
    i = j = 0
    
    while i < len(ransom_sorted) and j < len(magazine_sorted):
        if ransom_sorted[i] == magazine_sorted[j]:
            i += 1
            j += 1
        elif ransom_sorted[i] > magazine_sorted[j]:
            j += 1
        else:
            return False
    
    return i == len(ransom_sorted)


def can_construct_array(ransom_note, magazine):
    """
    Check using character array (assuming lowercase letters only).
    
    Args:
        ransom_note: String representing the ransom note
        magazine: String representing the magazine
    
    Returns:
        True if ransom note can be constructed, False otherwise
    """
    if not ransom_note:
        return True
    if not magazine or len(ransom_note) > len(magazine):
        return False
    
    # Count characters using array (for lowercase letters)
    char_count = [0] * 26
    
    # Count characters in magazine
    for char in magazine:
        char_count[ord(char) - ord('a')] += 1
    
    # Check if we can construct ransom note
    for char in ransom_note:
        index = ord(char) - ord('a')
        if char_count[index] == 0:
            return False
        char_count[index] -= 1
    
    return True


def can_construct_remove(ransom_note, magazine):
    """
    Check by removing characters from magazine string.
    
    Args:
        ransom_note: String representing the ransom note
        magazine: String representing the magazine
    
    Returns:
        True if ransom note can be constructed, False otherwise
    """
    if not ransom_note:
        return True
    if not magazine or len(ransom_note) > len(magazine):
        return False
    
    # Convert to list for easier manipulation
    magazine_chars = list(magazine)
    
    for char in ransom_note:
        if char in magazine_chars:
            magazine_chars.remove(char)
        else:
            return False
    
    return True


# Test cases
if __name__ == "__main__":
    # Test case 1
    ransom1, magazine1 = "a", "b"
    result1a = can_construct(ransom1, magazine1)
    result1b = can_construct_counter(ransom1, magazine1)
    result1c = can_construct_sorting(ransom1, magazine1)
    result1d = can_construct_array(ransom1, magazine1)
    result1e = can_construct_remove(ransom1, magazine1)
    print(f"Test 1 - Expected: False, HashMap: {result1a}, Counter: {result1b}, Sorting: {result1c}, Array: {result1d}, Remove: {result1e}")
    
    # Test case 2
    ransom2, magazine2 = "aa", "ab"
    result2a = can_construct(ransom2, magazine2)
    result2b = can_construct_counter(ransom2, magazine2)
    result2c = can_construct_sorting(ransom2, magazine2)
    result2d = can_construct_array(ransom2, magazine2)
    result2e = can_construct_remove(ransom2, magazine2)
    print(f"Test 2 - Expected: False, HashMap: {result2a}, Counter: {result2b}, Sorting: {result2c}, Array: {result2d}, Remove: {result2e}")
    
    # Test case 3
    ransom3, magazine3 = "aa", "aab"
    result3a = can_construct(ransom3, magazine3)
    result3b = can_construct_counter(ransom3, magazine3)
    result3c = can_construct_sorting(ransom3, magazine3)
    result3d = can_construct_array(ransom3, magazine3)
    result3e = can_construct_remove(ransom3, magazine3)
    print(f"Test 3 - Expected: True, HashMap: {result3a}, Counter: {result3b}, Sorting: {result3c}, Array: {result3d}, Remove: {result3e}")
    
    # Test case 4 - Empty ransom note
    ransom4, magazine4 = "", "abc"
    result4a = can_construct(ransom4, magazine4)
    result4b = can_construct_counter(ransom4, magazine4)
    result4c = can_construct_sorting(ransom4, magazine4)
    result4d = can_construct_array(ransom4, magazine4)
    result4e = can_construct_remove(ransom4, magazine4)
    print(f"Test 4 - Expected: True, HashMap: {result4a}, Counter: {result4b}, Sorting: {result4c}, Array: {result4d}, Remove: {result4e}")
    
    # Test case 5 - Same strings
    ransom5, magazine5 = "abc", "abc"
    result5a = can_construct(ransom5, magazine5)
    result5b = can_construct_counter(ransom5, magazine5)
    result5c = can_construct_sorting(ransom5, magazine5)
    result5d = can_construct_array(ransom5, magazine5)
    result5e = can_construct_remove(ransom5, magazine5)
    print(f"Test 5 - Expected: True, HashMap: {result5a}, Counter: {result5b}, Sorting: {result5c}, Array: {result5d}, Remove: {result5e}")
    
    # Test case 6 - More complex
    ransom6, magazine6 = "aab", "baa"
    result6a = can_construct(ransom6, magazine6)
    result6b = can_construct_counter(ransom6, magazine6)
    result6c = can_construct_sorting(ransom6, magazine6)
    result6d = can_construct_array(ransom6, magazine6)
    result6e = can_construct_remove(ransom6, magazine6)
    print(f"Test 6 - Expected: True, HashMap: {result6a}, Counter: {result6b}, Sorting: {result6c}, Array: {result6d}, Remove: {result6e}")
    
    # Test case 7 - Large magazine
    ransom7, magazine7 = "abc", "aabbccdef"
    result7a = can_construct(ransom7, magazine7)
    result7b = can_construct_counter(ransom7, magazine7)
    result7c = can_construct_sorting(ransom7, magazine7)
    result7d = can_construct_array(ransom7, magazine7)
    result7e = can_construct_remove(ransom7, magazine7)
    print(f"Test 7 - Expected: True, HashMap: {result7a}, Counter: {result7b}, Sorting: {result7c}, Array: {result7d}, Remove: {result7e}")
    
    # Test case 8 - Single character
    ransom8, magazine8 = "z", "z"
    result8a = can_construct(ransom8, magazine8)
    result8b = can_construct_counter(ransom8, magazine8)
    result8c = can_construct_sorting(ransom8, magazine8)
    result8d = can_construct_array(ransom8, magazine8)
    result8e = can_construct_remove(ransom8, magazine8)
    print(f"Test 8 - Expected: True, HashMap: {result8a}, Counter: {result8b}, Sorting: {result8c}, Array: {result8d}, Remove: {result8e}")
    
    # Test case 9 - Missing character
    ransom9, magazine9 = "abc", "ab"
    result9a = can_construct(ransom9, magazine9)
    result9b = can_construct_counter(ransom9, magazine9)
    result9c = can_construct_sorting(ransom9, magazine9)
    result9d = can_construct_array(ransom9, magazine9)
    result9e = can_construct_remove(ransom9, magazine9)
    print(f"Test 9 - Expected: False, HashMap: {result9a}, Counter: {result9b}, Sorting: {result9c}, Array: {result9d}, Remove: {result9e}")
    
    # Test case 10 - Repeated characters
    ransom10, magazine10 = "aabbcc", "abcabc"
    result10a = can_construct(ransom10, magazine10)
    result10b = can_construct_counter(ransom10, magazine10)
    result10c = can_construct_sorting(ransom10, magazine10)
    result10d = can_construct_array(ransom10, magazine10)
    result10e = can_construct_remove(ransom10, magazine10)
    print(f"Test 10 - Expected: True, HashMap: {result10a}, Counter: {result10b}, Sorting: {result10c}, Array: {result10d}, Remove: {result10e}") 