"""
125. Valid Palindrome

Problem:
A phrase is a palindrome if, after converting all uppercase letters into lowercase letters
and removing all non-alphanumeric characters, it reads the same forward and backward.
Alphanumeric characters include letters and numbers.

Given a string s, return true if it is a palindrome, or false otherwise.

Example 1:
Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.

Example 2:
Input: s = "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.

Example 3:
Input: s = " "
Output: true
Explanation: s is an empty string "" after removing non-alphanumeric characters.

Time Complexity: O(n)
Space Complexity: O(1)
"""


def is_palindrome(s):
    """
    Check if string is a palindrome using two pointers.
    
    Args:
        s: Input string
    
    Returns:
        True if palindrome, False otherwise
    """
    left, right = 0, len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric characters from left
        while left < right and not s[left].isalnum():
            left += 1
        
        # Skip non-alphanumeric characters from right
        while left < right and not s[right].isalnum():
            right -= 1
        
        # Compare characters (case-insensitive)
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True


def is_palindrome_preprocess(s):
    """
    Check if string is a palindrome by preprocessing.
    
    Args:
        s: Input string
    
    Returns:
        True if palindrome, False otherwise
    """
    # Remove non-alphanumeric characters and convert to lowercase
    cleaned = ''.join(char.lower() for char in s if char.isalnum())
    
    # Check if cleaned string is equal to its reverse
    return cleaned == cleaned[::-1]


def is_palindrome_manual(s):
    """
    Check if string is a palindrome without using built-in functions.
    
    Args:
        s: Input string
    
    Returns:
        True if palindrome, False otherwise
    """
    left, right = 0, len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric characters from left
        while left < right and not is_alphanumeric(s[left]):
            left += 1
        
        # Skip non-alphanumeric characters from right
        while left < right and not is_alphanumeric(s[right]):
            right -= 1
        
        # Compare characters (case-insensitive)
        if to_lowercase(s[left]) != to_lowercase(s[right]):
            return False
        
        left += 1
        right -= 1
    
    return True


def is_alphanumeric(char):
    """Check if character is alphanumeric."""
    return (('a' <= char <= 'z') or 
            ('A' <= char <= 'Z') or 
            ('0' <= char <= '9'))


def to_lowercase(char):
    """Convert character to lowercase."""
    if 'A' <= char <= 'Z':
        return chr(ord(char) - ord('A') + ord('a'))
    return char


# Test cases
if __name__ == "__main__":
    # Test case 1
    s1 = "A man, a plan, a canal: Panama"
    result1a = is_palindrome(s1)
    result1b = is_palindrome_preprocess(s1)
    result1c = is_palindrome_manual(s1)
    print(f"Test 1 - Expected: True, Two Pointers: {result1a}, Preprocess: {result1b}, Manual: {result1c}")
    
    # Test case 2
    s2 = "race a car"
    result2a = is_palindrome(s2)
    result2b = is_palindrome_preprocess(s2)
    result2c = is_palindrome_manual(s2)
    print(f"Test 2 - Expected: False, Two Pointers: {result2a}, Preprocess: {result2b}, Manual: {result2c}")
    
    # Test case 3
    s3 = " "
    result3a = is_palindrome(s3)
    result3b = is_palindrome_preprocess(s3)
    result3c = is_palindrome_manual(s3)
    print(f"Test 3 - Expected: True, Two Pointers: {result3a}, Preprocess: {result3b}, Manual: {result3c}")
    
    # Test case 4
    s4 = "Madam"
    result4a = is_palindrome(s4)
    result4b = is_palindrome_preprocess(s4)
    result4c = is_palindrome_manual(s4)
    print(f"Test 4 - Expected: True, Two Pointers: {result4a}, Preprocess: {result4b}, Manual: {result4c}")
    
    # Test case 5
    s5 = "0P"
    result5a = is_palindrome(s5)
    result5b = is_palindrome_preprocess(s5)
    result5c = is_palindrome_manual(s5)
    print(f"Test 5 - Expected: False, Two Pointers: {result5a}, Preprocess: {result5b}, Manual: {result5c}")
    
    # Test case 6
    s6 = "ab_a"
    result6a = is_palindrome(s6)
    result6b = is_palindrome_preprocess(s6)
    result6c = is_palindrome_manual(s6)
    print(f"Test 6 - Expected: True, Two Pointers: {result6a}, Preprocess: {result6b}, Manual: {result6c}") 