"""
58. Length of Last Word

Problem:
Given a string s consisting of words and spaces, return the length of the last word in the string.

A word is a maximal substring consisting of non-space characters only.

Example 1:
Input: s = "Hello World"
Output: 5
Explanation: The last word is "World" with length 5.

Example 2:
Input: s = "   fly me   to   the moon  "
Output: 4
Explanation: The last word is "moon" with length 4.

Example 3:
Input: s = "luffy is still joyboy"
Output: 6
Explanation: The last word is "joyboy" with length 6.

Time Complexity: O(n) where n is the length of the string
Space Complexity: O(1) for optimal solution
"""


def length_of_last_word(s):
    """
    Find length of last word using reverse iteration.
    
    Args:
        s: Input string with words and spaces
    
    Returns:
        Length of the last word
    """
    # Strip trailing spaces and find length of last word
    i = len(s) - 1
    
    # Skip trailing spaces
    while i >= 0 and s[i] == ' ':
        i -= 1
    
    # Count characters in the last word
    length = 0
    while i >= 0 and s[i] != ' ':
        length += 1
        i -= 1
    
    return length


def length_of_last_word_split(s):
    """
    Find length of last word using split method.
    
    Args:
        s: Input string with words and spaces
    
    Returns:
        Length of the last word
    """
    words = s.split()
    return len(words[-1]) if words else 0


def length_of_last_word_strip(s):
    """
    Find length of last word using strip and reverse find.
    
    Args:
        s: Input string with words and spaces
    
    Returns:
        Length of the last word
    """
    # Strip trailing spaces
    s = s.rstrip()
    
    # Find the last space
    last_space_index = s.rfind(' ')
    
    # Return length from last space to end
    return len(s) - last_space_index - 1


def length_of_last_word_forward(s):
    """
    Find length of last word using forward iteration.
    
    Args:
        s: Input string with words and spaces
    
    Returns:
        Length of the last word
    """
    length = 0
    word_started = False
    
    for char in s:
        if char != ' ':
            if not word_started:
                word_started = True
                length = 1
            else:
                length += 1
        else:
            word_started = False
    
    return length


def length_of_last_word_regex(s):
    """
    Find length of last word using regular expressions.
    
    Args:
        s: Input string with words and spaces
    
    Returns:
        Length of the last word
    """
    import re
    
    # Find all words
    words = re.findall(r'\S+', s)
    
    return len(words[-1]) if words else 0


def length_of_last_word_two_pointers(s):
    """
    Find length of last word using two pointers approach.
    
    Args:
        s: Input string with words and spaces
    
    Returns:
        Length of the last word
    """
    n = len(s)
    end = n - 1
    
    # Find the end of the last word
    while end >= 0 and s[end] == ' ':
        end -= 1
    
    # Find the start of the last word
    start = end
    while start >= 0 and s[start] != ' ':
        start -= 1
    
    return end - start


def length_of_last_word_stack(s):
    """
    Find length of last word using stack approach.
    
    Args:
        s: Input string with words and spaces
    
    Returns:
        Length of the last word
    """
    stack = []
    current_word = []
    
    for char in s:
        if char != ' ':
            current_word.append(char)
        else:
            if current_word:
                stack.append(current_word)
                current_word = []
    
    # Handle the last word if it doesn't end with space
    if current_word:
        stack.append(current_word)
    
    return len(stack[-1]) if stack else 0


def length_of_last_word_manual(s):
    """
    Find length of last word using manual character-by-character processing.
    
    Args:
        s: Input string with words and spaces
    
    Returns:
        Length of the last word
    """
    last_word_length = 0
    current_word_length = 0
    
    for char in s:
        if char == ' ':
            if current_word_length > 0:
                last_word_length = current_word_length
                current_word_length = 0
        else:
            current_word_length += 1
    
    # If the string doesn't end with space, the last word is the current word
    return current_word_length if current_word_length > 0 else last_word_length


def length_of_last_word_functional(s):
    """
    Find length of last word using functional programming approach.
    
    Args:
        s: Input string with words and spaces
    
    Returns:
        Length of the last word
    """
    # Filter out empty strings and get the last word
    words = list(filter(None, s.split(' ')))
    return len(words[-1]) if words else 0


# Test cases
if __name__ == "__main__":
    # Test case 1
    s1 = "Hello World"
    result1a = length_of_last_word(s1)
    result1b = length_of_last_word_split(s1)
    result1c = length_of_last_word_strip(s1)
    result1d = length_of_last_word_forward(s1)
    result1e = length_of_last_word_regex(s1)
    result1f = length_of_last_word_two_pointers(s1)
    result1g = length_of_last_word_stack(s1)
    result1h = length_of_last_word_manual(s1)
    result1i = length_of_last_word_functional(s1)
    print(f"Test 1 - String: '{s1}', Expected: 5")
    print(f"Reverse: {result1a}, Split: {result1b}, Strip: {result1c}, Forward: {result1d}, Regex: {result1e}, TwoPointers: {result1f}, Stack: {result1g}, Manual: {result1h}, Functional: {result1i}")
    print()
    
    # Test case 2
    s2 = "   fly me   to   the moon  "
    result2a = length_of_last_word(s2)
    result2b = length_of_last_word_split(s2)
    result2c = length_of_last_word_strip(s2)
    result2d = length_of_last_word_forward(s2)
    result2e = length_of_last_word_regex(s2)
    result2f = length_of_last_word_two_pointers(s2)
    result2g = length_of_last_word_stack(s2)
    result2h = length_of_last_word_manual(s2)
    result2i = length_of_last_word_functional(s2)
    print(f"Test 2 - String: '{s2}', Expected: 4")
    print(f"Reverse: {result2a}, Split: {result2b}, Strip: {result2c}, Forward: {result2d}, Regex: {result2e}, TwoPointers: {result2f}, Stack: {result2g}, Manual: {result2h}, Functional: {result2i}")
    print()
    
    # Test case 3
    s3 = "luffy is still joyboy"
    result3a = length_of_last_word(s3)
    result3b = length_of_last_word_split(s3)
    result3c = length_of_last_word_strip(s3)
    result3d = length_of_last_word_forward(s3)
    result3e = length_of_last_word_regex(s3)
    result3f = length_of_last_word_two_pointers(s3)
    result3g = length_of_last_word_stack(s3)
    result3h = length_of_last_word_manual(s3)
    result3i = length_of_last_word_functional(s3)
    print(f"Test 3 - String: '{s3}', Expected: 6")
    print(f"Reverse: {result3a}, Split: {result3b}, Strip: {result3c}, Forward: {result3d}, Regex: {result3e}, TwoPointers: {result3f}, Stack: {result3g}, Manual: {result3h}, Functional: {result3i}")
    print()
    
    # Test case 4 - Single word
    s4 = "a"
    result4a = length_of_last_word(s4)
    result4b = length_of_last_word_split(s4)
    result4c = length_of_last_word_strip(s4)
    result4d = length_of_last_word_forward(s4)
    result4e = length_of_last_word_regex(s4)
    result4f = length_of_last_word_two_pointers(s4)
    result4g = length_of_last_word_stack(s4)
    result4h = length_of_last_word_manual(s4)
    result4i = length_of_last_word_functional(s4)
    print(f"Test 4 - String: '{s4}', Expected: 1")
    print(f"Reverse: {result4a}, Split: {result4b}, Strip: {result4c}, Forward: {result4d}, Regex: {result4e}, TwoPointers: {result4f}, Stack: {result4g}, Manual: {result4h}, Functional: {result4i}")
    print()
    
    # Test case 5 - Only spaces
    s5 = "   "
    result5a = length_of_last_word(s5)
    result5b = length_of_last_word_split(s5)
    result5c = length_of_last_word_strip(s5)
    result5d = length_of_last_word_forward(s5)
    result5e = length_of_last_word_regex(s5)
    result5f = length_of_last_word_two_pointers(s5)
    result5g = length_of_last_word_stack(s5)
    result5h = length_of_last_word_manual(s5)
    result5i = length_of_last_word_functional(s5)
    print(f"Test 5 - String: '{s5}', Expected: 0")
    print(f"Reverse: {result5a}, Split: {result5b}, Strip: {result5c}, Forward: {result5d}, Regex: {result5e}, TwoPointers: {result5f}, Stack: {result5g}, Manual: {result5h}, Functional: {result5i}")
    print()
    
    # Test case 6 - No trailing spaces
    s6 = "hello world"
    result6a = length_of_last_word(s6)
    result6b = length_of_last_word_split(s6)
    result6c = length_of_last_word_strip(s6)
    result6d = length_of_last_word_forward(s6)
    result6e = length_of_last_word_regex(s6)
    result6f = length_of_last_word_two_pointers(s6)
    result6g = length_of_last_word_stack(s6)
    result6h = length_of_last_word_manual(s6)
    result6i = length_of_last_word_functional(s6)
    print(f"Test 6 - String: '{s6}', Expected: 5")
    print(f"Reverse: {result6a}, Split: {result6b}, Strip: {result6c}, Forward: {result6d}, Regex: {result6e}, TwoPointers: {result6f}, Stack: {result6g}, Manual: {result6h}, Functional: {result6i}")
    print()
    
    # Test case 7 - Multiple spaces between words
    s7 = "a   b    c"
    result7a = length_of_last_word(s7)
    result7b = length_of_last_word_split(s7)
    result7c = length_of_last_word_strip(s7)
    result7d = length_of_last_word_forward(s7)
    result7e = length_of_last_word_regex(s7)
    result7f = length_of_last_word_two_pointers(s7)
    result7g = length_of_last_word_stack(s7)
    result7h = length_of_last_word_manual(s7)
    result7i = length_of_last_word_functional(s7)
    print(f"Test 7 - String: '{s7}', Expected: 1")
    print(f"Reverse: {result7a}, Split: {result7b}, Strip: {result7c}, Forward: {result7d}, Regex: {result7e}, TwoPointers: {result7f}, Stack: {result7g}, Manual: {result7h}, Functional: {result7i}")
    print()
    
    # Test case 8 - Leading and trailing spaces
    s8 = "  programming  "
    result8a = length_of_last_word(s8)
    result8b = length_of_last_word_split(s8)
    result8c = length_of_last_word_strip(s8)
    result8d = length_of_last_word_forward(s8)
    result8e = length_of_last_word_regex(s8)
    result8f = length_of_last_word_two_pointers(s8)
    result8g = length_of_last_word_stack(s8)
    result8h = length_of_last_word_manual(s8)
    result8i = length_of_last_word_functional(s8)
    print(f"Test 8 - String: '{s8}', Expected: 11")
    print(f"Reverse: {result8a}, Split: {result8b}, Strip: {result8c}, Forward: {result8d}, Regex: {result8e}, TwoPointers: {result8f}, Stack: {result8g}, Manual: {result8h}, Functional: {result8i}")
    print()
    
    # Test case 9 - Long word
    s9 = "Today is a beautiful day"
    result9a = length_of_last_word(s9)
    result9b = length_of_last_word_split(s9)
    result9c = length_of_last_word_strip(s9)
    result9d = length_of_last_word_forward(s9)
    result9e = length_of_last_word_regex(s9)
    result9f = length_of_last_word_two_pointers(s9)
    result9g = length_of_last_word_stack(s9)
    result9h = length_of_last_word_manual(s9)
    result9i = length_of_last_word_functional(s9)
    print(f"Test 9 - String: '{s9}', Expected: 3")
    print(f"Reverse: {result9a}, Split: {result9b}, Strip: {result9c}, Forward: {result9d}, Regex: {result9e}, TwoPointers: {result9f}, Stack: {result9g}, Manual: {result9h}, Functional: {result9i}")
    print()
    
    # Test case 10 - Single character with spaces
    s10 = "  a  "
    result10a = length_of_last_word(s10)
    result10b = length_of_last_word_split(s10)
    result10c = length_of_last_word_strip(s10)
    result10d = length_of_last_word_forward(s10)
    result10e = length_of_last_word_regex(s10)
    result10f = length_of_last_word_two_pointers(s10)
    result10g = length_of_last_word_stack(s10)
    result10h = length_of_last_word_manual(s10)
    result10i = length_of_last_word_functional(s10)
    print(f"Test 10 - String: '{s10}', Expected: 1")
    print(f"Reverse: {result10a}, Split: {result10b}, Strip: {result10c}, Forward: {result10d}, Regex: {result10e}, TwoPointers: {result10f}, Stack: {result10g}, Manual: {result10h}, Functional: {result10i}") 