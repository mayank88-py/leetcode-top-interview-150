"""
151. Reverse Words in a String

Problem:
Given an input string s, reverse the order of the words.

A word is defined as a sequence of non-space characters. The words in s will be separated by at least one space.

Return a string of the words in reverse order concatenated by a single space.

Note that s may contain leading or trailing spaces or multiple spaces between two words. The returned string should only have a single space separating the words. Do not include any extra spaces.

Example 1:
Input: s = "the sky is blue"
Output: "blue is sky the"

Example 2:
Input: s = "  hello world  "
Output: "world hello"
Explanation: Your reversed string should not contain leading or trailing spaces.

Example 3:
Input: s = "a good   example"
Output: "example good a"
Explanation: You need to reduce multiple spaces between two words to a single space in the reversed string.

Time Complexity: O(n) where n is the length of the string
Space Complexity: O(n) for the output string
"""


def reverse_words(s):
    """
    Reverse words in a string using built-in split and join.
    
    Args:
        s: Input string with words separated by spaces
    
    Returns:
        String with words in reverse order
    """
    # Split by spaces (handles multiple spaces) and reverse
    words = s.split()
    return ' '.join(reversed(words))


def reverse_words_manual(s):
    """
    Reverse words in a string using manual parsing.
    
    Args:
        s: Input string with words separated by spaces
    
    Returns:
        String with words in reverse order
    """
    words = []
    word = ""
    
    for char in s:
        if char != ' ':
            word += char
        else:
            if word:
                words.append(word)
                word = ""
    
    # Add the last word if it exists
    if word:
        words.append(word)
    
    # Reverse the words list
    words.reverse()
    
    return ' '.join(words)


def reverse_words_two_pass(s):
    """
    Reverse words in a string using two-pass approach.
    
    Args:
        s: Input string with words separated by spaces
    
    Returns:
        String with words in reverse order
    """
    # First pass: extract words
    words = []
    i = 0
    n = len(s)
    
    while i < n:
        # Skip spaces
        while i < n and s[i] == ' ':
            i += 1
        
        if i < n:
            # Extract word
            start = i
            while i < n and s[i] != ' ':
                i += 1
            words.append(s[start:i])
    
    # Second pass: reverse and join
    result = []
    for i in range(len(words) - 1, -1, -1):
        result.append(words[i])
    
    return ' '.join(result)


def reverse_words_stack(s):
    """
    Reverse words in a string using stack.
    
    Args:
        s: Input string with words separated by spaces
    
    Returns:
        String with words in reverse order
    """
    stack = []
    word = ""
    
    for char in s:
        if char != ' ':
            word += char
        else:
            if word:
                stack.append(word)
                word = ""
    
    # Add the last word
    if word:
        stack.append(word)
    
    # Pop from stack to get reversed order
    result = []
    while stack:
        result.append(stack.pop())
    
    return ' '.join(result)


def reverse_words_in_place(s):
    """
    Reverse words in a string using in-place approach (simulated).
    
    Args:
        s: Input string with words separated by spaces
    
    Returns:
        String with words in reverse order
    """
    # Convert to list for in-place operations
    chars = list(s)
    n = len(chars)
    
    # Remove extra spaces
    def clean_spaces():
        i = j = 0
        while i < n:
            # Skip leading spaces
            while i < n and chars[i] == ' ':
                i += 1
            
            # Copy word
            while i < n and chars[i] != ' ':
                chars[j] = chars[i]
                i += 1
                j += 1
            
            # Add single space after word (if not last word)
            if i < n:
                chars[j] = ' '
                j += 1
        
        return j
    
    # Reverse entire string
    def reverse_string(start, end):
        while start < end:
            chars[start], chars[end] = chars[end], chars[start]
            start += 1
            end -= 1
    
    # Reverse each word
    def reverse_words_in_cleaned():
        start = 0
        for i in range(actual_len + 1):
            if i == actual_len or chars[i] == ' ':
                reverse_string(start, i - 1)
                start = i + 1
    
    # Clean spaces
    actual_len = clean_spaces()
    
    # Reverse entire string
    reverse_string(0, actual_len - 1)
    
    # Reverse each word
    reverse_words_in_cleaned()
    
    return ''.join(chars[:actual_len])


def reverse_words_regex(s):
    """
    Reverse words in a string using regular expressions.
    
    Args:
        s: Input string with words separated by spaces
    
    Returns:
        String with words in reverse order
    """
    import re
    
    # Find all words
    words = re.findall(r'\S+', s)
    
    # Reverse and join
    return ' '.join(reversed(words))


def reverse_words_deque(s):
    """
    Reverse words in a string using deque.
    
    Args:
        s: Input string with words separated by spaces
    
    Returns:
        String with words in reverse order
    """
    from collections import deque
    
    dq = deque()
    word = ""
    
    for char in s:
        if char != ' ':
            word += char
        else:
            if word:
                dq.appendleft(word)
                word = ""
    
    # Add the last word
    if word:
        dq.appendleft(word)
    
    return ' '.join(dq)


def reverse_words_functional(s):
    """
    Reverse words in a string using functional programming.
    
    Args:
        s: Input string with words separated by spaces
    
    Returns:
        String with words in reverse order
    """
    # Filter out empty strings and reverse
    words = list(filter(None, s.split(' ')))
    return ' '.join(words[::-1])


def reverse_words_one_pass(s):
    """
    Reverse words in a string using one-pass approach.
    
    Args:
        s: Input string with words separated by spaces
    
    Returns:
        String with words in reverse order
    """
    result = []
    word = ""
    
    for i in range(len(s) - 1, -1, -1):
        if s[i] != ' ':
            word = s[i] + word
        else:
            if word:
                result.append(word)
                word = ""
    
    # Add the last word (first in original string)
    if word:
        result.append(word)
    
    return ' '.join(result)


# Test cases
if __name__ == "__main__":
    # Test case 1
    s1 = "the sky is blue"
    result1a = reverse_words(s1)
    result1b = reverse_words_manual(s1)
    result1c = reverse_words_two_pass(s1)
    result1d = reverse_words_stack(s1)
    result1e = reverse_words_in_place(s1)
    result1f = reverse_words_regex(s1)
    result1g = reverse_words_deque(s1)
    result1h = reverse_words_functional(s1)
    result1i = reverse_words_one_pass(s1)
    print(f"Test 1 - Input: '{s1}', Expected: 'blue is sky the'")
    print(f"Split: '{result1a}', Manual: '{result1b}', TwoPass: '{result1c}', Stack: '{result1d}', InPlace: '{result1e}', Regex: '{result1f}', Deque: '{result1g}', Functional: '{result1h}', OnePass: '{result1i}'")
    print()
    
    # Test case 2
    s2 = "  hello world  "
    result2a = reverse_words(s2)
    result2b = reverse_words_manual(s2)
    result2c = reverse_words_two_pass(s2)
    result2d = reverse_words_stack(s2)
    result2e = reverse_words_in_place(s2)
    result2f = reverse_words_regex(s2)
    result2g = reverse_words_deque(s2)
    result2h = reverse_words_functional(s2)
    result2i = reverse_words_one_pass(s2)
    print(f"Test 2 - Input: '{s2}', Expected: 'world hello'")
    print(f"Split: '{result2a}', Manual: '{result2b}', TwoPass: '{result2c}', Stack: '{result2d}', InPlace: '{result2e}', Regex: '{result2f}', Deque: '{result2g}', Functional: '{result2h}', OnePass: '{result2i}'")
    print()
    
    # Test case 3
    s3 = "a good   example"
    result3a = reverse_words(s3)
    result3b = reverse_words_manual(s3)
    result3c = reverse_words_two_pass(s3)
    result3d = reverse_words_stack(s3)
    result3e = reverse_words_in_place(s3)
    result3f = reverse_words_regex(s3)
    result3g = reverse_words_deque(s3)
    result3h = reverse_words_functional(s3)
    result3i = reverse_words_one_pass(s3)
    print(f"Test 3 - Input: '{s3}', Expected: 'example good a'")
    print(f"Split: '{result3a}', Manual: '{result3b}', TwoPass: '{result3c}', Stack: '{result3d}', InPlace: '{result3e}', Regex: '{result3f}', Deque: '{result3g}', Functional: '{result3h}', OnePass: '{result3i}'")
    print()
    
    # Test case 4 - Single word
    s4 = "hello"
    result4a = reverse_words(s4)
    result4b = reverse_words_manual(s4)
    result4c = reverse_words_two_pass(s4)
    result4d = reverse_words_stack(s4)
    result4e = reverse_words_in_place(s4)
    result4f = reverse_words_regex(s4)
    result4g = reverse_words_deque(s4)
    result4h = reverse_words_functional(s4)
    result4i = reverse_words_one_pass(s4)
    print(f"Test 4 - Input: '{s4}', Expected: 'hello'")
    print(f"Split: '{result4a}', Manual: '{result4b}', TwoPass: '{result4c}', Stack: '{result4d}', InPlace: '{result4e}', Regex: '{result4f}', Deque: '{result4g}', Functional: '{result4h}', OnePass: '{result4i}'")
    print()
    
    # Test case 5 - Empty string
    s5 = "   "
    result5a = reverse_words(s5)
    result5b = reverse_words_manual(s5)
    result5c = reverse_words_two_pass(s5)
    result5d = reverse_words_stack(s5)
    result5e = reverse_words_in_place(s5)
    result5f = reverse_words_regex(s5)
    result5g = reverse_words_deque(s5)
    result5h = reverse_words_functional(s5)
    result5i = reverse_words_one_pass(s5)
    print(f"Test 5 - Input: '{s5}', Expected: ''")
    print(f"Split: '{result5a}', Manual: '{result5b}', TwoPass: '{result5c}', Stack: '{result5d}', InPlace: '{result5e}', Regex: '{result5f}', Deque: '{result5g}', Functional: '{result5h}', OnePass: '{result5i}'")
    print()
    
    # Test case 6 - Two words
    s6 = "hello world"
    result6a = reverse_words(s6)
    result6b = reverse_words_manual(s6)
    result6c = reverse_words_two_pass(s6)
    result6d = reverse_words_stack(s6)
    result6e = reverse_words_in_place(s6)
    result6f = reverse_words_regex(s6)
    result6g = reverse_words_deque(s6)
    result6h = reverse_words_functional(s6)
    result6i = reverse_words_one_pass(s6)
    print(f"Test 6 - Input: '{s6}', Expected: 'world hello'")
    print(f"Split: '{result6a}', Manual: '{result6b}', TwoPass: '{result6c}', Stack: '{result6d}', InPlace: '{result6e}', Regex: '{result6f}', Deque: '{result6g}', Functional: '{result6h}', OnePass: '{result6i}'")
    print()
    
    # Test case 7 - Multiple spaces
    s7 = "  multiple   spaces   here  "
    result7a = reverse_words(s7)
    result7b = reverse_words_manual(s7)
    result7c = reverse_words_two_pass(s7)
    result7d = reverse_words_stack(s7)
    result7e = reverse_words_in_place(s7)
    result7f = reverse_words_regex(s7)
    result7g = reverse_words_deque(s7)
    result7h = reverse_words_functional(s7)
    result7i = reverse_words_one_pass(s7)
    print(f"Test 7 - Input: '{s7}', Expected: 'here spaces multiple'")
    print(f"Split: '{result7a}', Manual: '{result7b}', TwoPass: '{result7c}', Stack: '{result7d}', InPlace: '{result7e}', Regex: '{result7f}', Deque: '{result7g}', Functional: '{result7h}', OnePass: '{result7i}'")
    print()
    
    # Test case 8 - Long sentence
    s8 = "This is a very long sentence with many words"
    result8a = reverse_words(s8)
    result8b = reverse_words_manual(s8)
    result8c = reverse_words_two_pass(s8)
    result8d = reverse_words_stack(s8)
    result8e = reverse_words_in_place(s8)
    result8f = reverse_words_regex(s8)
    result8g = reverse_words_deque(s8)
    result8h = reverse_words_functional(s8)
    result8i = reverse_words_one_pass(s8)
    print(f"Test 8 - Input: '{s8}', Expected: 'words many with sentence long very a is This'")
    print(f"Split: '{result8a}', Manual: '{result8b}', TwoPass: '{result8c}', Stack: '{result8d}', InPlace: '{result8e}', Regex: '{result8f}', Deque: '{result8g}', Functional: '{result8h}', OnePass: '{result8i}'")
    print()
    
    # Test case 9 - Single character words
    s9 = "a b c d"
    result9a = reverse_words(s9)
    result9b = reverse_words_manual(s9)
    result9c = reverse_words_two_pass(s9)
    result9d = reverse_words_stack(s9)
    result9e = reverse_words_in_place(s9)
    result9f = reverse_words_regex(s9)
    result9g = reverse_words_deque(s9)
    result9h = reverse_words_functional(s9)
    result9i = reverse_words_one_pass(s9)
    print(f"Test 9 - Input: '{s9}', Expected: 'd c b a'")
    print(f"Split: '{result9a}', Manual: '{result9b}', TwoPass: '{result9c}', Stack: '{result9d}', InPlace: '{result9e}', Regex: '{result9f}', Deque: '{result9g}', Functional: '{result9h}', OnePass: '{result9i}'")
    print()
    
    # Test case 10 - Leading and trailing spaces
    s10 = "   leading and trailing   "
    result10a = reverse_words(s10)
    result10b = reverse_words_manual(s10)
    result10c = reverse_words_two_pass(s10)
    result10d = reverse_words_stack(s10)
    result10e = reverse_words_in_place(s10)
    result10f = reverse_words_regex(s10)
    result10g = reverse_words_deque(s10)
    result10h = reverse_words_functional(s10)
    result10i = reverse_words_one_pass(s10)
    print(f"Test 10 - Input: '{s10}', Expected: 'trailing and leading'")
    print(f"Split: '{result10a}', Manual: '{result10b}', TwoPass: '{result10c}', Stack: '{result10d}', InPlace: '{result10e}', Regex: '{result10f}', Deque: '{result10g}', Functional: '{result10h}', OnePass: '{result10i}'") 