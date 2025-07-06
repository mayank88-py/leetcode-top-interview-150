"""
68. Text Justification

Problem:
Given an array of strings words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.

You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces ' ' when necessary so that each line has exactly maxWidth characters.

Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line does not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.

For the last line of text, it should be left-justified, and no extra spaces are inserted between words.

Example 1:
Input: words = ["This", "is", "an", "example", "of", "text", "justification."], maxWidth = 16
Output:
[
   "This    is    an",
   "example  of text",
   "justification.  "
]

Example 2:
Input: words = ["What","must","be","acknowledgment","shall","be"], maxWidth = 16
Output:
[
  "What   must   be",
  "acknowledgment  ",
  "shall be        "
]

Example 3:
Input: words = ["Science","is","what","we","understand","well","enough","to","explain","to","a","computer.","Art","is","everything","else","we","do"], maxWidth = 20
Output:
[
  "Science  is  what we",
  "understand      well",
  "enough to explain to",
  "a  computer.  Art is",
  "everything  else  we",
  "do                  "
]

Time Complexity: O(n) where n is the total number of characters
Space Complexity: O(k) where k is the number of lines
"""


def full_justify(words, maxWidth):
    """
    Justify text to fit within given width.
    
    Args:
        words: List of words to justify
        maxWidth: Maximum width of each line
    
    Returns:
        List of justified lines
    """
    result = []
    i = 0
    
    while i < len(words):
        # Collect words for current line
        line_words = []
        line_length = 0
        
        # Pack as many words as possible
        while i < len(words) and line_length + len(words[i]) + len(line_words) <= maxWidth:
            line_words.append(words[i])
            line_length += len(words[i])
            i += 1
        
        # Justify the line
        if i == len(words):  # Last line
            result.append(left_justify(line_words, maxWidth))
        else:
            result.append(full_justify_line(line_words, maxWidth))
    
    return result


def full_justify_line(words, maxWidth):
    """
    Fully justify a single line.
    
    Args:
        words: List of words in the line
        maxWidth: Target width
    
    Returns:
        Justified line as string
    """
    if len(words) == 1:
        return words[0] + ' ' * (maxWidth - len(words[0]))
    
    total_word_length = sum(len(word) for word in words)
    total_spaces = maxWidth - total_word_length
    gaps = len(words) - 1
    
    # Calculate spaces per gap
    spaces_per_gap = total_spaces // gaps
    extra_spaces = total_spaces % gaps
    
    result = []
    for i in range(len(words)):
        result.append(words[i])
        
        if i < gaps:  # Not the last word
            result.append(' ' * spaces_per_gap)
            
            # Add extra space if needed
            if i < extra_spaces:
                result.append(' ')
    
    return ''.join(result)


def left_justify(words, maxWidth):
    """
    Left justify a line (for last line).
    
    Args:
        words: List of words in the line
        maxWidth: Target width
    
    Returns:
        Left-justified line as string
    """
    line = ' '.join(words)
    return line + ' ' * (maxWidth - len(line))


def full_justify_alternative(words, maxWidth):
    """
    Alternative implementation with different organization.
    
    Args:
        words: List of words to justify
        maxWidth: Maximum width of each line
    
    Returns:
        List of justified lines
    """
    result = []
    current_line = []
    current_length = 0
    
    for word in words:
        # Check if adding this word would exceed maxWidth
        if current_length + len(word) + len(current_line) > maxWidth:
            # Process current line
            if current_line:
                result.append(justify_line(current_line, maxWidth, False))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word)
    
    # Process last line
    if current_line:
        result.append(justify_line(current_line, maxWidth, True))
    
    return result


def justify_line(words, maxWidth, is_last_line):
    """
    Justify a single line.
    
    Args:
        words: List of words in the line
        maxWidth: Target width
        is_last_line: Whether this is the last line
    
    Returns:
        Justified line as string
    """
    if is_last_line or len(words) == 1:
        # Left justify
        line = ' '.join(words)
        return line + ' ' * (maxWidth - len(line))
    
    # Full justify
    total_word_length = sum(len(word) for word in words)
    total_spaces = maxWidth - total_word_length
    gaps = len(words) - 1
    
    spaces_per_gap = total_spaces // gaps
    extra_spaces = total_spaces % gaps
    
    result = []
    for i in range(len(words)):
        result.append(words[i])
        
        if i < gaps:
            result.append(' ' * spaces_per_gap)
            if i < extra_spaces:
                result.append(' ')
    
    return ''.join(result)


def full_justify_greedy(words, maxWidth):
    """
    Greedy approach to text justification.
    
    Args:
        words: List of words to justify
        maxWidth: Maximum width of each line
    
    Returns:
        List of justified lines
    """
    def can_fit(line_words, new_word):
        """Check if new word can fit in current line"""
        current_length = sum(len(word) for word in line_words)
        spaces_needed = len(line_words) - 1  # Minimum spaces between words
        return current_length + spaces_needed + 1 + len(new_word) <= maxWidth
    
    result = []
    current_line = []
    
    for word in words:
        if not current_line or can_fit(current_line, word):
            current_line.append(word)
        else:
            # Justify current line and start new line
            result.append(format_line(current_line, maxWidth, False))
            current_line = [word]
    
    # Handle last line
    if current_line:
        result.append(format_line(current_line, maxWidth, True))
    
    return result


def format_line(words, maxWidth, is_last):
    """
    Format a line according to justification rules.
    
    Args:
        words: List of words in the line
        maxWidth: Target width
        is_last: Whether this is the last line
    
    Returns:
        Formatted line as string
    """
    if is_last or len(words) == 1:
        # Left justify
        content = ' '.join(words)
        return content + ' ' * (maxWidth - len(content))
    
    # Full justify
    total_chars = sum(len(word) for word in words)
    total_spaces = maxWidth - total_chars
    gaps = len(words) - 1
    
    base_spaces = total_spaces // gaps
    extra_spaces = total_spaces % gaps
    
    line = []
    for i, word in enumerate(words):
        line.append(word)
        if i < gaps:
            line.append(' ' * base_spaces)
            if i < extra_spaces:
                line.append(' ')
    
    return ''.join(line)


def full_justify_functional(words, maxWidth):
    """
    Functional programming approach to text justification.
    
    Args:
        words: List of words to justify
        maxWidth: Maximum width of each line
    
    Returns:
        List of justified lines
    """
    def group_words():
        """Group words into lines"""
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            spaces_needed = len(current_line)  # Spaces between words
            
            if current_length + word_length + spaces_needed <= maxWidth:
                current_line.append(word)
                current_length += word_length
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [word]
                current_length = word_length
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def justify_lines(lines):
        """Justify all lines"""
        result = []
        for i, line in enumerate(lines):
            is_last = (i == len(lines) - 1)
            result.append(justify_single_line(line, maxWidth, is_last))
        return result
    
    def justify_single_line(line, width, is_last):
        """Justify a single line"""
        if is_last or len(line) == 1:
            content = ' '.join(line)
            return content + ' ' * (width - len(content))
        
        total_chars = sum(len(word) for word in line)
        total_spaces = width - total_chars
        gaps = len(line) - 1
        
        base_spaces = total_spaces // gaps
        extra_spaces = total_spaces % gaps
        
        justified = []
        for i, word in enumerate(line):
            justified.append(word)
            if i < gaps:
                justified.append(' ' * base_spaces)
                if i < extra_spaces:
                    justified.append(' ')
        
        return ''.join(justified)
    
    word_groups = group_words()
    return justify_lines(word_groups)


def full_justify_iterative(words, maxWidth):
    """
    Iterative approach with explicit state management.
    
    Args:
        words: List of words to justify
        maxWidth: Maximum width of each line
    
    Returns:
        List of justified lines
    """
    result = []
    word_index = 0
    
    while word_index < len(words):
        # Collect words for current line
        line_start = word_index
        line_char_count = 0
        
        # Find how many words fit in current line
        while word_index < len(words):
            word_len = len(words[word_index])
            spaces_needed = word_index - line_start  # Spaces between words
            
            if line_char_count + word_len + spaces_needed <= maxWidth:
                line_char_count += word_len
                word_index += 1
            else:
                break
        
        # Create line from collected words
        line_words = words[line_start:word_index]
        
        # Determine if this is the last line
        is_last_line = (word_index == len(words))
        
        # Justify the line
        justified_line = create_justified_line(line_words, maxWidth, is_last_line)
        result.append(justified_line)
    
    return result


def create_justified_line(words, max_width, is_last_line):
    """
    Create a justified line from words.
    
    Args:
        words: List of words for the line
        max_width: Maximum width of the line
        is_last_line: Whether this is the last line
    
    Returns:
        Justified line as string
    """
    if is_last_line or len(words) == 1:
        # Left justify
        line = ' '.join(words)
        return line + ' ' * (max_width - len(line))
    
    # Full justify
    total_word_chars = sum(len(word) for word in words)
    total_spaces_needed = max_width - total_word_chars
    gaps = len(words) - 1
    
    spaces_per_gap = total_spaces_needed // gaps
    extra_spaces = total_spaces_needed % gaps
    
    result_parts = []
    for i, word in enumerate(words):
        result_parts.append(word)
        
        if i < gaps:  # Not the last word
            result_parts.append(' ' * spaces_per_gap)
            
            # Add extra space if needed
            if i < extra_spaces:
                result_parts.append(' ')
    
    return ''.join(result_parts)


# Test cases
if __name__ == "__main__":
    # Test case 1
    words1 = ["This", "is", "an", "example", "of", "text", "justification."]
    maxWidth1 = 16
    result1a = full_justify(words1, maxWidth1)
    result1b = full_justify_alternative(words1, maxWidth1)
    result1c = full_justify_greedy(words1, maxWidth1)
    result1d = full_justify_functional(words1, maxWidth1)
    result1e = full_justify_iterative(words1, maxWidth1)
    print(f"Test 1 - Words: {words1}, MaxWidth: {maxWidth1}")
    print("Expected:")
    print('["This    is    an", "example  of text", "justification.  "]')
    print(f"Original: {result1a}")
    print(f"Alternative: {result1b}")
    print(f"Greedy: {result1c}")
    print(f"Functional: {result1d}")
    print(f"Iterative: {result1e}")
    print()
    
    # Test case 2
    words2 = ["What","must","be","acknowledgment","shall","be"]
    maxWidth2 = 16
    result2a = full_justify(words2, maxWidth2)
    result2b = full_justify_alternative(words2, maxWidth2)
    result2c = full_justify_greedy(words2, maxWidth2)
    result2d = full_justify_functional(words2, maxWidth2)
    result2e = full_justify_iterative(words2, maxWidth2)
    print(f"Test 2 - Words: {words2}, MaxWidth: {maxWidth2}")
    print("Expected:")
    print('["What   must   be", "acknowledgment  ", "shall be        "]')
    print(f"Original: {result2a}")
    print(f"Alternative: {result2b}")
    print(f"Greedy: {result2c}")
    print(f"Functional: {result2d}")
    print(f"Iterative: {result2e}")
    print()
    
    # Test case 3
    words3 = ["Science","is","what","we","understand","well","enough","to","explain","to","a","computer.","Art","is","everything","else","we","do"]
    maxWidth3 = 20
    result3a = full_justify(words3, maxWidth3)
    result3b = full_justify_alternative(words3, maxWidth3)
    result3c = full_justify_greedy(words3, maxWidth3)
    result3d = full_justify_functional(words3, maxWidth3)
    result3e = full_justify_iterative(words3, maxWidth3)
    print(f"Test 3 - Words: {words3[:5]}..., MaxWidth: {maxWidth3}")
    print("Expected:")
    print('["Science  is  what we", "understand      well", "enough to explain to", "a  computer.  Art is", "everything  else  we", "do                  "]')
    print(f"Original: {result3a}")
    print(f"Alternative: {result3b}")
    print(f"Greedy: {result3c}")
    print(f"Functional: {result3d}")
    print(f"Iterative: {result3e}")
    print()
    
    # Test case 4 - Single word
    words4 = ["Hello"]
    maxWidth4 = 10
    result4a = full_justify(words4, maxWidth4)
    result4b = full_justify_alternative(words4, maxWidth4)
    result4c = full_justify_greedy(words4, maxWidth4)
    result4d = full_justify_functional(words4, maxWidth4)
    result4e = full_justify_iterative(words4, maxWidth4)
    print(f"Test 4 - Words: {words4}, MaxWidth: {maxWidth4}")
    print("Expected:")
    print('["Hello     "]')
    print(f"Original: {result4a}")
    print(f"Alternative: {result4b}")
    print(f"Greedy: {result4c}")
    print(f"Functional: {result4d}")
    print(f"Iterative: {result4e}")
    print()
    
    # Test case 5 - Two words
    words5 = ["Hello", "World"]
    maxWidth5 = 10
    result5a = full_justify(words5, maxWidth5)
    result5b = full_justify_alternative(words5, maxWidth5)
    result5c = full_justify_greedy(words5, maxWidth5)
    result5d = full_justify_functional(words5, maxWidth5)
    result5e = full_justify_iterative(words5, maxWidth5)
    print(f"Test 5 - Words: {words5}, MaxWidth: {maxWidth5}")
    print("Expected:")
    print('["Hello World"]')
    print(f"Original: {result5a}")
    print(f"Alternative: {result5b}")
    print(f"Greedy: {result5c}")
    print(f"Functional: {result5d}")
    print(f"Iterative: {result5e}")
    print()
    
    # Test case 6 - Edge case with exact fit
    words6 = ["a", "b", "c", "d", "e"]
    maxWidth6 = 9
    result6a = full_justify(words6, maxWidth6)
    result6b = full_justify_alternative(words6, maxWidth6)
    result6c = full_justify_greedy(words6, maxWidth6)
    result6d = full_justify_functional(words6, maxWidth6)
    result6e = full_justify_iterative(words6, maxWidth6)
    print(f"Test 6 - Words: {words6}, MaxWidth: {maxWidth6}")
    print("Expected:")
    print('["a  b  c  d  e"]')
    print(f"Original: {result6a}")
    print(f"Alternative: {result6b}")
    print(f"Greedy: {result6c}")
    print(f"Functional: {result6d}")
    print(f"Iterative: {result6e}")
    print()
    
    # Test case 7 - Long word that takes entire line
    words7 = ["justification", "is", "hard"]
    maxWidth7 = 13
    result7a = full_justify(words7, maxWidth7)
    result7b = full_justify_alternative(words7, maxWidth7)
    result7c = full_justify_greedy(words7, maxWidth7)
    result7d = full_justify_functional(words7, maxWidth7)
    result7e = full_justify_iterative(words7, maxWidth7)
    print(f"Test 7 - Words: {words7}, MaxWidth: {maxWidth7}")
    print("Expected:")
    print('["justification", "is hard      "]')
    print(f"Original: {result7a}")
    print(f"Alternative: {result7b}")
    print(f"Greedy: {result7c}")
    print(f"Functional: {result7d}")
    print(f"Iterative: {result7e}")
    print()
    
    # Test case 8 - Single character words
    words8 = ["a", "b", "c", "d", "e", "f", "g"]
    maxWidth8 = 7
    result8a = full_justify(words8, maxWidth8)
    result8b = full_justify_alternative(words8, maxWidth8)
    result8c = full_justify_greedy(words8, maxWidth8)
    result8d = full_justify_functional(words8, maxWidth8)
    result8e = full_justify_iterative(words8, maxWidth8)
    print(f"Test 8 - Words: {words8}, MaxWidth: {maxWidth8}")
    print("Expected:")
    print('["a  b  c", "d  e  f", "g      "]')
    print(f"Original: {result8a}")
    print(f"Alternative: {result8b}")
    print(f"Greedy: {result8c}")
    print(f"Functional: {result8d}")
    print(f"Iterative: {result8e}")
    print()
    
    # Test case 9 - Empty words list
    words9 = []
    maxWidth9 = 10
    result9a = full_justify(words9, maxWidth9)
    result9b = full_justify_alternative(words9, maxWidth9)
    result9c = full_justify_greedy(words9, maxWidth9)
    result9d = full_justify_functional(words9, maxWidth9)
    result9e = full_justify_iterative(words9, maxWidth9)
    print(f"Test 9 - Words: {words9}, MaxWidth: {maxWidth9}")
    print("Expected:")
    print('[]')
    print(f"Original: {result9a}")
    print(f"Alternative: {result9b}")
    print(f"Greedy: {result9c}")
    print(f"Functional: {result9d}")
    print(f"Iterative: {result9e}")
    print()
    
    # Test case 10 - Very wide line
    words10 = ["The", "quick", "brown", "fox"]
    maxWidth10 = 30
    result10a = full_justify(words10, maxWidth10)
    result10b = full_justify_alternative(words10, maxWidth10)
    result10c = full_justify_greedy(words10, maxWidth10)
    result10d = full_justify_functional(words10, maxWidth10)
    result10e = full_justify_iterative(words10, maxWidth10)
    print(f"Test 10 - Words: {words10}, MaxWidth: {maxWidth10}")
    print("Expected:")
    print('["The     quick     brown     fox"]')
    print(f"Original: {result10a}")
    print(f"Alternative: {result10b}")
    print(f"Greedy: {result10c}")
    print(f"Functional: {result10d}")
    print(f"Iterative: {result10e}") 