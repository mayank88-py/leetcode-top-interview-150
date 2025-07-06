"""
30. Substring with Concatenation of All Words

Problem:
You are given a string s and an array of strings words. All the strings of words are of the same length.
A concatenated substring in s is a substring that contains all the strings of any permutation of words concatenated.

For example, if words = ["ab","cd","ef"], then "abcdef", "abefcd", "cdabef", "cdefab", "efabcd", 
and "efcdab" are all concatenated strings. "acdbef" is not a concatenated string because it is not 
the concatenation of any permutation of words.

Return the starting indices of all the concatenated substrings in s. You can return the answer in any order.

Example 1:
Input: s = "barfoothefoobarman", words = ["foo","bar"]
Output: [0,9]

Example 2:
Input: s = "wordgoodgoodgoodbestword", words = ["word","good","best","word"]
Output: []

Example 3:
Input: s = "barfoobar", words = ["foo","bar"]
Output: [0,3]

Time Complexity: O(N * M * L) where N is length of s, M is number of words, L is length of each word
Space Complexity: O(M * L)
"""


def find_substring(s, words):
    """
    Find all starting indices where concatenation of all words appears.
    
    Args:
        s: Input string
        words: List of words to concatenate
    
    Returns:
        List of starting indices
    """
    if not s or not words or not words[0]:
        return []
    
    word_len = len(words[0])
    word_count = len(words)
    total_len = word_len * word_count
    
    if len(s) < total_len:
        return []
    
    # Create word frequency map
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    result = []
    
    # Try each possible starting position
    for i in range(len(s) - total_len + 1):
        seen = {}
        j = 0
        
        # Check if we can form valid concatenation starting at position i
        while j < word_count:
            word = s[i + j * word_len:i + (j + 1) * word_len]
            
            if word not in word_freq:
                break
                
            seen[word] = seen.get(word, 0) + 1
            
            if seen[word] > word_freq[word]:
                break
                
            j += 1
        
        if j == word_count:
            result.append(i)
    
    return result


def find_substring_sliding_window(s, words):
    """
    Find all starting indices using sliding window optimization.
    
    Args:
        s: Input string
        words: List of words to concatenate
    
    Returns:
        List of starting indices
    """
    if not s or not words or not words[0]:
        return []
    
    word_len = len(words[0])
    word_count = len(words)
    total_len = word_len * word_count
    
    if len(s) < total_len:
        return []
    
    # Create word frequency map
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    result = []
    
    # Try each possible starting offset (0 to word_len - 1)
    for offset in range(word_len):
        left = offset
        right = offset
        seen = {}
        
        while right + word_len <= len(s):
            # Get the word at right pointer
            word = s[right:right + word_len]
            right += word_len
            
            if word not in word_freq:
                # Reset window
                seen.clear()
                left = right
                continue
            
            seen[word] = seen.get(word, 0) + 1
            
            # If we have too many of this word, shrink window from left
            while seen[word] > word_freq[word]:
                left_word = s[left:left + word_len]
                seen[left_word] -= 1
                if seen[left_word] == 0:
                    del seen[left_word]
                left += word_len
            
            # If window size matches total length, we found a match
            if right - left == total_len:
                result.append(left)
    
    return result


def find_substring_brute_force(s, words):
    """
    Find all starting indices using brute force with permutations.
    
    Args:
        s: Input string
        words: List of words to concatenate
    
    Returns:
        List of starting indices
    """
    if not s or not words or not words[0]:
        return []
    
    from itertools import permutations
    
    word_len = len(words[0])
    total_len = word_len * len(words)
    
    if len(s) < total_len:
        return []
    
    # Generate all permutations of words
    all_perms = set()
    for perm in permutations(words):
        all_perms.add(''.join(perm))
    
    result = []
    
    # Check each possible starting position
    for i in range(len(s) - total_len + 1):
        substring = s[i:i + total_len]
        if substring in all_perms:
            result.append(i)
    
    return result


# Test cases
if __name__ == "__main__":
    # Test case 1
    s1 = "barfoothefoobarman"
    words1 = ["foo", "bar"]
    result1a = find_substring(s1, words1)
    result1b = find_substring_sliding_window(s1, words1)
    result1c = find_substring_brute_force(s1, words1)
    print(f"Test 1 - Expected: [0,9], Basic: {result1a}, Sliding: {result1b}, Brute: {result1c}")
    
    # Test case 2
    s2 = "wordgoodgoodgoodbestword"
    words2 = ["word", "good", "best", "word"]
    result2a = find_substring(s2, words2)
    result2b = find_substring_sliding_window(s2, words2)
    result2c = find_substring_brute_force(s2, words2)
    print(f"Test 2 - Expected: [], Basic: {result2a}, Sliding: {result2b}, Brute: {result2c}")
    
    # Test case 3
    s3 = "barfoobar"
    words3 = ["foo", "bar"]
    result3a = find_substring(s3, words3)
    result3b = find_substring_sliding_window(s3, words3)
    result3c = find_substring_brute_force(s3, words3)
    print(f"Test 3 - Expected: [0,3], Basic: {result3a}, Sliding: {result3b}, Brute: {result3c}")
    
    # Test case 4 - Empty string
    s4 = ""
    words4 = ["foo", "bar"]
    result4a = find_substring(s4, words4)
    result4b = find_substring_sliding_window(s4, words4)
    result4c = find_substring_brute_force(s4, words4)
    print(f"Test 4 - Expected: [], Basic: {result4a}, Sliding: {result4b}, Brute: {result4c}")
    
    # Test case 5 - Single word
    s5 = "aaaa"
    words5 = ["aa"]
    result5a = find_substring(s5, words5)
    result5b = find_substring_sliding_window(s5, words5)
    result5c = find_substring_brute_force(s5, words5)
    print(f"Test 5 - Expected: [0,1,2], Basic: {result5a}, Sliding: {result5b}, Brute: {result5c}")
    
    # Test case 6 - No match
    s6 = "wordgoodgoodgoodbestword"
    words6 = ["word", "good", "best", "good"]
    result6a = find_substring(s6, words6)
    result6b = find_substring_sliding_window(s6, words6)
    result6c = find_substring_brute_force(s6, words6)
    print(f"Test 6 - Expected: [8], Basic: {result6a}, Sliding: {result6b}, Brute: {result6c}")
    
    # Test case 7 - Overlapping
    s7 = "lingmindraboofooowingdingbarrwingmonkeypoundcake"
    words7 = ["fooo", "barr", "wing", "ding", "wing"]
    result7a = find_substring(s7, words7)
    result7b = find_substring_sliding_window(s7, words7)
    result7c = find_substring_brute_force(s7, words7)
    print(f"Test 7 - Expected: [13], Basic: {result7a}, Sliding: {result7b}, Brute: {result7c}") 