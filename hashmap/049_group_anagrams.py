"""
49. Group Anagrams

Problem:
Given an array of strings strs, group the anagrams together. 
You can return the answer in any order.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, 
typically using all the original letters exactly once.

Example 1:
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Example 2:
Input: strs = [""]
Output: [[""]]

Example 3:
Input: strs = ["a"]
Output: [["a"]]

Time Complexity: O(n * m * log(m)) where n is number of strings, m is average length
Space Complexity: O(n * m) for storing the results
"""


def group_anagrams(strs):
    """
    Group anagrams together using sorted strings as keys.
    
    Args:
        strs: List of strings
    
    Returns:
        List of groups, where each group contains anagrams
    """
    anagram_groups = {}
    
    for s in strs:
        # Sort the string to create a key for anagrams
        sorted_str = ''.join(sorted(s))
        
        # Add to the appropriate group
        if sorted_str not in anagram_groups:
            anagram_groups[sorted_str] = []
        anagram_groups[sorted_str].append(s)
    
    # Return all groups as a list
    return list(anagram_groups.values())


def group_anagrams_char_count(strs):
    """
    Group anagrams using character count as key.
    
    Args:
        strs: List of strings
    
    Returns:
        List of groups, where each group contains anagrams
    """
    anagram_groups = {}
    
    for s in strs:
        # Count characters and create a tuple key
        char_count = [0] * 26
        for char in s:
            char_count[ord(char) - ord('a')] += 1
        
        # Use tuple as key (lists are not hashable)
        key = tuple(char_count)
        
        if key not in anagram_groups:
            anagram_groups[key] = []
        anagram_groups[key].append(s)
    
    return list(anagram_groups.values())


def group_anagrams_prime_product(strs):
    """
    Group anagrams using prime number products as keys.
    Each letter maps to a prime number, anagrams will have same product.
    
    Args:
        strs: List of strings
    
    Returns:
        List of groups, where each group contains anagrams
    """
    # Map each letter to a prime number
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 
              43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
    
    anagram_groups = {}
    
    for s in strs:
        # Calculate product of primes for each character
        product = 1
        for char in s:
            product *= primes[ord(char) - ord('a')]
        
        if product not in anagram_groups:
            anagram_groups[product] = []
        anagram_groups[product].append(s)
    
    return list(anagram_groups.values())


def group_anagrams_hash_function(strs):
    """
    Group anagrams using custom hash function.
    
    Args:
        strs: List of strings
    
    Returns:
        List of groups, where each group contains anagrams
    """
    def custom_hash(s):
        """Create custom hash for string"""
        # Use sum of character codes multiplied by their positions
        hash_value = 0
        for i, char in enumerate(s):
            hash_value += ord(char) * (i + 1)
        return hash_value
    
    anagram_groups = {}
    
    for s in strs:
        # Use length and character sum as composite key
        char_sum = sum(ord(c) for c in s)
        key = (len(s), char_sum)
        
        if key not in anagram_groups:
            anagram_groups[key] = []
        anagram_groups[key].append(s)
    
    # Filter out false positives by checking if strings are actually anagrams
    result = []
    for group in anagram_groups.values():
        if len(group) == 1:
            result.append(group)
        else:
            # Separate actual anagrams within this group
            sub_groups = {}
            for s in group:
                sorted_s = ''.join(sorted(s))
                if sorted_s not in sub_groups:
                    sub_groups[sorted_s] = []
                sub_groups[sorted_s].append(s)
            result.extend(sub_groups.values())
    
    return result


def group_anagrams_dictionary_key(strs):
    """
    Group anagrams using dictionary representation as key.
    
    Args:
        strs: List of strings
    
    Returns:
        List of groups, where each group contains anagrams
    """
    anagram_groups = {}
    
    for s in strs:
        # Create character frequency dictionary
        char_freq = {}
        for char in s:
            char_freq[char] = char_freq.get(char, 0) + 1
        
        # Convert to sorted tuple of items for hashing
        key = tuple(sorted(char_freq.items()))
        
        if key not in anagram_groups:
            anagram_groups[key] = []
        anagram_groups[key].append(s)
    
    return list(anagram_groups.values())


# Test cases
if __name__ == "__main__":
    # Test case 1
    strs1 = ["eat","tea","tan","ate","nat","bat"]
    result1a = group_anagrams(strs1)
    result1b = group_anagrams_char_count(strs1)
    result1c = group_anagrams_prime_product(strs1)
    result1d = group_anagrams_hash_function(strs1)
    result1e = group_anagrams_dictionary_key(strs1)
    print(f"Test 1 - Input: {strs1}")
    print(f"Sorted: {result1a}")
    print(f"CharCount: {result1b}")
    print(f"PrimeProduct: {result1c}")
    print(f"HashFunc: {result1d}")
    print(f"DictKey: {result1e}")
    print()
    
    # Test case 2
    strs2 = [""]
    result2a = group_anagrams(strs2)
    result2b = group_anagrams_char_count(strs2)
    result2c = group_anagrams_prime_product(strs2)
    result2d = group_anagrams_hash_function(strs2)
    result2e = group_anagrams_dictionary_key(strs2)
    print(f"Test 2 - Input: {strs2}")
    print(f"Sorted: {result2a}")
    print(f"CharCount: {result2b}")
    print(f"PrimeProduct: {result2c}")
    print(f"HashFunc: {result2d}")
    print(f"DictKey: {result2e}")
    print()
    
    # Test case 3
    strs3 = ["a"]
    result3a = group_anagrams(strs3)
    result3b = group_anagrams_char_count(strs3)
    result3c = group_anagrams_prime_product(strs3)
    result3d = group_anagrams_hash_function(strs3)
    result3e = group_anagrams_dictionary_key(strs3)
    print(f"Test 3 - Input: {strs3}")
    print(f"Sorted: {result3a}")
    print(f"CharCount: {result3b}")
    print(f"PrimeProduct: {result3c}")
    print(f"HashFunc: {result3d}")
    print(f"DictKey: {result3e}")
    print()
    
    # Test case 4 - No anagrams
    strs4 = ["abc", "def", "ghi"]
    result4a = group_anagrams(strs4)
    result4b = group_anagrams_char_count(strs4)
    result4c = group_anagrams_prime_product(strs4)
    result4d = group_anagrams_hash_function(strs4)
    result4e = group_anagrams_dictionary_key(strs4)
    print(f"Test 4 - Input: {strs4}")
    print(f"Sorted: {result4a}")
    print(f"CharCount: {result4b}")
    print(f"PrimeProduct: {result4c}")
    print(f"HashFunc: {result4d}")
    print(f"DictKey: {result4e}")
    print()
    
    # Test case 5 - All same letters
    strs5 = ["aaa", "aaa", "aaa"]
    result5a = group_anagrams(strs5)
    result5b = group_anagrams_char_count(strs5)
    result5c = group_anagrams_prime_product(strs5)
    result5d = group_anagrams_hash_function(strs5)
    result5e = group_anagrams_dictionary_key(strs5)
    print(f"Test 5 - Input: {strs5}")
    print(f"Sorted: {result5a}")
    print(f"CharCount: {result5b}")
    print(f"PrimeProduct: {result5c}")
    print(f"HashFunc: {result5d}")
    print(f"DictKey: {result5e}")
    print()
    
    # Test case 6 - Single character anagrams
    strs6 = ["a", "b", "a", "c", "b"]
    result6a = group_anagrams(strs6)
    result6b = group_anagrams_char_count(strs6)
    result6c = group_anagrams_prime_product(strs6)
    result6d = group_anagrams_hash_function(strs6)
    result6e = group_anagrams_dictionary_key(strs6)
    print(f"Test 6 - Input: {strs6}")
    print(f"Sorted: {result6a}")
    print(f"CharCount: {result6b}")
    print(f"PrimeProduct: {result6c}")
    print(f"HashFunc: {result6d}")
    print(f"DictKey: {result6e}")
    print()
    
    # Test case 7 - Mixed length anagrams
    strs7 = ["abc", "bca", "cab", "xyz", "zyx", "a"]
    result7a = group_anagrams(strs7)
    result7b = group_anagrams_char_count(strs7)
    result7c = group_anagrams_prime_product(strs7)
    result7d = group_anagrams_hash_function(strs7)
    result7e = group_anagrams_dictionary_key(strs7)
    print(f"Test 7 - Input: {strs7}")
    print(f"Sorted: {result7a}")
    print(f"CharCount: {result7b}")
    print(f"PrimeProduct: {result7c}")
    print(f"HashFunc: {result7d}")
    print(f"DictKey: {result7e}")
    print()
    
    # Test case 8 - Large group
    strs8 = ["listen", "silent", "enlist", "hello", "world", "act", "cat", "tac"]
    result8a = group_anagrams(strs8)
    result8b = group_anagrams_char_count(strs8)
    result8c = group_anagrams_prime_product(strs8)
    result8d = group_anagrams_hash_function(strs8)
    result8e = group_anagrams_dictionary_key(strs8)
    print(f"Test 8 - Input: {strs8}")
    print(f"Sorted: {result8a}")
    print(f"CharCount: {result8b}")
    print(f"PrimeProduct: {result8c}")
    print(f"HashFunc: {result8d}")
    print(f"DictKey: {result8e}")
    print()
    
    # Test case 9 - Empty list
    strs9 = []
    result9a = group_anagrams(strs9)
    result9b = group_anagrams_char_count(strs9)
    result9c = group_anagrams_prime_product(strs9)
    result9d = group_anagrams_hash_function(strs9)
    result9e = group_anagrams_dictionary_key(strs9)
    print(f"Test 9 - Input: {strs9}")
    print(f"Sorted: {result9a}")
    print(f"CharCount: {result9b}")
    print(f"PrimeProduct: {result9c}")
    print(f"HashFunc: {result9d}")
    print(f"DictKey: {result9e}")
    print()
    
    # Test case 10 - Repeated characters
    strs10 = ["aabb", "abab", "baba", "bbaa", "ccdd"]
    result10a = group_anagrams(strs10)
    result10b = group_anagrams_char_count(strs10)
    result10c = group_anagrams_prime_product(strs10)
    result10d = group_anagrams_hash_function(strs10)
    result10e = group_anagrams_dictionary_key(strs10)
    print(f"Test 10 - Input: {strs10}")
    print(f"Sorted: {result10a}")
    print(f"CharCount: {result10b}")
    print(f"PrimeProduct: {result10c}")
    print(f"HashFunc: {result10d}")
    print(f"DictKey: {result10e}") 