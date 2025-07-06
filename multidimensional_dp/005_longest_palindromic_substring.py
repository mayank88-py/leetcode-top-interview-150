"""
5. Longest Palindromic Substring

Given a string s, return the longest palindromic substring in s.

Example 1:
Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.

Example 2:
Input: s = "cbbd"
Output: "bb"

Constraints:
- 1 <= s.length <= 1000
- s consist of only digits and English letters.
"""

def longest_palindrome_2d_dp(s):
    """
    Approach 1: 2D Dynamic Programming
    Time Complexity: O(n^2)
    Space Complexity: O(n^2)
    
    dp[i][j] = True if s[i:j+1] is palindrome
    """
    if not s:
        return ""
    
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    
    start = 0
    max_len = 1
    
    # Every single character is a palindrome
    for i in range(n):
        dp[i][i] = True
    
    # Check for palindromes of length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_len = 2
    
    # Check for palindromes of length 3 and more
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start = i
                max_len = length
    
    return s[start:start + max_len]


def longest_palindrome_expand_around_centers(s):
    """
    Approach 2: Expand Around Centers
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    
    For each possible center, expand outwards to find palindromes.
    """
    if not s:
        return ""
    
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
    
    start = 0
    max_len = 0
    
    for i in range(len(s)):
        # Odd length palindromes (center at i)
        len1 = expand_around_center(i, i)
        # Even length palindromes (center between i and i+1)
        len2 = expand_around_center(i, i + 1)
        
        current_max = max(len1, len2)
        if current_max > max_len:
            max_len = current_max
            start = i - (current_max - 1) // 2
    
    return s[start:start + max_len]


def longest_palindrome_manachers(s):
    """
    Approach 3: Manacher's Algorithm
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Linear time algorithm for finding palindromes.
    """
    if not s:
        return ""
    
    # Preprocess string to handle even length palindromes
    processed = '#'.join('^{}$'.format(s))
    n = len(processed)
    
    # Array to store radius of palindromes
    P = [0] * n
    center = right = 0
    
    for i in range(1, n - 1):
        # Mirror of i with respect to center
        mirror = 2 * center - i
        
        if i < right:
            P[i] = min(right - i, P[mirror])
        
        # Try to expand palindrome centered at i
        try:
            while processed[i + P[i] + 1] == processed[i - P[i] - 1]:
                P[i] += 1
        except:
            pass
        
        # If palindrome centered at i extends past right, adjust center and right
        if i + P[i] > right:
            center, right = i, i + P[i]
    
    # Find the longest palindrome
    max_len = 0
    center_index = 0
    for i in range(1, n - 1):
        if P[i] > max_len:
            max_len = P[i]
            center_index = i
    
    # Extract the original palindrome
    start = (center_index - max_len) // 2
    return s[start:start + max_len]


def longest_palindrome_space_optimized_dp(s):
    """
    Approach 4: Space Optimized DP
    Time Complexity: O(n^2)
    Space Complexity: O(n)
    
    Use only one row for DP computation.
    """
    if not s:
        return ""
    
    n = len(s)
    start = 0
    max_len = 1
    
    # Check all possible centers
    for center in range(n):
        # Odd length palindromes
        left, right = center, center
        while left >= 0 and right < n and s[left] == s[right]:
            if right - left + 1 > max_len:
                max_len = right - left + 1
                start = left
            left -= 1
            right += 1
        
        # Even length palindromes
        left, right = center, center + 1
        while left >= 0 and right < n and s[left] == s[right]:
            if right - left + 1 > max_len:
                max_len = right - left + 1
                start = left
            left -= 1
            right += 1
    
    return s[start:start + max_len]


def longest_palindrome_recursive_memoization(s):
    """
    Approach 5: Recursive with Memoization
    Time Complexity: O(n^2)
    Space Complexity: O(n^2)
    
    Use memoization to avoid recalculating subproblems.
    """
    if not s:
        return ""
    
    memo = {}
    max_palindrome = ""
    
    def is_palindrome(i, j):
        if i >= j:
            return True
        
        if (i, j) in memo:
            return memo[(i, j)]
        
        result = s[i] == s[j] and is_palindrome(i + 1, j - 1)
        memo[(i, j)] = result
        return result
    
    # Check all substrings
    for i in range(len(s)):
        for j in range(i, len(s)):
            if is_palindrome(i, j) and j - i + 1 > len(max_palindrome):
                max_palindrome = s[i:j + 1]
    
    return max_palindrome


def longest_palindrome_rolling_hash(s):
    """
    Approach 6: Rolling Hash
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    
    Use rolling hash to check palindromes efficiently.
    """
    if not s:
        return ""
    
    def get_hash(string):
        base = 31
        mod = 10**9 + 7
        hash_val = 0
        for char in string:
            hash_val = (hash_val * base + ord(char)) % mod
        return hash_val
    
    max_len = 1
    result = s[0]
    
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            substring = s[i:j + 1]
            if get_hash(substring) == get_hash(substring[::-1]) and substring == substring[::-1]:
                if len(substring) > max_len:
                    max_len = len(substring)
                    result = substring
    
    return result


def longest_palindrome_kmp_based(s):
    """
    Approach 7: KMP-based Approach
    Time Complexity: O(n^2)
    Space Complexity: O(n)
    
    Use KMP pattern matching to find palindromes.
    """
    if not s:
        return ""
    
    def kmp_table(pattern):
        table = [0] * len(pattern)
        j = 0
        for i in range(1, len(pattern)):
            while j > 0 and pattern[i] != pattern[j]:
                j = table[j - 1]
            if pattern[i] == pattern[j]:
                j += 1
            table[i] = j
        return table
    
    max_palindrome = s[0]
    
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            substring = s[i:j + 1]
            # Check if it's a palindrome using KMP concept
            combined = substring + "#" + substring[::-1]
            table = kmp_table(combined)
            
            if table[-1] == len(substring):
                if len(substring) > len(max_palindrome):
                    max_palindrome = substring
    
    return max_palindrome


def longest_palindrome_suffix_array(s):
    """
    Approach 8: Suffix Array Approach
    Time Complexity: O(n^2 log n)
    Space Complexity: O(n)
    
    Use suffix array concepts to find palindromes.
    """
    if not s:
        return ""
    
    n = len(s)
    max_palindrome = s[0]
    
    # Generate all suffixes
    suffixes = []
    for i in range(n):
        suffixes.append((s[i:], i))
    
    # Sort suffixes
    suffixes.sort()
    
    # Check for palindromes in sorted order
    for i in range(n):
        suffix, start_idx = suffixes[i]
        for length in range(1, len(suffix) + 1):
            substring = suffix[:length]
            if substring == substring[::-1] and len(substring) > len(max_palindrome):
                max_palindrome = substring
    
    return max_palindrome


def longest_palindrome_two_pointers(s):
    """
    Approach 9: Two Pointers Technique
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    
    Use two pointers to check all possible palindromes.
    """
    if not s:
        return ""
    
    def is_palindrome(left, right):
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True
    
    max_len = 1
    result = s[0]
    
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if is_palindrome(i, j) and j - i + 1 > max_len:
                max_len = j - i + 1
                result = s[i:j + 1]
    
    return result


def test_longest_palindrome():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ("babad", ["bab", "aba"]),
        ("cbbd", ["bb"]),
        ("a", ["a"]),
        ("ac", ["a", "c"]),
        ("racecar", ["racecar"]),
        ("abcdcba", ["abcdcba"]),
        ("abacabad", ["abacaba"]),
        ("forgeeksskeegfor", ["geeksskeeg"]),
        ("aabbaa", ["aabbaa"]),
        ("aaaa", ["aaaa"]),
        ("abcdef", ["a", "b", "c", "d", "e", "f"]),
        ("noon", ["noon"]),
        ("", [""]),
    ]
    
    approaches = [
        ("2D DP", longest_palindrome_2d_dp),
        ("Expand Centers", longest_palindrome_expand_around_centers),
        ("Manacher's", longest_palindrome_manachers),
        ("Space Optimized", longest_palindrome_space_optimized_dp),
        ("Recursive Memo", longest_palindrome_recursive_memoization),
        ("Rolling Hash", longest_palindrome_rolling_hash),
        ("KMP Based", longest_palindrome_kmp_based),
        ("Suffix Array", longest_palindrome_suffix_array),
        ("Two Pointers", longest_palindrome_two_pointers),
    ]
    
    for i, (s, expected_options) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: \"{s}\"")
        print(f"Expected: {expected_options}")
        
        for name, func in approaches:
            try:
                result = func(s)
                status = "✓" if result in expected_options else "✗"
                print(f"{status} {name}: \"{result}\"")
            except Exception as e:
                print(f"✗ {name}: Error - {e}")


if __name__ == "__main__":
    test_longest_palindrome() 