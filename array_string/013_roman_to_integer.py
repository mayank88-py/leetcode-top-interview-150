"""
13. Roman to Integer

Problem:
Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000

For example, 2 is written as II in Roman numeral, just two ones added together. 
12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. 
Instead, the number four is written as IV. Because the one is before the five we subtract it making four. 
The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.

Given a roman numeral, convert it to an integer.

Example 1:
Input: s = "III"
Output: 3
Explanation: III = 3.

Example 2:
Input: s = "LVIII"
Output: 58
Explanation: L = 50, V = 5, III = 3.

Example 3:
Input: s = "MCMXC"
Output: 1994
Explanation: M = 1000, CM = 900, XC = 90.

Time Complexity: O(n) where n is the length of the string
Space Complexity: O(1)
"""


def roman_to_int(s):
    """
    Convert Roman numeral to integer using forward scanning.
    
    Args:
        s: Roman numeral string
    
    Returns:
        Integer value of the Roman numeral
    """
    roman_map = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    
    result = 0
    i = 0
    
    while i < len(s):
        # If this is the last character or current character value >= next character value
        if i + 1 < len(s) and roman_map[s[i]] < roman_map[s[i + 1]]:
            # Subtraction case (e.g., IV, IX, XL, XC, CD, CM)
            result += roman_map[s[i + 1]] - roman_map[s[i]]
            i += 2
        else:
            # Normal case
            result += roman_map[s[i]]
            i += 1
    
    return result


def roman_to_int_reverse(s):
    """
    Convert Roman numeral to integer using reverse scanning.
    
    Args:
        s: Roman numeral string
    
    Returns:
        Integer value of the Roman numeral
    """
    roman_map = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    
    result = 0
    prev_value = 0
    
    for char in reversed(s):
        value = roman_map[char]
        
        if value < prev_value:
            # Subtraction case
            result -= value
        else:
            # Addition case
            result += value
        
        prev_value = value
    
    return result


def roman_to_int_substring(s):
    """
    Convert Roman numeral to integer using substring replacement.
    
    Args:
        s: Roman numeral string
    
    Returns:
        Integer value of the Roman numeral
    """
    # Define subtraction cases first
    subtraction_map = {
        'IV': 4, 'IX': 9, 'XL': 40, 'XC': 90, 'CD': 400, 'CM': 900
    }
    
    roman_map = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    
    result = 0
    i = 0
    
    while i < len(s):
        # Check for subtraction cases first
        if i + 1 < len(s) and s[i:i+2] in subtraction_map:
            result += subtraction_map[s[i:i+2]]
            i += 2
        else:
            result += roman_map[s[i]]
            i += 1
    
    return result


def roman_to_int_dictionary(s):
    """
    Convert Roman numeral to integer using comprehensive dictionary.
    
    Args:
        s: Roman numeral string
    
    Returns:
        Integer value of the Roman numeral
    """
    # Comprehensive mapping including subtraction cases
    roman_values = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000,
        'IV': 4, 'IX': 9, 'XL': 40, 'XC': 90, 'CD': 400, 'CM': 900
    }
    
    result = 0
    i = 0
    
    while i < len(s):
        # Try two-character substring first
        if i + 1 < len(s) and s[i:i+2] in roman_values:
            result += roman_values[s[i:i+2]]
            i += 2
        else:
            result += roman_values[s[i]]
            i += 1
    
    return result


def roman_to_int_stack(s):
    """
    Convert Roman numeral to integer using stack approach.
    
    Args:
        s: Roman numeral string
    
    Returns:
        Integer value of the Roman numeral
    """
    roman_map = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    
    stack = []
    
    for char in s:
        value = roman_map[char]
        
        # If stack is not empty and current value is greater than top of stack
        if stack and value > stack[-1]:
            # This is a subtraction case
            stack[-1] = value - stack[-1]
        else:
            stack.append(value)
    
    return sum(stack)


def roman_to_int_mathematical(s):
    """
    Convert Roman numeral to integer using mathematical approach.
    
    Args:
        s: Roman numeral string
    
    Returns:
        Integer value of the Roman numeral
    """
    roman_map = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    
    result = 0
    
    for i in range(len(s)):
        value = roman_map[s[i]]
        
        # If this is not the last character and current value < next value
        if i + 1 < len(s) and value < roman_map[s[i + 1]]:
            result -= value  # Subtract for subtraction case
        else:
            result += value  # Add normally
    
    return result


# Test cases
if __name__ == "__main__":
    # Test case 1
    s1 = "III"
    result1a = roman_to_int(s1)
    result1b = roman_to_int_reverse(s1)
    result1c = roman_to_int_substring(s1)
    result1d = roman_to_int_dictionary(s1)
    result1e = roman_to_int_stack(s1)
    result1f = roman_to_int_mathematical(s1)
    print(f"Test 1 - Roman: {s1}, Expected: 3")
    print(f"Forward: {result1a}, Reverse: {result1b}, Substring: {result1c}, Dictionary: {result1d}, Stack: {result1e}, Mathematical: {result1f}")
    print()
    
    # Test case 2
    s2 = "LVIII"
    result2a = roman_to_int(s2)
    result2b = roman_to_int_reverse(s2)
    result2c = roman_to_int_substring(s2)
    result2d = roman_to_int_dictionary(s2)
    result2e = roman_to_int_stack(s2)
    result2f = roman_to_int_mathematical(s2)
    print(f"Test 2 - Roman: {s2}, Expected: 58")
    print(f"Forward: {result2a}, Reverse: {result2b}, Substring: {result2c}, Dictionary: {result2d}, Stack: {result2e}, Mathematical: {result2f}")
    print()
    
    # Test case 3
    s3 = "MCMXC"
    result3a = roman_to_int(s3)
    result3b = roman_to_int_reverse(s3)
    result3c = roman_to_int_substring(s3)
    result3d = roman_to_int_dictionary(s3)
    result3e = roman_to_int_stack(s3)
    result3f = roman_to_int_mathematical(s3)
    print(f"Test 3 - Roman: {s3}, Expected: 1994")
    print(f"Forward: {result3a}, Reverse: {result3b}, Substring: {result3c}, Dictionary: {result3d}, Stack: {result3e}, Mathematical: {result3f}")
    print()
    
    # Test case 4 - Simple subtraction cases
    s4 = "IV"
    result4a = roman_to_int(s4)
    result4b = roman_to_int_reverse(s4)
    result4c = roman_to_int_substring(s4)
    result4d = roman_to_int_dictionary(s4)
    result4e = roman_to_int_stack(s4)
    result4f = roman_to_int_mathematical(s4)
    print(f"Test 4 - Roman: {s4}, Expected: 4")
    print(f"Forward: {result4a}, Reverse: {result4b}, Substring: {result4c}, Dictionary: {result4d}, Stack: {result4e}, Mathematical: {result4f}")
    print()
    
    # Test case 5
    s5 = "IX"
    result5a = roman_to_int(s5)
    result5b = roman_to_int_reverse(s5)
    result5c = roman_to_int_substring(s5)
    result5d = roman_to_int_dictionary(s5)
    result5e = roman_to_int_stack(s5)
    result5f = roman_to_int_mathematical(s5)
    print(f"Test 5 - Roman: {s5}, Expected: 9")
    print(f"Forward: {result5a}, Reverse: {result5b}, Substring: {result5c}, Dictionary: {result5d}, Stack: {result5e}, Mathematical: {result5f}")
    print()
    
    # Test case 6 - Large number
    s6 = "MMMCMXCIX"
    result6a = roman_to_int(s6)
    result6b = roman_to_int_reverse(s6)
    result6c = roman_to_int_substring(s6)
    result6d = roman_to_int_dictionary(s6)
    result6e = roman_to_int_stack(s6)
    result6f = roman_to_int_mathematical(s6)
    print(f"Test 6 - Roman: {s6}, Expected: 3999")
    print(f"Forward: {result6a}, Reverse: {result6b}, Substring: {result6c}, Dictionary: {result6d}, Stack: {result6e}, Mathematical: {result6f}")
    print()
    
    # Test case 7 - Single character
    s7 = "M"
    result7a = roman_to_int(s7)
    result7b = roman_to_int_reverse(s7)
    result7c = roman_to_int_substring(s7)
    result7d = roman_to_int_dictionary(s7)
    result7e = roman_to_int_stack(s7)
    result7f = roman_to_int_mathematical(s7)
    print(f"Test 7 - Roman: {s7}, Expected: 1000")
    print(f"Forward: {result7a}, Reverse: {result7b}, Substring: {result7c}, Dictionary: {result7d}, Stack: {result7e}, Mathematical: {result7f}")
    print()
    
    # Test case 8 - Multiple subtraction cases
    s8 = "CDXLIV"
    result8a = roman_to_int(s8)
    result8b = roman_to_int_reverse(s8)
    result8c = roman_to_int_substring(s8)
    result8d = roman_to_int_dictionary(s8)
    result8e = roman_to_int_stack(s8)
    result8f = roman_to_int_mathematical(s8)
    print(f"Test 8 - Roman: {s8}, Expected: 444")
    print(f"Forward: {result8a}, Reverse: {result8b}, Substring: {result8c}, Dictionary: {result8d}, Stack: {result8e}, Mathematical: {result8f}")
    print()
    
    # Test case 9 - All basic symbols
    s9 = "MDCLXVI"
    result9a = roman_to_int(s9)
    result9b = roman_to_int_reverse(s9)
    result9c = roman_to_int_substring(s9)
    result9d = roman_to_int_dictionary(s9)
    result9e = roman_to_int_stack(s9)
    result9f = roman_to_int_mathematical(s9)
    print(f"Test 9 - Roman: {s9}, Expected: 1666")
    print(f"Forward: {result9a}, Reverse: {result9b}, Substring: {result9c}, Dictionary: {result9d}, Stack: {result9e}, Mathematical: {result9f}")
    print()
    
    # Test case 10 - Year example
    s10 = "MMXXI"
    result10a = roman_to_int(s10)
    result10b = roman_to_int_reverse(s10)
    result10c = roman_to_int_substring(s10)
    result10d = roman_to_int_dictionary(s10)
    result10e = roman_to_int_stack(s10)
    result10f = roman_to_int_mathematical(s10)
    print(f"Test 10 - Roman: {s10}, Expected: 2021")
    print(f"Forward: {result10a}, Reverse: {result10b}, Substring: {result10c}, Dictionary: {result10d}, Stack: {result10e}, Mathematical: {result10f}") 