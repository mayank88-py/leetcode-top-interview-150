"""
12. Integer to Roman

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

Given an integer, convert it to a roman numeral.

Example 1:
Input: num = 3
Output: "III"
Explanation: 3 is represented as 3 ones.

Example 2:
Input: num = 58
Output: "LVIII"
Explanation: L = 50, V = 5, III = 3.

Example 3:
Input: num = 1994
Output: "MCMXCIV"
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.

Time Complexity: O(1) since we have fixed number of symbols
Space Complexity: O(1)
"""


def int_to_roman(num):
    """
    Convert integer to Roman numeral using greedy approach.
    
    Args:
        num: Integer to convert (1 <= num <= 3999)
    
    Returns:
        Roman numeral string
    """
    # Define values and symbols in descending order
    values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    symbols = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    
    result = ""
    
    for i in range(len(values)):
        count = num // values[i]
        if count:
            result += symbols[i] * count
            num -= values[i] * count
    
    return result


def int_to_roman_hardcoded(num):
    """
    Convert integer to Roman numeral using hardcoded approach.
    
    Args:
        num: Integer to convert (1 <= num <= 3999)
    
    Returns:
        Roman numeral string
    """
    # Define mappings for each digit position
    thousands = ["", "M", "MM", "MMM"]
    hundreds = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
    tens = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
    ones = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
    
    return (thousands[num // 1000] + 
            hundreds[(num % 1000) // 100] + 
            tens[(num % 100) // 10] + 
            ones[num % 10])


def int_to_roman_recursive(num):
    """
    Convert integer to Roman numeral using recursive approach.
    
    Args:
        num: Integer to convert (1 <= num <= 3999)
    
    Returns:
        Roman numeral string
    """
    if num == 0:
        return ""
    
    # Check for special cases first
    if num >= 1000:
        return "M" + int_to_roman_recursive(num - 1000)
    elif num >= 900:
        return "CM" + int_to_roman_recursive(num - 900)
    elif num >= 500:
        return "D" + int_to_roman_recursive(num - 500)
    elif num >= 400:
        return "CD" + int_to_roman_recursive(num - 400)
    elif num >= 100:
        return "C" + int_to_roman_recursive(num - 100)
    elif num >= 90:
        return "XC" + int_to_roman_recursive(num - 90)
    elif num >= 50:
        return "L" + int_to_roman_recursive(num - 50)
    elif num >= 40:
        return "XL" + int_to_roman_recursive(num - 40)
    elif num >= 10:
        return "X" + int_to_roman_recursive(num - 10)
    elif num >= 9:
        return "IX" + int_to_roman_recursive(num - 9)
    elif num >= 5:
        return "V" + int_to_roman_recursive(num - 5)
    elif num >= 4:
        return "IV" + int_to_roman_recursive(num - 4)
    else:
        return "I" + int_to_roman_recursive(num - 1)


def int_to_roman_dictionary(num):
    """
    Convert integer to Roman numeral using dictionary lookup.
    
    Args:
        num: Integer to convert (1 <= num <= 3999)
    
    Returns:
        Roman numeral string
    """
    # Dictionary mapping values to roman numerals
    val_to_roman = {
        1000: "M", 900: "CM", 500: "D", 400: "CD",
        100: "C", 90: "XC", 50: "L", 40: "XL",
        10: "X", 9: "IX", 5: "V", 4: "IV", 1: "I"
    }
    
    result = ""
    
    for value in sorted(val_to_roman.keys(), reverse=True):
        count = num // value
        if count:
            result += val_to_roman[value] * count
            num -= value * count
    
    return result


def int_to_roman_modular(num):
    """
    Convert integer to Roman numeral using modular arithmetic.
    
    Args:
        num: Integer to convert (1 <= num <= 3999)
    
    Returns:
        Roman numeral string
    """
    def convert_digit(digit, one, five, ten):
        """Convert a single digit to Roman numeral"""
        if digit == 0:
            return ""
        elif digit <= 3:
            return one * digit
        elif digit == 4:
            return one + five
        elif digit == 5:
            return five
        elif digit <= 8:
            return five + one * (digit - 5)
        else:  # digit == 9
            return one + ten
    
    result = ""
    
    # Thousands
    result += "M" * (num // 1000)
    num %= 1000
    
    # Hundreds
    result += convert_digit(num // 100, "C", "D", "M")
    num %= 100
    
    # Tens
    result += convert_digit(num // 10, "X", "L", "C")
    num %= 10
    
    # Ones
    result += convert_digit(num, "I", "V", "X")
    
    return result


def int_to_roman_stack(num):
    """
    Convert integer to Roman numeral using stack-based approach.
    
    Args:
        num: Integer to convert (1 <= num <= 3999)
    
    Returns:
        Roman numeral string
    """
    # Define values and symbols
    mapping = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")
    ]
    
    stack = []
    
    for value, symbol in mapping:
        while num >= value:
            stack.append(symbol)
            num -= value
    
    return "".join(stack)


# Test cases
if __name__ == "__main__":
    # Test case 1
    num1 = 3
    result1a = int_to_roman(num1)
    result1b = int_to_roman_hardcoded(num1)
    result1c = int_to_roman_recursive(num1)
    result1d = int_to_roman_dictionary(num1)
    result1e = int_to_roman_modular(num1)
    result1f = int_to_roman_stack(num1)
    print(f"Test 1 - Number: {num1}, Expected: 'III'")
    print(f"Greedy: {result1a}, Hardcoded: {result1b}, Recursive: {result1c}, Dictionary: {result1d}, Modular: {result1e}, Stack: {result1f}")
    print()
    
    # Test case 2
    num2 = 58
    result2a = int_to_roman(num2)
    result2b = int_to_roman_hardcoded(num2)
    result2c = int_to_roman_recursive(num2)
    result2d = int_to_roman_dictionary(num2)
    result2e = int_to_roman_modular(num2)
    result2f = int_to_roman_stack(num2)
    print(f"Test 2 - Number: {num2}, Expected: 'LVIII'")
    print(f"Greedy: {result2a}, Hardcoded: {result2b}, Recursive: {result2c}, Dictionary: {result2d}, Modular: {result2e}, Stack: {result2f}")
    print()
    
    # Test case 3
    num3 = 1994
    result3a = int_to_roman(num3)
    result3b = int_to_roman_hardcoded(num3)
    result3c = int_to_roman_recursive(num3)
    result3d = int_to_roman_dictionary(num3)
    result3e = int_to_roman_modular(num3)
    result3f = int_to_roman_stack(num3)
    print(f"Test 3 - Number: {num3}, Expected: 'MCMXCIV'")
    print(f"Greedy: {result3a}, Hardcoded: {result3b}, Recursive: {result3c}, Dictionary: {result3d}, Modular: {result3e}, Stack: {result3f}")
    print()
    
    # Test case 4
    num4 = 4
    result4a = int_to_roman(num4)
    result4b = int_to_roman_hardcoded(num4)
    result4c = int_to_roman_recursive(num4)
    result4d = int_to_roman_dictionary(num4)
    result4e = int_to_roman_modular(num4)
    result4f = int_to_roman_stack(num4)
    print(f"Test 4 - Number: {num4}, Expected: 'IV'")
    print(f"Greedy: {result4a}, Hardcoded: {result4b}, Recursive: {result4c}, Dictionary: {result4d}, Modular: {result4e}, Stack: {result4f}")
    print()
    
    # Test case 5
    num5 = 9
    result5a = int_to_roman(num5)
    result5b = int_to_roman_hardcoded(num5)
    result5c = int_to_roman_recursive(num5)
    result5d = int_to_roman_dictionary(num5)
    result5e = int_to_roman_modular(num5)
    result5f = int_to_roman_stack(num5)
    print(f"Test 5 - Number: {num5}, Expected: 'IX'")
    print(f"Greedy: {result5a}, Hardcoded: {result5b}, Recursive: {result5c}, Dictionary: {result5d}, Modular: {result5e}, Stack: {result5f}")
    print()
    
    # Test case 6
    num6 = 3999
    result6a = int_to_roman(num6)
    result6b = int_to_roman_hardcoded(num6)
    result6c = int_to_roman_recursive(num6)
    result6d = int_to_roman_dictionary(num6)
    result6e = int_to_roman_modular(num6)
    result6f = int_to_roman_stack(num6)
    print(f"Test 6 - Number: {num6}, Expected: 'MMMCMXCIX'")
    print(f"Greedy: {result6a}, Hardcoded: {result6b}, Recursive: {result6c}, Dictionary: {result6d}, Modular: {result6e}, Stack: {result6f}")
    print()
    
    # Test case 7
    num7 = 1
    result7a = int_to_roman(num7)
    result7b = int_to_roman_hardcoded(num7)
    result7c = int_to_roman_recursive(num7)
    result7d = int_to_roman_dictionary(num7)
    result7e = int_to_roman_modular(num7)
    result7f = int_to_roman_stack(num7)
    print(f"Test 7 - Number: {num7}, Expected: 'I'")
    print(f"Greedy: {result7a}, Hardcoded: {result7b}, Recursive: {result7c}, Dictionary: {result7d}, Modular: {result7e}, Stack: {result7f}")
    print()
    
    # Test case 8
    num8 = 444
    result8a = int_to_roman(num8)
    result8b = int_to_roman_hardcoded(num8)
    result8c = int_to_roman_recursive(num8)
    result8d = int_to_roman_dictionary(num8)
    result8e = int_to_roman_modular(num8)
    result8f = int_to_roman_stack(num8)
    print(f"Test 8 - Number: {num8}, Expected: 'CDXLIV'")
    print(f"Greedy: {result8a}, Hardcoded: {result8b}, Recursive: {result8c}, Dictionary: {result8d}, Modular: {result8e}, Stack: {result8f}")
    print()
    
    # Test case 9
    num9 = 1000
    result9a = int_to_roman(num9)
    result9b = int_to_roman_hardcoded(num9)
    result9c = int_to_roman_recursive(num9)
    result9d = int_to_roman_dictionary(num9)
    result9e = int_to_roman_modular(num9)
    result9f = int_to_roman_stack(num9)
    print(f"Test 9 - Number: {num9}, Expected: 'M'")
    print(f"Greedy: {result9a}, Hardcoded: {result9b}, Recursive: {result9c}, Dictionary: {result9d}, Modular: {result9e}, Stack: {result9f}")
    print()
    
    # Test case 10
    num10 = 2021
    result10a = int_to_roman(num10)
    result10b = int_to_roman_hardcoded(num10)
    result10c = int_to_roman_recursive(num10)
    result10d = int_to_roman_dictionary(num10)
    result10e = int_to_roman_modular(num10)
    result10f = int_to_roman_stack(num10)
    print(f"Test 10 - Number: {num10}, Expected: 'MMXXI'")
    print(f"Greedy: {result10a}, Hardcoded: {result10b}, Recursive: {result10c}, Dictionary: {result10d}, Modular: {result10e}, Stack: {result10f}") 