"""
17. Letter Combinations of a Phone Number

Problem:
Given a string containing digits from 2-9 inclusive, return all possible letter combinations 
that the number could represent. Return the answer in any order.

A mapping of digits to letters (just like on the telephone buttons) is given below. 
Note that 1 does not map to any letters.

2: abc
3: def
4: ghi
5: jkl
6: mno
7: pqrs
8: tuv
9: wxyz

Example 1:
Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]

Example 2:
Input: digits = ""
Output: []

Example 3:
Input: digits = "2"
Output: ["a","b","c"]

Time Complexity: O(3^m * 4^n) where m is number of digits with 3 letters, n is number with 4 letters
Space Complexity: O(3^m * 4^n) for storing all combinations
"""


def letter_combinations_backtrack(digits):
    """
    Backtracking approach - optimal solution.
    
    Time Complexity: O(3^m * 4^n) where m is digits with 3 letters, n is digits with 4 letters
    Space Complexity: O(3^m * 4^n) for storing results + O(len(digits)) for recursion stack
    
    Algorithm:
    1. Use backtracking to explore all possible combinations
    2. At each step, try all letters corresponding to current digit
    3. Build path incrementally and backtrack when complete
    """
    if not digits:
        return []
    
    # Digit to letters mapping
    phone_map = {
        '2': 'abc',
        '3': 'def', 
        '4': 'ghi',
        '5': 'jkl',
        '6': 'mno',
        '7': 'pqrs',
        '8': 'tuv',
        '9': 'wxyz'
    }
    
    result = []
    
    def backtrack(index, path):
        # Base case: if we've processed all digits
        if index == len(digits):
            result.append(path)
            return
        
        # Get current digit and its corresponding letters
        current_digit = digits[index]
        letters = phone_map[current_digit]
        
        # Try each letter for current digit
        for letter in letters:
            backtrack(index + 1, path + letter)
    
    backtrack(0, "")
    return result


def letter_combinations_iterative(digits):
    """
    Iterative approach using queue (BFS-style).
    
    Time Complexity: O(3^m * 4^n)
    Space Complexity: O(3^m * 4^n)
    
    Algorithm:
    1. Use a queue to build combinations iteratively
    2. For each digit, expand all current combinations
    3. Add each letter to all existing combinations
    """
    if not digits:
        return []
    
    phone_map = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }
    
    # Start with empty string
    queue = [""]
    
    for digit in digits:
        letters = phone_map[digit]
        new_queue = []
        
        # For each existing combination
        for combination in queue:
            # Add each possible letter
            for letter in letters:
                new_queue.append(combination + letter)
        
        queue = new_queue
    
    return queue


def letter_combinations_recursive(digits):
    """
    Pure recursive approach without explicit backtracking.
    
    Time Complexity: O(3^m * 4^n)
    Space Complexity: O(3^m * 4^n)
    
    Algorithm:
    1. Recursively process each digit
    2. Combine results from current digit with remaining digits
    3. Build final combinations by cartesian product
    """
    if not digits:
        return []
    
    phone_map = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }
    
    def generate(remaining_digits):
        # Base case: no more digits
        if not remaining_digits:
            return [""]
        
        # Get first digit and process remaining
        first_digit = remaining_digits[0]
        first_letters = phone_map[first_digit]
        rest_combinations = generate(remaining_digits[1:])
        
        result = []
        for letter in first_letters:
            for rest in rest_combinations:
                result.append(letter + rest)
        
        return result
    
    return generate(digits)


def letter_combinations_product(digits):
    """
    Using itertools.product approach (Pythonic).
    
    Time Complexity: O(3^m * 4^n)
    Space Complexity: O(3^m * 4^n)
    
    Algorithm:
    1. Map each digit to its letters
    2. Use itertools.product to get cartesian product
    3. Join letters to form strings
    """
    if not digits:
        return []
    
    from itertools import product
    
    phone_map = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }
    
    # Get letter groups for each digit
    letter_groups = [phone_map[digit] for digit in digits]
    
    # Generate all combinations using cartesian product
    combinations = product(*letter_groups)
    
    # Join letters to form strings
    return [''.join(combo) for combo in combinations]


def test_letter_combinations():
    """Test all implementations with various test cases."""
    
    test_cases = [
        ("23", ["ad","ae","af","bd","be","bf","cd","ce","cf"]),
        ("", []),
        ("2", ["a","b","c"]),
        ("7", ["p","q","r","s"]),
        ("234", ["adg","adh","adi","aeg","aeh","aei","afg","afh","afi",
                 "bdg","bdh","bdi","beg","beh","bei","bfg","bfh","bfi",
                 "cdg","cdh","cdi","ceg","ceh","cei","cfg","cfh","cfi"])
    ]
    
    implementations = [
        ("Backtracking", letter_combinations_backtrack),
        ("Iterative", letter_combinations_iterative),
        ("Recursive", letter_combinations_recursive),
        ("Product", letter_combinations_product)
    ]
    
    print("Testing Letter Combinations of a Phone Number...")
    
    for i, (digits, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: digits = '{digits}'")
        print(f"Expected length: {len(expected)}")
        
        for impl_name, impl_func in implementations:
            result = impl_func(digits)
            # Sort both for comparison since order may vary
            result_sorted = sorted(result)
            expected_sorted = sorted(expected)
            
            is_correct = result_sorted == expected_sorted
            print(f"{impl_name:15} | Length: {len(result):3} | {'✓' if is_correct else '✗'}")
            
            if not is_correct and len(result) <= 20:
                print(f"                  Got: {result}")
                print(f"                  Expected: {expected}")


if __name__ == "__main__":
    test_letter_combinations() 