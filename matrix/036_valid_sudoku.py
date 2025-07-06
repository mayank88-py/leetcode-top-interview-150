"""
36. Valid Sudoku

Problem:
Determine if a 9x9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:
1. Each row must contain the digits 1-9 without repetition.
2. Each column must contain the digits 1-9 without repetition.
3. Each of the nine 3x3 sub-boxes of the grid must contain the digits 1-9 without repetition.

Note:
- A Sudoku board (partially filled) could be valid but is not necessarily solvable.
- Only the filled cells need to be validated according to the mentioned rules.

Example 1:
Input: board = 
[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: true

Time Complexity: O(1) - constant 9x9 board
Space Complexity: O(1) - constant space for tracking
"""


def is_valid_sudoku(board):
    """
    Check if a 9x9 Sudoku board is valid using hash sets.
    
    Args:
        board: 9x9 list of lists representing the sudoku board
    
    Returns:
        True if the board is valid, False otherwise
    """
    # Track seen numbers in rows, columns, and boxes
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    
    for i in range(9):
        for j in range(9):
            if board[i][j] != '.':
                num = board[i][j]
                box_index = (i // 3) * 3 + (j // 3)
                
                # Check if number already exists in row, column, or box
                if num in rows[i] or num in cols[j] or num in boxes[box_index]:
                    return False
                
                # Add number to respective sets
                rows[i].add(num)
                cols[j].add(num)
                boxes[box_index].add(num)
    
    return True


def is_valid_sudoku_single_pass(board):
    """
    Check if a 9x9 Sudoku board is valid using single dictionary.
    
    Args:
        board: 9x9 list of lists representing the sudoku board
    
    Returns:
        True if the board is valid, False otherwise
    """
    seen = set()
    
    for i in range(9):
        for j in range(9):
            if board[i][j] != '.':
                num = board[i][j]
                
                # Create unique identifiers for row, column, and box
                row_id = f"row{i}-{num}"
                col_id = f"col{j}-{num}"
                box_id = f"box{i//3}{j//3}-{num}"
                
                # Check if any identifier already exists
                if row_id in seen or col_id in seen or box_id in seen:
                    return False
                
                # Add identifiers to seen set
                seen.add(row_id)
                seen.add(col_id)
                seen.add(box_id)
    
    return True


def is_valid_sudoku_arrays(board):
    """
    Check if a 9x9 Sudoku board is valid using boolean arrays.
    
    Args:
        board: 9x9 list of lists representing the sudoku board
    
    Returns:
        True if the board is valid, False otherwise
    """
    # Use boolean arrays to track seen numbers (1-9)
    rows = [[False] * 9 for _ in range(9)]
    cols = [[False] * 9 for _ in range(9)]
    boxes = [[False] * 9 for _ in range(9)]
    
    for i in range(9):
        for j in range(9):
            if board[i][j] != '.':
                num = int(board[i][j]) - 1  # Convert to 0-8 index
                box_index = (i // 3) * 3 + (j // 3)
                
                # Check if number already exists
                if rows[i][num] or cols[j][num] or boxes[box_index][num]:
                    return False
                
                # Mark number as seen
                rows[i][num] = True
                cols[j][num] = True
                boxes[box_index][num] = True
    
    return True


def is_valid_sudoku_bit_manipulation(board):
    """
    Check if a 9x9 Sudoku board is valid using bit manipulation.
    
    Args:
        board: 9x9 list of lists representing the sudoku board
    
    Returns:
        True if the board is valid, False otherwise
    """
    # Use integers as bit vectors
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9
    
    for i in range(9):
        for j in range(9):
            if board[i][j] != '.':
                num = int(board[i][j])
                bit = 1 << num  # Create bit mask for this number
                box_index = (i // 3) * 3 + (j // 3)
                
                # Check if bit is already set
                if (rows[i] & bit) or (cols[j] & bit) or (boxes[box_index] & bit):
                    return False
                
                # Set the bit
                rows[i] |= bit
                cols[j] |= bit
                boxes[box_index] |= bit
    
    return True


def is_valid_sudoku_functional(board):
    """
    Check if a 9x9 Sudoku board is valid using functional approach.
    
    Args:
        board: 9x9 list of lists representing the sudoku board
    
    Returns:
        True if the board is valid, False otherwise
    """
    def get_numbers(cells):
        """Extract non-empty numbers from cells"""
        return [cell for cell in cells if cell != '.']
    
    def is_valid_unit(numbers):
        """Check if a unit (row/column/box) is valid"""
        return len(numbers) == len(set(numbers))
    
    # Check all rows
    for row in board:
        if not is_valid_unit(get_numbers(row)):
            return False
    
    # Check all columns
    for j in range(9):
        col = [board[i][j] for i in range(9)]
        if not is_valid_unit(get_numbers(col)):
            return False
    
    # Check all 3x3 boxes
    for box_row in range(3):
        for box_col in range(3):
            box = []
            for i in range(3):
                for j in range(3):
                    box.append(board[box_row * 3 + i][box_col * 3 + j])
            if not is_valid_unit(get_numbers(box)):
                return False
    
    return True


# Test cases
if __name__ == "__main__":
    # Test case 1 - Valid sudoku
    board1 = [
        ["5","3",".",".","7",".",".",".","."],
        ["6",".",".","1","9","5",".",".","."],
        [".","9","8",".",".",".",".","6","."],
        ["8",".",".",".","6",".",".",".","3"],
        ["4",".",".","8",".","3",".",".","1"],
        ["7",".",".",".","2",".",".",".","6"],
        [".","6",".",".",".",".","2","8","."],
        [".",".",".","4","1","9",".",".","5"],
        [".",".",".",".","8",".",".","7","9"]
    ]
    result1a = is_valid_sudoku(board1)
    result1b = is_valid_sudoku_single_pass(board1)
    result1c = is_valid_sudoku_arrays(board1)
    result1d = is_valid_sudoku_bit_manipulation(board1)
    result1e = is_valid_sudoku_functional(board1)
    print(f"Test 1 - Expected: True")
    print(f"HashSet: {result1a}, SinglePass: {result1b}, Arrays: {result1c}, BitManip: {result1d}, Functional: {result1e}")
    print()
    
    # Test case 2 - Invalid sudoku (duplicate in row)
    board2 = [
        ["8","3",".",".","7",".",".",".","."],
        ["6",".",".","1","9","5",".",".","."],
        [".","9","8",".",".",".",".","6","."],
        ["8",".",".",".","6",".",".",".","3"],
        ["4",".",".","8",".","3",".",".","1"],
        ["7",".",".",".","2",".",".",".","6"],
        [".","6",".",".",".",".","2","8","."],
        [".",".",".","4","1","9",".",".","5"],
        [".",".",".",".","8",".",".","7","9"]
    ]
    result2a = is_valid_sudoku(board2)
    result2b = is_valid_sudoku_single_pass(board2)
    result2c = is_valid_sudoku_arrays(board2)
    result2d = is_valid_sudoku_bit_manipulation(board2)
    result2e = is_valid_sudoku_functional(board2)
    print(f"Test 2 - Expected: False (duplicate 8 in first row)")
    print(f"HashSet: {result2a}, SinglePass: {result2b}, Arrays: {result2c}, BitManip: {result2d}, Functional: {result2e}")
    print()
    
    # Test case 3 - Invalid sudoku (duplicate in column)
    board3 = [
        ["5","3",".",".","7",".",".",".","."],
        ["6",".",".","1","9","5",".",".","."],
        [".","9","8",".",".",".",".","6","."],
        ["8",".",".",".","6",".",".",".","3"],
        ["4",".",".","8",".","3",".",".","1"],
        ["7",".",".",".","2",".",".",".","6"],
        [".","6",".",".",".",".","2","8","."],
        [".",".",".","4","1","9",".",".","5"],
        ["5",".",".",".","8",".",".","7","9"]
    ]
    result3a = is_valid_sudoku(board3)
    result3b = is_valid_sudoku_single_pass(board3)
    result3c = is_valid_sudoku_arrays(board3)
    result3d = is_valid_sudoku_bit_manipulation(board3)
    result3e = is_valid_sudoku_functional(board3)
    print(f"Test 3 - Expected: False (duplicate 5 in first column)")
    print(f"HashSet: {result3a}, SinglePass: {result3b}, Arrays: {result3c}, BitManip: {result3d}, Functional: {result3e}")
    print()
    
    # Test case 4 - Invalid sudoku (duplicate in box)
    board4 = [
        ["5","3",".",".","7",".",".",".","."],
        ["6",".",".","1","9","5",".",".","."],
        [".","9","8",".",".",".",".","6","."],
        ["8",".",".",".","6",".",".",".","3"],
        ["4",".",".","8",".","3",".",".","1"],
        ["7",".",".",".","2",".",".",".","6"],
        [".","6",".",".",".",".","2","8","."],
        [".",".",".","4","1","9",".",".","5"],
        [".",".",".",".","8",".",".","7","9"]
    ]
    # Modify to create duplicate in box
    board4[0][2] = "5"  # This creates duplicate 5 in top-left box
    result4a = is_valid_sudoku(board4)
    result4b = is_valid_sudoku_single_pass(board4)
    result4c = is_valid_sudoku_arrays(board4)
    result4d = is_valid_sudoku_bit_manipulation(board4)
    result4e = is_valid_sudoku_functional(board4)
    print(f"Test 4 - Expected: False (duplicate 5 in top-left box)")
    print(f"HashSet: {result4a}, SinglePass: {result4b}, Arrays: {result4c}, BitManip: {result4d}, Functional: {result4e}")
    print()
    
    # Test case 5 - Empty board
    board5 = [["." for _ in range(9)] for _ in range(9)]
    result5a = is_valid_sudoku(board5)
    result5b = is_valid_sudoku_single_pass(board5)
    result5c = is_valid_sudoku_arrays(board5)
    result5d = is_valid_sudoku_bit_manipulation(board5)
    result5e = is_valid_sudoku_functional(board5)
    print(f"Test 5 - Expected: True (empty board)")
    print(f"HashSet: {result5a}, SinglePass: {result5b}, Arrays: {result5c}, BitManip: {result5d}, Functional: {result5e}")
    print()
    
    # Test case 6 - Single number
    board6 = [["." for _ in range(9)] for _ in range(9)]
    board6[0][0] = "5"
    result6a = is_valid_sudoku(board6)
    result6b = is_valid_sudoku_single_pass(board6)
    result6c = is_valid_sudoku_arrays(board6)
    result6d = is_valid_sudoku_bit_manipulation(board6)
    result6e = is_valid_sudoku_functional(board6)
    print(f"Test 6 - Expected: True (single number)")
    print(f"HashSet: {result6a}, SinglePass: {result6b}, Arrays: {result6c}, BitManip: {result6d}, Functional: {result6e}")
    print()
    
    # Test case 7 - Full valid row
    board7 = [["." for _ in range(9)] for _ in range(9)]
    board7[0] = ["1","2","3","4","5","6","7","8","9"]
    result7a = is_valid_sudoku(board7)
    result7b = is_valid_sudoku_single_pass(board7)
    result7c = is_valid_sudoku_arrays(board7)
    result7d = is_valid_sudoku_bit_manipulation(board7)
    result7e = is_valid_sudoku_functional(board7)
    print(f"Test 7 - Expected: True (full valid row)")
    print(f"HashSet: {result7a}, SinglePass: {result7b}, Arrays: {result7c}, BitManip: {result7d}, Functional: {result7e}")
    print()
    
    # Test case 8 - Full valid column
    board8 = [["." for _ in range(9)] for _ in range(9)]
    for i in range(9):
        board8[i][0] = str(i + 1)
    result8a = is_valid_sudoku(board8)
    result8b = is_valid_sudoku_single_pass(board8)
    result8c = is_valid_sudoku_arrays(board8)
    result8d = is_valid_sudoku_bit_manipulation(board8)
    result8e = is_valid_sudoku_functional(board8)
    print(f"Test 8 - Expected: True (full valid column)")
    print(f"HashSet: {result8a}, SinglePass: {result8b}, Arrays: {result8c}, BitManip: {result8d}, Functional: {result8e}")
    print()
    
    # Test case 9 - Full valid box
    board9 = [["." for _ in range(9)] for _ in range(9)]
    num = 1
    for i in range(3):
        for j in range(3):
            board9[i][j] = str(num)
            num += 1
    result9a = is_valid_sudoku(board9)
    result9b = is_valid_sudoku_single_pass(board9)
    result9c = is_valid_sudoku_arrays(board9)
    result9d = is_valid_sudoku_bit_manipulation(board9)
    result9e = is_valid_sudoku_functional(board9)
    print(f"Test 9 - Expected: True (full valid box)")
    print(f"HashSet: {result9a}, SinglePass: {result9b}, Arrays: {result9c}, BitManip: {result9d}, Functional: {result9e}")
    print()
    
    # Test case 10 - Complex valid case
    board10 = [
        ["1","2","3","4","5","6","7","8","9"],
        ["4","5","6","7","8","9","1","2","3"],
        ["7","8","9","1","2","3","4","5","6"],
        ["2","3","4","5","6","7","8","9","1"],
        ["5","6","7","8","9","1","2","3","4"],
        ["8","9","1","2","3","4","5","6","7"],
        ["3","4","5","6","7","8","9","1","2"],
        ["6","7","8","9","1","2","3","4","5"],
        ["9","1","2","3","4","5","6","7","8"]
    ]
    result10a = is_valid_sudoku(board10)
    result10b = is_valid_sudoku_single_pass(board10)
    result10c = is_valid_sudoku_arrays(board10)
    result10d = is_valid_sudoku_bit_manipulation(board10)
    result10e = is_valid_sudoku_functional(board10)
    print(f"Test 10 - Expected: True (complete valid sudoku)")
    print(f"HashSet: {result10a}, SinglePass: {result10b}, Arrays: {result10c}, BitManip: {result10d}, Functional: {result10e}") 