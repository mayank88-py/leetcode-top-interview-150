"""
51. N-Queens

Problem:
The n-queens puzzle is the problem of placing n queens on an n×n chessboard such that no two queens attack each other.

Given an integer n, return all distinct solutions to the n-queens puzzle. You may return the answer in any order.

Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' 
both indicate a queen and an empty space, respectively.

Example 1:
Input: n = 4
Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]

Example 2:
Input: n = 1
Output: [["Q"]]

Time Complexity: O(N!) - there are N choices for first row, N-1 for second, etc.
Space Complexity: O(N^2) for storing the board
"""


def solve_n_queens_backtrack(n):
    """
    Classic backtracking approach.
    
    Time Complexity: O(N!)
    Space Complexity: O(N^2) for board + O(N) for recursion stack
    
    Algorithm:
    1. Try placing queen in each column of current row
    2. Check if placement is safe (no conflicts with previous queens)
    3. If safe, place queen and recursively solve for next row
    4. If no solution found, backtrack and try next column
    """
    result = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    
    def is_safe(row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        
        # Check diagonal (top-left to bottom-right)
        for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
            if board[i][j] == 'Q':
                return False
        
        # Check diagonal (top-right to bottom-left)
        for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
            if board[i][j] == 'Q':
                return False
        
        return True
    
    def backtrack(row):
        if row == n:
            # Convert board to required format
            solution = [''.join(row) for row in board]
            result.append(solution)
            return
        
        for col in range(n):
            if is_safe(row, col):
                # Place queen
                board[row][col] = 'Q'
                # Recursively solve for next row
                backtrack(row + 1)
                # Remove queen (backtrack)
                board[row][col] = '.'
    
    backtrack(0)
    return result


def solve_n_queens_optimized(n):
    """
    Optimized backtracking with conflict tracking.
    
    Time Complexity: O(N!)
    Space Complexity: O(N)
    
    Algorithm:
    1. Track conflicts using sets instead of checking board
    2. Use column, diagonal1, diagonal2 sets for O(1) conflict checking
    3. diagonal1: row - col is constant
    4. diagonal2: row + col is constant
    """
    result = []
    cols = set()
    diag1 = set()  # row - col
    diag2 = set()  # row + col
    
    def backtrack(row, current_board):
        if row == n:
            result.append(current_board[:])
            return
        
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            
            # Place queen
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            
            # Build board row
            board_row = '.' * col + 'Q' + '.' * (n - col - 1)
            current_board.append(board_row)
            
            # Recurse
            backtrack(row + 1, current_board)
            
            # Backtrack
            current_board.pop()
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)
    
    backtrack(0, [])
    return result


def solve_n_queens_positions(n):
    """
    Store only queen positions instead of full board.
    
    Time Complexity: O(N!)
    Space Complexity: O(N)
    
    Algorithm:
    1. Store only column positions of queens
    2. Check conflicts using position arithmetic
    3. Build board representation only at the end
    """
    result = []
    
    def is_safe(positions, row, col):
        for i in range(row):
            # Check column conflict
            if positions[i] == col:
                return False
            
            # Check diagonal conflicts
            if abs(positions[i] - col) == abs(i - row):
                return False
        
        return True
    
    def backtrack(row, positions):
        if row == n:
            # Convert positions to board representation
            board = []
            for r in range(n):
                row_str = '.' * positions[r] + 'Q' + '.' * (n - positions[r] - 1)
                board.append(row_str)
            result.append(board)
            return
        
        for col in range(n):
            if is_safe(positions, row, col):
                positions[row] = col
                backtrack(row + 1, positions)
    
    backtrack(0, [-1] * n)
    return result


def solve_n_queens_iterative(n):
    """
    Iterative solution using stack.
    
    Time Complexity: O(N!)
    Space Complexity: O(N^2) for stack states
    
    Algorithm:
    1. Use stack to store (row, positions, conflicts)
    2. Process each state and generate next valid states
    3. Add to result when all queens are placed
    """
    result = []
    # Stack stores (row, positions, cols_used, diag1_used, diag2_used)
    stack = [(0, [], set(), set(), set())]
    
    while stack:
        row, positions, cols, diag1, diag2 = stack.pop()
        
        if row == n:
            # Convert positions to board
            board = []
            for r in range(n):
                row_str = '.' * positions[r] + 'Q' + '.' * (n - positions[r] - 1)
                board.append(row_str)
            result.append(board)
            continue
        
        for col in range(n):
            if col not in cols and (row - col) not in diag1 and (row + col) not in diag2:
                new_positions = positions + [col]
                new_cols = cols | {col}
                new_diag1 = diag1 | {row - col}
                new_diag2 = diag2 | {row + col}
                stack.append((row + 1, new_positions, new_cols, new_diag1, new_diag2))
    
    return result


def test_n_queens():
    """Test all implementations with various test cases."""
    
    test_cases = [
        (1, [["Q"]]),
        (4, [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]),
        (8, None),  # Too many solutions to list
        (2, []),   # No solution exists
        (3, [])    # No solution exists
    ]
    
    implementations = [
        ("Backtracking", solve_n_queens_backtrack),
        ("Optimized", solve_n_queens_optimized),
        ("Positions Only", solve_n_queens_positions),
        ("Iterative", solve_n_queens_iterative)
    ]
    
    print("Testing N-Queens...")
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: n = {n}")
        
        if expected is not None:
            print(f"Expected solutions: {len(expected)}")
        
        for impl_name, impl_func in implementations:
            result = impl_func(n)
            
            if expected is not None:
                # Sort for comparison since order may vary
                result_sorted = [sorted(solution) for solution in result]
                result_sorted.sort()
                expected_sorted = [sorted(solution) for solution in expected]
                expected_sorted.sort()
                
                is_correct = result_sorted == expected_sorted
                print(f"{impl_name:15} | Count: {len(result):2} | {'✓' if is_correct else '✗'}")
                
                if not is_correct and len(result) <= 4:
                    print(f"                  Got: {result}")
                    print(f"                  Expected: {expected}")
            else:
                # For larger n, just show count
                print(f"{impl_name:15} | Count: {len(result):2}")


if __name__ == "__main__":
    test_n_queens() 