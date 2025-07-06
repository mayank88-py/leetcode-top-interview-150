"""
149. Max Points on a Line

Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane, 
return the maximum number of points that lie on the same straight line.

Example 1:
Input: points = [[1,1],[2,2],[3,3]]
Output: 3

Example 2:
Input: points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
Output: 4

Constraints:
- 1 <= points.length <= 300
- points[i].length == 2
- -10^4 <= xi, yi <= 10^4
- All the points are unique.
"""

def max_points_on_line_slope_map(points):
    """
    Approach 1: Slope Map with GCD
    Time Complexity: O(n^2)
    Space Complexity: O(n)
    
    For each point, calculate slopes to all other points and count.
    Use GCD to handle precision issues with fractions.
    """
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def get_slope(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        if dx == 0:
            return (float('inf'), 0)  # Vertical line
        if dy == 0:
            return (0, 1)  # Horizontal line
        
        # Reduce fraction using GCD
        g = gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        
        # Ensure consistent sign
        if dx < 0:
            dx, dy = -dx, -dy
        
        return (dy, dx)
    
    if len(points) <= 2:
        return len(points)
    
    max_points = 2
    
    for i in range(len(points)):
        slope_count = {}
        duplicates = 0
        local_max = 0
        
        for j in range(i + 1, len(points)):
            if points[i] == points[j]:
                duplicates += 1
                continue
            
            slope = get_slope(points[i], points[j])
            slope_count[slope] = slope_count.get(slope, 0) + 1
            local_max = max(local_max, slope_count[slope])
        
        max_points = max(max_points, local_max + duplicates + 1)
    
    return max_points


def max_points_on_line_float_slope(points):
    """
    Approach 2: Float Slope (with precision handling)
    Time Complexity: O(n^2)
    Space Complexity: O(n)
    
    Use floating point slopes with precision handling.
    """
    if len(points) <= 2:
        return len(points)
    
    max_points = 2
    
    for i in range(len(points)):
        slope_count = {}
        vertical_count = 0
        duplicates = 0
        local_max = 0
        
        for j in range(i + 1, len(points)):
            if points[i] == points[j]:
                duplicates += 1
                continue
            
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            
            if dx == 0:
                vertical_count += 1
                local_max = max(local_max, vertical_count)
            else:
                slope = dy / dx
                # Round to handle floating point precision
                slope = round(slope, 9)
                slope_count[slope] = slope_count.get(slope, 0) + 1
                local_max = max(local_max, slope_count[slope])
        
        max_points = max(max_points, local_max + duplicates + 1)
    
    return max_points


def max_points_on_line_line_equation(points):
    """
    Approach 3: Line Equation Method
    Time Complexity: O(n^3)
    Space Complexity: O(1)
    
    For each pair of points, check how many other points lie on the same line.
    """
    def points_on_line(p1, p2, p3):
        """Check if three points are collinear using cross product."""
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) == (p2[1] - p1[1]) * (p3[0] - p1[0])
    
    n = len(points)
    if n <= 2:
        return n
    
    max_points = 2
    
    for i in range(n):
        for j in range(i + 1, n):
            count = 2  # Current pair
            
            for k in range(n):
                if k != i and k != j:
                    if points_on_line(points[i], points[j], points[k]):
                        count += 1
            
            max_points = max(max_points, count)
    
    return max_points


def max_points_on_line_determinant(points):
    """
    Approach 4: Determinant Method
    Time Complexity: O(n^3)
    Space Complexity: O(1)
    
    Use determinant to check if three points are collinear.
    """
    def are_collinear(p1, p2, p3):
        """Check if three points are collinear using determinant."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        # Calculate determinant of the matrix
        # | x1 y1 1 |
        # | x2 y2 1 |
        # | x3 y3 1 |
        det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        return det == 0
    
    n = len(points)
    if n <= 2:
        return n
    
    max_points = 2
    
    for i in range(n):
        for j in range(i + 1, n):
            count = 2
            
            for k in range(n):
                if k != i and k != j:
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
            
            max_points = max(max_points, count)
    
    return max_points


def max_points_on_line_optimized(points):
    """
    Approach 5: Optimized Slope Calculation
    Time Complexity: O(n^2)
    Space Complexity: O(n)
    
    Optimized version with better slope calculation and duplicate handling.
    """
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def get_slope_key(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        if dx == 0 and dy == 0:
            return None  # Same point
        
        if dx == 0:
            return 'inf'  # Vertical line
        
        if dy == 0:
            return 0  # Horizontal line
        
        # Normalize the slope
        g = gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        
        # Ensure consistent representation
        if dx < 0:
            dx, dy = -dx, -dy
        
        return (dy, dx)
    
    n = len(points)
    if n <= 2:
        return n
    
    max_count = 1
    
    for i in range(n):
        slope_map = {}
        duplicates = 0
        
        for j in range(i + 1, n):
            slope_key = get_slope_key(points[i], points[j])
            
            if slope_key is None:
                duplicates += 1
            else:
                slope_map[slope_key] = slope_map.get(slope_key, 0) + 1
        
        # Find maximum points on any line through point i
        current_max = duplicates + 1  # Include point i itself
        
        if slope_map:
            current_max = max(current_max, max(slope_map.values()) + duplicates + 1)
        
        max_count = max(max_count, current_max)
    
    return max_count


def max_points_on_line_brute_force(points):
    """
    Approach 6: Brute Force
    Time Complexity: O(n^3)
    Space Complexity: O(1)
    
    Check every possible line and count points on it.
    """
    def on_same_line(p1, p2, p3):
        # Using cross product to avoid division
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    n = len(points)
    if n <= 2:
        return n
    
    max_points = 1
    
    for i in range(n):
        for j in range(i + 1, n):
            count = 2
            for k in range(n):
                if k != i and k != j and on_same_line(points[i], points[j], points[k]):
                    count += 1
            max_points = max(max_points, count)
    
    return max_points


def test_max_points_on_line():
    """Test all approaches with various test cases."""
    
    test_cases = [
        ([[1, 1], [2, 2], [3, 3]], 3),
        ([[1, 1], [3, 2], [5, 3], [4, 1], [2, 3], [1, 4]], 4),
        ([[1, 1]], 1),
        ([[1, 1], [2, 2]], 2),
        ([[0, 0], [1, 1], [0, 0]], 3),  # Duplicate points
        ([[1, 1], [1, 2], [1, 3]], 3),  # Vertical line
        ([[1, 1], [2, 1], [3, 1]], 3),  # Horizontal line
        ([[0, 0], [1, 1], [2, 2], [3, 4]], 3),
        ([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], 5),
        ([[0, 0], [1, 0], [0, 1]], 2),
        ([[2, 3], [3, 3], [-5, 3]], 3),
        ([[0, 1], [0, 0], [0, 4], [0, -2], [0, -1], [0, 3], [0, -4]], 7),
    ]
    
    approaches = [
        ("Slope Map with GCD", max_points_on_line_slope_map),
        ("Float Slope", max_points_on_line_float_slope),
        ("Line Equation", max_points_on_line_line_equation),
        ("Determinant", max_points_on_line_determinant),
        ("Optimized", max_points_on_line_optimized),
        ("Brute Force", max_points_on_line_brute_force),
    ]
    
    for i, (points, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: {points}")
        print(f"Expected: {expected}")
        
        for name, func in approaches:
            result = func(points)
            status = "✓" if result == expected else "✗"
            print(f"{status} {name}: {result}")


if __name__ == "__main__":
    test_max_points_on_line() 