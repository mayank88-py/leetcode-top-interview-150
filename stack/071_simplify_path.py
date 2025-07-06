"""
71. Simplify Path

Problem:
Given a string path, which is an absolute path (starting with a slash '/') to a file or directory
in a Unix-style file system, convert it to the simplified canonical path.

In a Unix-style file system, a period '.' refers to the current directory, a double period '..' 
refers to the directory up a level, and any multiple consecutive slashes (i.e. '//') are treated 
as a single slash '/'. For this problem, any other format of periods such as '...' are treated 
as file/directory names.

The canonical path should have the following format:
- The path starts with a single slash '/'.
- Any two directories are separated by a single slash '/'.
- The path does not end with a trailing '/' unless the path is the root directory.
- The path only contains the directories on the path from the root directory to the target file or directory.

Return the simplified canonical path.

Example 1:
Input: path = "/home/"
Output: "/home"

Example 2:
Input: path = "/../"
Output: "/"

Example 3:
Input: path = "/home//foo/"
Output: "/home/foo"

Time Complexity: O(n)
Space Complexity: O(n)
"""


def simplify_path(path):
    """
    Simplify the given Unix-style path using stack.
    
    Args:
        path: Unix-style path string
    
    Returns:
        Simplified canonical path
    """
    if not path:
        return "/"
    
    # Split path by '/' and filter out empty strings
    components = [component for component in path.split('/') if component]
    
    stack = []
    
    for component in components:
        if component == '.':
            # Current directory, do nothing
            continue
        elif component == '..':
            # Parent directory, go back one level if possible
            if stack:
                stack.pop()
        else:
            # Regular directory or file name
            stack.append(component)
    
    # Build the result path
    return '/' + '/'.join(stack)


def simplify_path_manual(path):
    """
    Simplify path manually without using split.
    
    Args:
        path: Unix-style path string
    
    Returns:
        Simplified canonical path
    """
    if not path:
        return "/"
    
    stack = []
    i = 0
    
    while i < len(path):
        if path[i] == '/':
            i += 1
            continue
        
        # Extract the next component
        component = ""
        while i < len(path) and path[i] != '/':
            component += path[i]
            i += 1
        
        if component == '.':
            # Current directory, do nothing
            continue
        elif component == '..':
            # Parent directory, go back one level if possible
            if stack:
                stack.pop()
        else:
            # Regular directory or file name
            stack.append(component)
    
    # Build the result path
    return '/' + '/'.join(stack)


def simplify_path_deque(path):
    """
    Simplify path using deque for potentially better performance.
    
    Args:
        path: Unix-style path string
    
    Returns:
        Simplified canonical path
    """
    from collections import deque
    
    if not path:
        return "/"
    
    # Split path by '/' and filter out empty strings
    components = [component for component in path.split('/') if component]
    
    queue = deque()
    
    for component in components:
        if component == '.':
            # Current directory, do nothing
            continue
        elif component == '..':
            # Parent directory, go back one level if possible
            if queue:
                queue.pop()
        else:
            # Regular directory or file name
            queue.append(component)
    
    # Build the result path
    return '/' + '/'.join(queue)


# Test cases
if __name__ == "__main__":
    # Test case 1
    path1 = "/home/"
    result1a = simplify_path(path1)
    result1b = simplify_path_manual(path1)
    result1c = simplify_path_deque(path1)
    print(f"Test 1 - Expected: /home, Stack: {result1a}, Manual: {result1b}, Deque: {result1c}")
    
    # Test case 2
    path2 = "/../"
    result2a = simplify_path(path2)
    result2b = simplify_path_manual(path2)
    result2c = simplify_path_deque(path2)
    print(f"Test 2 - Expected: /, Stack: {result2a}, Manual: {result2b}, Deque: {result2c}")
    
    # Test case 3
    path3 = "/home//foo/"
    result3a = simplify_path(path3)
    result3b = simplify_path_manual(path3)
    result3c = simplify_path_deque(path3)
    print(f"Test 3 - Expected: /home/foo, Stack: {result3a}, Manual: {result3b}, Deque: {result3c}")
    
    # Test case 4 - Complex path
    path4 = "/a/./b/../../c/"
    result4a = simplify_path(path4)
    result4b = simplify_path_manual(path4)
    result4c = simplify_path_deque(path4)
    print(f"Test 4 - Expected: /c, Stack: {result4a}, Manual: {result4b}, Deque: {result4c}")
    
    # Test case 5 - Multiple slashes
    path5 = "/a//b////c/d//././/.."
    result5a = simplify_path(path5)
    result5b = simplify_path_manual(path5)
    result5c = simplify_path_deque(path5)
    print(f"Test 5 - Expected: /a/b/c, Stack: {result5a}, Manual: {result5b}, Deque: {result5c}")
    
    # Test case 6 - Root directory
    path6 = "/"
    result6a = simplify_path(path6)
    result6b = simplify_path_manual(path6)
    result6c = simplify_path_deque(path6)
    print(f"Test 6 - Expected: /, Stack: {result6a}, Manual: {result6b}, Deque: {result6c}")
    
    # Test case 7 - Only dots
    path7 = "/../../.."
    result7a = simplify_path(path7)
    result7b = simplify_path_manual(path7)
    result7c = simplify_path_deque(path7)
    print(f"Test 7 - Expected: /, Stack: {result7a}, Manual: {result7b}, Deque: {result7c}")
    
    # Test case 8 - Special file names
    path8 = "/a/b/c/..."
    result8a = simplify_path(path8)
    result8b = simplify_path_manual(path8)
    result8c = simplify_path_deque(path8)
    print(f"Test 8 - Expected: /a/b/c/..., Stack: {result8a}, Manual: {result8b}, Deque: {result8c}")
    
    # Test case 9 - Hidden files
    path9 = "/a/.b/../c"
    result9a = simplify_path(path9)
    result9b = simplify_path_manual(path9)
    result9c = simplify_path_deque(path9)
    print(f"Test 9 - Expected: /a/c, Stack: {result9a}, Manual: {result9b}, Deque: {result9c}")
    
    # Test case 10 - Complex nested
    path10 = "/a/b/c/../../../.."
    result10a = simplify_path(path10)
    result10b = simplify_path_manual(path10)
    result10c = simplify_path_deque(path10)
    print(f"Test 10 - Expected: /, Stack: {result10a}, Manual: {result10b}, Deque: {result10c}") 