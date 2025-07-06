"""
380. Insert Delete GetRandom O(1)

Problem:
Implement the RandomizedSet class:

RandomizedSet() Initializes the RandomizedSet object.
bool insert(int val) Inserts an item val into the set if not present. Returns true if the item was not present, false otherwise.
bool remove(int val) Removes an item val from the set if present. Returns true if the item was present, false otherwise.
int getRandom() Returns a random element from the current set of elements (it's guaranteed that at least one element exists when this method is called). Each element must have the same probability of being returned.

You must implement the functions of the class such that each function works in average O(1) time complexity.

Example 1:
Input
["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
[[], [1], [2], [2], [], [1], [2], []]
Output
[null, true, false, true, 2, true, false, 2]

Explanation
RandomizedSet randomizedSet = new RandomizedSet();
randomizedSet.insert(1); // Inserts 1 to the set. Returns true as 1 was inserted successfully.
randomizedSet.remove(2); // Returns false as 2 does not exist in the set.
randomizedSet.insert(2); // Inserts 2 to the set, returns true. Set now contains [1,2].
randomizedSet.getRandom(); // getRandom() should return either 1 or 2 randomly.
randomizedSet.remove(1); // Removes 1 from the set, returns true. Set now contains [2].
randomizedSet.insert(2); // 2 was already in the set, so return false.
randomizedSet.getRandom(); // Since 2 is the only number in the set, getRandom() will always return 2.

Time Complexity: O(1) average for all operations
Space Complexity: O(n) where n is the number of elements
"""


import random


class RandomizedSet:
    """
    Randomized Set implementation using array and hash map.
    """
    
    def __init__(self):
        """Initialize the data structure."""
        self.vals = []  # Array to store values
        self.indices = {}  # Hash map: value -> index in array
    
    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        
        Args:
            val: Value to insert
        
        Returns:
            True if inserted, False if already present
        """
        if val in self.indices:
            return False
        
        # Add to end of array and record index
        self.vals.append(val)
        self.indices[val] = len(self.vals) - 1
        
        return True
    
    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        
        Args:
            val: Value to remove
        
        Returns:
            True if removed, False if not present
        """
        if val not in self.indices:
            return False
        
        # Get index of element to remove
        last_element = self.vals[-1]
        idx_to_remove = self.indices[val]
        
        # Move last element to the position of element to remove
        self.vals[idx_to_remove] = last_element
        self.indices[last_element] = idx_to_remove
        
        # Remove last element
        self.vals.pop()
        del self.indices[val]
        
        return True
    
    def getRandom(self):
        """
        Get a random element from the set.
        
        Returns:
            Random element from the set
        """
        return random.choice(self.vals)


class RandomizedSetWithList:
    """
    Alternative implementation using only list with linear remove.
    """
    
    def __init__(self):
        """Initialize the data structure."""
        self.vals = []
    
    def insert(self, val):
        """Insert a value to the set."""
        if val in self.vals:
            return False
        
        self.vals.append(val)
        return True
    
    def remove(self, val):
        """Remove a value from the set."""
        if val not in self.vals:
            return False
        
        self.vals.remove(val)
        return True
    
    def getRandom(self):
        """Get a random element from the set."""
        return random.choice(self.vals)


class RandomizedSetWithSet:
    """
    Implementation using built-in set (not O(1) for getRandom).
    """
    
    def __init__(self):
        """Initialize the data structure."""
        self.vals = set()
    
    def insert(self, val):
        """Insert a value to the set."""
        if val in self.vals:
            return False
        
        self.vals.add(val)
        return True
    
    def remove(self, val):
        """Remove a value from the set."""
        if val not in self.vals:
            return False
        
        self.vals.remove(val)
        return True
    
    def getRandom(self):
        """Get a random element from the set."""
        return random.choice(list(self.vals))


class RandomizedSetDoublyLinked:
    """
    Implementation using doubly linked list and hash map.
    """
    
    class Node:
        def __init__(self, val):
            self.val = val
            self.prev = None
            self.next = None
    
    def __init__(self):
        """Initialize the data structure."""
        self.nodes = {}  # value -> node
        self.vals = []   # For O(1) random access
        self.indices = {}  # value -> index in vals
    
    def insert(self, val):
        """Insert a value to the set."""
        if val in self.nodes:
            return False
        
        # Create new node
        node = self.Node(val)
        self.nodes[val] = node
        
        # Add to array for random access
        self.vals.append(val)
        self.indices[val] = len(self.vals) - 1
        
        return True
    
    def remove(self, val):
        """Remove a value from the set."""
        if val not in self.nodes:
            return False
        
        # Remove from hash map
        del self.nodes[val]
        
        # Remove from array (swap with last element)
        last_element = self.vals[-1]
        idx_to_remove = self.indices[val]
        
        self.vals[idx_to_remove] = last_element
        self.indices[last_element] = idx_to_remove
        
        self.vals.pop()
        del self.indices[val]
        
        return True
    
    def getRandom(self):
        """Get a random element from the set."""
        return random.choice(self.vals)


class RandomizedSetBST:
    """
    Implementation using binary search tree (not O(1) average case).
    """
    
    class TreeNode:
        def __init__(self, val):
            self.val = val
            self.left = None
            self.right = None
    
    def __init__(self):
        """Initialize the data structure."""
        self.root = None
        self.vals = []
        self.indices = {}
    
    def insert(self, val):
        """Insert a value to the set."""
        if val in self.indices:
            return False
        
        # Insert into BST
        self.root = self._insert_bst(self.root, val)
        
        # Add to array for random access
        self.vals.append(val)
        self.indices[val] = len(self.vals) - 1
        
        return True
    
    def _insert_bst(self, node, val):
        """Helper method to insert into BST."""
        if not node:
            return self.TreeNode(val)
        
        if val < node.val:
            node.left = self._insert_bst(node.left, val)
        else:
            node.right = self._insert_bst(node.right, val)
        
        return node
    
    def remove(self, val):
        """Remove a value from the set."""
        if val not in self.indices:
            return False
        
        # Remove from BST
        self.root = self._remove_bst(self.root, val)
        
        # Remove from array
        last_element = self.vals[-1]
        idx_to_remove = self.indices[val]
        
        self.vals[idx_to_remove] = last_element
        self.indices[last_element] = idx_to_remove
        
        self.vals.pop()
        del self.indices[val]
        
        return True
    
    def _remove_bst(self, node, val):
        """Helper method to remove from BST."""
        if not node:
            return None
        
        if val < node.val:
            node.left = self._remove_bst(node.left, val)
        elif val > node.val:
            node.right = self._remove_bst(node.right, val)
        else:
            # Node to be deleted found
            if not node.left:
                return node.right
            elif not node.right:
                return node.left
            else:
                # Find inorder successor
                min_node = node.right
                while min_node.left:
                    min_node = min_node.left
                
                node.val = min_node.val
                node.right = self._remove_bst(node.right, min_node.val)
        
        return node
    
    def getRandom(self):
        """Get a random element from the set."""
        return random.choice(self.vals)


# Test cases
if __name__ == "__main__":
    # Test basic operations
    def test_randomized_set(RandomizedSetClass, name):
        print(f"\nTesting {name}:")
        rs = RandomizedSetClass()
        
        # Test insert
        print(f"Insert 1: {rs.insert(1)}")  # Should return True
        print(f"Insert 2: {rs.insert(2)}")  # Should return True
        print(f"Insert 1: {rs.insert(1)}")  # Should return False (duplicate)
        
        # Test getRandom
        print(f"Random element: {rs.getRandom()}")  # Should be 1 or 2
        
        # Test remove
        print(f"Remove 1: {rs.remove(1)}")  # Should return True
        print(f"Remove 1: {rs.remove(1)}")  # Should return False (not present)
        
        # Test getRandom after removal
        print(f"Random element after removal: {rs.getRandom()}")  # Should be 2
        
        # Test edge cases
        print(f"Insert 3: {rs.insert(3)}")  # Should return True
        print(f"Insert 4: {rs.insert(4)}")  # Should return True
        print(f"Insert 5: {rs.insert(5)}")  # Should return True
        
        # Test multiple getRandom calls
        print("Multiple random calls:")
        for i in range(5):
            print(f"  Random {i+1}: {rs.getRandom()}")
        
        # Test remove all
        print(f"Remove 2: {rs.remove(2)}")  # Should return True
        print(f"Remove 3: {rs.remove(3)}")  # Should return True
        print(f"Remove 4: {rs.remove(4)}")  # Should return True
        print(f"Remove 5: {rs.remove(5)}")  # Should return True
        
        # Test insert after removing all
        print(f"Insert 10: {rs.insert(10)}")  # Should return True
        print(f"Random element: {rs.getRandom()}")  # Should be 10
    
    # Test all implementations
    test_randomized_set(RandomizedSet, "RandomizedSet (Optimal)")
    test_randomized_set(RandomizedSetWithList, "RandomizedSetWithList")
    test_randomized_set(RandomizedSetWithSet, "RandomizedSetWithSet")
    test_randomized_set(RandomizedSetDoublyLinked, "RandomizedSetDoublyLinked")
    test_randomized_set(RandomizedSetBST, "RandomizedSetBST")
    
    # Test sequence from problem example
    print("\n\nTesting problem example sequence:")
    rs = RandomizedSet()
    
    operations = [
        ("insert", 1),
        ("remove", 2),
        ("insert", 2),
        ("getRandom", None),
        ("remove", 1),
        ("insert", 2),
        ("getRandom", None)
    ]
    
    results = []
    for op, val in operations:
        if op == "insert":
            result = rs.insert(val)
            results.append(result)
            print(f"insert({val}) = {result}")
        elif op == "remove":
            result = rs.remove(val)
            results.append(result)
            print(f"remove({val}) = {result}")
        elif op == "getRandom":
            result = rs.getRandom()
            results.append(result)
            print(f"getRandom() = {result}")
    
    print(f"\nResults: {results}")
    
    # Test large dataset
    print("\n\nTesting large dataset:")
    rs = RandomizedSet()
    
    # Insert many elements
    for i in range(1000):
        rs.insert(i)
    
    # Test random distribution
    random_counts = {}
    for _ in range(10000):
        val = rs.getRandom()
        random_counts[val] = random_counts.get(val, 0) + 1
    
    # Check if distribution is roughly uniform
    avg_count = 10000 / 1000  # 10
    min_count = min(random_counts.values())
    max_count = max(random_counts.values())
    
    print(f"Random distribution test:")
    print(f"  Expected average: {avg_count}")
    print(f"  Actual range: [{min_count}, {max_count}]")
    print(f"  Deviation: {max(abs(min_count - avg_count), abs(max_count - avg_count))}")
    
    # Remove half the elements
    for i in range(0, 1000, 2):
        rs.remove(i)
    
    # Test random after removal
    random_after_removal = set()
    for _ in range(100):
        random_after_removal.add(rs.getRandom())
    
    print(f"Elements after removal (sample): {sorted(list(random_after_removal))[:10]}")
    
    print("\nAll tests completed successfully!") 