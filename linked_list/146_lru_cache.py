"""
146. LRU Cache

Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:
- LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
- int get(int key) Return the value of the key if the key exists, otherwise return -1.
- void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity, evict the least recently used key.

The functions get and put must each run in O(1) average time complexity.

Example 1:
Input
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // cache is {1=1}
lRUCache.put(2, 2); // cache is {1=1, 2=2}
lRUCache.get(1);    // return 1
lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
lRUCache.get(2);    // returns -1 (not found)
lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {3=3, 4=4}
lRUCache.get(1);    // return -1 (not found)
lRUCache.get(3);    // return 3
lRUCache.get(4);    // return 4

Constraints:
- 1 <= capacity <= 3000
- 0 <= key <= 10^4
- 0 <= value <= 10^5
- At most 2 * 10^5 calls will be made to get and put
"""

from typing import Dict, Optional
from collections import OrderedDict


class Node:
    """Doubly linked list node for LRU cache."""
    def __init__(self, key: int = 0, value: int = 0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    """
    LRU Cache implementation using HashMap + Doubly Linked List.
    
    Time Complexity: O(1) for both get and put operations
    Space Complexity: O(capacity) for storing the cache
    
    Algorithm:
    1. Use HashMap for O(1) key lookup
    2. Use doubly linked list for O(1) insertion/deletion
    3. Keep track of head (most recent) and tail (least recent)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> node
        
        # Create dummy head and tail nodes
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node: Node) -> None:
        """Add node right after head."""
        node.prev = self.head
        node.next = self.head.next
        
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node: Node) -> None:
        """Remove an existing node from the linked list."""
        prev_node = node.prev
        next_node = node.next
        
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _move_to_head(self, node: Node) -> None:
        """Move node to head (mark as most recently used)."""
        self._remove_node(node)
        self._add_node(node)
    
    def _pop_tail(self) -> Node:
        """Remove the last node (least recently used)."""
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node
    
    def get(self, key: int) -> int:
        """Get value by key and mark as most recently used."""
        node = self.cache.get(key)
        
        if not node:
            return -1
        
        # Move the accessed node to head
        self._move_to_head(node)
        
        return node.value
    
    def put(self, key: int, value: int) -> None:
        """Put key-value pair and manage capacity."""
        node = self.cache.get(key)
        
        if not node:
            # New key
            new_node = Node(key, value)
            
            if len(self.cache) >= self.capacity:
                # Remove LRU node
                tail = self._pop_tail()
                del self.cache[tail.key]
            
            # Add new node
            self.cache[key] = new_node
            self._add_node(new_node)
        else:
            # Update existing key
            node.value = value
            self._move_to_head(node)


class LRUCacheOrderedDict:
    """
    LRU Cache implementation using OrderedDict.
    
    Time Complexity: O(1) for both get and put operations
    Space Complexity: O(capacity) for storing the cache
    
    Algorithm:
    1. Use OrderedDict which maintains insertion order
    2. Move accessed items to end for LRU behavior
    3. Remove from beginning when capacity exceeded
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: int) -> int:
        """Get value by key and mark as most recently used."""
        if key not in self.cache:
            return -1
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: int, value: int) -> None:
        """Put key-value pair and manage capacity."""
        if key in self.cache:
            # Update existing key
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # New key
            if len(self.cache) >= self.capacity:
                # Remove LRU item (first item)
                self.cache.popitem(last=False)
            
            self.cache[key] = value


class LRUCacheArray:
    """
    LRU Cache implementation using arrays (less efficient but educational).
    
    Time Complexity: O(n) for get and put operations
    Space Complexity: O(capacity) for storing the cache
    
    Algorithm:
    1. Use arrays to store keys and values
    2. Use timestamps to track access order
    3. Linear search for operations
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.keys = []
        self.values = []
        self.timestamps = []
        self.time = 0
    
    def get(self, key: int) -> int:
        """Get value by key and update timestamp."""
        try:
            index = self.keys.index(key)
            self.timestamps[index] = self.time
            self.time += 1
            return self.values[index]
        except ValueError:
            return -1
    
    def put(self, key: int, value: int) -> None:
        """Put key-value pair and manage capacity."""
        try:
            # Update existing key
            index = self.keys.index(key)
            self.values[index] = value
            self.timestamps[index] = self.time
            self.time += 1
        except ValueError:
            # New key
            if len(self.keys) >= self.capacity:
                # Remove LRU item
                lru_index = self.timestamps.index(min(self.timestamps))
                del self.keys[lru_index]
                del self.values[lru_index]
                del self.timestamps[lru_index]
            
            self.keys.append(key)
            self.values.append(value)
            self.timestamps.append(self.time)
            self.time += 1


class LRUCacheDict:
    """
    LRU Cache implementation using dict + access tracking.
    
    Time Complexity: O(1) for get, O(n) for put when eviction needed
    Space Complexity: O(capacity) for storing the cache
    
    Algorithm:
    1. Use regular dict for storage
    2. Track access order with timestamps
    3. Find LRU item when eviction needed
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.access_time = {}
        self.time = 0
    
    def get(self, key: int) -> int:
        """Get value by key and update access time."""
        if key not in self.cache:
            return -1
        
        self.access_time[key] = self.time
        self.time += 1
        return self.cache[key]
    
    def put(self, key: int, value: int) -> None:
        """Put key-value pair and manage capacity."""
        if key in self.cache:
            # Update existing key
            self.cache[key] = value
            self.access_time[key] = self.time
            self.time += 1
        else:
            # New key
            if len(self.cache) >= self.capacity:
                # Remove LRU item
                lru_key = min(self.access_time, key=self.access_time.get)
                del self.cache[lru_key]
                del self.access_time[lru_key]
            
            self.cache[key] = value
            self.access_time[key] = self.time
            self.time += 1


class LRUCacheLinkedList:
    """
    LRU Cache implementation using HashMap + Singly Linked List.
    
    Time Complexity: O(1) for get, O(n) for put when eviction needed
    Space Complexity: O(capacity) for storing the cache
    
    Algorithm:
    1. Use HashMap for O(1) key lookup
    2. Use singly linked list for order tracking
    3. Move accessed items to front
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []  # List to maintain order
    
    def get(self, key: int) -> int:
        """Get value by key and move to front."""
        if key not in self.cache:
            return -1
        
        # Move to front (most recently used)
        self.order.remove(key)
        self.order.append(key)
        
        return self.cache[key]
    
    def put(self, key: int, value: int) -> None:
        """Put key-value pair and manage capacity."""
        if key in self.cache:
            # Update existing key
            self.cache[key] = value
            self.order.remove(key)
            self.order.append(key)
        else:
            # New key
            if len(self.cache) >= self.capacity:
                # Remove LRU item
                lru_key = self.order.pop(0)
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.order.append(key)


class LRUCacheDeque:
    """
    LRU Cache implementation using deque for order tracking.
    
    Time Complexity: O(n) for operations due to deque operations
    Space Complexity: O(capacity) for storing the cache
    
    Algorithm:
    1. Use deque to maintain access order
    2. Move accessed items to right (most recent)
    3. Remove from left when capacity exceeded
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        
        from collections import deque
        self.order = deque()
    
    def get(self, key: int) -> int:
        """Get value by key and move to most recent."""
        if key not in self.cache:
            return -1
        
        # Move to most recent
        self.order.remove(key)
        self.order.append(key)
        
        return self.cache[key]
    
    def put(self, key: int, value: int) -> None:
        """Put key-value pair and manage capacity."""
        if key in self.cache:
            # Update existing key
            self.cache[key] = value
            self.order.remove(key)
            self.order.append(key)
        else:
            # New key
            if len(self.cache) >= self.capacity:
                # Remove LRU item
                lru_key = self.order.popleft()
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.order.append(key)


def test_lru_cache():
    """Test all LRU cache implementations."""
    
    implementations = [
        ("Doubly Linked List", LRUCache),
        ("OrderedDict", LRUCacheOrderedDict),
        ("Array", LRUCacheArray),
        ("Dict + Timestamps", LRUCacheDict),
        ("Linked List", LRUCacheLinkedList),
        ("Deque", LRUCacheDeque),
    ]
    
    print("Testing LRU Cache implementations:")
    print("=" * 60)
    
    # Test case 1: Basic functionality
    print("\nTest Case 1: Basic functionality")
    operations = [
        ("put", 1, 1),
        ("put", 2, 2),
        ("get", 1, 1),
        ("put", 3, 3),
        ("get", 2, -1),
        ("put", 4, 4),
        ("get", 1, -1),
        ("get", 3, 3),
        ("get", 4, 4),
    ]
    
    for name, cache_class in implementations:
        try:
            cache = cache_class(2)
            results = []
            
            for op, key, expected in operations:
                if op == "put":
                    cache.put(key, expected)
                    results.append(None)
                else:  # get
                    result = cache.get(key)
                    results.append(result)
                    if result != expected:
                        print(f"✗ {name}: get({key}) = {result}, expected {expected}")
                        break
            else:
                print(f"✓ {name}: All operations successful")
        except Exception as e:
            print(f"✗ {name}: Error - {e}")
    
    # Test case 2: Single capacity
    print("\nTest Case 2: Single capacity")
    for name, cache_class in implementations:
        try:
            cache = cache_class(1)
            
            cache.put(1, 1)
            assert cache.get(1) == 1
            
            cache.put(2, 2)
            assert cache.get(1) == -1
            assert cache.get(2) == 2
            
            cache.put(3, 3)
            assert cache.get(2) == -1
            assert cache.get(3) == 3
            
            print(f"✓ {name}: Single capacity test passed")
        except Exception as e:
            print(f"✗ {name}: Error - {e}")
    
    # Test case 3: Update existing key
    print("\nTest Case 3: Update existing key")
    for name, cache_class in implementations:
        try:
            cache = cache_class(2)
            
            cache.put(1, 1)
            cache.put(2, 2)
            cache.put(1, 10)  # Update key 1
            
            assert cache.get(1) == 10
            assert cache.get(2) == 2
            
            cache.put(3, 3)  # Should evict key 2
            assert cache.get(2) == -1
            assert cache.get(1) == 10
            assert cache.get(3) == 3
            
            print(f"✓ {name}: Update existing key test passed")
        except Exception as e:
            print(f"✗ {name}: Error - {e}")
    
    # Test case 4: Large capacity
    print("\nTest Case 4: Large capacity")
    for name, cache_class in implementations:
        try:
            cache = cache_class(100)
            
            # Fill cache
            for i in range(100):
                cache.put(i, i * 10)
            
            # Access some keys
            for i in range(0, 50, 2):
                assert cache.get(i) == i * 10
            
            # Add more keys to trigger eviction
            for i in range(100, 150):
                cache.put(i, i * 10)
            
            # Check that some old keys are evicted
            evicted_count = 0
            for i in range(100):
                if cache.get(i) == -1:
                    evicted_count += 1
            
            assert evicted_count == 50  # Should evict 50 keys
            
            print(f"✓ {name}: Large capacity test passed")
        except Exception as e:
            print(f"✗ {name}: Error - {e}")
    
    # Performance comparison
    print("\n" + "=" * 60)
    print("Performance Analysis:")
    print("=" * 60)
    
    import time
    
    test_scenarios = [
        ("Small operations", 100, 10),
        ("Medium operations", 1000, 100),
        ("Large operations", 10000, 1000),
    ]
    
    for scenario_name, operations_count, capacity in test_scenarios:
        print(f"\n{scenario_name}: {operations_count} operations, capacity {capacity}")
        
        for name, cache_class in implementations:
            try:
                cache = cache_class(capacity)
                
                start_time = time.time()
                
                # Mixed operations
                for i in range(operations_count):
                    if i % 3 == 0:
                        cache.put(i, i * 10)
                    else:
                        cache.get(i % capacity)
                
                end_time = time.time()
                
                print(f"  {name}: {end_time - start_time:.4f} seconds")
            except Exception as e:
                print(f"  {name}: Error - {e}")


if __name__ == "__main__":
    test_lru_cache() 