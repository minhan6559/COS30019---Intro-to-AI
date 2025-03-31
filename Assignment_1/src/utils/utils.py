import functools
import heapq
import math


def distance(a, b):
    """The distance between two (x, y) points."""
    xA, yA = a
    xB, yB = b
    return math.sqrt((xA - xB) ** 2 + (yA - yB) ** 2)


def memoize(fn, slot=None, maxsize=32):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, use lru_cache for caching the values."""
    if slot:

        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val

    else:

        @functools.lru_cache(maxsize=maxsize)
        def memoized_fn(*args):
            return fn(*args)

    return memoized_fn


class PriorityQueue:
    """A Queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first. Efficiently supports item lookup and updates."""

    def __init__(self, order="min", f=lambda x: x):
        self.heap = []
        # Map from items to their heap index and priority value
        self.entry_finder = {}
        self.counter = 0  # Unique sequence count for tiebreaking
        if order == "min":
            self.f = f
        elif order == "max":
            self.f = lambda x: -f(x)
        else:
            raise ValueError("Order must be either 'min' or 'max'.")

    def append(self, item):
        """Insert item at its correct position."""
        # If item already in queue, remove it first
        if item in self.entry_finder:
            self.remove_item(item)

        # Get priority and add entry counter for stable ordering
        priority = self.f(item)
        count = self.counter
        self.counter += 1

        # Add to heap and remember entry
        entry = [priority, count, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.heap, entry)

    def extend(self, items):
        """Insert each item in items at its correct position."""
        for item in items:
            self.append(item)

    def pop(self):
        """Pop and return the item with lowest f(x) value."""
        while self.heap:
            priority, count, item = heapq.heappop(self.heap)
            if item in self.entry_finder:
                del self.entry_finder[item]
                return item
        raise Exception("Trying to pop from empty PriorityQueue.")

    def remove_item(self, item):
        """Mark an existing item as removed. Raises KeyError if not found."""
        if item in self.entry_finder:
            entry = self.entry_finder[item]
            # Mark as removed by pointing to None
            entry[-1] = None
            del self.entry_finder[item]
        else:
            raise KeyError(f"{item} not in priority queue")

    def __len__(self):
        """Return current capacity of PriorityQueue."""
        return len(self.entry_finder)

    def __contains__(self, item):
        """Return True if the key is in PriorityQueue."""
        return item in self.entry_finder

    def __getitem__(self, item):
        """Returns the priority value associated with item."""
        if item in self.entry_finder:
            return self.entry_finder[item][0]
        raise KeyError(f"{item} not in priority queue")

    def __delitem__(self, item):
        """Remove item from queue."""
        self.remove_item(item)
