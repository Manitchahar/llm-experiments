""" Python DSA Made Playful! Learn with Stories & Simple Examples """

#%% 1.  Python Playground Basics
#  Storing Things (Variables)
my_toys = 3          # Whole numbers are like counting toys (int)
pie_slice = 3.14     # Numbers with dots are like pizza slices (float)
my_name = "Buddy"    # Words in quotes are like name tags (string)
i_love_icecream = True # Yes/No values (boolean)

# ️ Let's Draw with Python!
print(f"{my_name} has {my_toys} toys and eats {pie_slice:.2f} of a pie!")

#%% 2.  Toy Boxes (Data Structures)
#  List - Your Toy Collection (Changeable)
toys = ["teddy", "car", "ball"]
toys.append("puzzle")       # Add new toy to the end
toy_shelves = [["bear", "rabbit"], ["cars", "trains"]]  # Shelf of shelves

#  Tuple - Mystery Box (Can't Change)
secret_box = ("magic", "wand")  # Once packed, stays same

#  Set - Unique Candy Jar
candy_types = {"lollipop", "gummy", "chocolate", "gummy"}  # Duplicates vanish!

#  Dictionary - Toy Catalog
toy_info = {
    "name": "Super Truck",
    "features": ["lights", "sounds"],
    "age_group": "3+"
}

#%% 3.  Making Choices & Loops
#  List Magic - Make candy squares!
even_candies = [x**2 for x in range(10) if x % 2 == 0]

#  Carousel Ride (Loop with Numbers)
for seat_number, toy in enumerate(toys, 1):
    print(f"Seat {seat_number} has: {toy}")

#%% 4.  Playground Slides (Functions)
def count_toys(n):
    """Recursive countdown like sliding down!"""
    if n <= 0:
        print("Base! ")
    else:
        print(n)
        count_toys(n-1)

#  Quick Candy Calculator
add_candies = lambda a, b: a + b

#%% 5.  Finding Games (Algorithms)
# ️♂️ Treasure Hunt (Binary Search)
from bisect import bisect_left

def find_toy(sorted_toys, target):
    """Find toy in sorted list using bisect"""
    idx = bisect_left(sorted_toys, target)
    return idx if idx < len(sorted_toys) and sorted_toys[idx] == target else -1

#  Puzzle Sorter (QuickSort)
def sort_toys(toys):
    if len(toys) <= 1:
        return toys
    pivot = toys[0]
    return (
        sort_toys([t for t in toys[1:] if t <= pivot]) 
        + [pivot] 
        + sort_toys([t for t in toys[1:] if t > pivot])
    )

#%% 6.  LeetCode Playtime
#  Two Cookies Problem
def find_cookie_pairs(cookies, target):
    cookie_jars = {}
    for jar_idx, cookie in enumerate(cookies):
        needed = target - cookie
        if needed in cookie_jars:
            return [cookie_jars[needed], jar_idx]
        cookie_jars[cookie] = jar_idx
    return []

#%% 7.  Coding Carnival (Tips & Tricks)
# 1.  Pairing Toys with zip()
toy_names = ["Teddy", "Car"]
prices = [10, 15]
for name, price in zip(toy_names, prices):
    print(f"{name} costs ${price}")

# 2.  itertools Merry-Go-Round
from itertools import permutations
print("Toy combinations:", list(permutations('ABC', 2)))

# 3.  Candy Counter
from collections import Counter
candy_bag = Counter("abracadabra")
print("Candy letters:", candy_bag)

# 4. ️ Playroom Organizer
from collections import defaultdict
toy_boxes = defaultdict(list)
for room, toy in [('blue', 'car'), ('red', 'doll'), ('blue', 'ball')]:
    toy_boxes[room].append(toy)
print("Organized toys:", dict(toy_boxes))

#%% 8.  Playground Adventures (Graphs)
#  Bus Route Finder (BFS)
def find_friends(graph, start):
    visited = set()
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current not in visited:
            visited.add(current)
            queue.extend(graph[current] - visited)
    return visited

#  Tree House Search (DFS)
def explore_tree(graph, start, visited=None):
    visited = visited or set()
    visited.add(start)
    for neighbor in graph[start] - visited:
        explore_tree(graph, neighbor, visited)
    return visited

#%% 9.  Storytime Patterns (DP)
#  Fibonacci Bunnies
from functools import lru_cache

@lru_cache(maxsize=None)
def bunny_growth(n):
    """Each bunny pair makes new pair every month"""
    return n if n < 2 else bunny_growth(n-1) + bunny_growth(n-2)

#  Picnic Basket Packing
def pack_snacks(snacks, max_weight):
    dp = [0] * (max_weight + 1)
    for weight, value in snacks:
        for w in range(max_weight, weight-1, -1):
            dp[w] = max(dp[w], dp[w - weight] + value)
    return dp[max_weight]

#%% 10.  Magic Tricks (Advanced)
# ♂️ Matrix Flip
toy_grid = [[1, 2], [3, 4], [5, 6]]
flipped = list(zip(*toy_grid))
print("Flipped grid:", flipped)

#  Walrus Operator
if (toy_count := len(toys)) > 2:
    print(f"Wow! {toy_count} toys to play with!")

#  Fast String Building
from io import StringIO
builder = StringIO()
for _ in range(3):
    builder.write("play ")
result = builder.getvalue()

#  Train Queue
train = deque(maxlen=3)
train.append("Engine")
train.append("Carriage")
train.append("Caboose")

#  Safe Sharing
def share_apples(total, friends):
    try:
        return total / friends
    except ZeroDivisionError:
        print("Oops! Can't share with no friends!")
        return 0

#  Cookie Jar Sorting
import heapq
def sort_cookies(jar):
    heapq.heapify(jar)
    return [heapq.heappop(jar) for _ in range(len(jar))]

if __name__ == "__main__":
    print("\n=== Let's Play! ===")
    count_toys(3)
    print("Sorted toys:", sort_toys(["ball", "car", "apple"]))
    print("Cookie pairs:", find_cookie_pairs([2, 7, 11, 15], 9))
    
    print("\n Storybook Examples:")
    print("Bunny families after 5 months:", bunny_growth(5))
    print("Cookie sorting:", sort_cookies([3, 1, 4]))
    print("Sharing 5 apples with 0 friends:", share_apples(5, 0))
    
    # Playground map
    play_areas = {
        'Slide': {'Swing', 'Sandbox'},
        'Swing': {'Slide', 'Merry-go-round'},
        'Sandbox': {'Slide'},
        'Merry-go-round': {'Swing'}
    }
    print("Play areas to visit:", find_friends(play_areas, 'Slide'))
