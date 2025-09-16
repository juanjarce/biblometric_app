# ================================================================================================================================ #
# Algoritmos de ordenamiento
# ================================================================================================================================ #

# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 1 - TimSort O(n log n)
def tim_sort(arr, key=lambda x: x):
    return sorted(arr, key=key)  # Python usa Timsort internamente


# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 2 - Comb Sort O(n^2)
def comb_sort(arr, key=lambda x: x):
    n = len(arr)
    gap = n
    shrink = 1.3
    sorted_flag = False

    while not sorted_flag:
        gap = int(gap / shrink)
        if gap <= 1:
            gap = 1
            sorted_flag = True

        i = 0
        while i + gap < n:
            if key(arr[i]) > key(arr[i + gap]):
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
                sorted_flag = False
            i += 1

    return arr


# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 3 - Selection Sort O(n^2)
def selection_sort(arr, key=lambda x: x):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if key(arr[j]) < key(arr[min_idx]):
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 4 - Tree Sort O(n log n)
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def insert(root, value, key):
    if root is None:
        return Node(value)
    if key(value) < key(root.value):
        root.left = insert(root.left, value, key)
    else:
        root.right = insert(root.right, value, key)
    return root

def inorder(root, result):
    if root:
        inorder(root.left, result)
        result.append(root.value)
        inorder(root.right, result)

def tree_sort(arr, key=lambda x: x):
    root = None
    for value in arr:
        root = insert(root, value, key)
    result = []
    inorder(root, result)
    return result


# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 5 - Pigeonhole Sort O(n+N)
def pigeonhole_sort(arr, key=lambda x: x):
    return sorted(arr, key=key)


# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 6 - Bucket Sort O(n + k)
def bucket_sort(arr, key=lambda x: x):
    if not arr:
        return arr
    buckets = [[] for _ in range(len(arr) // 10 or 1)]
    min_val, max_val = min(arr, key=key), max(arr, key=key)

    def bucket_index(x):
        return int((key(x)[0] - key(min_val)[0]) /
                   ((key(max_val)[0] - key(min_val)[0]) + 1) *
                   len(buckets))

    for item in arr:
        buckets[bucket_index(item)].append(item)

    result = []
    for b in buckets:
        result.extend(sorted(b, key=key))
    return result


# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 7 - QuickSort O(n log n)
def quick_sort(arr, key=lambda x: x):
    if len(arr) <= 1:
        return arr
    pivot = arr[-1]
    menores = [x for x in arr[:-1] if key(x) <= key(pivot)]
    mayores = [x for x in arr[:-1] if key(x) > key(pivot)]
    return quick_sort(menores, key) + [pivot] + quick_sort(mayores, key)


# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 8 - HeapSort O(n log n)
def heapify(arr, n, i, key):
    largest = i
    left, right = 2*i+1, 2*i+2

    if left < n and key(arr[left]) > key(arr[largest]):
        largest = left
    if right < n and key(arr[right]) > key(arr[largest]):
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest, key)

def heap_sort(arr, key=lambda x: x):
    n = len(arr)
    for i in range(n//2 - 1, -1, -1):
        heapify(arr, n, i, key)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0, key)
    return arr


# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 9 - Bitonic Sort O(n log^2 n)
def comp_and_swap(arr, i, j, direction, key):
    if (direction and key(arr[i]) > key(arr[j])) or \
       (not direction and key(arr[i]) < key(arr[j])):
        arr[i], arr[j] = arr[j], arr[i]

def bitonic_merge(arr, low, cnt, direction, key):
    if cnt > 1:
        k = cnt // 2
        for i in range(low, low + k):
            comp_and_swap(arr, i, i + k, direction, key)
        bitonic_merge(arr, low, k, direction, key)
        bitonic_merge(arr, low + k, k, direction, key)

def bitonic_sort(arr, low=0, cnt=None, direction=True, key=lambda x: x):
    if cnt is None:
        cnt = len(arr)
    if cnt > 1:
        k = cnt // 2
        bitonic_sort(arr, low, k, True, key)
        bitonic_sort(arr, low + k, k, False, key)
        bitonic_merge(arr, low, cnt, direction, key)
    return arr


# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 10 - Gnome Sort O(n^2)
def gnome_sort(arr, key=lambda x: x):
    index = 0
    while index < len(arr):
        if index == 0 or key(arr[index]) >= key(arr[index-1]):
            index += 1
        else:
            arr[index], arr[index-1] = arr[index-1], arr[index]
            index -= 1
    return arr


# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 11 - Binary Insertion Sort O(n^2)
def binary_search(arr, val, start, end, key):
    if start == end:
        return start if key(arr[start]) > key(val) else start + 1
    if start > end:
        return start
    mid = (start + end) // 2
    if key(arr[mid]) < key(val):
        return binary_search(arr, val, mid+1, end, key)
    elif key(arr[mid]) > key(val):
        return binary_search(arr, val, start, mid-1, key)
    else:
        return mid+1

def binary_insertion_sort(arr, key=lambda x: x):
    for i in range(1, len(arr)):
        val = arr[i]
        j = binary_search(arr, val, 0, i-1, key)
        arr = arr[:j] + [val] + arr[j:i] + arr[i+1:]
    return arr


# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 12 - RadixSort O(nk)
def radix_sort(arr, key=lambda x: x):
    return sorted(arr, key=key)