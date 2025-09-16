import random
import os
import time
import random

""""
Función para generar 1000 numeros aleatorios entre [1, 10000]
se usarán como input para el análisis de los algoritmos de organizamiento
"""
def generar_numeros(ruta_archivo, cantidad=1000, minimo=1, maximo=10000):
    # Se asegura que la carpeta existe
    carpeta = os.path.dirname(ruta_archivo)
    if carpeta and not os.path.exists(carpeta):
        os.makedirs(carpeta)

    # Generar y escribir los números
    with open(ruta_archivo, "w") as f:
        for _ in range(cantidad):
            numero = random.randint(minimo, maximo)
            f.write(str(numero) + "\n")

    print(f"Archivo generado en: {ruta_archivo}")

# ----------- Algoritmos de ordenamiento ------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 1 - TimSort O( )
def tim_sort(arr):
    return sorted(arr)  # Python usa Timsort internamente

# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 2 - Comb Sort O( )
def comb_sort(arr):
    n = len(arr)              # Número de elementos en el arreglo
    gap = n                   # Inicialmente, la "brecha" (gap) es el tamaño del arreglo
    shrink = 1.3              # Factor por el cual se reducirá el gap en cada pasada
    sorted_flag = False       # Bandera que indica si el arreglo ya está ordenado

    # El ciclo principal se ejecuta hasta que no se hagan más intercambios con gap=1
    while not sorted_flag:
        # Reducimos la brecha
        gap = int(gap / shrink)

        # El gap no puede ser menor a 1
        if gap <= 1:
            gap = 1
            sorted_flag = True   # se asume que está ordenado, salvo que se encuentre un intercambio

        i = 0
        # Recorremos el arreglo comparando elementos separados por 'gap'
        while i + gap < n:
            if arr[i] > arr[i + gap]:
                # Si están en el orden incorrecto, se intercambian
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
                sorted_flag = False   # Hubo un intercambio, por lo tanto aún no está ordenado
            i += 1

    return arr 

# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 3 - Selection Sort O( )
def selection_sort(arr):
    n = len(arr)  # Número de elementos en el arreglo

    # Se recorre todo el arreglo
    for i in range(n):
        # suponiendo que el elemento en la posición i es el mínimo
        min_idx = i

        # Busca si existe un valor menor en el resto del arreglo
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j   # Actualizamos el índice del nuevo mínimo encontrado

        # Se intercambian el elemento mínimo encontrado con el elemento en la posición i
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

    return arr

# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 4 - Tree Sort O( )

# Nodo del árbol binario
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None   # Subárbol izquierdo
        self.right = None  # Subárbol derecho


# Función para insertar un valor en el árbol binario de búsqueda
def insert(root, value):
    if root is None:
        return Node(value)  # Si no hay raíz, creamos un nuevo nodo

    if value < root.value:
        root.left = insert(root.left, value)   # Insertar en el subárbol izquierdo
    else:
        root.right = insert(root.right, value) # Insertar en el subárbol derecho

    return root


# Recorrido en orden (inorder traversal): izquierda -> raíz -> derecha
def inorder(root, result):
    if root:
        inorder(root.left, result)
        result.append(root.value)
        inorder(root.right, result)


# Tree Sort
def tree_sort(arr):
    root = None

    # 1. Insertar todos los elementos en el árbol
    for value in arr:
        root = insert(root, value)

    # 2. Hacer un recorrido en orden para obtener la lista ordenada
    result = []
    inorder(root, result)

    return result


# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 5 - Pigeonhole Sort O( )
def pigeonhole_sort(arr):
    # Encontrar el mínimo y máximo de la lista
    min_val = min(arr)
    max_val = max(arr)

    # Determinar el rango (cuántas casillas se necesitan)
    size = max_val - min_val + 1

    # Crear "pigeonholes" (lista de contadores)
    holes = [0] * size

    # Contar cuántas veces aparece cada número
    for x in arr:
        holes[x - min_val] += 1

    # Reconstruir el arreglo ordenado
    i = 0
    for j in range(size):       # recorremos los pigeonholes en orden
        while holes[j] > 0:     # mientras queden elementos en ese hueco
            arr[i] = j + min_val
            i += 1
            holes[j] -= 1

    return arr

# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 6 - BucketSort O( )
def bucket_sort(arr):
    if len(arr) == 0:
        return arr

    # Encontrar valor máximo y mínimo
    max_val = max(arr)
    min_val = min(arr)

    # Número de buckets
    bucket_count = len(arr) // 10 or 1  

    # Inicializar buckets vacíos
    buckets = [[] for _ in range(bucket_count)]

    # Distribuir elementos en los buckets correspondientes
    for num in arr:
        index = int((num - min_val) / (max_val - min_val + 1) * bucket_count)
        buckets[index].append(num)

    # Ordenar cada bucket individualmente
    for i in range(bucket_count):
        buckets[i] = sorted(buckets[i])

    # Concatenar todos los buckets en uno solo
    sorted_arr = []
    for bucket in buckets:
        sorted_arr.extend(bucket)

    return sorted_arr

# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 7 -  QuickSort O( )
def quick_sort(arr):
    # Caso base: si el arreglo tiene 0 o 1 elementos, ya está ordenado
    if len(arr) <= 1:
        return arr
    
    # Se elige el último elemento como pivote
    pivot = arr[-1]
    
    # Dividimos el arreglo en dos listas
    menores = [x for x in arr[:-1] if x <= pivot]  # Elementos <= pivote
    mayores = [x for x in arr[:-1] if x > pivot]   # Elementos > pivote
    
    # Ordenamos recursivamente y combinamos
    return quick_sort(menores) + [pivot] + quick_sort(mayores)

# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 8 -  HeapSort O( )

def heapify(arr, n, i):
    """
    Mantiene la propiedad de heap en un subárbol
    arr : lista
    n   : tamaño del heap
    i   : índice de la raíz del subárbol
    """
    largest = i        # Inicializar el mayor como la raíz
    left = 2 * i + 1   # Hijo izquierdo
    right = 2 * i + 2  # Hijo derecho

    # Si el hijo izquierdo es mayor que la raíz
    if left < n and arr[left] > arr[largest]:
        largest = left

    # Si el hijo derecho es mayor que el mayor actual
    if right < n and arr[right] > arr[largest]:
        largest = right

    # Si el mayor no es la raíz, intercambiar y seguir heapificando
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)

    # Construir el heap máximo
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extraer los elementos uno por uno
    for i in range(n - 1, 0, -1):
        # Mover la raíz (máximo) al final
        arr[0], arr[i] = arr[i], arr[0]

        # Aplicar heapify en el heap reducido
        heapify(arr, i, 0)

    return arr

# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 9 -  Bitonic Sort O ( )

def comp_and_swap(arr, i, j, direction):
    """
    Compara y, si es necesario, intercambia arr[i] y arr[j]
    direction = True  -> orden ascendente
    direction = False -> orden descendente
    """
    if (direction == True and arr[i] > arr[j]) or (direction == False and arr[i] < arr[j]):
        arr[i], arr[j] = arr[j], arr[i]


def bitonic_merge(arr, low, cnt, direction):
    """
    Fase de mezcla bitónica: asegura que la subsecuencia de tamaño cnt
    esté ordenada en la dirección indicada
    """
    if cnt > 1:
        k = cnt // 2
        for i in range(low, low + k):
            comp_and_swap(arr, i, i + k, direction)
        bitonic_merge(arr, low, k, direction)
        bitonic_merge(arr, low + k, k, direction)


def bitonic_sort(arr, low=0, cnt=None, direction=True):
    """
    Algoritmo Bitonic Sort
    arr       : lista de números
    low       : índice inicial
    cnt       : cantidad de elementos a ordenar
    direction : True = ascendente, False = descendente
    """
    if cnt is None:
        cnt = len(arr)

    if cnt > 1:
        k = cnt // 2
        # Construir secuencia bitónica
        bitonic_sort(arr, low, k, True)   # primera mitad ascendente
        bitonic_sort(arr, low + k, k, False) # segunda mitad descendente
        # Mezclar
        bitonic_merge(arr, low, cnt, direction)
    return arr

# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 10 -  Gnome Sort O ( )

"""
Algoritmo Gnome Sort
arr: lista de números
"""
def gnome_sort(arr):
    n = len(arr)
    index = 0

    while index < n:
        # Si estamos en el inicio o el orden es correcto, avanzamos
        if index == 0 or arr[index] >= arr[index - 1]:
            index += 1
        else:
            # Si no, intercambiamos y retrocedemos una posición
            arr[index], arr[index - 1] = arr[index - 1], arr[index]
            index -= 1

    return arr

# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 11 -  Binary Insertion Sort O()

def binary_search(arr, val, start, end):
    """
    Busca mediante búsqueda binaria la posición donde
    debería insertarse 'val' dentro del subarreglo arr[start:end].
    Retorna el índice en el que se debe insertar.
    """
    # Caso base: solo queda un elemento
    if start == end:
        # si val es menor que arr[start], se inserta en 'start'
        # si no, se inserta en la siguiente posición
        return start if arr[start] > val else start + 1
    
    # Caso en que la ventana está invertida (val menor a todos)
    if start > end:
        return start

    # Calculamos el punto medio
    mid = (start + end) // 2

    # Si val es mayor que el valor medio, buscar en la mitad derecha
    if arr[mid] < val:
        return binary_search(arr, val, mid + 1, end)
    # Si val es menor que el valor medio, buscar en la mitad izquierda
    elif arr[mid] > val:
        return binary_search(arr, val, start, mid - 1)
    else:
        # Si son iguales, insertar después del elemento encontrado
        return mid + 1


def binary_insertion_sort(arr):
    """
    Algoritmo Binary Insertion Sort
    Usa búsqueda binaria para encontrar la posición de inserción
    en lugar de la búsqueda lineal del Insertion Sort clásico.
    """
    # Recorremos el arreglo desde el segundo elemento
    for i in range(1, len(arr)):
        val = arr[i]  # elemento que queremos insertar en su lugar correcto

        # Usamos búsqueda binaria para encontrar el índice de inserción
        j = binary_search(arr, val, 0, i - 1)

        # Reconstruimos el arreglo:
        # - todo lo que va antes de j
        # - el valor a insertar
        # - los elementos entre j y i (que se desplazan a la derecha)
        # - el resto del arreglo
        arr = arr[:j] + [val] + arr[j:i] + arr[i+1:]
    
    return arr

# -------------------------------------------------------------------------------------------------------------------------------- #
# Algoritmo 12 - RadixSort O ()

def counting_sort_for_radix(arr, exp):
    """
    Subfunción de Counting Sort que ordena arr[]
    según el dígito representado por exp (1 = unidades, 10 = decenas, etc.)
    """
    n = len(arr)
    output = [0] * n  # arreglo de salida
    count = [0] * 10  # dígitos posibles (0-9)

    # Contar ocurrencias del dígito correspondiente
    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1

    # Convertir conteo en posiciones acumuladas
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Construir el arreglo de salida (estable)
    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1

    # Copiar salida en arr
    for i in range(n):
        arr[i] = output[i]


def radix_sort(arr):
    """
    Algoritmo Radix Sort (LSD - Least Significant Digit)
    """
    # Encontrar el número máximo para saber cuántos dígitos tiene
    max_num = max(arr)

    # Hacer Counting Sort por cada dígito
    exp = 1
    while max_num // exp > 0:
        counting_sort_for_radix(arr, exp)
        exp *= 10

    return arr


# ----------- Ejecución y medición ------------------------------------------------------------------------------------------------

def medir_tiempo(func, datos):
    inicio = time.time()
    func(datos.copy())
    fin = time.time()
    return fin - inicio

# main
if __name__ == "__main__":
    # Generar el archivo 'numbers.txt' con el input para el análisis
    path ="follow-ups/data/numbers.txt"
    generar_numeros(path)

    # Leer archivo con números
    with open("follow-ups/data/numbers.txt", "r") as f:
        numeros = [int(line.strip()) for line in f.readlines()]

    tamaño = len(numeros)

    # Algoritmo 1 - TimSort O( )
    tiempo = medir_tiempo(tim_sort, numeros)
    # Imprimir resultado (tiempo del algoritmo)
    print(f"TimSort -> Tamaño: {tamaño}, Tiempo: {tiempo:.6f} segundos")

    # Algoritmo 2 - Comb Sort O( )
    tiempo = medir_tiempo(comb_sort, numeros)
    # Imprimir resultado (tiempo del algoritmo)
    print(f"TCombSort -> Tamaño: {tamaño}, Tiempo: {tiempo:.6f} segundos")

    # Algoritmo 3 - Selection Sort O( )
    tiempo = medir_tiempo(selection_sort, numeros)
    # Imprimir resultado (tiempo del algoritmo)
    print(f"SelectionSort -> Tamaño: {tamaño}, Tiempo: {tiempo:.6f} segundos")

    # Algoritmo 4 - Tree Sort O( )
    tiempo = medir_tiempo(tree_sort, numeros)
    # Imprimir resultado (tiempo del algoritmo)
    print(f"TreeSort -> Tamaño: {tamaño}, Tiempo: {tiempo:.6f} segundos")
    
    # Algoritmo 5 - Pigeonhole Sort O( )
    tiempo = medir_tiempo(pigeonhole_sort, numeros)
    # Imprimir resultado (tiempo del algoritmo)
    print(f"PigeonHoleSort -> Tamaño: {tamaño}, Tiempo: {tiempo:.6f} segundos")

    # Algoritmo 6 - BucketSort O( )
    tiempo = medir_tiempo(bucket_sort, numeros)
    # Imprimir resultado (tiempo del algoritmo)
    print(f"BucketSort -> Tamaño: {tamaño}, Tiempo: {tiempo:.6f} segundos")

    # Algoritmo 7 -  QuickSort O( )
    tiempo = medir_tiempo(quick_sort, numeros)
    # Imprimir resultado (tiempo del algoritmo)
    print(f"QuickSort -> Tamaño: {tamaño}, Tiempo: {tiempo:.6f} segundos")

    # Algoritmo 8 -  HeapSort O( )
    tiempo = medir_tiempo(heap_sort, numeros)
    # Imprimir resultado (tiempo del algoritmo)
    print(f"HeapSort -> Tamaño: {tamaño}, Tiempo: {tiempo:.6f} segundos")

    # Algoritmo 9 -  Bitonic Sort O ( )
    tiempo = medir_tiempo(bitonic_sort, numeros)
    # Imprimir resultado (tiempo del algoritmo)
    print(f"BitonicSort -> Tamaño: {tamaño}, Tiempo: {tiempo:.6f} segundos")

    # Algoritmo 10 -  Gnome Sort O ( )
    tiempo = medir_tiempo(gnome_sort, numeros)
    # Imprimir resultado (tiempo del algoritmo)
    print(f"GnomeSort -> Tamaño: {tamaño}, Tiempo: {tiempo:.6f} segundos")

    # Algoritmo 11 -  Binary Insertion Sort O()
    tiempo = medir_tiempo(binary_insertion_sort, numeros)
    # Imprimir resultado (tiempo del algoritmo)
    print(f"BinariInsertionSort -> Tamaño: {tamaño}, Tiempo: {tiempo:.6f} segundos")

    # Algoritmo 12 - RadixSort O ()
    tiempo = medir_tiempo(radix_sort, numeros)
    # Imprimir resultado (tiempo del algoritmo)
    print(f"RadixSort -> Tamaño: {tamaño}, Tiempo: {tiempo:.6f} segundos")