#!/usr/bin/env python3
"""
Unit 7 Assignment: Sorting Algorithm Benchmarking
Author: Andres Camacho
Course: IN450 Advanced Software Development
Description: Implements and benchmarks sorting algorithms with optimization analysis

Algorithm Sources:
- Bubble Sort: Classic algorithm, implementation based on Cormen et al. (2009)
  "Introduction to Algorithms" - https://mitpress.mit.edu/9780262046305/
- Quicksort: Divide-and-conquer algorithm by C.A.R. Hoare (1961)
  Python implementation adapted from Python documentation examples
"""

import time
import random
import copy


# =============================================================================
# DATA SET GENERATION
# =============================================================================

def generate_dataset(size, seed=42):
    """
    Generate a random dataset of integers for sorting.
    
    Args:
        size (int): Number of elements in the dataset
        seed (int): Random seed for reproducibility
        
    Returns:
        list: List of random integers between 1 and 10000
    """
    random.seed(seed)
    return [random.randint(1, 10000) for _ in range(size)]


def create_datasets():
    """
    Create three datasets of different sizes for benchmarking.
    
    Returns:
        dict: Dictionary containing small, medium, and large datasets
    """
    datasets = {
        'small': generate_dataset(10),
        'medium': generate_dataset(1000),
        'large': generate_dataset(10000)
    }
    return datasets


# =============================================================================
# BUBBLE SORT - ORIGINAL IMPLEMENTATION
# =============================================================================

def bubble_sort_original(arr):
    """
    Original bubble sort implementation.
    Compares adjacent elements and swaps if out of order.
    
    Time Complexity: O(n²) - all cases
    Space Complexity: O(1) - in-place sorting
    
    Args:
        arr (list): List to sort
        
    Returns:
        tuple: (sorted list, number of comparisons, number of swaps)
    """
    n = len(arr)
    comparisons = 0
    swaps = 0
    
    for i in range(n):
        for j in range(0, n - i - 1):
            comparisons += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swaps += 1
                
    return arr, comparisons, swaps


# =============================================================================
# BUBBLE SORT - OPTIMIZED IMPLEMENTATION
# =============================================================================

def bubble_sort_optimized(arr):
    """
    Optimized bubble sort with early termination.
    Stops if no swaps occur during a pass (array is sorted).
    
    Time Complexity: 
        - Best case: O(n) - already sorted
        - Average/Worst case: O(n²)
    Space Complexity: O(1) - in-place sorting
    
    Args:
        arr (list): List to sort
        
    Returns:
        tuple: (sorted list, number of comparisons, number of swaps)
    """
    n = len(arr)
    comparisons = 0
    swaps = 0
    
    for i in range(n):
        swapped = False
        
        for j in range(0, n - i - 1):
            comparisons += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swaps += 1
                swapped = True
        
        # Early termination: if no swaps occurred, array is sorted
        if not swapped:
            break
            
    return arr, comparisons, swaps


# =============================================================================
# QUICKSORT - ALTERNATIVE ALGORITHM
# =============================================================================

# Global counters for quicksort metrics
qs_comparisons = 0
qs_swaps = 0


def quicksort(arr):
    """
    Quicksort implementation using divide-and-conquer.
    Uses middle element as pivot for better average performance.
    
    Time Complexity:
        - Best/Average case: O(n log n)
        - Worst case: O(n²) - rare with good pivot selection
    Space Complexity: O(n log n) - recursive call stack
    
    Args:
        arr (list): List to sort
        
    Returns:
        tuple: (sorted list, number of comparisons, number of swaps)
    """
    global qs_comparisons, qs_swaps
    qs_comparisons = 0
    qs_swaps = 0
    
    result = _quicksort_recursive(arr.copy())
    return result, qs_comparisons, qs_swaps


def _quicksort_recursive(arr):
    """Internal recursive quicksort function."""
    global qs_comparisons, qs_swaps
    
    if len(arr) <= 1:
        return arr
    
    # Choose pivot as middle element
    pivot_index = len(arr) // 2
    pivot = arr[pivot_index]
    
    left = []
    middle = []
    right = []
    
    for item in arr:
        qs_comparisons += 1
        if item < pivot:
            left.append(item)
        elif item > pivot:
            right.append(item)
        else:
            middle.append(item)
        qs_swaps += 1  # Count element placement as a "move"
    
    return _quicksort_recursive(left) + middle + _quicksort_recursive(right)


# =============================================================================
# BENCHMARKING FUNCTIONS
# =============================================================================

def measure_time(sort_func, data):
    """
    Measure execution time for a sorting function.
    
    Args:
        sort_func (callable): Sorting function to benchmark
        data (list): Data to sort
        
    Returns:
        tuple: (elapsed_time, sorted_data, comparisons, swaps)
    """
    # Make a deep copy to preserve original data
    data_copy = copy.deepcopy(data)
    
    # Measure execution time
    start_time = time.time()
    result, comparisons, swaps = sort_func(data_copy)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    return elapsed_time, result, comparisons, swaps


def verify_sorted(arr):
    """Verify that an array is correctly sorted."""
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True


def run_benchmark(algorithm_name, sort_func, datasets, big_o):
    """
    Run benchmarks for a sorting algorithm across all datasets.
    
    Args:
        algorithm_name (str): Name of the algorithm
        sort_func (callable): Sorting function
        datasets (dict): Dictionary of datasets
        big_o (str): Big-O notation string
        
    Returns:
        dict: Benchmark results
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: {algorithm_name}")
    print(f"Big-O Complexity: {big_o}")
    print(f"{'='*70}")
    
    results = {}
    
    for size_name, data in datasets.items():
        elapsed, sorted_data, comparisons, swaps = measure_time(sort_func, data)
        is_sorted = verify_sorted(sorted_data)
        
        results[size_name] = {
            'elapsed_time': elapsed,
            'comparisons': comparisons,
            'swaps': swaps,
            'data_items': len(data),
            'verified': is_sorted
        }
        
        print(f"\n{size_name.upper()} Dataset ({len(data)} items):")
        print(f"  Elapsed Time: {elapsed:.6f} seconds")
        print(f"  Comparisons:  {comparisons:,}")
        print(f"  Swaps/Moves:  {swaps:,}")
        print(f"  Correctly Sorted: {'Yes' if is_sorted else 'NO - ERROR!'}")
    
    return results


def display_summary(all_results):
    """
    Display a summary comparison of all algorithms.
    
    Args:
        all_results (dict): Results from all benchmarks
    """
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY - ALL ALGORITHMS")
    print("="*80)
    
    # Header
    print(f"\n{'Algorithm':<30} {'Dataset':<10} {'Items':<10} {'Time (sec)':<15} {'Big-O':<15}")
    print("-"*80)
    
    for algo_name, algo_data in all_results.items():
        big_o = algo_data['big_o']
        results = algo_data['results']
        
        for size_name, metrics in results.items():
            print(f"{algo_name:<30} {size_name:<10} {metrics['data_items']:<10} "
                  f"{metrics['elapsed_time']:<15.6f} {big_o:<15}")
    
    # Performance comparison
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Compare large dataset times
    print("\nLarge Dataset (10,000 items) Comparison:")
    print("-"*50)
    
    times = []
    for algo_name, algo_data in all_results.items():
        large_time = algo_data['results']['large']['elapsed_time']
        times.append((algo_name, large_time))
    
    times.sort(key=lambda x: x[1])
    
    fastest_time = times[0][1]
    for algo_name, elapsed in times:
        speedup = elapsed / fastest_time if fastest_time > 0 else 1
        print(f"  {algo_name}: {elapsed:.6f} sec ({speedup:.1f}x baseline)")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run all benchmarks."""
    print("="*70)
    print("UNIT 7: SORTING ALGORITHM BENCHMARKING")
    print("Author: Andres Camacho")
    print("Course: IN450 Advanced Software Development")
    print("="*70)
    
    # Create datasets
    print("\nGenerating datasets...")
    datasets = create_datasets()
    
    print(f"  Small dataset:  {len(datasets['small'])} items")
    print(f"  Medium dataset: {len(datasets['medium'])} items")
    print(f"  Large dataset:  {len(datasets['large'])} items")
    
    # Store all results
    all_results = {}
    
    # Benchmark 1: Original Bubble Sort
    results = run_benchmark(
        "Bubble Sort (Original)",
        bubble_sort_original,
        datasets,
        "O(n²)"
    )
    all_results["Bubble Sort (Original)"] = {
        'results': results,
        'big_o': "O(n²)"
    }
    
    # Benchmark 2: Optimized Bubble Sort
    results = run_benchmark(
        "Bubble Sort (Optimized)",
        bubble_sort_optimized,
        datasets,
        "O(n²) avg, O(n) best"
    )
    all_results["Bubble Sort (Optimized)"] = {
        'results': results,
        'big_o': "O(n²) avg, O(n) best"
    }
    
    # Benchmark 3: Quicksort (Alternative Algorithm)
    results = run_benchmark(
        "Quicksort",
        quicksort,
        datasets,
        "O(n log n) avg"
    )
    all_results["Quicksort"] = {
        'results': results,
        'big_o': "O(n log n) avg"
    }
    
    # Display summary
    display_summary(all_results)
    
    # Optimization analysis
    print("\n" + "="*80)
    print("OPTIMIZATION ANALYSIS")
    print("="*80)
    
    # Compare original vs optimized bubble sort
    orig_large = all_results["Bubble Sort (Original)"]['results']['large']['elapsed_time']
    opt_large = all_results["Bubble Sort (Optimized)"]['results']['large']['elapsed_time']
    improvement = ((orig_large - opt_large) / orig_large) * 100 if orig_large > 0 else 0
    
    print(f"\nBubble Sort Optimization (Large Dataset):")
    print(f"  Original:  {orig_large:.6f} seconds")
    print(f"  Optimized: {opt_large:.6f} seconds")
    print(f"  Improvement: {improvement:.1f}%")
    
    # Compare bubble sort vs quicksort
    qs_large = all_results["Quicksort"]['results']['large']['elapsed_time']
    speedup = orig_large / qs_large if qs_large > 0 else 0
    
    print(f"\nQuicksort vs Original Bubble Sort (Large Dataset):")
    print(f"  Bubble Sort: {orig_large:.6f} seconds")
    print(f"  Quicksort:   {qs_large:.6f} seconds")
    print(f"  Speedup:     {speedup:.1f}x faster")
    
    print("\n" + "="*80)
    print("BENCHMARKING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
