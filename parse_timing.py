#!/usr/bin/env python3
"""
Parse clock cycle timing data from out2.txt and calculate average statistics.
Mimics the calculation logic from many_linear.cu
"""

import re
import sys
import argparse
from collections import defaultdict

# Constants from many_linear.cu
NUM_LAYERS = 30
REDUCTION_SIZE = 1024
REDUCTION_SIZE_DIV_128 = REDUCTION_SIZE // 128  # = 8
NUM_RUNS_TO_PROCESS = 1

def parse_timing_file(filename):
    """Parse the timing file and extract clock cycle arrays."""
    clock_cycles_mem = defaultdict(list)
    clock_cycles_compute = defaultdict(list)
    
    with open(filename, 'r') as f:
        for line in f:
            # Parse l_clock_cycles_mem[i]: value
            match = re.match(r'l_clock_cycles_mem\[(\d+)\]:\s*(\d+)', line)
            if match:
                idx = int(match.group(1))
                value = int(match.group(2))
                clock_cycles_mem[idx].append(value)
                continue
            
            # Parse l_clock_cycles_mem_launch[i]: value
            match = re.match(r'l_clock_cycles_mem_launch\[(\d+)\]:\s*(\d+)', line)
            if match:
                idx = int(match.group(1))
                value = int(match.group(2))
                clock_cycles_mem[idx].append(value)
                continue
            
            # Parse l_clock_cycles_compute[i]: value
            match = re.match(r'l_clock_cycles_compute\[(\d+)\]:\s*(\d+)', line)
            if match:
                idx = int(match.group(1))
                value = int(match.group(2))
                clock_cycles_compute[idx].append(value)
                continue
    
    return clock_cycles_mem, clock_cycles_compute

def calculate_statistics(clock_cycles_mem, clock_cycles_compute):
    """Calculate average statistics following the C++ logic."""
    
    # Convert to flat arrays - each index should have multiple values
    # We need to organize them into groups of 8 (REDUCTION_SIZE_DIV_128)
    
    # Determine total number of measurements
    common_length = len(clock_cycles_compute[0])
    for idx in range(8):
        assert len(clock_cycles_compute[idx]) == common_length
        assert len(clock_cycles_mem[idx]) == common_length
    
    assert common_length >= NUM_LAYERS * (NUM_RUNS_TO_PROCESS)
    total_measurements = NUM_LAYERS * NUM_RUNS_TO_PROCESS
    
    print(f"Total measurements per index: {total_measurements}")
    
    # Initialize accumulators
    compute = 0
    launch = 0
    wait = 0
    warmup_launch = 0
    warmup_wait = 0
    
    # Process in groups of 8 (REDUCTION_SIZE_DIV_128)
    num_groups = total_measurements
    
    for group_idx in range(num_groups):
        # Get values for this group (indices 0-7)
        h_clock_cycles_compute = []
        h_clock_cycles_mem = []
        
        for j in range(8):
            assert j in clock_cycles_compute and j in clock_cycles_mem
            clock_cycles_compute_j = clock_cycles_compute[j][-num_groups:]
            clock_cycles_mem_j = clock_cycles_mem[j][-num_groups:]
            h_clock_cycles_compute.append(clock_cycles_compute_j[group_idx])
            h_clock_cycles_mem.append(clock_cycles_mem_j[group_idx])
        
        # Apply the C++ logic for each layer
        # for (int i = 0; i < NUM_LAYERS * (REDUCTION_SIZE / 128); i += REDUCTION_SIZE / 128)
        # In this case, each group of 8 consecutive measurements represents one layer iteration
        
        # warmup_wait += h_clock_cycles_compute[i + 5]
        warmup_wait += h_clock_cycles_compute[5]
        
        # warmup_launch += h_clock_cycles_compute[i + 4]
        warmup_launch += h_clock_cycles_compute[4]
        
        # compute += (h_clock_cycles_compute[i + 6] + h_clock_cycles_compute[i + 7] + h_clock_cycles_compute[i + 3]) / 3
        compute += (h_clock_cycles_compute[6] + h_clock_cycles_compute[7] + h_clock_cycles_compute[3]) / 3
        
        # Accumulate launch and wait from mem array
        acc_launch = 0
        acc_wait = 0
        for j in range(8):
            if j % 2 == 0:
                acc_wait += h_clock_cycles_mem[j]
            else:
                acc_launch += h_clock_cycles_mem[j]
        
        # launch += (acc_launch / ((REDUCTION_SIZE / 128) / 2))
        launch += (acc_launch / 4)
        
        # wait += (acc_wait / ((REDUCTION_SIZE / 128) / 2))
        wait += (acc_wait / 4)
    
    # Divide by number of groups (which should be NUM_LAYERS * NUM_TRIALS)
    # But we divide by NUM_LAYERS to get average per layer
    num_layers_measured = num_groups
    
    compute /= num_layers_measured
    warmup_wait /= num_layers_measured
    warmup_launch /= num_layers_measured
    launch /= num_layers_measured
    wait /= num_layers_measured
    
    return {
        'compute': int(compute),
        'warmup_wait': int(warmup_wait),
        'warmup_launch': int(warmup_launch),
        'launch': int(launch),
        'wait': int(wait)
    }

def main():
    parser = argparse.ArgumentParser(
        description='Parse clock cycle timing data and calculate average statistics'
    )
    parser.add_argument(
        'filename',
        nargs='?',
        default='out2.txt',
        help='Input file containing timing data (default: out2.txt)'
    )
    args = parser.parse_args()
    
    print(f"Parsing timing data from {args.filename}...")
    clock_cycles_mem, clock_cycles_compute = parse_timing_file(args.filename)
    
    print(f"Parsed {len(clock_cycles_mem)} mem indices and {len(clock_cycles_compute)} compute indices")
    for idx in range(8):
        if idx in clock_cycles_mem:
            print(f"  Index {idx}: {len(clock_cycles_mem[idx])} mem entries, {len(clock_cycles_compute[idx])} compute entries")
    
    print("\nCalculating statistics...")
    stats = calculate_statistics(clock_cycles_mem, clock_cycles_compute)
    
    print("\nReporting average clock cycles for each layer:")
    print(f"compute = {stats['compute']}")
    print(f"warmup_wait = {stats['warmup_wait']}")
    print(f"warmup_launch = {stats['warmup_launch']}")
    print(f"launch = {stats['launch']}")
    print(f"wait = {stats['wait']}")

if __name__ == '__main__':
    main()

