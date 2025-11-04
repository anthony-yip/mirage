#!/usr/bin/env python3
"""
Parse out_0.6_warmup.txt and calculate:
total (warmup launch time + wait time) / total cycles
"""

import re
import sys

def parse_warmup_file(filepath):
    """Parse the warmup file and extract metrics."""
    total_cycles = 0
    total_warmup_launch_time = 0
    total_warmup_wait_time = 0
    num_entries = 0
    
    # Pattern to match lines with cycle data
    pattern = r'(\d+) cycles\|.*?\|(\d+) warmup launch time\|(\d+) warmup wait time\|'
    
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                cycles = int(match.group(1))
                warmup_launch_time = int(match.group(2))
                warmup_wait_time = int(match.group(3))
                
                total_cycles += cycles
                total_warmup_launch_time += warmup_launch_time
                total_warmup_wait_time += warmup_wait_time
                num_entries += 1
    
    return {
        'total_cycles': total_cycles,
        'total_warmup_launch_time': total_warmup_launch_time,
        'total_warmup_wait_time': total_warmup_wait_time,
        'num_entries': num_entries
    }

def main():
    filepath = 'out_0.6_warmup.txt'
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    
    print(f"Parsing {filepath}...")
    print()
    
    results = parse_warmup_file(filepath)
    
    print(f"Number of entries: {results['num_entries']}")
    print(f"Total cycles: {results['total_cycles']:,}")
    print(f"Total warmup launch time: {results['total_warmup_launch_time']:,}")
    print(f"Total warmup wait time: {results['total_warmup_wait_time']:,}")
    print()
    
    total_warmup_time = results['total_warmup_launch_time'] + results['total_warmup_wait_time']
    print(f"Total warmup time (launch + wait): {total_warmup_time:,}")
    print()
    
    if results['total_cycles'] > 0:
        ratio = total_warmup_time / results['total_cycles']
        print(f"Ratio (warmup launch time + wait time) / total cycles: {ratio:.6f}")
        print(f"Percentage: {ratio * 100:.4f}%")
    else:
        print("No cycles data found")

if __name__ == '__main__':
    main()

