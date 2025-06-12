#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Length Analysis
===================

Script to analyze file lengths in the Supreme Court dataset and determine
minimum length thresholds for filtering.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import statistics

from utils import get_project_root


def read_file_content(file_path: Path) -> str:
    """
    Read file content with multiple encoding attempts
    
    Args:
        file_path: Path to the file
        
    Returns:
        File content as string, empty string if failed
    """
    encodings = ['utf-8', 'windows-1255', 'iso-8859-8', 'cp1255']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"⚠️ Error reading {file_path} with {encoding}: {e}")
            continue
    
    print(f"❌ Failed to read {file_path} with any encoding")
    return ""


def count_words(text: str) -> int:
    """
    Count words in text (Hebrew and English)
    
    Args:
        text: Text content
        
    Returns:
        Number of words
    """
    if not text.strip():
        return 0
    
    # Remove extra whitespace and split by whitespace
    # This works for both Hebrew and English text
    words = text.split()
    
    # Filter out very short "words" (likely punctuation only)
    meaningful_words = [word for word in words if len(word.strip()) >= 2]
    
    return len(meaningful_words)


def analyze_specific_files(file_names: List[str], data_dir: Path) -> Dict[str, int]:
    """
    Analyze word counts for specific files
    
    Args:
        file_names: List of file names to analyze
        data_dir: Directory containing the files
        
    Returns:
        Dictionary mapping file names to word counts
    """
    results = {}
    
    print(f"📁 Analyzing files in: {data_dir}")
    print(f"🔍 Looking for {len(file_names)} specific files...")
    
    for file_name in file_names:
        file_path = data_dir / file_name
        
        if not file_path.exists():
            print(f"❌ File not found: {file_name}")
            results[file_name] = 0
            continue
        
        try:
            content = read_file_content(file_path)
            word_count = count_words(content)
            results[file_name] = word_count
            
            print(f"✅ {file_name}: {word_count:,} words")
            
        except Exception as e:
            print(f"❌ Error analyzing {file_name}: {e}")
            results[file_name] = 0
    
    return results


def calculate_statistics(word_counts: Dict[str, int]) -> Tuple[int, int, int]:
    """
    Calculate statistics from word counts
    
    Args:
        word_counts: Dictionary of file names to word counts
        
    Returns:
        Tuple of (median, maximum, maximum + 10)
    """
    # Filter out zero counts (failed files)
    valid_counts = [count for count in word_counts.values() if count > 0]
    
    if not valid_counts:
        print("❌ No valid word counts found!")
        return 0, 0, 10
    
    median_count = int(statistics.median(valid_counts))
    max_count = max(valid_counts)
    max_plus_10 = max_count + 10
    
    print(f"\n📊 Statistics from {len(valid_counts)} valid files:")
    print(f"📈 Median word count: {median_count:,}")
    print(f"🔝 Maximum word count: {max_count:,}")
    print(f"🎯 Threshold (max + 10): {max_plus_10:,}")
    
    return median_count, max_count, max_plus_10


def update_constants_file(median: int, maximum: int, threshold: int) -> None:
    """
    Update the constants file with file length statistics
    
    Args:
        median: Median word count
        maximum: Maximum word count  
        threshold: Minimum threshold for filtering (max + 10)
    """
    constants_path = Path(__file__).parent / 'constants.py'
    
    if not constants_path.exists():
        print(f"❌ Constants file not found: {constants_path}")
        return
    
    # Read current constants file
    with open(constants_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define the new constants section
    file_length_constants = f'''
# ============================================================
# File Length Filtering Configuration
# ============================================================

# File length statistics from sample analysis
FILE_LENGTH_MEDIAN: int = {median}
FILE_LENGTH_MAXIMUM: int = {maximum}
FILE_LENGTH_MIN_THRESHOLD: int = {threshold}  # Files with fewer words will be filtered out

# Sample files used for analysis
SAMPLE_FILES_FOR_LENGTH_ANALYSIS = [
    "00003780-A03.txt",
    "00003890-I02.txt", 
    "00003900-G02.txt",
    "00003920-A02.txt",
    "00003980-G03.txt",
    "00003990-V04.txt"
]
'''
    
    # Check if file length section already exists
    if 'FILE_LENGTH_MEDIAN' in content:
        # Replace existing section
        # Find the start and end of the file length section
        start_marker = '# File Length Filtering Configuration'
        end_marker = '# ============================================================'
        
        lines = content.split('\n')
        start_idx = None
        end_idx = None
        
        for i, line in enumerate(lines):
            if start_marker in line:
                # Find the previous section marker
                for j in range(i-1, -1, -1):
                    if lines[j].strip().startswith('# ============================================================'):
                        start_idx = j
                        break
                break
        
        if start_idx is not None:
            # Find the end of this section
            for i in range(start_idx + 1, len(lines)):
                if lines[i].strip().startswith('# ============================================================') and i > start_idx + 2:
                    end_idx = i
                    break
            
            if end_idx is not None:
                # Replace the section
                new_lines = lines[:start_idx] + file_length_constants.strip().split('\n') + lines[end_idx:]
                content = '\n'.join(new_lines)
            else:
                print("⚠️ Could not find end of file length section, appending instead")
                content += file_length_constants
        else:
            print("⚠️ Could not find start of file length section, appending instead")
            content += file_length_constants
    else:
        # Add new section at the end
        content += file_length_constants
    
    # Write updated content
    with open(constants_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Updated constants file: {constants_path}")


def find_data_directory() -> Path:
    """
    Find the data directory containing text files
    
    Returns:
        Path to data directory
    """
    project_root = get_project_root()
    
    # Common possible locations for text files
    possible_paths = [
        project_root / 'data' / 'raw_texts',  # Added this path
        project_root / 'data' / 'raw',
        project_root / 'data' / 'processed', 
        project_root / 'data' / 'text_files',
        project_root / 'data',
        project_root / 'text_files',
        project_root / 'raw_data',
        project_root / 'processed_data'
    ]
    
    for path in possible_paths:
        if path.exists() and any(path.glob('*.txt')):
            print(f"✅ Found data directory: {path}")
            return path
    
    # If not found, ask user to specify
    print("❌ Could not find data directory automatically.")
    print("🔍 Searched in:")
    for path in possible_paths:
        print(f"   - {path}")
    
    print("\nPlease check that the text files are in one of these directories,")
    print("or update the paths in this script.")
    
    # Return the most likely path for manual checking
    return project_root / 'data' / 'raw'


def main():
    """Main function to analyze file lengths and update constants"""
    print("📝 File Length Analysis for Supreme Court Dataset")
    print("=" * 60)
    
    # List of specific files to analyze
    target_files = [
        "00003780-A03.txt",
        "00003890-I02.txt", 
        "00003900-G02.txt",
        "00003920-A02.txt",
        "00003980-G03.txt",
        "00003990-V04.txt"
    ]
    
    # Find data directory
    data_dir = find_data_directory()
    
    if not data_dir.exists():
        print(f"❌ Data directory does not exist: {data_dir}")
        print("Please ensure the text files are in the correct location.")
        return
    
    # Analyze specific files
    word_counts = analyze_specific_files(target_files, data_dir)
    
    if not any(count > 0 for count in word_counts.values()):
        print("❌ No valid files found or all files are empty!")
        print("Please check the file paths and try again.")
        return
    
    # Calculate statistics
    median, maximum, threshold = calculate_statistics(word_counts)
    
    # Display detailed results
    print(f"\n📋 Detailed Results:")
    print(f"{'File Name':<20} {'Word Count':<12}")
    print("-" * 35)
    for file_name, count in word_counts.items():
        status = "✅" if count > 0 else "❌"
        print(f"{status} {file_name:<18} {count:>10,}")
    
    # Update constants file
    print(f"\n💾 Updating constants file...")
    update_constants_file(median, maximum, threshold)
    
    print(f"\n🎉 Analysis completed!")
    print(f"📊 Files with fewer than {threshold:,} words will be filtered out in LDA analysis.")


if __name__ == "__main__":
    main() 