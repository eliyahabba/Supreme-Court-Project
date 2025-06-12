#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDA Analysis Utilities
======================

Shared utility functions for LDA analysis modules.
"""

from typing import List, Optional
from pathlib import Path

def get_project_root() -> Path:
    """Get project root directory, handling different execution contexts"""
    current_dir = Path.cwd()
    
    # If running from lda_analysis directory, go up one level
    if current_dir.name == 'lda_analysis':
        return current_dir.parent
    
    # Otherwise assume we're in project root
    return current_dir

def get_full_paths(project_root: Path) -> dict:
    """Get full paths based on project root"""
    from constants import DEFAULT_PATHS
    return {
        key: project_root / path
        for key, path in DEFAULT_PATHS.items()
    }

def filter_topics_from_data(data, excluded_topics: List[int] = None, 
                           included_topics: List[int] = None, 
                           topic_column: str = 'strongest_topic') -> 'DataFrame':
    """
    Filter topics from data based on inclusion/exclusion lists.
    
    Args:
        data: DataFrame to filter
        excluded_topics: List of topic IDs to exclude
        included_topics: List of topic IDs to include (if specified, only these will be kept)
        topic_column: Name of the column containing topic IDs
        
    Returns:
        Filtered DataFrame
    """
    import pandas as pd
    
    if data.empty:
        return data
    
    # Make a copy to avoid modifying original data
    filtered_data = data.copy()
    
    # Handle topic column as string or int
    if topic_column in filtered_data.columns:
        # Convert topic column to int for consistent comparison
        filtered_data[f'{topic_column}_int'] = filtered_data[topic_column].astype(int)
        
        # Apply inclusion filter (if specified)
        if included_topics is not None:
            filtered_data = filtered_data[
                filtered_data[f'{topic_column}_int'].isin(included_topics)
            ]
        
        # Apply exclusion filter
        if excluded_topics is not None:
            filtered_data = filtered_data[
                ~filtered_data[f'{topic_column}_int'].isin(excluded_topics)
            ]
        
        # Remove temporary column
        filtered_data = filtered_data.drop(f'{topic_column}_int', axis=1)
    
    return filtered_data

def get_filtered_topic_mappings(topic_mappings, excluded_topics: List[int] = None,
                               included_topics: List[int] = None) -> 'DataFrame':
    """
    Filter topic mappings based on inclusion/exclusion lists.
    
    Args:
        topic_mappings: DataFrame with topic mappings
        excluded_topics: List of topic IDs to exclude
        included_topics: List of topic IDs to include
        
    Returns:
        Filtered topic mappings DataFrame
    """
    if topic_mappings.empty:
        return topic_mappings
    
    filtered_mappings = topic_mappings.copy()
    
    # Convert topic numbers to int for consistent comparison
    filtered_mappings['topic_int'] = filtered_mappings['מספר נושא'].astype(int)
    
    # Apply inclusion filter (if specified)
    if included_topics is not None:
        filtered_mappings = filtered_mappings[
            filtered_mappings['topic_int'].isin(included_topics)
        ]
    
    # Apply exclusion filter
    if excluded_topics is not None:
        filtered_mappings = filtered_mappings[
            ~filtered_mappings['topic_int'].isin(excluded_topics)
        ]
    
    # Remove temporary column
    filtered_mappings = filtered_mappings.drop('topic_int', axis=1)
    
    return filtered_mappings

def print_filtering_info(excluded_topics: List[int] = None, 
                        included_topics: List[int] = None):
    """Print information about topic filtering"""
    if included_topics is not None:
        print(f"🎯 Including ONLY topics: {sorted(included_topics)}")
    elif excluded_topics:
        print(f"❌ Excluding topics: {sorted(excluded_topics)}")
    else:
        print("✅ No topic filtering applied")

def filter_files_by_length(files_data, min_word_count: int = None) -> 'DataFrame':
    """
    Filter files based on minimum word count.
    
    Args:
        files_data: DataFrame containing file information
        min_word_count: Minimum number of words (if None, uses constant)
        
    Returns:
        Filtered DataFrame
    """
    import pandas as pd
    
    if files_data.empty:
        return files_data
    
    # Use default threshold if not specified
    if min_word_count is None:
        # Try to get from constants, fallback to reasonable default
        try:
            from constants import FILE_LENGTH_MIN_THRESHOLD
            min_word_count = FILE_LENGTH_MIN_THRESHOLD
        except ImportError:
            min_word_count = 100  # Fallback default
            print(f"⚠️ FILE_LENGTH_MIN_THRESHOLD not set, using default: {min_word_count}")
    
    print(f"📏 Filtering files with minimum {min_word_count} words...")
    
    # Check if word count column exists
    word_count_columns = ['word_count', 'words', 'length', 'num_words']
    word_col = None
    
    for col in word_count_columns:
        if col in files_data.columns:
            word_col = col
            break
    
    if word_col is None:
        print("⚠️ No word count column found in data, skipping length filtering")
        return files_data
    
    initial_count = len(files_data)
    filtered_data = files_data[files_data[word_col] >= min_word_count].copy()
    final_count = len(filtered_data)
    
    removed_count = initial_count - final_count
    print(f"📊 Length filtering: {initial_count} → {final_count} files ({removed_count} removed)")
    
    return filtered_data

def get_file_word_count(file_path: Path) -> int:
    """
    Count words in a text file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Number of words in the file
    """
    encodings = ['utf-8', 'windows-1255', 'iso-8859-8', 'cp1255']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            if not content.strip():
                return 0
            
            # Split by whitespace and filter meaningful words
            words = content.split()
            meaningful_words = [word for word in words if len(word.strip()) >= 2]
            
            return len(meaningful_words)
            
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    
    return 0  # Return 0 if all encodings fail

def check_file_length_constants():
    """Check if file length constants are properly set"""
    try:
        from constants import FILE_LENGTH_MEDIAN, FILE_LENGTH_MAXIMUM, FILE_LENGTH_MIN_THRESHOLD
        print(f"📊 File length constants:")
        print(f"   Median: {FILE_LENGTH_MEDIAN:,} words")
        print(f"   Maximum: {FILE_LENGTH_MAXIMUM:,} words") 
        print(f"   Min threshold: {FILE_LENGTH_MIN_THRESHOLD:,} words")
        return True
    except ImportError:
        print("⚠️ File length constants not set. Run analyze_file_lengths.py first.")
        return False 