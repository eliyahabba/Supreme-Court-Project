#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Length Filter for LDA Analysis
==================================

Shared utilities for filtering files by word count before LDA analysis.
This module provides functions to integrate file length filtering into the LDA pipeline.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

from utils import (
    get_project_root, filter_files_by_length, get_file_word_count,
    check_file_length_constants
)


def add_word_counts_to_dataframe(df: pd.DataFrame, text_files_dir: Path = None) -> pd.DataFrame:
    """
    Add word count column to DataFrame containing filenames.
    
    Args:
        df: DataFrame with 'filename' column
        text_files_dir: Directory containing text files (auto-detected if None)
        
    Returns:
        DataFrame with added 'word_count' column
    """
    if 'filename' not in df.columns:
        print("⚠️ DataFrame must have 'filename' column")
        return df
    
    # Auto-detect text files directory if not provided
    if text_files_dir is None:
        text_files_dir = find_text_files_directory()
    
    if not text_files_dir.exists():
        print(f"❌ Text files directory not found: {text_files_dir}")
        return df
    
    print(f"📁 Adding word counts from: {text_files_dir}")
    print(f"🔄 Processing {len(df)} files...")
    
    word_counts = []
    for filename in df['filename']:
        file_path = text_files_dir / filename
        word_count = get_file_word_count(file_path)
        word_counts.append(word_count)
    
    df_with_counts = df.copy()
    df_with_counts['word_count'] = word_counts
    
    # Show statistics
    total_files = len(df_with_counts)
    files_with_words = len(df_with_counts[df_with_counts['word_count'] > 0])
    avg_words = df_with_counts['word_count'].mean()
    
    print(f"✅ Word count analysis complete:")
    print(f"   📄 Total files: {total_files}")
    print(f"   ✅ Files with content: {files_with_words}")
    print(f"   📊 Average words: {avg_words:.1f}")
    
    return df_with_counts


def find_text_files_directory() -> Path:
    """
    Find the directory containing the original text files.
    
    Returns:
        Path to text files directory
    """
    project_root = get_project_root()
    
    possible_paths = [
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
            return path
    
    # Return most likely default
    return project_root / 'data' / 'raw'


def apply_length_filtering(df: pd.DataFrame, min_word_count: int = None, 
                          add_word_counts: bool = True, text_files_dir: Path = None) -> Tuple[pd.DataFrame, dict]:
    """
    Apply file length filtering to a DataFrame.
    
    Args:
        df: Input DataFrame
        min_word_count: Minimum word count threshold
        add_word_counts: Whether to add word counts if not present
        text_files_dir: Directory with text files
        
    Returns:
        Tuple of (filtered_dataframe, filtering_stats)
    """
    print("📏 Applying file length filtering...")
    
    # Check if file length constants are set
    constants_ok = check_file_length_constants()
    if not constants_ok and min_word_count is None:
        print("⚠️ Using fallback minimum word count: 100")
        min_word_count = 100
    
    # Add word counts if needed and requested
    if add_word_counts and 'word_count' not in df.columns:
        print("📊 Word counts not found, calculating...")
        df = add_word_counts_to_dataframe(df, text_files_dir)
    
    # Apply filtering
    initial_count = len(df)
    
    if 'word_count' in df.columns:
        filtered_df = filter_files_by_length(df, min_word_count)
        final_count = len(filtered_df)
        
        # Calculate statistics
        if 'word_count' in df.columns:
            removed_files = initial_count - final_count
            avg_words_before = df['word_count'].mean()
            avg_words_after = filtered_df['word_count'].mean() if final_count > 0 else 0
            min_words_before = df['word_count'].min()
            min_words_after = filtered_df['word_count'].min() if final_count > 0 else 0
            
            stats = {
                'initial_count': initial_count,
                'final_count': final_count,
                'removed_count': removed_files,
                'removal_percentage': (removed_files / initial_count * 100) if initial_count > 0 else 0,
                'avg_words_before': avg_words_before,
                'avg_words_after': avg_words_after,
                'min_words_before': min_words_before,
                'min_words_after': min_words_after,
                'threshold_used': min_word_count
            }
        else:
            stats = {
                'initial_count': initial_count,
                'final_count': final_count,
                'removed_count': 0,
                'removal_percentage': 0,
                'threshold_used': min_word_count,
                'note': 'No word count available'
            }
    else:
        print("⚠️ No word count column available, skipping length filtering")
        filtered_df = df
        stats = {
            'initial_count': initial_count,
            'final_count': initial_count,
            'removed_count': 0,
            'removal_percentage': 0,
            'note': 'No filtering applied - word counts not available'
        }
    
    return filtered_df, stats


def print_filtering_summary(stats: dict):
    """
    Print a summary of file length filtering results.
    
    Args:
        stats: Statistics dictionary from apply_length_filtering
    """
    print("\n📋 File Length Filtering Summary:")
    print("=" * 50)
    print(f"📄 Initial files: {stats['initial_count']:,}")
    print(f"✅ Remaining files: {stats['final_count']:,}")
    print(f"❌ Removed files: {stats['removed_count']:,}")
    print(f"📊 Removal rate: {stats['removal_percentage']:.1f}%")
    
    if 'threshold_used' in stats:
        threshold = stats['threshold_used']
        print(f"🎯 Minimum word threshold: {threshold:,}")
    
    if 'avg_words_before' in stats:
        print(f"📈 Average words (before): {stats['avg_words_before']:.1f}")
        print(f"📈 Average words (after): {stats['avg_words_after']:.1f}")
        print(f"📉 Min words (before): {stats['min_words_before']:,}")
        print(f"📉 Min words (after): {stats['min_words_after']:,}")
    
    if 'note' in stats:
        print(f"💡 Note: {stats['note']}")


def filter_lda_input_data(doc_mappings: pd.DataFrame, years_df: pd.DataFrame, 
                         min_word_count: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Filter LDA input data by file length.
    
    Args:
        doc_mappings: Document-topic mappings DataFrame
        years_df: Years extraction DataFrame
        min_word_count: Minimum word count threshold
        
    Returns:
        Tuple of (filtered_doc_mappings, filtered_years_df, filtering_stats)
    """
    print("🔄 Filtering LDA input data by file length...")
    
    # Filter years data first (it's more likely to be complete)
    filtered_years_df, stats = apply_length_filtering(
        years_df, 
        min_word_count=min_word_count,
        add_word_counts=True
    )
    
    # Filter doc mappings to match
    if not filtered_years_df.empty:
        valid_filenames = set(filtered_years_df['filename'])
        initial_doc_count = len(doc_mappings)
        
        filtered_doc_mappings = doc_mappings[
            doc_mappings['filename'].isin(valid_filenames)
        ].copy()
        
        final_doc_count = len(filtered_doc_mappings)
        
        print(f"📊 Document mappings: {initial_doc_count} → {final_doc_count}")
        
        # Update stats
        stats['doc_mappings_initial'] = initial_doc_count
        stats['doc_mappings_final'] = final_doc_count
        stats['doc_mappings_removed'] = initial_doc_count - final_doc_count
    else:
        print("⚠️ No files passed length filtering!")
        filtered_doc_mappings = pd.DataFrame()
    
    return filtered_doc_mappings, filtered_years_df, stats


def main():
    """Test function for file length filtering"""
    print("🧪 Testing File Length Filtering")
    print("=" * 40)
    
    # Check if constants are set
    check_file_length_constants()
    
    # Find text files directory
    text_dir = find_text_files_directory()
    print(f"📁 Text files directory: {text_dir}")
    
    # Create sample DataFrame for testing
    sample_files = [
        "00003780-A03.txt",
        "00003890-I02.txt", 
        "00003900-G02.txt",
        "00003920-A02.txt",
        "00003980-G03.txt",
        "00003990-V04.txt"
    ]
    
    test_df = pd.DataFrame({'filename': sample_files})
    print(f"\n🔬 Testing with {len(sample_files)} sample files:")
    
    # Apply filtering
    filtered_df, stats = apply_length_filtering(test_df)
    
    # Print results
    print_filtering_summary(stats)
    
    if not filtered_df.empty:
        print(f"\n✅ Remaining files:")
        for filename in filtered_df['filename']:
            word_count = filtered_df[filtered_df['filename'] == filename]['word_count'].iloc[0]
            print(f"   📄 {filename}: {word_count:,} words")


if __name__ == "__main__":
    main() 