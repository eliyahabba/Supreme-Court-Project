#!/usr/bin/env python3
"""
Year Extraction from Legal Verdict Documents
Extracts years from Israeli Supreme Court verdict text files.
"""

import os
import re
import argparse
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict
import time
from tqdm import tqdm

# Constants - Using absolute paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw_texts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MAX_FILES = None  # Process all files if None
DEFAULT_MAX_WORKERS = 4

# Year extraction patterns
DATE_PATTERNS = [
    r'\b\d{1,2}/\d{1,2}/(\d{2}|\d{4})\b',  # DD/MM/YY or DD/MM/YYYY
    r'\b\d{1,2}\.\d{1,2}\.(\d{2}|\d{4})\b'  # DD.MM.YY or DD.MM.YYYY
]
YEAR_PATTERN = r'\b(\d{4})\b'  # YYYY format

# Legal verdict keyword patterns for proximity-based extraction
VERDICT_KEYWORDS = [
    r'ניתן\s+היום',          # "ניתן היום"
    r'ניתן\s+ביום',          # "ניתן ביום" 
    r'ניתנה\s+היום',         # "ניתנה היום"
    r'ניתנה\s+ביום',         # "ניתנה ביום"
    r'פסק\s+דין\s+מיום',     # "פסק דין מיום"
    r'פסק[-\s]דין\s+מיום',   # "פסק-דין מיום"
    r'הוחלט\s+כאמור',        # "הוחלט כאמור" (ממצא נוסף)
    r'נפסק\s+כי',            # "נפסק כי" (ממצא נוסף)
]

MAX_WORD_DISTANCE = 10  # Maximum distance in words from keyword to year

MIN_VALID_YEAR = 1948  # Israel's founding year
MAX_VALID_YEAR = 2025  # Future buffer
CENTURY_CUTOFF = 50     # Years 00-50 are 2000s, 51-99 are 1900s


def extract_years_from_text(text: str, use_proximity: bool = False) -> List[str]:
    """Extract all valid years from text using regex patterns.
    
    Args:
        text: Input text to search
        use_proximity: If True, prioritize years near verdict keywords
        
    Returns:
        List of extracted years as strings
    """
    if use_proximity:
        return _extract_years_with_proximity(text)
    else:
        return _extract_years_traditional(text)


def _extract_years_traditional(text: str) -> List[str]:
    """Traditional year extraction method (original logic)."""
    years = []
    
    # Try date patterns first (DD/MM/YY format)
    for pattern in DATE_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            year = _normalize_year(match)
            if year:
            years.append(year)

    # If no years found in dates, try standalone year pattern
    if not years:
        matches = re.findall(YEAR_PATTERN, text)
        for match in matches:
            if _is_valid_year(match):
                years.append(match)
    
    return years


def _extract_years_with_proximity(text: str) -> List[str]:
    """Extract years prioritizing those near verdict keywords."""
    all_years = []
    years_with_proximity = []
    
    # First, find all years using traditional method
    traditional_years = _extract_years_traditional(text)
    
    # If no traditional years found, return empty
    if not traditional_years:
        return []
    
    # Find all year positions in text
    year_positions = []
    words = text.split()
    
    for i, word in enumerate(words):
        # Check for years in dates
        for pattern in DATE_PATTERNS:
            matches = re.finditer(pattern, word)
            for match in matches:
                year = _normalize_year(match.group(1))
                if year:
                    year_positions.append((year, i))
        
        # Check for standalone years
        if re.match(YEAR_PATTERN, word) and _is_valid_year(word):
            year_positions.append((word, i))
    
    # Find keyword positions
    text_for_search = ' '.join(words)
    keyword_positions = []
    
    for keyword_pattern in VERDICT_KEYWORDS:
        matches = re.finditer(keyword_pattern, text_for_search, re.IGNORECASE)
        for match in matches:
            # Convert character position to approximate word position
            start_chars = match.start()
            word_pos = len(text_for_search[:start_chars].split()) - 1
            keyword_positions.append(word_pos)
    
    # Calculate proximity scores for each year
    for year, year_pos in year_positions:
        min_distance = float('inf')
        
        for keyword_pos in keyword_positions:
            distance = abs(year_pos - keyword_pos)
            min_distance = min(min_distance, distance)
        
        if min_distance <= MAX_WORD_DISTANCE:
            years_with_proximity.append((year, min_distance))
        
        # Also keep all years for fallback
        all_years.append(year)
    
    # If we found years near keywords, return the closest one
    if years_with_proximity:
        # Sort by proximity (closest first), then by year (latest first)
        years_with_proximity.sort(key=lambda x: (x[1], -int(x[0])))
        closest_years = [year for year, _ in years_with_proximity]
        return closest_years
    
    # Fallback to traditional years if no proximity matches
    return traditional_years


def _normalize_year(year_str: str) -> Optional[str]:
    """Convert 2-digit years to 4-digit years and validate."""
    if len(year_str) == 2:
        year_int = int(year_str)
        if year_int <= CENTURY_CUTOFF:
            year_str = '20' + year_str
        else:
            year_str = '19' + year_str
    
    return year_str if _is_valid_year(year_str) else None


def _is_valid_year(year_str: str) -> bool:
    """Check if year is within valid range."""
    try:
        year = int(year_str)
        return MIN_VALID_YEAR <= year <= MAX_VALID_YEAR
    except ValueError:
        return False


def process_single_file(file_path: Path, use_proximity: bool = False) -> Tuple[str, List[str]]:
    """Process a single text file and extract years."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            years = extract_years_from_text(text, use_proximity)
            return file_path.name, years
    except Exception as e:
        # Using tqdm.write to maintain progress bar display
        tqdm.write(f"Error processing {file_path}: {e}")
        return file_path.name, []


def get_max_year_info(years_list: List[str]) -> Tuple[str, int]:
    """Get the maximum year and count of unique years."""
    if not years_list:
        return '', 0
    
    valid_years = [int(year) for year in years_list if _is_valid_year(year)]
    if not valid_years:
        return '', 0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    
    return str(max(valid_years)), len(set(valid_years))


def process_files_parallel(input_dir: Path, max_files: Optional[int] = None, 
                          max_workers: int = DEFAULT_MAX_WORKERS,
                          use_proximity: bool = False) -> pd.DataFrame:
    """Process multiple files in parallel and extract years."""
    
    # Get all text files
    text_files = list(input_dir.glob('*.txt'))
    
    if max_files:
        text_files = text_files[:max_files]
        print(f"Processing first {len(text_files)} files...")
    else:
        print(f"Processing all {len(text_files)} files...")
    
    if not text_files:
        print("No text files found in the input directory!")
        return pd.DataFrame()
    
    print(f"Using {'proximity-based' if use_proximity else 'traditional'} year extraction method")
    
    results = []
    
    # Process files in parallel with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, file_path, use_proximity): file_path 
            for file_path in text_files
        }
        
        # Collect results with progress bar
        with tqdm(total=len(text_files), desc="Processing files", unit="file") as pbar:
            for future in as_completed(future_to_file):
                filename, years = future.result()
                results.append({
                    'filename': filename,
                    'years': ', '.join(years),
                    'years_list': years
                })
                pbar.update(1)
    
    # Create DataFrame
    print("Creating DataFrame and calculating statistics...")
    df = pd.DataFrame(results)
    
    # Add max year and count columns with progress bar
    tqdm.pandas(desc="Calculating max years")
    df[['max_year', 'num_unique_years']] = df['years_list'].progress_apply(
        lambda years: pd.Series(get_max_year_info(years))
    )
    
    # Remove temporary column
    df = df.drop('years_list', axis=1)
    
    return df


def save_results(df: pd.DataFrame, output_dir: Path) -> str:
    """Save results to CSV file with timestamp."""
    timestamp = int(time.time())
    output_file = output_dir / f"extracted_years_{timestamp}.csv"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving results to: {output_file}")
    df.to_csv(output_file, index=False, encoding='utf-8')
    return str(output_file)


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Extract years from legal verdict text files"
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help=f'Input directory containing text files (default: {DEFAULT_INPUT_DIR})'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f'Output directory for results (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--max-files', '-n',
        type=int,
        default=DEFAULT_MAX_FILES,
        help='Maximum number of files to process (default: process all files)'
    )
    
    parser.add_argument(
        '--max-workers', '-w',
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f'Maximum number of worker threads (default: {DEFAULT_MAX_WORKERS})'
    )
    
    parser.add_argument(
        '--use-proximity', '-p',
        action='store_true', default=False,
        help='Use proximity-based extraction (prioritize years near verdict keywords)'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute Path objects
    input_dir = Path(args.input_dir).absolute()
    output_dir = Path(args.output_dir).absolute()
    
    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist!")
        print(f"Please create the directory or place your text files there.")
        return
    
    print("=" * 60)
    print("YEAR EXTRACTION FROM LEGAL VERDICTS")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max files: {args.max_files if args.max_files else 'All'}")
    print(f"Max workers: {args.max_workers}")
    print(f"Extraction method: {'Proximity-based' if args.use_proximity else 'Traditional'}")
    if args.use_proximity:
        print(f"Max word distance: {MAX_WORD_DISTANCE}")
        print(f"Verdict keywords: {len(VERDICT_KEYWORDS)} patterns")
    print("-" * 60)
    
    # Process files
    start_time = time.time()
    df = process_files_parallel(input_dir, args.max_files, args.max_workers, args.use_proximity)
    
    if df.empty:
        print("No files were processed. Exiting.")
        return
    
    # Save results
    output_file = save_results(df, output_dir)
    
    # Print summary
    end_time = time.time()
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Files processed: {len(df)}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Results saved to: {output_file}")
    
    # Show sample results
    if len(df) > 0:
        print(f"\nSample results:")
        print("-" * 60)
        print(df.head().to_string(index=False))
        
        # Summary statistics
        print(f"\nSummary Statistics:")
        print("-" * 60)
        print(f"Files with years found: {len(df[df['years'] != ''])}")
        print(f"Files without years: {len(df[df['years'] == ''])}")
        if len(df[df['max_year'] != '']) > 0:
            years_series = df[df['max_year'] != '']['max_year'].astype(int)
            print(f"Year range: {years_series.min()} - {years_series.max()}")
            print(f"Most common year: {years_series.mode().iloc[0] if not years_series.mode().empty else 'N/A'}")


if __name__ == "__main__":
    main()
