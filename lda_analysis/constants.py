#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDA Analysis Constants
======================

Shared constants and configuration for LDA analysis modules.
"""

from typing import List, Optional
from pathlib import Path

# ============================================================
# Path Configuration
# ============================================================

# Default paths relative to project root
DEFAULT_PATHS = {
    'lda_model_dir': Path('LDA Best Result') / '1693294471',
    'years_data': Path('data') / 'processed' / 'extracted_years.csv',
    'custom_topics': Path('LDA Best Result') / '1693294471' / 'topics_with_claude.csv',
    'output_dir': Path('data') / 'results' / 'lda'
}

# ============================================================
# Topic Filtering Configuration
# ============================================================

# Topics to exclude from analysis by default
# נושאים להחריג מהניתוח כברירת מחדל
DEFAULT_EXCLUDED_TOPICS: List[int] = [11]

# Optional: Topics to include (if specified, only these will be analyzed)
# אופציונלי: נושאים לכלול (אם מצוין, רק אלה ינותחו)
DEFAULT_INCLUDED_TOPICS: Optional[List[int]] = None

# ============================================================
# Analysis Configuration
# ============================================================

# Default threshold for multi-topic analysis
DEFAULT_MULTI_TOPIC_THRESHOLD: float = 0.1

# Default number of top topics to include per document (regardless of threshold)
DEFAULT_TOP_K_TOPICS: int = 3

# Default minimum year for analysis
DEFAULT_MIN_YEAR: int = 1948

# Default analysis modes
ANALYSIS_MODES = {
    'single': 'single-topic',
    'multi': 'multi-topic',
    'both': 'both'
}

# ============================================================
# Visualization Configuration
# ============================================================

# Default chart dimensions
DEFAULT_CHART_WIDTH: int = 1200
DEFAULT_CHART_HEIGHT: int = 700

# Wide chart dimensions
WIDE_CHART_WIDTH: int = 1400
WIDE_CHART_HEIGHT: int = 800

# Color configuration
COLOR_SATURATION_BASE: float = 0.8
COLOR_VALUE_BASE: float = 0.7

# ============================================================
# Export Configuration
# ============================================================

# Default file encoding for CSV exports
DEFAULT_ENCODING: str = 'utf-8-sig'

# ============================================================
# File Length Filtering Configuration
# ============================================================

# File length statistics from sample analysis
FILE_LENGTH_MEDIAN: int = 125
FILE_LENGTH_MAXIMUM: int = 292
FILE_LENGTH_MIN_THRESHOLD: int = 302  # Files with fewer words will be filtered out

# Sample files used for analysis
SAMPLE_FILES_FOR_LENGTH_ANALYSIS = [
    "00003780-A03.txt",
    "00003890-I02.txt", 
    "00003900-G02.txt",
    "00003920-A02.txt",
    "00003980-G03.txt",
    "00003990-V04.txt"
]
