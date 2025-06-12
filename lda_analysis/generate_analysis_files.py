#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDA Analysis - File Generation Script
====================================

Script for generating LDA analysis files and reports.
Uses shared visualization functions for consistency.
Updated to support topic filtering using shared constants.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional
import argparse

import pandas as pd
from gensim import models

# Import shared constants
from constants import (
    DEFAULT_PATHS, DEFAULT_EXCLUDED_TOPICS, DEFAULT_INCLUDED_TOPICS, DEFAULT_MIN_YEAR,
    DEFAULT_MULTI_TOPIC_THRESHOLD, DEFAULT_TOP_K_TOPICS, DEFAULT_ENCODING
)
from utils import (
    get_project_root, get_full_paths, filter_topics_from_data, get_filtered_topic_mappings,
    print_filtering_info, check_file_length_constants
)
from file_length_filter import filter_lda_input_data, print_filtering_summary

from lda_visualizations import (
    create_wordcloud_grid,
    plot_absolute_topics_trend,
    plot_stacked_yearly_distribution,
    plot_topics_histogram,
    plot_topics_trend,
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LDAFileGenerator:
    """Class for generating LDA analysis files"""

    def __init__(self, excluded_topics: List[int] = None, included_topics: List[int] = None):
        """
        Initialize file generator with topic filtering options
        
        Args:
            excluded_topics: List of topic IDs to exclude (default: from constants)
            included_topics: List of topic IDs to include (if specified, only these will be analyzed)
        """
        self.excluded_topics = excluded_topics if excluded_topics is not None else DEFAULT_EXCLUDED_TOPICS
        self.included_topics = included_topics
        
        # Print filtering configuration
        print_filtering_info(self.excluded_topics, self.included_topics)
        
        self.setup_paths()
        
        self.model = None
        self.doc_mappings = pd.DataFrame()
        self.topic_mappings = pd.DataFrame()
        self.years_df = pd.DataFrame()
        self.merged_df = pd.DataFrame()
        self.yr_agg_df = pd.DataFrame()

    def setup_paths(self, mode_suffix: str = ""):
        """Setup all required paths using shared constants"""
        self.project_root = get_project_root()
        self.paths = get_full_paths(self.project_root)
        
        # Add mode suffix to output directory if specified
        if mode_suffix:
            self.paths['output_dir'] = self.paths['output_dir'] / mode_suffix

        self.paths['output_dir'].mkdir(parents=True, exist_ok=True)

        print("📁 Paths configured:")
        for key, path in self.paths.items():
            exists = "✅" if path.exists() else "❌"  
            print(f"  {key}: {exists} {path}")

    def load_data(self, apply_length_filter: bool = True):
        """Load all required data with optional file length filtering"""
        print("🔄 Loading LDA model and data...")

        # Load model
        model_path = self.paths['lda_model_dir'] / "model"
        self.model = models.ldamodel.LdaModel.load(str(model_path))

        # Load document mappings
        doc_mappings_path = self.paths['lda_model_dir'] / "docs_topics.csv"
        doc_mappings_original = pd.read_csv(doc_mappings_path)

        # Load topic mappings
        if self.paths['custom_topics'].exists():
            self.topic_mappings = pd.read_csv(self.paths['custom_topics'], encoding='utf-8')
        else:
            self.topic_mappings = pd.DataFrame()

        # Load years data
        years_df_original = pd.read_csv(self.paths['years_data'])

        print(f"✅ Loaded model with {self.model.num_topics} topics")
        print(f"✅ Loaded {len(doc_mappings_original)} document mappings")
        print(f"✅ Loaded {len(years_df_original)} year records")

        # Apply file length filtering if requested
        if apply_length_filter:
            print("\n📏 Applying file length filtering...")
            check_file_length_constants()
            
            self.doc_mappings, self.years_df, filtering_stats = filter_lda_input_data(
                doc_mappings_original, years_df_original
            )
            
            print_filtering_summary(filtering_stats)
        else:
            print("⚠️ Skipping file length filtering")
            self.doc_mappings = doc_mappings_original
            self.years_df = years_df_original

    def create_single_topic_data(self):
        """Create single-topic analysis data with topic filtering"""
        print("🔄 Creating single-topic data...")
        
        # Merge years with document mappings
        self.merged_df = self.years_df.merge(self.doc_mappings, on='filename', how='inner')
        
        # Find topic columns
        topic_cols = [col for col in self.merged_df.columns if col.isdigit()]
        
        if topic_cols:
            self.merged_df['strongest_topic'] = self.merged_df[topic_cols].idxmax(axis=1)
            self.merged_df['strongest_topic_prob'] = self.merged_df[topic_cols].max(axis=1)
        else:
            self.merged_df['strongest_topic'] = '0'
            self.merged_df['strongest_topic_prob'] = 1.0

        self.merged_df['year'] = self.merged_df['max_year'].astype(int)
        
        # Apply topic filtering
        print(f"📊 Before filtering: {len(self.merged_df)} documents")
        self.merged_df = filter_topics_from_data(
            self.merged_df, 
            self.excluded_topics, 
            self.included_topics, 
            'strongest_topic'
        )
        print(f"📊 After filtering: {len(self.merged_df)} documents")

        # Create aggregated data
        if not self.merged_df.empty:
            yr_topic_agg = self.merged_df.groupby(['year', 'strongest_topic']).size().reset_index(name='verdicts')
            yr_total = self.merged_df.groupby(['year']).size().reset_index(name='total_verdicts')
            self.yr_agg_df = yr_topic_agg.merge(yr_total, on='year')
            self.yr_agg_df['topic_percentage'] = (self.yr_agg_df['verdicts'] * 100.0 / self.yr_agg_df['total_verdicts']).round(2)
        else:
            self.yr_agg_df = pd.DataFrame()

        print(f"✅ Created single-topic data: {len(self.merged_df)} documents")

    def create_multi_topic_data(self, threshold: float = DEFAULT_MULTI_TOPIC_THRESHOLD, top_k: int = DEFAULT_TOP_K_TOPICS):
        """Create multi-topic analysis data with topic filtering and top-K support"""
        print(f"🔄 Creating multi-topic data with threshold {threshold} and top-{top_k} topics...")
        
        topic_cols = [col for col in self.doc_mappings.columns if col.isdigit()]
        
        if not topic_cols:
            print("⚠️ No topic columns found")
            return False
        
        multi_topic_data = []
        
        for _, row in self.doc_mappings.iterrows():
            filename = row['filename']
            
            # Get all topic probabilities for this document (after filtering)
            filtered_topics = []
            for topic_col in topic_cols:
                topic_prob = row[topic_col]
                topic_id = int(topic_col)
                
                # Apply topic filtering here
                if self.included_topics is not None and topic_id not in self.included_topics:
                    continue
                if self.excluded_topics and topic_id in self.excluded_topics:
                    continue
                
                filtered_topics.append((topic_id, topic_prob))
            
            if not filtered_topics:
                continue  # Skip if no topics passed filtering
            
            # Sort topics by probability (highest first)
            filtered_topics.sort(key=lambda x: x[1], reverse=True)
            
            # Get topics above threshold
            topics_above_threshold = [(topic_id, prob) for topic_id, prob in filtered_topics if prob >= threshold]
            
            # Get top-K topics (regardless of threshold)
            top_k_topics = filtered_topics[:top_k]
            
            # Combine both approaches: use threshold OR top-K (whichever gives more topics)
            if topics_above_threshold:
                # If we have topics above threshold, include all of them
                selected_topics = topics_above_threshold
                # But also ensure we have at least top-K if threshold gives us fewer
                if len(topics_above_threshold) < top_k:
                    for topic_id, prob in top_k_topics:
                        if (topic_id, prob) not in topics_above_threshold:
                            selected_topics.append((topic_id, prob))
                            if len(selected_topics) >= top_k:
                                break
            else:
                # If no topics above threshold, fall back to top-K
                selected_topics = top_k_topics
            
            # Get the strongest topic for comparison
            max_prob = filtered_topics[0][1] if filtered_topics else 0
            
            # Add selected topics to data
            for topic_id, topic_prob in selected_topics:
                multi_topic_data.append({
                    'filename': filename,
                    'topic_id': topic_id,
                    'topic_probability': round(topic_prob, 4),
                    'is_strongest': topic_prob == max_prob,
                    'selection_method': 'threshold' if topic_prob >= threshold else 'top_k'
                })
        
        self.merged_df = pd.DataFrame(multi_topic_data)
        
        if not self.merged_df.empty:
            # Add year data
            self.merged_df = self.merged_df.merge(
                self.years_df[['filename', 'max_year']], 
                on='filename', 
                how='left'
            )
            self.merged_df['year'] = self.merged_df['max_year'].astype(int)
            self.merged_df = self.merged_df.drop('max_year', axis=1)
            
            # Create aggregated data
            yr_topic_agg = self.merged_df.groupby(['year', 'topic_id']).size().reset_index(name='verdicts')
            yr_topic_agg.rename(columns={'topic_id': 'strongest_topic'}, inplace=True)
            yr_total = self.merged_df.groupby(['year']).size().reset_index(name='total_verdicts')
            self.yr_agg_df = yr_topic_agg.merge(yr_total, on='year')
            self.yr_agg_df['topic_percentage'] = (self.yr_agg_df['verdicts'] * 100.0 / self.yr_agg_df['total_verdicts']).round(2)
            
            print(f"✅ Created multi-topic data: {len(self.merged_df)} topic-document pairs")
            return True
        
        print("⚠️ No data remaining after filtering")
        return False

    def generate_visualizations(self, min_year: int = DEFAULT_MIN_YEAR):
        """Generate all visualization files with topic filtering"""
        print("📊 Generating visualizations...")

        if self.yr_agg_df.empty:
            print("⚠️ No aggregated data available for visualizations")
            return

        # Individual charts
        print("📈 Creating trends chart...")
        trend_fig = plot_topics_trend(
            self.yr_agg_df, self.topic_mappings, min_year,
            self.excluded_topics, self.included_topics
        )
        trend_path = self.paths['output_dir'] / 'topics_trend.html'
        trend_fig.write_html(trend_path)

        print("📊 Creating absolute trends chart...")
        absolute_fig = plot_absolute_topics_trend(
            self.yr_agg_df, self.topic_mappings, min_year,
            self.excluded_topics, self.included_topics
        )
        absolute_path = self.paths['output_dir'] / 'absolute_trends.html'
        absolute_fig.write_html(absolute_path)

        print("📚 Creating stacked distribution chart...")
        stacked_fig = plot_stacked_yearly_distribution(
            self.yr_agg_df, self.topic_mappings, min_year,
            self.excluded_topics, self.included_topics
        )
        stacked_path = self.paths['output_dir'] / 'stacked_distribution.html'
        stacked_fig.write_html(stacked_path)

        print("📊 Creating histogram...")
        hist_fig = plot_topics_histogram(
            self.yr_agg_df, self.topic_mappings,
            self.excluded_topics, self.included_topics
        )
        hist_path = self.paths['output_dir'] / 'topics_histogram.html'
        hist_fig.write_html(hist_path)

        print("🎨 Creating word clouds...")
        try:
            wordcloud_fig = create_wordcloud_grid(
                self.model, self.topic_mappings,
                self.excluded_topics, self.included_topics
            )
            if wordcloud_fig:
                wordcloud_path = self.paths['output_dir'] / 'topics_wordcloud.png'
                wordcloud_fig.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
                print(f"✅ Word cloud saved to {wordcloud_path}")
        except Exception as e:
            print(f"⚠️ Error creating word cloud: {e}")

        print("✅ All visualizations generated")

    def create_combined_html(self, min_year: int) -> str:
        """Create combined HTML file"""
        mode_name = "multi-topic" if 'topic_id' in self.merged_df.columns else "single-topic"
        
        return f"""
<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
    <meta charset="utf-8">
    <title>Supreme Court LDA Analysis - {mode_name}</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            margin: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            text-align: center;
            background-color: #2c3e50;
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .chart-container {{
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin: 30px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .chart-iframe {{
            width: 100%;
            height: 700px;
            border: none;
            border-radius: 8px;
        }}
        .info-box {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚖️ ניתוח נושאי בית המשפט העליון</h1>
        <h2>{mode_name.replace('-', ' ').title()}</h2>
        <p>ניתוח מקיף של פסקי דין לפי נושא ושנה</p>
    </div>
    
    <div class="info-box">
        <h3>📖 אודות הניתוח</h3>
        <p><strong>סוג ניתוח:</strong> {mode_name.replace('-', ' ')}</p>
        <p><strong>טווח נתונים:</strong> {min_year} - היום</p>
        <p><strong>סה"כ נושאים:</strong> {self.model.num_topics}</p>
        <p><strong>סה"כ מסמכים:</strong> {len(self.merged_df) if 'topic_id' not in self.merged_df.columns else self.merged_df['filename'].nunique()}</p>
    </div>

    <div class="chart-container">
        <h2>📈 מגמות יחסיות</h2>
        <iframe src="topics_trend.html" class="chart-iframe"></iframe>
    </div>

    <div class="chart-container">
        <h2>📊 מגמות מוחלטות</h2>
        <iframe src="absolute_trends.html" class="chart-iframe"></iframe>
    </div>

    <div class="chart-container">
        <h2>📚 התפלגות מוערמת</h2>
        <iframe src="stacked_distribution.html" class="chart-iframe"></iframe>
    </div>

    <div class="chart-container">
        <h2>📋 התפלגות כללית</h2>
        <iframe src="topics_histogram.html" class="chart-iframe"></iframe>
    </div>
</body>
</html>
        """

    def export_data_files(self):
        """Export data files"""
        print("💾 Exporting data files...")

        # Main processed data
        if 'topic_id' in self.merged_df.columns:
            # Multi-topic export - clean format
            export_data = self.merged_df[['filename', 'year', 'topic_id', 'topic_probability', 'is_strongest']].copy()
            export_data.rename(columns={
                'topic_id': 'strongest_topic',
                'topic_probability': 'strongest_topic_prob'
            }, inplace=True)
            
            # Add topic descriptions
            if not self.topic_mappings.empty:
                topic_title_map = {}
                topic_desc_map = {}
                for _, row in self.topic_mappings.iterrows():
                    topic_num = int(row['מספר נושא'])
                    topic_title_map[topic_num] = row['כותרת מוצעת']
                    topic_desc_map[topic_num] = row['רשימת המילים']
                
                export_data['topic_title'] = export_data['strongest_topic'].map(topic_title_map)
                export_data['topic_description'] = export_data['strongest_topic'].map(topic_desc_map)
            else:
                export_data['topic_title'] = export_data['strongest_topic'].map(lambda x: f"Topic {x}")
                export_data['topic_description'] = export_data['strongest_topic'].map(lambda x: f"Topic {x} words")
            
            # Fill any missing mappings
            export_data['topic_title'] = export_data['topic_title'].fillna(
                export_data['strongest_topic'].astype(str)
            )
            export_data['topic_description'] = export_data['topic_description'].fillna("No description available")
            
            processed_path = self.paths['output_dir'] / 'comprehensive_topic_data.csv'
            export_data.to_csv(processed_path, index=False, encoding='utf-8-sig')
        else:
            # Single-topic export - clean format
            export_data = self.merged_df[['filename', 'year', 'strongest_topic', 'strongest_topic_prob']].copy()
            
            # Add topic descriptions
            if not self.topic_mappings.empty:
                topic_title_map = {}
                topic_desc_map = {}
                for _, row in self.topic_mappings.iterrows():
                    topic_num = str(int(row['מספר נושא']))
                    topic_title_map[topic_num] = row['כותרת מוצעת']
                    topic_desc_map[topic_num] = row['רשימת המילים']
                
                export_data['topic_title'] = export_data['strongest_topic'].map(topic_title_map)
                export_data['topic_description'] = export_data['strongest_topic'].map(topic_desc_map)
            else:
                export_data['topic_title'] = export_data['strongest_topic'].map(lambda x: f"Topic {x}")
                export_data['topic_description'] = export_data['strongest_topic'].map(lambda x: f"Topic {x} words")
            
            # Fill any missing mappings
            export_data['topic_title'] = export_data['topic_title'].fillna(
                export_data['strongest_topic'].astype(str)
            )
            export_data['topic_description'] = export_data['topic_description'].fillna("No description available")

            processed_path = self.paths['output_dir'] / 'comprehensive_topic_data.csv'
            export_data.to_csv(processed_path, index=False, encoding='utf-8-sig')

        # Yearly aggregation
        aggregation_path = self.paths['output_dir'] / 'yearly_topic_aggregation.csv'
        self.yr_agg_df.to_csv(aggregation_path, index=False)

        # Topic mappings reference
        if not self.topic_mappings.empty:
            topics_reference_path = self.paths['output_dir'] / 'topic_mappings_reference.csv'
            reference_data = []
            for _, row in self.topic_mappings.sort_values('מספר נושא').iterrows():
                reference_data.append({
                    'topic_id': int(row['מספר נושא']),
                    'topic_title': row['כותרת מוצעת'],
                    'topic_words': row['רשימת המילים']
                })
            
            pd.DataFrame(reference_data).to_csv(topics_reference_path, index=False, encoding='utf-8-sig')

        print(f"✅ Data files exported to {self.paths['output_dir']}")
        return processed_path, aggregation_path

    def create_summary(self) -> Dict:
        """Create analysis summary"""
        if 'topic_id' in self.merged_df.columns:
            total_documents = self.merged_df['filename'].nunique()
            avg_confidence = self.merged_df['topic_probability'].mean().round(3)
            most_common = self.merged_df['topic_id'].value_counts().head(5).to_dict()
        else:
            total_documents = len(self.merged_df)
            avg_confidence = self.merged_df['strongest_topic_prob'].mean().round(3)
            most_common = self.merged_df['strongest_topic'].value_counts().head(5).to_dict()
        
        return {
            'total_documents': total_documents,
            'total_topics': self.model.num_topics,
            'year_range': f"{self.merged_df['year'].min()} - {self.merged_df['year'].max()}",
            'avg_topic_confidence': avg_confidence,
            'most_common_topics': most_common
        }

    def run_analysis(self, multi_topic_mode: bool = False, min_year: int = DEFAULT_MIN_YEAR, 
                    threshold: float = DEFAULT_MULTI_TOPIC_THRESHOLD, top_k: int = DEFAULT_TOP_K_TOPICS,
                    apply_length_filter: bool = True):
        """Run complete analysis with topic filtering, top-K support, and file length filtering"""
        mode_name = "multi-topic" if multi_topic_mode else "single-topic"
        mode_suffix = "multi_topic" if multi_topic_mode else "single_topic"
        
        print(f"\n🚀 Starting {mode_name} file generation...")
        print("=" * 60)
        
        # Setup paths for this mode
        self.setup_paths(mode_suffix)
        
        # Load data (with optional file length filtering)
        self.load_data(apply_length_filter)
        
        # Process data
        if multi_topic_mode:
            success = self.create_multi_topic_data(threshold, top_k)
            if not success:
                print("❌ Failed to create multi-topic data")
                return
        else:
            self.create_single_topic_data()
        
        # Generate visualizations
        vis_paths = self.generate_visualizations(min_year)
        
        # Export data files  
        data_paths = self.export_data_files()
        
        # Create summary
        summary = self.create_summary()
        
        print(f"\n📊 {mode_name.title()} Analysis Summary:")
        print("=" * 40)
        for key, value in summary.items():
            if key == 'most_common_topics':
                print("🏆 Most common topics:")
                for topic, count in value.items():
                    print(f"    Topic {topic}: {count} documents")
            else:
                print(f"📈 {key}: {value}")
        
        print(f"\n✅ {mode_name.title()} analysis completed!")
        print(f"📁 Files saved to: {self.paths['output_dir']}")


def main():
    """Main function with topic filtering and file length filtering support"""
    parser = argparse.ArgumentParser(description="LDA Analysis File Generator with Topic Filtering and File Length Filtering")
    parser.add_argument('--mode', choices=['single', 'multi', 'both'], default='both',
                       help='Analysis mode: single, multi, or both (default: both)')
    parser.add_argument('--exclude-topics', type=int, nargs='*', 
                       help=f'Topics to exclude from analysis (default: {DEFAULT_EXCLUDED_TOPICS})')
    parser.add_argument('--include-topics', type=int, nargs='*',
                       help='Topics to include in analysis (if specified, only these will be analyzed)')
    parser.add_argument('--min-year', type=int, default=DEFAULT_MIN_YEAR,
                       help=f'Minimum year for analysis (default: {DEFAULT_MIN_YEAR})')
    parser.add_argument('--threshold', type=float, default=DEFAULT_MULTI_TOPIC_THRESHOLD,
                       help=f'Probability threshold for multi-topic analysis (default: {DEFAULT_MULTI_TOPIC_THRESHOLD})')
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K_TOPICS,
                       help=f'Number of top topics to include per document (default: {DEFAULT_TOP_K_TOPICS})')
    parser.add_argument('--no-length-filter', action='store_true',
                       help='Disable file length filtering (include all files regardless of length)')
    
    args = parser.parse_args()
    
    # Handle topic filtering arguments
    excluded_topics = args.exclude_topics if args.exclude_topics is not None else DEFAULT_EXCLUDED_TOPICS
    included_topics = args.include_topics
    
    # Set up file length filtering
    apply_length_filter = not args.no_length_filter
    
    print("🔬 LDA Analysis File Generator")
    print("=" * 40)
    print(f"📊 Analysis mode: {args.mode}")
    print(f"📅 Minimum year: {args.min_year}")
    print(f"🎯 Multi-topic threshold: {args.threshold}")
    print(f"🔝 Top-K topics per document: {args.top_k}")
    print(f"📏 File length filtering: {'Enabled' if apply_length_filter else 'Disabled'}")

    try:
        if args.mode == 'both':
            # Run both analyses
            print("\n📊 Running single-topic analysis...")
            generator_single = LDAFileGenerator(excluded_topics, included_topics)
            generator_single.run_analysis(multi_topic_mode=False, min_year=args.min_year, 
                                        threshold=args.threshold, top_k=args.top_k, 
                                        apply_length_filter=apply_length_filter)
            
            print("\n🎯 Running multi-topic analysis...")
            generator_multi = LDAFileGenerator(excluded_topics, included_topics)
            generator_multi.run_analysis(multi_topic_mode=True, min_year=args.min_year, 
                                       threshold=args.threshold, top_k=args.top_k, 
                                       apply_length_filter=apply_length_filter)
            
            print("\n🎉 Both analyses completed!")
        else:
            generator = LDAFileGenerator(excluded_topics, included_topics)
            multi_mode = args.mode == 'multi'
            generator.run_analysis(multi_topic_mode=multi_mode, min_year=args.min_year, 
                                 threshold=args.threshold, top_k=args.top_k, 
                                 apply_length_filter=apply_length_filter)

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 