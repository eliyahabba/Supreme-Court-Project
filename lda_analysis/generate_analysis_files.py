#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDA Analysis - File Generation Script
====================================

Script for generating LDA analysis files and reports.
Uses shared visualization functions for consistency.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict

import pandas as pd
from gensim import models

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

    def __init__(self):
        self.setup_paths()
        
        self.model = None
        self.doc_mappings = pd.DataFrame()
        self.topic_mappings = pd.DataFrame()
        self.years_df = pd.DataFrame()
        self.merged_df = pd.DataFrame()
        self.yr_agg_df = pd.DataFrame()

    def setup_paths(self, mode_suffix: str = ""):
        """Setup all required paths"""
        self.project_root = Path.cwd()
        if self.project_root.name == 'lda_analysis':
            self.project_root = self.project_root.parent

        output_dir = self.project_root / 'data' / 'results' / 'lda'
        if mode_suffix:
            output_dir = output_dir / mode_suffix

        self.paths = {
            'project_root': self.project_root,
            'lda_model_dir': self.project_root / 'LDA Best Result' / '1693294471',
            'output_dir': output_dir,
            'years_data': self.project_root / 'data' / 'processed' / 'extracted_years.csv',
            'custom_topics': self.project_root / 'LDA Best Result' / '1693294471' / 'topics_with_claude.csv'
        }

        self.paths['output_dir'].mkdir(parents=True, exist_ok=True)

        print("📁 Paths configured:")
        for key, path in self.paths.items():
            exists = "✅" if path.exists() else "❌"  
            print(f"  {key}: {exists} {path}")

    def load_data(self):
        """Load all required data"""
        print("🔄 Loading LDA model and data...")

        # Load model
        model_path = self.paths['lda_model_dir'] / "model"
        self.model = models.ldamodel.LdaModel.load(str(model_path))

        # Load document mappings
        doc_mappings_path = self.paths['lda_model_dir'] / "docs_topics.csv"
        self.doc_mappings = pd.read_csv(doc_mappings_path)

        # Load topic mappings
        if self.paths['custom_topics'].exists():
            self.topic_mappings = pd.read_csv(self.paths['custom_topics'], encoding='utf-8')
        else:
            self.topic_mappings = pd.DataFrame()

        # Load years data
        self.years_df = pd.read_csv(self.paths['years_data'])

        print(f"✅ Loaded model with {self.model.num_topics} topics")
        print(f"✅ Loaded {len(self.doc_mappings)} document mappings")
        print(f"✅ Loaded {len(self.years_df)} year records")

    def create_single_topic_data(self):
        """Create single-topic analysis data"""
        self.merged_df = self.years_df.merge(self.doc_mappings, on='filename', how='inner')
        
        topic_cols = [col for col in self.merged_df.columns if col.isdigit()]
        
        if topic_cols:
            self.merged_df['strongest_topic'] = self.merged_df[topic_cols].idxmax(axis=1)
            self.merged_df['strongest_topic_prob'] = self.merged_df[topic_cols].max(axis=1)
        else:
            self.merged_df['strongest_topic'] = '0'
            self.merged_df['strongest_topic_prob'] = 1.0

        self.merged_df['year'] = self.merged_df['max_year'].astype(int)

        # Create aggregated data
        yr_topic_agg = self.merged_df.groupby(['year', 'strongest_topic']).size().reset_index(name='verdicts')
        yr_total = self.merged_df.groupby(['year']).size().reset_index(name='total_verdicts')
        self.yr_agg_df = yr_topic_agg.merge(yr_total, on='year')
        self.yr_agg_df['topic_percentage'] = (self.yr_agg_df['verdicts'] * 100.0 / self.yr_agg_df['total_verdicts']).round(2)

        print(f"✅ Created single-topic data: {len(self.merged_df)} documents")

    def create_multi_topic_data(self, threshold: float = 0.3):
        """Create multi-topic analysis data"""
        topic_cols = [col for col in self.doc_mappings.columns if col.isdigit()]
        
        if not topic_cols:
            print("⚠️ No topic columns found")
            return False
        
        multi_topic_data = []
        
        for _, row in self.doc_mappings.iterrows():
            filename = row['filename']
            topics_above_threshold = []
            max_prob = -1
            max_topic = None

            for topic_col in topic_cols:
                topic_prob = row[topic_col]
                if topic_prob >= threshold:
                    topics_above_threshold.append((int(topic_col), topic_prob))
                if topic_prob > max_prob:
                    max_prob = topic_prob
                    max_topic = int(topic_col)

            if topics_above_threshold:
                for topic_id, topic_prob in topics_above_threshold:
                    multi_topic_data.append({
                        'filename': filename,
                        'topic_id': topic_id,
                        'topic_probability': round(topic_prob, 4),
                        'is_strongest': topic_prob == row[topic_cols].max()
                    })
            else:
                multi_topic_data.append({
                    'filename': filename,
                    'topic_id': max_topic,
                    'topic_probability': round(max_prob, 4),
                    'is_strongest': True
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
        
        return False

    def generate_visualizations(self, min_year: int = 1948):
        """Generate all visualization files"""
        print("📊 Generating visualizations...")

        # Individual charts
        print("📈 Creating trends chart...")
        trend_fig = plot_topics_trend(self.yr_agg_df, self.topic_mappings, min_year)
        trend_path = self.paths['output_dir'] / 'topics_trend.html'
        trend_fig.write_html(trend_path)

        print("📊 Creating absolute trends chart...")
        absolute_fig = plot_absolute_topics_trend(self.yr_agg_df, self.topic_mappings, min_year)
        absolute_path = self.paths['output_dir'] / 'absolute_trends.html'
        absolute_fig.write_html(absolute_path)

        print("📚 Creating stacked distribution chart...")
        stacked_fig = plot_stacked_yearly_distribution(self.yr_agg_df, self.topic_mappings, min_year)
        stacked_path = self.paths['output_dir'] / 'stacked_distribution.html'
        stacked_fig.write_html(stacked_path)

        print("📋 Creating histogram...")
        hist_fig = plot_topics_histogram(self.yr_agg_df, self.topic_mappings)
        hist_path = self.paths['output_dir'] / 'topics_histogram.html'
        hist_fig.write_html(hist_path)

        # Word clouds
        print("🎨 Creating word clouds...")
        wordcloud_path = self.paths['output_dir'] / 'topics_wordcloud.png'
        wordcloud_fig = create_wordcloud_grid(self.model, self.topic_mappings)
        if wordcloud_fig:
            wordcloud_fig.savefig(wordcloud_path, dpi=300, bbox_inches='tight', facecolor='white')

        # Combined HTML
        combined_html = self.create_combined_html(min_year)
        combined_path = self.paths['output_dir'] / 'comprehensive_analysis.html'
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write(combined_html)

        print(f"✅ Visualizations saved to {self.paths['output_dir']}")
        return trend_path, absolute_path, stacked_path, hist_path, wordcloud_path, combined_path

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
            # Multi-topic export
            processed_path = self.paths['output_dir'] / 'multi_topic_data.csv'
            self.merged_df.to_csv(processed_path, index=False, encoding='utf-8-sig')
        else:
            # Single-topic export
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

    def run_analysis(self, multi_topic_mode: bool = False, min_year: int = 1948):
        """Run complete analysis"""
        mode_name = "multi-topic" if multi_topic_mode else "single-topic"
        mode_suffix = "multi_topic" if multi_topic_mode else "single_topic"
        
        print(f"\n🚀 Starting {mode_name} file generation...")
        print("=" * 60)
        
        # Setup paths for this mode
        self.setup_paths(mode_suffix)
        
        # Load data
        self.load_data()
        
        # Process data
        if multi_topic_mode:
            success = self.create_multi_topic_data()
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
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LDA Analysis File Generator")
    parser.add_argument('--mode', choices=['single', 'multi', 'both'], default='both',
                       help='Analysis mode')
    parser.add_argument('--min-year', type=int, default=1948,
                       help='Minimum year for analysis')
    args = parser.parse_args()
    
    print("🔬 LDA Analysis File Generator")
    print("=" * 40)

    try:
        if args.mode == 'both':
            # Run both analyses
            generator = LDAFileGenerator()
            generator.run_analysis(multi_topic_mode=False, min_year=args.min_year)
            
            generator = LDAFileGenerator()
            generator.run_analysis(multi_topic_mode=True, min_year=args.min_year)
            
            print("\n🎉 Both analyses completed!")
        else:
            generator = LDAFileGenerator()
            multi_mode = args.mode == 'multi'
            generator.run_analysis(multi_topic_mode=multi_mode, min_year=args.min_year)

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 