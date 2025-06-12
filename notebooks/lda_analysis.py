#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDA Model Analysis Script - Updated Version with File Length Filtering
=====================================================================

Detailed analysis of LDA model selected based on its coherence score.
The script is organized with functions and relative paths for improved convenience and maintenance.

Features:
- ✅ Relative paths for portability
- ✅ Organized code with functions
- ✅ Error handling
- ✅ Compatibility with new year data format
- ✅ Modern Python methods with type hints
- ✅ Interactive visualizations with Plotly
- ✅ Hebrew text support in word clouds
- ✅ File length filtering support (NEW!)

File Length Filtering:
- Automatically filters out files that are too short for meaningful analysis
- Uses pre-calculated statistics from sample files to determine minimum word threshold
- Can be disabled with --no-length-filter flag

Usage Examples:
    # Run with file length filtering (default)
    python lda_analysis.py --mode both
    
    # Run without file length filtering
    python lda_analysis.py --mode both --no-length-filter
    
    # Run only single-topic analysis with filtering
    python lda_analysis.py --mode single

Prerequisites for file length filtering:
    1. Run: cd ../lda_analysis && python analyze_file_lengths.py
    2. This will analyze sample files and set up filtering constants

Author: Updated analysis script
Date: 2024
"""

import logging
import math
import warnings
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
# Import visualization libraries
import plotly.express as px
import plotly.graph_objects as go
from gensim import models
# Import Gensim libraries
from gensim.models import LdaModel
from wordcloud import WordCloud

# Add lda_analysis module to path for importing shared components
sys.path.append(str(Path(__file__).parent.parent / 'lda_analysis'))

try:
            from utils import check_file_length_constants
    from file_length_filter import filter_lda_input_data, print_filtering_summary
    print("✅ Imported file length filtering modules")
except ImportError as e:
    print(f"⚠️ Could not import file length filtering: {e}")
    print("File length filtering will be disabled.")
    filter_lda_input_data = None
    print_filtering_summary = None
    check_file_length_constants = None

# Disable warnings for cleaner appearance
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LDAAnalyzer:
    """Class for LDA model analysis"""

    def __init__(self, single_topic_mode: bool = None, apply_length_filter: bool = True):
        """
        Initialize LDA analyzer.
        
        Args:
            single_topic_mode: If None, run both analyses. If True, only strongest topic. If False, multi-topic.
            apply_length_filter: Whether to apply file length filtering
        """
        self.single_topic_mode = single_topic_mode
        self.apply_length_filter = apply_length_filter
        self.setup_paths()
        
        # Initialize data containers
        self.model = None
        self.doc_mappings = pd.DataFrame()
        self.topic_mappings = pd.DataFrame()
        self.years_df = pd.DataFrame()
        self.merged_df = pd.DataFrame()
        self.yr_agg_df = pd.DataFrame()

    def setup_paths(self, mode_suffix: str = ""):
        """Setup all required paths using pathlib for better path handling"""
        # Get project root
        self.project_root = Path.cwd()
        if self.project_root.name == 'notebooks':
            self.project_root = self.project_root.parent

        # Base output directory with mode suffix
        output_dir = self.project_root / 'data' / 'results' / 'lda'
        if mode_suffix:
            output_dir = output_dir / mode_suffix

        self.paths = {
            'project_root': self.project_root,
            'data_dir': self.project_root / 'data',
            'lda_model_dir': self.project_root / 'LDA Best Result' / '1693294471' ,
            'output_dir': output_dir,
            'years_data': self.project_root / 'data' / 'processed' / 'extracted_years.csv',
            'custom_topics': self.project_root / 'LDA Best Result' / '1693294471' / 'topics_with_claude.csv'
        }

        # Create output directory if it doesn't exist
        self.paths['output_dir'].mkdir(parents=True, exist_ok=True)

        print("📁 Paths configured:")
        for key, path in self.paths.items():
            exists = "✅" if path.exists() else "❌"
            print(f"  {key}: {exists} {path}")

    def load_lda_model(self) -> Tuple[LdaModel, pd.DataFrame, pd.DataFrame]:
        """
        Load LDA model and related data files with optional file length filtering.
        
        Returns:
            Tuple of (model, doc_mappings, topic_mappings)
        """
        try:
            print(f"🔄 Loading LDA model from {self.paths['lda_model_dir']}...")

            # Load the model
            model_path = self.paths['lda_model_dir'] / "model"
            self.model = models.ldamodel.LdaModel.load(str(model_path))

            # Load document mappings (original)
            doc_mappings_path = self.paths['lda_model_dir'] / "docs_topics.csv"
            doc_mappings_original = pd.read_csv(doc_mappings_path)

            # Load topic mappings - try custom topics first, then fallback to Excel
            if self.paths['custom_topics'].exists():
                print(f"✅ Using custom topics file: {self.paths['custom_topics']}")
                self.topic_mappings = pd.read_csv(self.paths['custom_topics'], encoding='utf-8')
                print(f"✅ Loaded {len(self.topic_mappings)} custom topic mappings")
            else:
                # Fallback to original Excel file if custom topics don't exist
                topics_path = self.paths['lda_model_dir'] / "topics.xlsx"
                if topics_path.exists():
                    self.topic_mappings = pd.read_excel(topics_path)
                    print(f"✅ Loaded {len(self.topic_mappings)} topic mappings from Excel")
                else:
                    print("⚠️ No topic mappings file found")
                    self.topic_mappings = pd.DataFrame()

            print(f"✅ Loaded LDA model with {self.model.num_topics} topics")
            print(f"✅ Loaded {len(doc_mappings_original)} document mappings")

            # Apply file length filtering if enabled and available
            if self.apply_length_filter and filter_lda_input_data is not None:
                print("\n📏 Applying file length filtering...")
                try:
                    if check_file_length_constants is not None:
                        check_file_length_constants()
                    
                    # Load years data for filtering
                    years_df_original = pd.read_csv(self.paths['years_data'])
                    
                    # Apply filtering
                    self.doc_mappings, years_filtered, filtering_stats = filter_lda_input_data(
                        doc_mappings_original, years_df_original
                    )
                    
                    if print_filtering_summary is not None:
                        print_filtering_summary(filtering_stats)
                        
                except Exception as e:
                    print(f"⚠️ File length filtering failed: {e}")
                    print("Continuing with all files...")
                    self.doc_mappings = doc_mappings_original
            else:
                if not self.apply_length_filter:
                    print("📄 File length filtering disabled")
                else:
                    print("⚠️ File length filtering modules not available")
                self.doc_mappings = doc_mappings_original

            print(f"✅ Final document mappings: {len(self.doc_mappings)}")

            return self.model, self.doc_mappings, self.topic_mappings

        except Exception as e:
            print(f"❌ Error loading LDA model: {e}")
            raise

    def load_years_data(self) -> pd.DataFrame:
        """
        Load year extraction data.
        
        Returns:
            DataFrame with year data
        """
        try:
            print(f"🔄 Loading year data from {self.paths['years_data']}...")
            self.years_df = pd.read_csv(self.paths['years_data'])
            print(f"✅ Loaded year data for {len(self.years_df)} files")
            print(f"📊 Year range: {self.years_df['max_year'].min()} - {self.years_df['max_year'].max()}")
            return self.years_df
        except Exception as e:
            print(f"❌ Error loading year data: {e}")
            raise

    def merge_data(self, multi_topic_mode: bool = False) -> pd.DataFrame:
        """
        Merge year data with topic mappings.
        
        Args:
            multi_topic_mode: If True, create multi-topic version of merged data
        
        Returns:
            Merged DataFrame with all relevant information
        """
        try:
            print("🔄 Merging data...")
            print(f"Year data shape: {self.years_df.shape}")
            print(f"Document mappings shape: {self.doc_mappings.shape}")

            if multi_topic_mode:
                # Create multi-topic merged data
                print("🎯 Creating multi-topic merged data...")
                multi_topic_df = self.create_multi_topic_data(threshold=0.3)
                
                if not multi_topic_df.empty:
                    self.merged_df = multi_topic_df
                    print(f"✅ Multi-topic data merged: {len(self.merged_df)} topic-document pairs")
                    print(f"📊 Average topics per document: {len(self.merged_df) / len(self.doc_mappings):.2f}")
                else:
                    print("❌ Failed to create multi-topic data, falling back to single topic")
                    multi_topic_mode = False

            if not multi_topic_mode:
                # Original single-topic logic
                self.merged_df = self.years_df.merge(self.doc_mappings, on='filename', how='inner')
                print(f"Shape after merge: {self.merged_df.shape}")

                # Find topic columns (should be numeric columns)
                topic_cols = [col for col in self.merged_df.columns if col.isdigit()]

                if not topic_cols:
                    # If no numeric columns, search for topic columns by pattern
                    topic_cols = [col for col in self.merged_df.columns
                                  if 'topic' in col.lower() or col.startswith('Topic') or col.startswith('0')]

                print(f"Found {len(topic_cols)} topic columns: {topic_cols[:5]}...")

                if topic_cols:
                    # Find strongest topic for each document
                    self.merged_df['strongest_topic'] = self.merged_df[topic_cols].idxmax(axis=1)
                    self.merged_df['strongest_topic_prob'] = self.merged_df[topic_cols].max(axis=1)
                else:
                    print("⚠️ No topic columns found in data, using default values")
                    self.merged_df['strongest_topic'] = '0'
                    self.merged_df['strongest_topic_prob'] = 1.0

                # Ensure year is integer
                self.merged_df['year'] = self.merged_df['max_year'].astype(int)

                # Add row index for counting
                self.merged_df = self.merged_df.reset_index(drop=True)

                print(f"✅ Single-topic data merged: {len(self.merged_df)} documents")
                print(f"Topic columns found: {len(topic_cols)}")

            return self.merged_df

        except Exception as e:
            print(f"❌ Error merging data: {e}")
            raise

    def create_aggregated_data(self, multi_topic_mode: bool = False) -> pd.DataFrame:
        """
        Create aggregated data for analysis.
        
        Args:
            multi_topic_mode: If True, handle multi-topic aggregation
        
        Returns:
            Aggregated DataFrame with topic percentages by year
        """
        try:
            print("🔄 Creating aggregated data...")

            if multi_topic_mode:
                # For multi-topic: count documents by topic (simple counting)
                print("🎯 Creating multi-topic aggregation...")
                
                # Simple counting - how many documents contain each topic above threshold
                yr_topic_agg = self.merged_df.groupby(['year', 'topic_id']).size().reset_index(name='verdicts')
                yr_topic_agg.rename(columns={'topic_id': 'strongest_topic'}, inplace=True)

                # Total document appearances by year (can be more than unique documents)
                yr_total = self.merged_df.groupby(['year']).size().reset_index(name='total_verdicts')
            else:
                # Original single-topic logic
                print("🎯 Creating single-topic aggregation...")
                
                # Count documents by year and topic
                yr_topic_agg = self.merged_df.groupby(['year', 'strongest_topic']).size().reset_index(name='verdicts')

                # Count total documents by year
                yr_total = self.merged_df.groupby(['year']).size().reset_index(name='total_verdicts')

            # Merge to get percentages
            self.yr_agg_df = yr_topic_agg.merge(yr_total, on='year')
            self.yr_agg_df['topic_percentage'] = (
                        self.yr_agg_df['verdicts'] * 100.0 / self.yr_agg_df['total_verdicts']).round(2)

            print(f"✅ Created aggregated data: {len(self.yr_agg_df)} year-topic combinations")
            print(f"Year range: {self.yr_agg_df['year'].min()} - {self.yr_agg_df['year'].max()}")

            return self.yr_agg_df

        except Exception as e:
            print(f"❌ Error creating aggregated data: {e}")
            raise

    def generate_distinct_colors(self, n_colors: int) -> list:
        """
        Generate n distinct colors for better visualization of many topics
        
        Args:
            n_colors: Number of colors needed
            
        Returns:
            List of color strings
        """
        import colorsys
        
        # Generate colors using HSV color space for maximum distinction
        colors = []
        for i in range(n_colors):
            # Distribute hues evenly around the color wheel
            hue = i / n_colors
            # Use high saturation and value for vivid colors
            saturation = 0.8 + (i % 3) * 0.05  # Slight variation in saturation
            value = 0.7 + (i % 4) * 0.075       # Slight variation in brightness
            
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            # Convert to hex color
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255), 
                int(rgb[1] * 255), 
                int(rgb[2] * 255)
            )
            colors.append(hex_color)
        
        return colors

    def plot_topics_trend(self, min_year: int = 1948) -> go.Figure:
        """Plot topic trends over years"""
        filtered_data = self.yr_agg_df[self.yr_agg_df['year'] >= min_year].copy()

        # Add Hebrew topic names if available
        if not self.topic_mappings.empty:
            # Create topic name mapping
            topic_name_map = {}
            for _, row in self.topic_mappings.iterrows():
                topic_id = str(int(row['מספר נושא']))
                topic_name = row['כותרת מוצעת']
                topic_name_map[topic_id] = f"{topic_id}: {topic_name}"
            
            # Apply mapping to data
            filtered_data['topic_display'] = filtered_data['strongest_topic'].astype(str).map(topic_name_map)
            # Fill any missing mappings with the original topic number
            filtered_data['topic_display'] = filtered_data['topic_display'].fillna(
                filtered_data['strongest_topic'].astype(str)
            )
            color_column = 'topic_display'
        else:
            # Fallback to topic numbers
            filtered_data['topic_display'] = filtered_data['strongest_topic'].astype(str)
            color_column = 'topic_display'

        # Sort topics by number for consistent legend order
        unique_topics = sorted(filtered_data['strongest_topic'].unique(), key=lambda x: int(x))
        n_topics = len(unique_topics)
        distinct_colors = self.generate_distinct_colors(n_topics)
        
        # Create color mapping based on sorted topic order
        color_mapping = {}
        for i, topic in enumerate(unique_topics):
            topic_display = filtered_data[filtered_data['strongest_topic'] == topic][color_column].iloc[0]
            color_mapping[topic_display] = distinct_colors[i % len(distinct_colors)]

        fig = px.line(
            filtered_data,
            x='year',
            y='topic_percentage',
            color=color_column,
            title=f'📈 Topic Trends Over Years (from {min_year})',
            labels={
                'topic_percentage': 'Topic Percentage (%)',
                'year': 'Year',
                color_column: 'Topic'
            },
            width=1200,
            height=700,
            color_discrete_map=color_mapping,
            category_orders={color_column: [filtered_data[filtered_data['strongest_topic'] == topic][color_column].iloc[0] for topic in unique_topics]}
        )

        fig.update_layout(
            title_x=0.5,
            legend_title_text='Topic',
            hovermode='closest',  # Show only the specific point being hovered
            template='plotly_white',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": [True] * len(unique_topics)}],
                            label="Show All",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [False] * len(unique_topics)}],
                            label="Hide All",
                            method="restyle"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                ),
            ]
        )
        
        # Improve line visibility and hover info
        fig.update_traces(
            line=dict(width=2.5),  # Make lines thicker
            marker=dict(size=6),   # Make markers slightly larger
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Year: %{x}<br>' +
                         'Percentage: %{y:.1f}%<br>' +
                         '<extra></extra>'  # Remove the default box
        )

        return fig

    def plot_topics_histogram(self) -> go.Figure:
        """Plot topic histogram for all years"""
        topic_totals = self.yr_agg_df.groupby('strongest_topic')['verdicts'].sum().reset_index()
        topic_totals['strongest_topic'] = topic_totals['strongest_topic'].astype(str)
        
        # Add Hebrew topic names if available
        if not self.topic_mappings.empty:
            # Create topic name mapping
            topic_name_map = {}
            for _, row in self.topic_mappings.iterrows():
                topic_id = str(int(row['מספר נושא']))
                topic_name = row['כותרת מוצעת']
                topic_name_map[topic_id] = f"{topic_id}: {topic_name}"
            
            # Apply mapping to data
            topic_totals['topic_display'] = topic_totals['strongest_topic'].map(topic_name_map)
            # Fill any missing mappings with the original topic number
            topic_totals['topic_display'] = topic_totals['topic_display'].fillna(
                topic_totals['strongest_topic']
            )
            x_column = 'topic_display'
        else:
            # Fallback to topic numbers
            topic_totals['topic_display'] = topic_totals['strongest_topic']
            x_column = 'topic_display'
        
        # Sort by topic number (not by verdicts count) for consistent order
        topic_totals['topic_num'] = topic_totals['strongest_topic'].astype(int)
        topic_totals = topic_totals.sort_values('topic_num')
        
        # Generate distinct colors for all topics in sorted order
        n_topics = len(topic_totals)
        distinct_colors = self.generate_distinct_colors(n_topics)
        
        # Create color mapping based on sorted topic order (not frequency)
        color_mapping = {}
        for i, row in topic_totals.iterrows():
            topic_display = row[x_column]
            topic_index = int(row['topic_num'])  # Use topic number for color consistency
            color_mapping[topic_display] = distinct_colors[topic_index % len(distinct_colors)]

        fig = px.bar(
            topic_totals,
            x=x_column,
            y='verdicts',
            title='📊 Topic Distribution for All Years',
            labels={
                'verdicts': 'Number of Verdicts',
                x_column: 'Topic'
            },
            width=1200,  
            height=600,
            color=x_column,
            color_discrete_map=color_mapping,
            category_orders={x_column: topic_totals[x_column].tolist()}
        )

        fig.update_layout(
            title_x=0.5,
            template='plotly_white',
            xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
            showlegend=False  # Hide legend for histogram since x-axis shows the topics
        )
        
        # Improve hover info
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>' +
                         'Documents: %{y}<br>' +
                         '<extra></extra>'  # Remove the default box
        )

        return fig

    def create_wordcloud_grid(self, output_path: Path = None) -> plt.Figure:
        """Create a grid of word clouds for all topics"""
        try:
            print("🎨 Creating word clouds...")

            # Hebrew fonts for macOS and other systems
            hebrew_fonts = [
                '/System/Library/Fonts/SFHebrew.ttf',  # macOS San Francisco Hebrew
                '/System/Library/Fonts/Arial Unicode MS.ttf',  # macOS fallback
                'C:/Windows/Fonts/arial.ttf',  # Windows
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'  # Linux
            ]

            font_path = None
            for font in hebrew_fonts:
                if Path(font).exists():
                    font_path = font
                    break

            # WordCloud settings
            wordcloud_kwargs = {
                'background_color': 'white',
                'width': 400,
                'height': 300,
                'max_words': 20,
                'colormap': 'tab20',
                'relative_scaling': 0.5,
                'random_state': 42
            }

            if font_path:
                wordcloud_kwargs['font_path'] = font_path
                print(f"✅ Using Hebrew font: {font_path}")
            else:
                print("⚠️ Hebrew font not found, using default")

            cloud = WordCloud(**wordcloud_kwargs)

            # Get topics
            n_topics = self.model.num_topics
            topics = self.model.show_topics(formatted=False, num_topics=n_topics, num_words=20)

            # Calculate grid size
            sqrt_topics = int(math.ceil(n_topics ** 0.5))

            # Create figure
            fig, axes = plt.subplots(sqrt_topics, sqrt_topics, figsize=(20, 20))
            if sqrt_topics == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for i, ax in enumerate(axes):
                if i >= n_topics:
                    ax.axis('off')
                    continue

                # Get topic words - use them as is, don't reverse Hebrew text
                topic_words = dict(topics[i][1])

                # Create word cloud
                if topic_words:
                    try:
                        cloud.generate_from_frequencies(topic_words)
                        ax.imshow(cloud, interpolation='bilinear')
                    except Exception as e:
                        print(f"⚠️ Cannot create word cloud for topic {i}: {e}")
                        ax.text(0.5, 0.5, f'Topic {i}\n(Error creating word cloud)',
                                ha='center', va='center', transform=ax.transAxes)

                # Set title with Hebrew topic name if available
                if not self.topic_mappings.empty:
                    # Find Hebrew topic name
                    topic_name = None
                    for _, row in self.topic_mappings.iterrows():
                        if int(row['מספר נושא']) == i:
                            topic_name = row['כותרת מוצעת']
                            break
                    
                    if topic_name:
                        ax.set_title(f'{i}: {topic_name}', fontsize=12, fontweight='bold')
                    else:
                        ax.set_title(f'Topic {i}', fontsize=12, fontweight='bold')
                else:
                    ax.set_title(f'Topic {i}', fontsize=12, fontweight='bold')
                
                ax.axis('off')

            plt.suptitle(' 🎨 LDA Topic Word Clouds', fontsize=20, fontweight='bold')
            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"✅ Word cloud saved to {output_path}")

            return fig

        except Exception as e:
            print(f"❌ Error creating word cloud: {e}")
            return None

    def display_topics_overview(self):
        """Display topics overview"""
        print("=" * 60)
        print("🎯 Topics Overview")
        print("=" * 60)
        topic_num = self.model.num_topics
        print(f"📊 Number of topics: {topic_num}")

        print("\n📋 Topic mappings:")
        if not self.topic_mappings.empty:
            print("✅ Using custom Hebrew topic descriptions")
            print("\n🔤 Topics and their Hebrew titles:")
            
            # Sort topic mappings by topic number for consistent display order
            sorted_mappings = self.topic_mappings.sort_values('מספר נושא')
            
            for _, row in sorted_mappings.iterrows():
                topic_id = int(row['מספר נושא'])
                topic_title = row['כותרת מוצעת']
                topic_words = row['רשימת המילים'][:100] + "..." if len(row['רשימת המילים']) > 100 else row['רשימת המילים']
                print(f"\n📌 Topic {topic_id}: {topic_title}")
                print(f"   Words: {topic_words}")
        else:
            print("⚠️ Using model topic words (no custom mappings)")
            print("\n🔤 Topics and their prominent words:")
            topics_info = self.model.show_topics(num_topics=min(topic_num, 10), num_words=8, formatted=True)
            for topic_id, words in topics_info:
                print(f"\n📌 Topic {topic_id}: {words}")

    def calculate_yearly_stats(self) -> pd.DataFrame:
        """Calculate yearly statistics"""
        try:
            if 'topic_id' in self.merged_df.columns:
                # Multi-topic mode
                yearly_stats = self.merged_df.groupby('year').agg({
                    'filename': 'nunique',  # Count unique documents
                    'topic_probability': ['mean', 'std']
                }).round(3)
                
                yearly_stats.columns = ['total_verdicts', 'avg_topic_confidence', 'std_topic_confidence']
            else:
                # Single-topic mode  
                yearly_stats = self.merged_df.groupby('year').agg({
                    'filename': 'count',
                    'strongest_topic_prob': ['mean', 'std']
                }).round(3)
                
                yearly_stats.columns = ['total_verdicts', 'avg_topic_confidence', 'std_topic_confidence']
            
            yearly_stats = yearly_stats.reset_index()
            return yearly_stats
            
        except Exception as e:
            print(f"⚠️ Error calculating yearly stats: {e}")
            # Fallback to simple count
            yearly_stats = self.merged_df.groupby('year').size().reset_index(name='total_verdicts')
            return yearly_stats

    def create_summary(self) -> Dict:
        """Create summary report"""
        try:
            if 'topic_id' in self.merged_df.columns:
                # Multi-topic mode
                total_documents = self.merged_df['filename'].nunique()
                avg_confidence = self.merged_df['topic_probability'].mean().round(3)
                most_common = self.merged_df['topic_id'].value_counts().head(5).to_dict()
            else:
                # Single-topic mode
                total_documents = len(self.merged_df)
                avg_confidence = self.merged_df['strongest_topic_prob'].mean().round(3)
                most_common = self.merged_df['strongest_topic'].value_counts().head(5).to_dict()
            
            summary = {
                'total_documents': total_documents,
                'total_topics': self.model.num_topics,
                'year_range': f"{self.merged_df['year'].min()} - {self.merged_df['year'].max()}",
                'avg_topic_confidence': avg_confidence,
                'most_common_topics': most_common
            }
            return summary
            
        except Exception as e:
            print(f"⚠️ Error creating summary: {e}")
            return {'error': str(e)}

    def create_multi_topic_data(self, threshold: float = 0.3) -> pd.DataFrame:
        """
        Create dataset with multiple topics per document above threshold.
        
        Args:
            threshold: Minimum probability threshold for including a topic (default: 0.3 = 30%)
            
        Returns:
            DataFrame with multiple rows per document for significant topics
        """
        try:
            print(f"🔄 Creating multi-topic data with threshold {threshold}...")
            
            # Get topic columns (should be numeric columns)
            topic_cols = [col for col in self.doc_mappings.columns if col.isdigit()]
            
            if not topic_cols:
                print("⚠️ No topic columns found")
                return pd.DataFrame()
            
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
                    # No topic above threshold: add the strongest topic anyway
                    multi_topic_data.append({
                        'filename': filename,
                        'topic_id': max_topic,
                        'topic_probability': round(max_prob, 4),
                        'is_strongest': True
                    })
            
            multi_df = pd.DataFrame(multi_topic_data)
            
            if not multi_df.empty:
                # Add year data
                if hasattr(self, 'years_df'):
                    multi_df = multi_df.merge(
                        self.years_df[['filename', 'max_year']], 
                        on='filename', 
                        how='left'
                    )
                    multi_df['year'] = multi_df['max_year'].astype(int)
                    multi_df = multi_df.drop('max_year', axis=1)
                
                # Add topic descriptions
                if not self.topic_mappings.empty:
                    topic_title_map = {}
                    topic_desc_map = {}
                    for _, topic_row in self.topic_mappings.iterrows():
                        topic_num = int(topic_row['מספר נושא'])
                        topic_title_map[topic_num] = topic_row['כותרת מוצעת']
                        topic_desc_map[topic_num] = topic_row['רשימת המילים']
                    
                    multi_df['topic_title'] = multi_df['topic_id'].map(topic_title_map)
                    multi_df['topic_description'] = multi_df['topic_id'].map(topic_desc_map)
                
                print(f"✅ Created multi-topic data: {len(multi_df)} topic-document pairs")
                print(f"📊 Average topics per document: {len(multi_df) / len(self.doc_mappings):.2f}")
                
                return multi_df
            else:
                print("⚠️ No topic-document pairs above threshold")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error creating multi-topic data: {e}")
            return pd.DataFrame()

    def export_data(self):
        """Export processed data for single-topic mode only"""
        print("💾 Exporting single-topic data files...")

        # 1. Processed data with topic descriptions (strongest topic only)
        export_data = self.merged_df[['filename', 'year', 'strongest_topic', 'strongest_topic_prob']].copy()
        
        # Add topic descriptions from custom topics file
        if not self.topic_mappings.empty:
            print("✅ Using custom Hebrew topic descriptions")
            # Create mapping from the custom topics file
            topic_title_map = {}
            topic_desc_map = {}
            for _, row in self.topic_mappings.iterrows():
                topic_num = str(int(row['מספר נושא']))  # Convert to string, ensuring integer first
                topic_title_map[topic_num] = row['כותרת מוצעת']
                topic_desc_map[topic_num] = row['רשימת המילים']
            
            # Add topic descriptions to export data
            export_data['topic_title'] = export_data['strongest_topic'].map(topic_title_map)
            export_data['topic_description'] = export_data['strongest_topic'].map(topic_desc_map)
        else:
            # Fallback: use topic words from model
            print("⚠️ No topic mappings found, using model topic words")
            topic_words_map = {}
            topics_info = self.model.show_topics(num_topics=self.model.num_topics, num_words=5, formatted=True)
            for topic_id, words in topics_info:
                topic_words_map[str(topic_id)] = words
            export_data['topic_title'] = export_data['strongest_topic'].map(lambda x: f"Topic {x}")
            export_data['topic_description'] = export_data['strongest_topic'].map(topic_words_map)

        # Save comprehensive dataset (strongest topic only)
        comprehensive_path = self.paths['output_dir'] / 'comprehensive_topic_data.csv'
        export_data.to_csv(comprehensive_path, index=False, encoding='utf-8-sig')
        print(f"✅ Comprehensive dataset saved to {comprehensive_path}")

        # 2. Topic mappings file for reference
        topics_reference_path = self.paths['output_dir'] / 'topic_mappings_reference.csv'
        
        if not self.topic_mappings.empty:
            # Create a clean reference file from custom topics
            reference_data = []
            # Sort topic mappings by topic number for consistent export order
            sorted_mappings = self.topic_mappings.sort_values('מספר נושא')
            for _, row in sorted_mappings.iterrows():
                reference_data.append({
                    'topic_id': int(row['מספר נושא']),
                    'topic_title': row['כותרת מוצעת'],
                    'topic_words': row['רשימת המילים']
                })
            
            topic_reference_df = pd.DataFrame(reference_data)
            topic_reference_df.to_csv(topics_reference_path, index=False, encoding='utf-8-sig')
        else:
            # Fallback: create topic mappings from model
            topics_info = self.model.show_topics(num_topics=self.model.num_topics, num_words=10, formatted=False)
            topic_reference_data = []
            
            for topic_id, word_probs in topics_info:
                # Get top words for this topic
                top_words = [word for word, prob in word_probs[:10]]
                words_str = ', '.join(top_words)
                
                topic_reference_data.append({
                    'topic_id': topic_id,
                    'topic_title': f"Topic {topic_id}",
                    'topic_words': words_str
                })
            
            topic_reference_df = pd.DataFrame(topic_reference_data)
            topic_reference_df.to_csv(topics_reference_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ Topic mappings reference saved to {topics_reference_path}")

        # 3. Original processed data (for backwards compatibility)
        original_export_data = self.merged_df[['filename', 'year', 'strongest_topic', 'strongest_topic_prob']]
        processed_path = self.paths['output_dir'] / 'processed_topic_data.csv'
        original_export_data.to_csv(processed_path, index=False)

        # 4. Yearly aggregation
        aggregation_path = self.paths['output_dir'] / 'yearly_topic_aggregation.csv'
        self.yr_agg_df.to_csv(aggregation_path, index=False)

        # 5. Yearly statistics
        yearly_stats = self.calculate_yearly_stats()
        stats_path = self.paths['output_dir'] / 'yearly_statistics.csv'
        yearly_stats.to_csv(stats_path, index=False)

        return processed_path, aggregation_path, stats_path, comprehensive_path, topics_reference_path

    def run_single_analysis(self, multi_topic_mode: bool = False):
        """
        Run analysis for a single mode (either single-topic or multi-topic).
        
        Args:
            multi_topic_mode: If True, run multi-topic analysis
        """
        mode_name = "multi-topic" if multi_topic_mode else "single-topic"
        mode_suffix = "multi_topic" if multi_topic_mode else "single_topic"
        
        print(f"\n🚀 Starting {mode_name} analysis...")
        print("=" * 70)
        
        # Update paths for this mode
        self.setup_paths(mode_suffix)

        # 1. Load data (same for both modes)
        print("\n1️⃣ Loading data...")
        self.load_lda_model()
        self.load_years_data()

        # Display basic information
        print("\n📊 Data overview:")
        print(f"🎯 Model topics: {self.model.num_topics}")
        print(f"📄 Documents with topics: {len(self.doc_mappings)}")
        print(f"📅 Documents with years: {len(self.years_df)}")
        print(f"📈 Year range: {self.years_df['max_year'].min()} - {self.years_df['max_year'].max()}")

        # 2. Merge and process (different logic for each mode)
        print(f"\n2️⃣ Merging and processing data ({mode_name})...")
        self.merge_data(multi_topic_mode=multi_topic_mode)
        self.create_aggregated_data(multi_topic_mode=multi_topic_mode)

        if multi_topic_mode:
            print(f"\n✅ Processed multi-topic data:")
            print(f"📄 Topic-document pairs: {len(self.merged_df)}")
            print(f"📊 Year-topic combinations: {len(self.yr_agg_df)}")
            print(f"📅 Years with data: {len(self.merged_df['year'].unique())}")
            print(f"🎯 Unique topics: {len(self.merged_df['topic_id'].unique())}")
        else:
            print(f"\n✅ Processed single-topic data:")
            print(f"📄 Merged documents: {len(self.merged_df)}")
            print(f"📊 Year-topic combinations: {len(self.yr_agg_df)}")
            print(f"📅 Years with data: {len(self.merged_df['year'].unique())}")
            print(f"🎯 Unique topics: {len(self.merged_df['strongest_topic'].unique())}")

        # 3. Topics overview
        print("\n3️⃣ Topics overview...")
        self.display_topics_overview()

        # 4. Visualizations
        print(f"\n4️⃣ Creating visualizations ({mode_name})...")

        # Create combined HTML with all visualizations
        print("📊 Creating comprehensive visualization dashboard...")
        combined_html = self.create_combined_visualization_html(min_year=1948)
        combined_path = self.paths['output_dir'] / f'comprehensive_analysis_{mode_suffix}.html'
        
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write(combined_html)
        print(f"✅ Comprehensive dashboard saved to {combined_path}")

        # Also create individual charts for backwards compatibility
        print("📈 Creating individual chart files...")
        
        # Topic trends (relative)
        trend_fig = self.plot_topics_trend(min_year=1948)
        trend_path = self.paths['output_dir'] / f'topics_trend_{mode_suffix}.html'
        trend_fig.write_html(trend_path)
        
        # Topic histogram
        hist_fig = self.plot_topics_histogram()
        hist_path = self.paths['output_dir'] / f'topics_histogram_{mode_suffix}.html'
        hist_fig.write_html(hist_path)

        # Absolute trends
        absolute_fig = self.plot_absolute_topics_trend(min_year=1948)
        absolute_path = self.paths['output_dir'] / f'absolute_trends_{mode_suffix}.html'
        absolute_fig.write_html(absolute_path)

        # Stacked distribution
        stacked_fig = self.plot_stacked_yearly_distribution(min_year=1948)
        stacked_path = self.paths['output_dir'] / f'stacked_distribution_{mode_suffix}.html'
        stacked_fig.write_html(stacked_path)
        
        print(f"✅ Individual charts saved")

        # Word clouds
        print("🎨 Creating word clouds...")
        wordcloud_path = self.paths['output_dir'] / f'topics_wordcloud_{mode_suffix}.png'
        wordcloud_fig = self.create_wordcloud_grid(wordcloud_path)

        # 5. Yearly statistics
        print(f"\n5️⃣ Calculating yearly statistics ({mode_name})...")
        yearly_stats = self.calculate_yearly_stats()
        print("📈 Yearly statistics (last 10 years):")
        print(yearly_stats.tail(10).to_string())

        # 6. Summary and export
        print(f"\n6️⃣ Summary and export ({mode_name})...")
        summary = self.create_summary()

        print("\n" + "=" * 60)
        print(f"📊 {mode_name.title()} Analysis Summary")
        print("=" * 60)
        for key, value in summary.items():
            if key == 'most_common_topics':
                print(f"🏆 Most common topics:")
                for topic, count in value.items():
                    print(f"    Topic {topic}: {count} documents")
            else:
                print(f"📈 {key}: {value}")

        # Export data
        if multi_topic_mode:
            # For multi-topic, export the multi-topic dataset as main
            multi_topic_path = self.paths['output_dir'] / 'multi_topic_data.csv'
            self.merged_df.to_csv(multi_topic_path, index=False, encoding='utf-8-sig')
            
            aggregation_path = self.paths['output_dir'] / 'yearly_topic_aggregation.csv'
            self.yr_agg_df.to_csv(aggregation_path, index=False)
            
            yearly_stats = self.calculate_yearly_stats()
            stats_path = self.paths['output_dir'] / 'yearly_statistics.csv'
            yearly_stats.to_csv(stats_path, index=False)
            
            print(f"✅ Multi-topic dataset: {multi_topic_path}")
            print(f"✅ Yearly aggregation: {aggregation_path}")
            print(f"✅ Yearly statistics: {stats_path}")
        else:
            # For single-topic, use the regular export
            processed_path, aggregation_path, stats_path, comprehensive_path, topics_reference_path = self.export_data()
            
            print(f"✅ Processed data: {processed_path}")
            print(f"✅ Comprehensive dataset: {comprehensive_path}")
            print(f"✅ Yearly aggregation: {aggregation_path}")
            print(f"✅ Yearly statistics: {stats_path}")
            print(f"✅ Topic mappings reference: {topics_reference_path}")

        print(f"\n🎨 Visualizations ({mode_name}):")
        print(f"✅ Comprehensive Dashboard: {combined_path}")
        print(f"✅ Relative trends: {trend_path}")
        print(f"✅ Absolute trends: {absolute_path}")
        print(f"✅ Stacked distribution: {stacked_path}")
        print(f"✅ Topic histogram: {hist_path}")
        if wordcloud_path.exists():
            print(f"✅ Word clouds: {wordcloud_path}")

        print(f"\n🎉 {mode_name.title()} analysis completed!")
        print(f"📁 Results saved to: {self.paths['output_dir']}")

    def run_full_analysis(self):
        """Run full analysis - either single mode or both modes"""
        if self.single_topic_mode is None:
            # Run both analyses
            print("🔬 Running BOTH single-topic and multi-topic analyses")
            print("=" * 70)
            
            # Run single-topic analysis
            self.run_single_analysis(multi_topic_mode=False)
            
            # Reset data containers and run multi-topic analysis
            self.merged_df = pd.DataFrame()
            self.yr_agg_df = pd.DataFrame()
            self.run_single_analysis(multi_topic_mode=True)
            
            print("\n" + "=" * 70)
            print("🎉 BOTH ANALYSES COMPLETED!")
            print("📁 Results saved in separate directories:")
            print(f"   Single-topic: {self.project_root / 'data' / 'results' / 'lda' / 'single_topic'}")
            print(f"   Multi-topic:  {self.project_root / 'data' / 'results' / 'lda' / 'multi_topic'}")
            print("🔍 Compare results between directories to see the differences!")
            
        elif self.single_topic_mode:
            # Run only single-topic analysis
            self.run_single_analysis(multi_topic_mode=False)
        else:
            # Run only multi-topic analysis
            self.run_single_analysis(multi_topic_mode=True)

    def plot_absolute_topics_trend(self, min_year: int = 1948) -> go.Figure:
        """Plot absolute topic trends over years (number of documents, not percentages)"""
        filtered_data = self.yr_agg_df[self.yr_agg_df['year'] >= min_year].copy()

        # Add Hebrew topic names if available
        if not self.topic_mappings.empty:
            # Create topic name mapping
            topic_name_map = {}
            for _, row in self.topic_mappings.iterrows():
                topic_id = str(int(row['מספר נושא']))
                topic_name = row['כותרת מוצעת']
                topic_name_map[topic_id] = f"{topic_id}: {topic_name}"
            
            # Apply mapping to data
            filtered_data['topic_display'] = filtered_data['strongest_topic'].astype(str).map(topic_name_map)
            # Fill any missing mappings with the original topic number
            filtered_data['topic_display'] = filtered_data['topic_display'].fillna(
                filtered_data['strongest_topic'].astype(str)
            )
            color_column = 'topic_display'
        else:
            # Fallback to topic numbers
            filtered_data['topic_display'] = filtered_data['strongest_topic'].astype(str)
            color_column = 'topic_display'

        # Sort topics by number for consistent legend order
        unique_topics = sorted(filtered_data['strongest_topic'].unique(), key=lambda x: int(x))
        n_topics = len(unique_topics)
        distinct_colors = self.generate_distinct_colors(n_topics)
        
        # Create color mapping based on sorted topic order
        color_mapping = {}
        for i, topic in enumerate(unique_topics):
            topic_display = filtered_data[filtered_data['strongest_topic'] == topic][color_column].iloc[0]
            color_mapping[topic_display] = distinct_colors[i % len(distinct_colors)]

        fig = px.line(
            filtered_data,
            x='year',
            y='verdicts',  # Use absolute numbers instead of percentages
            color=color_column,
            title=f'📊 Absolute Topic Trends Over Years (from {min_year})',
            labels={
                'verdicts': 'Number of Documents',
                'year': 'Year',
                color_column: 'Topic'
            },
            width=1400,
            height=800,
            color_discrete_map=color_mapping,
            category_orders={color_column: [filtered_data[filtered_data['strongest_topic'] == topic][color_column].iloc[0] for topic in unique_topics]}
        )

        fig.update_layout(
            title_x=0.5,
            legend_title_text='Topic',
            hovermode='closest',
            template='plotly_white',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": [True] * len(unique_topics)}],
                            label="Show All",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [False] * len(unique_topics)}],
                            label="Hide All",
                            method="restyle"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                ),
            ]
        )
        
        # Improve line visibility and hover info
        fig.update_traces(
            line=dict(width=3),  # Make lines thicker for better visibility
            marker=dict(size=8),   # Make markers larger
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Year: %{x}<br>' +
                         'Documents: %{y}<br>' +
                         '<extra></extra>'
        )

        return fig

    def plot_stacked_yearly_distribution(self, min_year: int = 1948) -> go.Figure:
        """Plot stacked bar chart showing topic distribution by year"""
        filtered_data = self.yr_agg_df[self.yr_agg_df['year'] >= min_year].copy()

        # Add Hebrew topic names if available
        if not self.topic_mappings.empty:
            # Create topic name mapping
            topic_name_map = {}
            for _, row in self.topic_mappings.iterrows():
                topic_id = str(int(row['מספר נושא']))
                topic_name = row['כותרת מוצעת']
                topic_name_map[topic_id] = f"{topic_id}: {topic_name}"
            
            # Apply mapping to data
            filtered_data['topic_display'] = filtered_data['strongest_topic'].astype(str).map(topic_name_map)
            # Fill any missing mappings with the original topic number
            filtered_data['topic_display'] = filtered_data['topic_display'].fillna(
                filtered_data['strongest_topic'].astype(str)
            )
        else:
            # Fallback to topic numbers
            filtered_data['topic_display'] = filtered_data['strongest_topic'].astype(str)

        # Sort topics by number for consistent legend order
        unique_topics = sorted(filtered_data['strongest_topic'].unique(), key=lambda x: int(x))
        n_topics = len(unique_topics)
        distinct_colors = self.generate_distinct_colors(n_topics)
        
        # Create color mapping based on sorted topic order
        color_mapping = {}
        for i, topic in enumerate(unique_topics):
            topic_display = filtered_data[filtered_data['strongest_topic'] == topic]['topic_display'].iloc[0]
            color_mapping[topic_display] = distinct_colors[i % len(distinct_colors)]

        # Create figure manually using go.Figure for better control
        fig = go.Figure()

        # Get all unique years and sort them
        years = sorted(filtered_data['year'].unique())
        
        # Track which topics have been added to legend to avoid duplicates
        topics_in_legend = set()
        
        # Process each year separately to create stacked bars
        for year in years:
            year_data = filtered_data[filtered_data['year'] == year].copy()
            
            # Sort topics by frequency within this specific year (descending)
            year_data = year_data.sort_values('verdicts', ascending=False)
            
            # Calculate cumulative positions for stacking
            cumulative_base = 0
            
            for _, row in year_data.iterrows():
                topic_display = row['topic_display']
                verdicts = row['verdicts']
                color = color_mapping[topic_display]
                
                # Determine if this topic should show in legend
                show_in_legend = topic_display not in topics_in_legend
                if show_in_legend:
                    topics_in_legend.add(topic_display)
                
                # Get topic number for legend ranking
                topic_num = int(row['strongest_topic'])
                
                # Add this topic segment to the stack
                fig.add_trace(go.Bar(
                    x=[year],
                    y=[verdicts],
                    base=[cumulative_base],
                    name=topic_display,
                    marker_color=color,
                    hovertemplate=f'<b>{topic_display}</b><br>' +
                                 f'Year: {year}<br>' +
                                 f'Documents: {verdicts}<br>' +
                                 f'Rank in year: {len(year_data) - list(year_data.index).index(row.name)}<br>' +
                                 '<extra></extra>',
                    showlegend=show_in_legend,  # Show in legend only the first time we see this topic
                    legendgroup=topic_display,  # Group legend items by topic
                    legendrank=topic_num  # Sort legend by topic number
                ))
                
                cumulative_base += verdicts

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'📊 Stacked Topic Distribution by Year (from {min_year})<br><sub>Each year sorted by topic frequency in that year</sub>',
                x=0.5,
                font=dict(size=16)
            ),
            barmode='stack',
            template='plotly_white',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                title="Topics (by number)",
                traceorder="normal"
            ),
            xaxis=dict(
                title='Year',
                tickmode='linear',
                dtick=2,  # Show every 2 years for better readability
                type='linear'
            ),
            yaxis=dict(
                title='Number of Documents'
            ),
            width=1400,
            height=800
        )

        return fig

    def create_combined_visualization_html(self, min_year: int = 1948) -> str:
        """Create a combined HTML file with all three visualizations"""
        mode_suffix = "multi_topic" if 'topic_id' in self.merged_df.columns else "single_topic"
        
        print("📊 Creating combined visualization with all charts...")
        
        # Create all three figures
        print("📈 Creating relative trends chart...")
        relative_fig = self.plot_topics_trend(min_year=min_year)
        
        print("📊 Creating absolute trends chart...")
        absolute_fig = self.plot_absolute_topics_trend(min_year=min_year)
        
        print("📚 Creating stacked distribution chart...")
        stacked_fig = self.plot_stacked_yearly_distribution(min_year=min_year)
        
        print("🎨 Creating histogram...")
        hist_fig = self.plot_topics_histogram()
        
        # Save individual charts first
        relative_path = f'topics_trend_{mode_suffix}.html'
        absolute_path = f'absolute_trends_{mode_suffix}.html'
        stacked_path = f'stacked_distribution_{mode_suffix}.html'
        hist_path = f'topics_histogram_{mode_suffix}.html'
        
        relative_fig.write_html(self.paths['output_dir'] / relative_path)
        absolute_fig.write_html(self.paths['output_dir'] / absolute_path)
        stacked_fig.write_html(self.paths['output_dir'] / stacked_path)
        hist_fig.write_html(self.paths['output_dir'] / hist_path)
        
        # Create combined HTML with links to individual files
        combined_html = f"""
<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
    <meta charset="utf-8">
    <title>Supreme Court LDA Analysis - {mode_suffix.replace('_', ' ').title()}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
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
            text-align: center;
        }}
        .chart-title {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
        }}
        .chart-description {{
            color: #555;
            margin-bottom: 20px;
            font-style: italic;
            font-size: 1.1em;
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
        .button-link {{
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            margin: 10px;
            font-weight: bold;
        }}
        .button-link:hover {{
            background-color: #2980b9;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Supreme Court Topic Analysis</h1>
        <h2>{mode_suffix.replace('_', ' ').title()} Mode</h2>
        <p>Comprehensive analysis of Supreme Court verdicts by topic and year</p>
    </div>
    
    <div class="info-box">
        <h3>📖 About This Analysis</h3>
        <p><strong>Analysis Type:</strong> {mode_suffix.replace('_', ' ').title()}</p>
        <p><strong>Data Range:</strong> {min_year} - Present</p>
        <p><strong>Total Topics:</strong> {self.model.num_topics}</p>
        <p><strong>Total Documents:</strong> {len(self.merged_df) if 'topic_id' not in self.merged_df.columns else self.merged_df['filename'].nunique()}</p>
        
        <h4>🔗 Individual Chart Links</h4>
        <a href="{relative_path}" class="button-link" target="_blank">📈 Relative Trends</a>
        <a href="{absolute_path}" class="button-link" target="_blank">📊 Absolute Trends</a>
        <a href="{stacked_path}" class="button-link" target="_blank">📚 Stacked Distribution</a>
        <a href="{hist_path}" class="button-link" target="_blank">📋 Topic Histogram</a>
    </div>

    <div class="chart-container">
        <div class="chart-title">📈 Relative Topic Trends (Percentages)</div>
        <div class="chart-description">
            Shows how each topic's relative importance changes over time as a percentage of all topics in each year
        </div>
        <iframe src="{relative_path}" class="chart-iframe"></iframe>
    </div>

    <div class="chart-container">
        <div class="chart-title">📊 Absolute Topic Trends (Document Counts)</div>
        <div class="chart-description">
            Shows the actual number of documents for each topic over time - reveals absolute growth or decline
        </div>
        <iframe src="{absolute_path}" class="chart-iframe"></iframe>
    </div>

    <div class="chart-container">
        <div class="chart-title">📚 Stacked Yearly Distribution</div>
        <div class="chart-description">
            Shows the total volume of documents per year and the distribution of topics within each year
        </div>
        <iframe src="{stacked_path}" class="chart-iframe"></iframe>
    </div>

    <div class="chart-container">
        <div class="chart-title">📋 Overall Topic Distribution</div>
        <div class="chart-description">
            Shows the total number of documents for each topic across all years
        </div>
        <iframe src="{hist_path}" class="chart-iframe"></iframe>
    </div>

    <div class="info-box">
        <h3>🔍 How to Interpret the Charts</h3>
        <ul>
            <li><strong>Relative Trends:</strong> Use this to see which topics become more or less dominant over time</li>
            <li><strong>Absolute Trends:</strong> Use this to see which topics are actually growing or declining in absolute numbers</li>
            <li><strong>Stacked Distribution:</strong> Use this to see overall document volume trends and dominant topics per year</li>
            <li><strong>Overall Distribution:</strong> Use this to understand which topics are most common overall</li>
        </ul>
        
        <p><strong>💡 Tip:</strong> If a chart doesn't display properly in the iframe, click the individual chart links above to open them in separate windows.</p>
    </div>
</body>
</html>
        """
        
        return combined_html


def main():
    """Main function with file length filtering support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LDA Model Analysis with File Length Filtering")
    parser.add_argument('--mode', choices=['single', 'multi', 'both'], default='both',
                       help='Analysis mode: single (strongest topic only), multi (all significant topics), both (run both analyses)')
    parser.add_argument('--no-length-filter', action='store_true',
                       help='Disable file length filtering (include all files regardless of length)')
    args = parser.parse_args()
    
    # Convert mode to single_topic_mode parameter
    if args.mode == 'single':
        single_topic_mode = True
        print("🎯 Running SINGLE-TOPIC analysis only")
    elif args.mode == 'multi':
        single_topic_mode = False
        print("🎯 Running MULTI-TOPIC analysis only")
    else:  # both
        single_topic_mode = None
        print("🎯 Running BOTH analyses (default)")
    
    # Set up file length filtering
    apply_length_filter = not args.no_length_filter
    
    print("🔬 LDA Model Analysis")
    print("=" * 50)
    print(f"📊 Analysis mode: {args.mode}")
    print(f"📏 File length filtering: {'Enabled' if apply_length_filter else 'Disabled'}")

    try:
        analyzer = LDAAnalyzer(single_topic_mode=single_topic_mode, apply_length_filter=apply_length_filter)
        analyzer.run_full_analysis()

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
