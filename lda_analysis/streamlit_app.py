#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App for LDA Analysis
=============================

Interactive web app for viewing LDA topic analysis results.
Updated to support topic filtering using shared constants.
"""

import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st
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


@st.cache_data
def load_lda_data(apply_length_filter=True):
    """Load LDA model and data with caching and optional file length filtering"""
    # Setup paths using constants
    project_root = get_project_root()
    paths = get_full_paths(project_root)
    
    # Load model
    model_path = paths['lda_model_dir'] / "model"
    model = models.ldamodel.LdaModel.load(str(model_path))
    
    # Load document mappings and years data  
    doc_mappings_path = paths['lda_model_dir'] / "docs_topics.csv"
    doc_mappings_original = pd.read_csv(doc_mappings_path)
    years_df_original = pd.read_csv(paths['years_data'])
    
    # Apply file length filtering if requested
    if apply_length_filter:
        try:
            doc_mappings, years_df, filtering_stats = filter_lda_input_data(
                doc_mappings_original, years_df_original
            )
            st.info(f"📏 File Length Filtering Applied: {filtering_stats['initial_count']:,} → {filtering_stats['final_count']:,} files ({filtering_stats['removed_count']:,} removed)")
        except Exception as e:
            st.warning(f"⚠️ File length filtering failed: {e}. Using all files.")
            doc_mappings = doc_mappings_original
            years_df = years_df_original
    else:
        doc_mappings = doc_mappings_original  
        years_df = years_df_original
    
    # Load topic mappings
    if paths['custom_topics'].exists():
        topic_mappings = pd.read_csv(paths['custom_topics'], encoding='utf-8')
    else:
        topic_mappings = pd.DataFrame()
    
    return model, doc_mappings, topic_mappings, years_df


def create_single_topic_data(doc_mappings, years_df, excluded_topics=None, included_topics=None):
    """Create single-topic analysis data with topic filtering"""
    # Merge with years
    merged_df = years_df.merge(doc_mappings, on='filename', how='inner')
    
    # Find topic columns
    topic_cols = [col for col in merged_df.columns if col.isdigit()]
    
    if topic_cols:
        merged_df['strongest_topic'] = merged_df[topic_cols].idxmax(axis=1)
        merged_df['strongest_topic_prob'] = merged_df[topic_cols].max(axis=1)
    else:
        merged_df['strongest_topic'] = '0'
        merged_df['strongest_topic_prob'] = 1.0
    
    merged_df['year'] = merged_df['max_year'].astype(int)
    
    # Apply topic filtering
    merged_df = filter_topics_from_data(merged_df, excluded_topics, included_topics, 'strongest_topic')
    
    if merged_df.empty:
        return merged_df, pd.DataFrame()
    
    # Create aggregated data
    yr_topic_agg = merged_df.groupby(['year', 'strongest_topic']).size().reset_index(name='verdicts')
    yr_total = merged_df.groupby(['year']).size().reset_index(name='total_verdicts')
    yr_agg_df = yr_topic_agg.merge(yr_total, on='year')
    yr_agg_df['topic_percentage'] = (yr_agg_df['verdicts'] * 100.0 / yr_agg_df['total_verdicts']).round(2)
    
    return merged_df, yr_agg_df


def create_multi_topic_data(doc_mappings, years_df, threshold=DEFAULT_MULTI_TOPIC_THRESHOLD,
                          top_k=DEFAULT_TOP_K_TOPICS, excluded_topics=None, included_topics=None):
    """Create multi-topic analysis data with topic filtering and top-K support"""
    topic_cols = [col for col in doc_mappings.columns if col.isdigit()]
    
    if not topic_cols:
        return pd.DataFrame(), pd.DataFrame()
    
    multi_topic_data = []
    
    for _, row in doc_mappings.iterrows():
        filename = row['filename']
        
        # Get all topic probabilities for this document (after filtering)
        filtered_topics = []
        for topic_col in topic_cols:
            topic_prob = row[topic_col]
            topic_id = int(topic_col)
            
            # Apply topic filtering
            if included_topics is not None and topic_id not in included_topics:
                continue
            if excluded_topics and topic_id in excluded_topics:
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
    
    multi_df = pd.DataFrame(multi_topic_data)
    
    if not multi_df.empty:
        # Add year data
        multi_df = multi_df.merge(
            years_df[['filename', 'max_year']], 
            on='filename', 
            how='left'
        )
        multi_df['year'] = multi_df['max_year'].astype(int)
        multi_df = multi_df.drop('max_year', axis=1)
        
        # Create aggregated data for multi-topic
        yr_topic_agg = multi_df.groupby(['year', 'topic_id']).size().reset_index(name='verdicts')
        yr_topic_agg.rename(columns={'topic_id': 'strongest_topic'}, inplace=True)
        yr_total = multi_df.groupby(['year']).size().reset_index(name='total_verdicts')
        yr_agg_df = yr_topic_agg.merge(yr_total, on='year')
        yr_agg_df['topic_percentage'] = (yr_agg_df['verdicts'] * 100.0 / yr_agg_df['total_verdicts']).round(2)
        
        return multi_df, yr_agg_df
    
    return pd.DataFrame(), pd.DataFrame()


def get_available_topics(model, topic_mappings):
    """Get list of available topics for filtering interface"""
    topics = []
    for i in range(model.num_topics):
        if not topic_mappings.empty:
            # Try to find Hebrew name
            topic_name = None
            for _, row in topic_mappings.iterrows():
                if int(row['מספר נושא']) == i:
                    topic_name = row['כותרת מוצעת']
                    break
            
            if topic_name:
                topics.append(f"{i}: {topic_name}")
            else:
                topics.append(f"Topic {i}")
        else:
            topics.append(f"Topic {i}")
    
    return topics


def prepare_export_data(merged_df, topic_mappings, analysis_mode):
    """Prepare clean data for export with topic titles and descriptions"""
    if analysis_mode == "Single Topic":
        # For single topic mode
        export_data = merged_df[['filename', 'year', 'strongest_topic', 'strongest_topic_prob']].copy()
        
        # Add topic descriptions
        if not topic_mappings.empty:
            # Create mapping from the custom topics file
            topic_title_map = {}
            topic_desc_map = {}
            for _, row in topic_mappings.iterrows():
                topic_num = str(int(row['מספר נושא']))
                topic_title_map[topic_num] = row['כותרת מוצעת']
                topic_desc_map[topic_num] = row['רשימת המילים']
            
            # Add topic descriptions to export data
            export_data['topic_title'] = export_data['strongest_topic'].map(topic_title_map)
            export_data['topic_description'] = export_data['strongest_topic'].map(topic_desc_map)
        else:
            # Fallback to generic titles
            export_data['topic_title'] = export_data['strongest_topic'].map(lambda x: f"Topic {x}")
            export_data['topic_description'] = export_data['strongest_topic'].map(lambda x: f"Topic {x} words")
        
        # Fill any missing mappings
        export_data['topic_title'] = export_data['topic_title'].fillna(
            export_data['strongest_topic'].astype(str)
        )
        export_data['topic_description'] = export_data['topic_description'].fillna("No description available")
        
        return export_data
    
    else:
        # For multi topic mode
        export_data = merged_df[['filename', 'year', 'topic_id', 'topic_probability', 'is_strongest']].copy()
        export_data.rename(columns={
            'topic_id': 'strongest_topic',
            'topic_probability': 'strongest_topic_prob'
        }, inplace=True)
        
        # Add topic descriptions
        if not topic_mappings.empty:
            # Create mapping from the custom topics file
            topic_title_map = {}
            topic_desc_map = {}
            for _, row in topic_mappings.iterrows():
                topic_num = int(row['מספר נושא'])
                topic_title_map[topic_num] = row['כותרת מוצעת']
                topic_desc_map[topic_num] = row['רשימת המילים']
            
            # Add topic descriptions to export data
            export_data['topic_title'] = export_data['strongest_topic'].map(topic_title_map)
            export_data['topic_description'] = export_data['strongest_topic'].map(topic_desc_map)
        else:
            # Fallback to generic titles
            export_data['topic_title'] = export_data['strongest_topic'].map(lambda x: f"Topic {x}")
            export_data['topic_description'] = export_data['strongest_topic'].map(lambda x: f"Topic {x} words")
        
        # Fill any missing mappings
        export_data['topic_title'] = export_data['topic_title'].fillna(
            export_data['strongest_topic'].astype(str)
        )
        export_data['topic_description'] = export_data['topic_description'].fillna("No description available")
        
        return export_data


def main():
    st.set_page_config(
        page_title="Supreme Court Topic Analysis",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚖️ Supreme Court Topic Analysis")
    st.markdown("### LDA Analysis of Court Verdicts with Topic and Length Filtering")
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Settings")
    
    # File length filtering option
    st.sidebar.subheader("📏 File Length Filtering")
    apply_length_filter = st.sidebar.checkbox(
        "Apply file length filtering",
        value=True,
        help="Filter out files that are too short based on pre-calculated statistics"
    )
    
    if apply_length_filter:
        # Show file length constants if available
        try:
            check_file_length_constants()
            st.sidebar.success("✅ File length filtering enabled")
        except NameError:
            st.sidebar.warning("⚠️ File length constants not set. Run analyze_file_lengths.py first.")
            apply_length_filter = False
    else:
        st.sidebar.info("📄 Using all files regardless of length")
    
    # Load data
    with st.spinner("Loading data..."):
        try:
            model, doc_mappings, topic_mappings, years_df = load_lda_data(apply_length_filter)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()
    
    # Topic filtering section
    st.sidebar.subheader("🎯 Topic Filtering")
    
    # Get available topics
    available_topics = get_available_topics(model, topic_mappings)
    topic_numbers = list(range(model.num_topics))
    
    # Filtering mode selection
    filter_mode = st.sidebar.radio(
        "Filtering Mode:",
        ["Exclude Topics", "Include Only", "No Filtering"],
        help="Choose how to filter topics"
    )
    
    excluded_topics = None
    included_topics = None
    
    if filter_mode == "Exclude Topics":
        # Topic exclusion interface
        default_excluded = [i for i in DEFAULT_EXCLUDED_TOPICS if i < model.num_topics]
        excluded_indices = st.sidebar.multiselect(
            "Topics to Exclude:",
            options=topic_numbers,
            default=default_excluded,
            format_func=lambda x: available_topics[x],
            help=f"Default excludes topics: {DEFAULT_EXCLUDED_TOPICS}"
        )
        excluded_topics = excluded_indices if excluded_indices else None
        
    elif filter_mode == "Include Only":
        # Topic inclusion interface  
        included_indices = st.sidebar.multiselect(
            "Topics to Include:",
            options=topic_numbers,
            format_func=lambda x: available_topics[x],
            help="If selected, only these topics will be analyzed"
        )
        included_topics = included_indices if included_indices else None
    
    # Analysis mode selection
    analysis_mode = st.sidebar.selectbox(
        "Select Analysis Mode:",
        ["Single Topic", "Multi Topic"],
        help="Single Topic: Each document assigned to strongest topic only. Multi Topic: Documents can belong to multiple topics"
    )
    
    # Year range selection
    min_year = st.sidebar.slider(
        "Minimum Year:",
        min_value=int(years_df['max_year'].min()),
        max_value=int(years_df['max_year'].max()),
        value=DEFAULT_MIN_YEAR,
        step=1
    )
    
    # Multi-topic parameters (only for multi-topic mode)
    if analysis_mode == "Multi Topic":
        threshold = st.sidebar.slider(
            "Topic Inclusion Threshold:",
            min_value=0.05,
            max_value=0.8,
            value=DEFAULT_MULTI_TOPIC_THRESHOLD,
            step=0.05,
            help="Minimum probability threshold for including a topic in a document"
        )
        
        top_k = st.sidebar.slider(
            "Top-K Topics per Document:",
            min_value=1,
            max_value=10,
            value=DEFAULT_TOP_K_TOPICS,
            step=1,
            help="Number of top topics to include per document (regardless of threshold)"
        )
        
        st.sidebar.info(f"📋 Documents will include topics above {threshold:.2f} threshold OR top-{top_k} topics (whichever gives more)")
    else:
        threshold = DEFAULT_MULTI_TOPIC_THRESHOLD
        top_k = DEFAULT_TOP_K_TOPICS
    
    # Display filtering info
    if excluded_topics:
        st.sidebar.info(f"🚫 Excluding topics: {excluded_topics}")
    elif included_topics:
        st.sidebar.info(f"✅ Including only topics: {included_topics}")
    else:
        st.sidebar.info("📊 No topic filtering applied")
    
    # Process data based on selected mode
    with st.spinner("Processing data..."):
        if analysis_mode == "Single Topic":
            merged_df, yr_agg_df = create_single_topic_data(doc_mappings, years_df, excluded_topics, included_topics)
            if merged_df.empty:
                st.error("No data found after applying topic filters")
                st.stop()
            mode_info = f"Single Topic - {len(merged_df)} documents"
        else:
            merged_df, yr_agg_df = create_multi_topic_data(doc_mappings, years_df, threshold, top_k, excluded_topics, included_topics)
            if merged_df.empty:
                st.error("No data found for multi-topic analysis with the selected parameters and filters")
                st.stop()
            
            # Enhanced info for multi-topic
            unique_docs = merged_df['filename'].nunique()
            avg_topics_per_doc = len(merged_df) / unique_docs if unique_docs > 0 else 0
            threshold_count = len(merged_df[merged_df.get('selection_method', '') == 'threshold']) if 'selection_method' in merged_df.columns else 0
            top_k_count = len(merged_df[merged_df.get('selection_method', '') == 'top_k']) if 'selection_method' in merged_df.columns else 0
            
            mode_info = f"Multi Topic - {len(merged_df)} topic-document pairs ({unique_docs} unique documents, avg {avg_topics_per_doc:.1f} topics/doc)"
            if 'selection_method' in merged_df.columns:
                mode_info += f"\n📊 Selected by threshold: {threshold_count}, by top-K: {top_k_count}"
    
    # Display basic info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 General Info")
    st.sidebar.markdown(f"**Model:** {model.num_topics} topics")
    st.sidebar.markdown(f"**Data:** {mode_info}")
    st.sidebar.markdown(f"**Year Range:** {merged_df['year'].min()} - {merged_df['year'].max()}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Relative Trends", 
        "📊 Absolute Trends", 
        "📚 Stacked Distribution", 
        "📋 Histogram",
        "🎨 Word Clouds"
    ])
    
    with tab1:
        st.header("📈 Relative Topic Trends (Percentages)")
        st.markdown("This chart shows how the relative importance of each topic changes over time as a percentage")
        
        with st.spinner("Creating relative trends chart..."):
            fig1 = plot_topics_trend(yr_agg_df, topic_mappings, min_year, excluded_topics, included_topics)
            st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        st.header("📊 Absolute Topic Trends (Document Counts)")
        st.markdown("This chart shows the actual number of documents for each topic over time")
        
        with st.spinner("Creating absolute trends chart..."):
            fig2 = plot_absolute_topics_trend(yr_agg_df, topic_mappings, min_year, excluded_topics, included_topics)
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.header("📚 Stacked Topic Distribution by Year")
        st.markdown("This chart shows the total document volume per year and the distribution of topics within each year")
        
        with st.spinner("Creating stacked distribution chart..."):
            fig3 = plot_stacked_yearly_distribution(yr_agg_df, topic_mappings, min_year, excluded_topics, included_topics)
            st.plotly_chart(fig3, use_container_width=True)
    
    with tab4:
        st.header("📋 Overall Topic Distribution")
        st.markdown("This chart shows the total number of documents for each topic across all years")
        
        with st.spinner("Creating histogram..."):
            fig4 = plot_topics_histogram(yr_agg_df, topic_mappings, excluded_topics, included_topics)
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab5:
        st.header("🎨 Topic Word Clouds")
        st.markdown("Word clouds showing the most prominent words in each topic")
        
        with st.spinner("Creating word clouds..."):
            try:
                fig5 = create_wordcloud_grid(model, topic_mappings, excluded_topics, included_topics)
                st.pyplot(fig5, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating word clouds: {e}")
    
    # Data export section
    st.markdown("---")
    st.header("💾 Data Download")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Download Processed Data"):
            # Prepare clean export data
            export_data = prepare_export_data(merged_df, topic_mappings, analysis_mode)
            csv = export_data.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"lda_analysis_{analysis_mode.replace(' ', '_')}.csv",
                mime="text/csv",
                help="Clean processed data with topics, years, and topic descriptions"
            )
    
    with col2:
        if st.button("📊 Download Aggregated Data"):
            csv = yr_agg_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"lda_yearly_agg_{analysis_mode.replace(' ', '_')}.csv",
                mime="text/csv",
                help="Aggregated data by year and topic"
            )


if __name__ == "__main__":
    main() 