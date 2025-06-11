#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App for LDA Analysis
=============================

Interactive web app for viewing LDA topic analysis results.
"""

import warnings
from pathlib import Path

import pandas as pd
import streamlit as st
from gensim import models

from lda_visualizations import (
    create_wordcloud_grid,
    plot_absolute_topics_trend,
    plot_stacked_yearly_distribution,
    plot_topics_histogram,
    plot_topics_trend,
)

warnings.filterwarnings('ignore')


@st.cache_data
def load_lda_data():
    """Load LDA model and data with caching"""
    # Setup paths
    project_root = Path.cwd()
    if project_root.name == 'lda_analysis':
        project_root = project_root.parent

    paths = {
        'lda_model_dir': project_root / 'LDA Best Result' / '1693294471',
        'years_data': project_root / 'data' / 'processed' / 'extracted_years.csv',
        'custom_topics': project_root / 'LDA Best Result' / '1693294471' / 'topics_with_claude.csv'
    }
    
    # Load model
    model_path = paths['lda_model_dir'] / "model"
    model = models.ldamodel.LdaModel.load(str(model_path))
    
    # Load document mappings
    doc_mappings_path = paths['lda_model_dir'] / "docs_topics.csv"
    doc_mappings = pd.read_csv(doc_mappings_path)
    
    # Load topic mappings
    if paths['custom_topics'].exists():
        topic_mappings = pd.read_csv(paths['custom_topics'], encoding='utf-8')
    else:
        topic_mappings = pd.DataFrame()
    
    # Load years data
    years_df = pd.read_csv(paths['years_data'])
    
    return model, doc_mappings, topic_mappings, years_df


def create_single_topic_data(doc_mappings, years_df):
    """Create single-topic analysis data"""
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
    
    # Create aggregated data
    yr_topic_agg = merged_df.groupby(['year', 'strongest_topic']).size().reset_index(name='verdicts')
    yr_total = merged_df.groupby(['year']).size().reset_index(name='total_verdicts')
    yr_agg_df = yr_topic_agg.merge(yr_total, on='year')
    yr_agg_df['topic_percentage'] = (yr_agg_df['verdicts'] * 100.0 / yr_agg_df['total_verdicts']).round(2)
    
    return merged_df, yr_agg_df


def create_multi_topic_data(doc_mappings, years_df, threshold=0.3):
    """Create multi-topic analysis data"""
    topic_cols = [col for col in doc_mappings.columns if col.isdigit()]
    
    if not topic_cols:
        return pd.DataFrame(), pd.DataFrame()
    
    multi_topic_data = []
    
    for _, row in doc_mappings.iterrows():
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


def main():
    st.set_page_config(
        page_title="Supreme Court Topic Analysis",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚖️ Supreme Court Topic Analysis")
    st.markdown("### LDA Analysis of Court Verdicts")
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Settings")
    
    # Load data
    with st.spinner("Loading data..."):
        try:
            model, doc_mappings, topic_mappings, years_df = load_lda_data()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()
    
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
        value=1990,
        step=1
    )
    
    # Multi-topic threshold (only for multi-topic mode)
    if analysis_mode == "Multi Topic":
        threshold = st.sidebar.slider(
            "Topic Inclusion Threshold:",
            min_value=0.1,
            max_value=0.8,
            value=0.3,
            step=0.05,
            help="Minimum probability threshold for including a topic in a document"
        )
    
    # Process data based on selected mode
    with st.spinner("Processing data..."):
        if analysis_mode == "Single Topic":
            merged_df, yr_agg_df = create_single_topic_data(doc_mappings, years_df)
            mode_info = f"Single Topic - {len(merged_df)} documents"
        else:
            merged_df, yr_agg_df = create_multi_topic_data(doc_mappings, years_df, threshold)
            if merged_df.empty:
                st.error("No data found for multi-topic analysis with the selected threshold")
                st.stop()
            mode_info = f"Multi Topic - {len(merged_df)} topic-document pairs ({merged_df['filename'].nunique()} unique documents)"
    
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
            fig1 = plot_topics_trend(yr_agg_df, topic_mappings, min_year)
            st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        st.header("📊 Absolute Topic Trends (Document Counts)")
        st.markdown("This chart shows the actual number of documents for each topic over time")
        
        with st.spinner("Creating absolute trends chart..."):
            fig2 = plot_absolute_topics_trend(yr_agg_df, topic_mappings, min_year)
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.header("📚 Stacked Topic Distribution by Year")
        st.markdown("This chart shows the total document volume per year and the distribution of topics within each year")
        
        with st.spinner("Creating stacked distribution chart..."):
            fig3 = plot_stacked_yearly_distribution(yr_agg_df, topic_mappings, min_year)
            st.plotly_chart(fig3, use_container_width=True)
    
    with tab4:
        st.header("📋 Overall Topic Distribution")
        st.markdown("This chart shows the total number of documents for each topic across all years")
        
        with st.spinner("Creating histogram..."):
            fig4 = plot_topics_histogram(yr_agg_df, topic_mappings)
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab5:
        st.header("🎨 Topic Word Clouds")
        st.markdown("Word clouds showing the most prominent words in each topic")
        
        with st.spinner("Creating word clouds..."):
            try:
                fig5 = create_wordcloud_grid(model, topic_mappings)
                st.pyplot(fig5, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating word clouds: {e}")
    
    # Data export section
    st.markdown("---")
    st.header("💾 Data Download")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Download Processed Data"):
            csv = merged_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"lda_analysis_{analysis_mode.replace(' ', '_')}.csv",
                mime="text/csv",
                help="Processed data with topics and years"
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