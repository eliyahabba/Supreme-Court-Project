#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDA Visualization Functions
==========================

Extracted visualization functions for use in both Streamlit app and file generation.
Updated to support topic filtering using shared constants.
"""

import colorsys
import math
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gensim.models import LdaModel
from wordcloud import WordCloud

# Import shared constants
from constants import (
    DEFAULT_CHART_WIDTH, DEFAULT_CHART_HEIGHT, WIDE_CHART_WIDTH, WIDE_CHART_HEIGHT,
    COLOR_SATURATION_BASE, COLOR_VALUE_BASE
)
from utils import filter_topics_from_data, get_filtered_topic_mappings, print_filtering_info

warnings.filterwarnings('ignore')


def generate_distinct_colors(n_colors: int) -> list:
    """Generate n distinct colors for better visualization of many topics"""
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        saturation = COLOR_SATURATION_BASE + (i % 3) * 0.05
        value = COLOR_VALUE_BASE + (i % 4) * 0.075
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), 
            int(rgb[1] * 255), 
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    
    return colors


def add_topic_display_names(data: pd.DataFrame, topic_mappings: pd.DataFrame, 
                           topic_col: str = 'strongest_topic',
                           excluded_topics: List[int] = None,
                           included_topics: List[int] = None) -> pd.DataFrame:
    """Add Hebrew topic display names to data and apply topic filtering"""
    data = data.copy()
    
    # Apply topic filtering first
    data = filter_topics_from_data(data, excluded_topics, included_topics, topic_col)
    
    if data.empty:
        print("⚠️ No data remaining after topic filtering")
        return data
    
    # Filter topic mappings as well
    filtered_topic_mappings = get_filtered_topic_mappings(topic_mappings, excluded_topics, included_topics)
    
    if not filtered_topic_mappings.empty:
        topic_name_map = {}
        for _, row in filtered_topic_mappings.iterrows():
            topic_id = str(int(row['מספר נושא']))
            topic_name = row['כותרת מוצעת']
            topic_name_map[topic_id] = f"{topic_id}: {topic_name}"
        
        data['topic_display'] = data[topic_col].astype(str).map(topic_name_map)
        data['topic_display'] = data['topic_display'].fillna(data[topic_col].astype(str))
    else:
        data['topic_display'] = data[topic_col].astype(str)
    
    # Add numeric topic column for proper sorting
    data['topic_num'] = data[topic_col].astype(int)
    
    return data


def plot_topics_trend(yr_agg_df: pd.DataFrame, topic_mappings: pd.DataFrame, 
                     min_year: int = 1948,
                     excluded_topics: List[int] = None,
                     included_topics: List[int] = None) -> go.Figure:
    """Plot topic trends over years (percentages)"""
    # Print filtering info
    print_filtering_info(excluded_topics, included_topics)
    
    filtered_data = yr_agg_df[yr_agg_df['year'] >= min_year].copy()
    filtered_data = add_topic_display_names(filtered_data, topic_mappings, 'strongest_topic', 
                                          excluded_topics, included_topics)
    
    if filtered_data.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available after applying topic filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title="📈 Topic Trends - No Data", width=DEFAULT_CHART_WIDTH, height=DEFAULT_CHART_HEIGHT)
        return fig
    
    # Sort topics and create colors
    unique_topics = sorted(filtered_data['strongest_topic'].unique(), key=lambda x: int(x))
    n_topics = len(unique_topics)
    distinct_colors = generate_distinct_colors(n_topics)
    
    color_mapping = {}
    for i, topic in enumerate(unique_topics):
        topic_display = filtered_data[filtered_data['strongest_topic'] == topic]['topic_display'].iloc[0]
        color_mapping[topic_display] = distinct_colors[i % len(distinct_colors)]

    # Create category order for proper sorting in legend
    sorted_topic_displays = [filtered_data[filtered_data['strongest_topic'] == topic]['topic_display'].iloc[0] 
                           for topic in unique_topics]

    fig = px.line(
        filtered_data,
        x='year',
        y='topic_percentage',
        color='topic_display',
        title=f'📈 Topic Trends Over Years (from {min_year})',
        labels={
            'topic_percentage': 'Topic Percentage (%)',
            'year': 'Year',
            'topic_display': 'Topic'
        },
        width=DEFAULT_CHART_WIDTH,
        height=DEFAULT_CHART_HEIGHT,
        color_discrete_map=color_mapping,
        category_orders={'topic_display': sorted_topic_displays}
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
            x=1.02,
            traceorder="normal"
        )
    )
    
    fig.update_traces(
        line=dict(width=2.5),
        marker=dict(size=6),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                     'Year: %{x}<br>' +
                     'Percentage: %{y:.1f}%<br>' +
                     '<extra></extra>'
    )

    return fig


def plot_absolute_topics_trend(yr_agg_df: pd.DataFrame, topic_mappings: pd.DataFrame, 
                              min_year: int = 1948,
                              excluded_topics: List[int] = None,
                              included_topics: List[int] = None) -> go.Figure:
    """Plot absolute topic trends over years (number of documents)"""
    # Print filtering info
    print_filtering_info(excluded_topics, included_topics)
    
    filtered_data = yr_agg_df[yr_agg_df['year'] >= min_year].copy()
    filtered_data = add_topic_display_names(filtered_data, topic_mappings, 'strongest_topic',
                                          excluded_topics, included_topics)
    
    if filtered_data.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available after applying topic filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title="📊 Absolute Topic Trends - No Data", width=WIDE_CHART_WIDTH, height=WIDE_CHART_HEIGHT)
        return fig
    
    # Sort topics and create colors
    unique_topics = sorted(filtered_data['strongest_topic'].unique(), key=lambda x: int(x))
    n_topics = len(unique_topics)
    distinct_colors = generate_distinct_colors(n_topics)
    
    color_mapping = {}
    for i, topic in enumerate(unique_topics):
        topic_display = filtered_data[filtered_data['strongest_topic'] == topic]['topic_display'].iloc[0]
        color_mapping[topic_display] = distinct_colors[i % len(distinct_colors)]

    # Create category order for proper sorting in legend
    sorted_topic_displays = [filtered_data[filtered_data['strongest_topic'] == topic]['topic_display'].iloc[0] 
                           for topic in unique_topics]

    fig = px.line(
        filtered_data,
        x='year',
        y='verdicts',
        color='topic_display',
        title=f'📊 Absolute Topic Trends Over Years (from {min_year})',
        labels={
            'verdicts': 'Number of Documents',
            'year': 'Year',
            'topic_display': 'Topic'
        },
        width=WIDE_CHART_WIDTH,
        height=WIDE_CHART_HEIGHT,
        color_discrete_map=color_mapping,
        category_orders={'topic_display': sorted_topic_displays}
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
            x=1.02,
            traceorder="normal"
        )
    )
    
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                     'Year: %{x}<br>' +
                     'Documents: %{y}<br>' +
                     '<extra></extra>'
    )

    return fig


def plot_topics_histogram(yr_agg_df: pd.DataFrame, topic_mappings: pd.DataFrame,
                         excluded_topics: List[int] = None,
                         included_topics: List[int] = None) -> go.Figure:
    """Plot topic histogram for all years"""
    # Print filtering info
    print_filtering_info(excluded_topics, included_topics)
    
    topic_totals = yr_agg_df.groupby('strongest_topic')['verdicts'].sum().reset_index()
    topic_totals['strongest_topic'] = topic_totals['strongest_topic'].astype(str)
    topic_totals = add_topic_display_names(topic_totals, topic_mappings, 'strongest_topic',
                                         excluded_topics, included_topics)
    
    if topic_totals.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available after applying topic filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title="📊 Topic Distribution - No Data", width=DEFAULT_CHART_WIDTH, height=600)
        return fig
    
    # Sort by topic number for consistent order
    topic_totals = topic_totals.sort_values('topic_num')
    
    # Generate distinct colors for all topics in sorted order
    n_topics = len(topic_totals)
    distinct_colors = generate_distinct_colors(n_topics)
    
    # Create color mapping based on sorted topic order (not frequency)
    color_mapping = {}
    for i, row in topic_totals.iterrows():
        topic_display = row['topic_display']
        topic_index = int(row['topic_num'])  # Use topic number for color consistency
        color_mapping[topic_display] = distinct_colors[topic_index % len(distinct_colors)]

    fig = px.bar(
        topic_totals,
        x='topic_display',
        y='verdicts',
        title='📊 Topic Distribution for All Years',
        labels={
            'verdicts': 'Number of Verdicts',
            'topic_display': 'Topic'
        },
        width=DEFAULT_CHART_WIDTH,  
        height=600,
        color='topic_display',
        color_discrete_map=color_mapping,
        category_orders={'topic_display': topic_totals['topic_display'].tolist()}
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


def plot_stacked_yearly_distribution(yr_agg_df: pd.DataFrame, topic_mappings: pd.DataFrame, 
                                    min_year: int = 1948,
                                    excluded_topics: List[int] = None,
                                    included_topics: List[int] = None) -> go.Figure:
    """Plot stacked bar chart showing topic distribution by year"""
    # Print filtering info
    print_filtering_info(excluded_topics, included_topics)
    
    filtered_data = yr_agg_df[yr_agg_df['year'] >= min_year].copy()
    filtered_data = add_topic_display_names(filtered_data, topic_mappings, 'strongest_topic',
                                          excluded_topics, included_topics)
    
    if filtered_data.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available after applying topic filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title="📚 Stacked Distribution - No Data", width=WIDE_CHART_WIDTH, height=WIDE_CHART_HEIGHT)
        return fig

    # Sort topics and create colors
    unique_topics = sorted(filtered_data['strongest_topic'].unique(), key=lambda x: int(x))
    n_topics = len(unique_topics)
    distinct_colors = generate_distinct_colors(n_topics)
    
    color_mapping = {}
    topic_order = {}  # To preserve numerical order in legend
    for i, topic in enumerate(unique_topics):
        topic_display = filtered_data[filtered_data['strongest_topic'] == topic]['topic_display'].iloc[0]
        color_mapping[topic_display] = distinct_colors[i % len(distinct_colors)]
        topic_order[topic_display] = int(topic)

    fig = go.Figure()
    years = sorted(filtered_data['year'].unique())
    topics_in_legend = set()
    
    # Add traces sorted by topic number for consistent legend order
    for topic in unique_topics:
        topic_display = filtered_data[filtered_data['strongest_topic'] == topic]['topic_display'].iloc[0]
        topic_num = int(topic)
        
        for year in years:
            year_data = filtered_data[(filtered_data['year'] == year) & 
                                    (filtered_data['strongest_topic'] == topic)]
            
            if not year_data.empty:
                verdicts = year_data['verdicts'].iloc[0]
                color = color_mapping[topic_display]
                
                # Calculate cumulative base for this year
                year_data_all = filtered_data[filtered_data['year'] == year].copy()
                year_data_all = year_data_all.sort_values('topic_num')
                cumulative_base = 0
                for _, prior_row in year_data_all.iterrows():
                    if int(prior_row['strongest_topic']) < topic_num:
                        cumulative_base += prior_row['verdicts']
                
                show_in_legend = topic_display not in topics_in_legend
                if show_in_legend:
                    topics_in_legend.add(topic_display)
                
                fig.add_trace(go.Bar(
                    x=[year],
                    y=[verdicts],
                    base=[cumulative_base],
                    name=topic_display,
                    marker_color=color,
                    hovertemplate=f'<b>{topic_display}</b><br>' +
                                 f'Year: {year}<br>' +
                                 f'Documents: {verdicts}<br>' +
                                 '<extra></extra>',
                    showlegend=show_in_legend,
                    legendgroup=topic_display,
                    legendrank=topic_num
                ))

    fig.update_layout(
        title=dict(
            text=f'📊 Stacked Topic Distribution by Year (from {min_year})',
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
            dtick=2,
            type='linear'
        ),
        yaxis=dict(
            title='Number of Documents'
        ),
                    width=WIDE_CHART_WIDTH,
            height=WIDE_CHART_HEIGHT
    )

    return fig


def create_wordcloud_grid(model: LdaModel, topic_mappings: pd.DataFrame,
                         excluded_topics: List[int] = None,
                         included_topics: List[int] = None) -> plt.Figure:
    """Create a grid of word clouds for all topics"""
    # Print filtering info
    print_filtering_info(excluded_topics, included_topics)
    
    # Filter topic mappings
    filtered_topic_mappings = get_filtered_topic_mappings(topic_mappings, excluded_topics, included_topics)
    # Hebrew fonts for different systems
    hebrew_fonts = [
        '/System/Library/Fonts/SFHebrew.ttf',
        '/System/Library/Fonts/Arial Unicode MS.ttf',
        'C:/Windows/Fonts/arial.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    ]

    font_path = None
    for font in hebrew_fonts:
        if Path(font).exists():
            font_path = font
            break

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

    cloud = WordCloud(**wordcloud_kwargs)

    # Get topics
    n_topics = model.num_topics
    topics = model.show_topics(formatted=False, num_topics=n_topics, num_words=20)

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

        # Get topic words
        topic_words = dict(topics[i][1])

        # Create word cloud
        if topic_words:
            try:
                cloud.generate_from_frequencies(topic_words)
                ax.imshow(cloud, interpolation='bilinear')
            except Exception:
                ax.text(0.5, 0.5, f'Topic {i}\n(Error creating word cloud)',
                        ha='center', va='center', transform=ax.transAxes)

        # Set title with Hebrew topic name if available
        if not topic_mappings.empty:
            topic_name = None
            for _, row in topic_mappings.iterrows():
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

    plt.suptitle('🎨 LDA Topic Word Clouds', fontsize=20, fontweight='bold')
    plt.tight_layout()

    return fig 