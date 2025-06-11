#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDA Visualization Functions
==========================

Extracted visualization functions for use in both Streamlit app and file generation.
"""

import colorsys
import math
import warnings
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gensim.models import LdaModel
from wordcloud import WordCloud

warnings.filterwarnings('ignore')


def generate_distinct_colors(n_colors: int) -> list:
    """Generate n distinct colors for better visualization of many topics"""
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        saturation = 0.8 + (i % 3) * 0.05
        value = 0.7 + (i % 4) * 0.075
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), 
            int(rgb[1] * 255), 
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    
    return colors


def add_topic_display_names(data: pd.DataFrame, topic_mappings: pd.DataFrame, topic_col: str = 'strongest_topic') -> pd.DataFrame:
    """Add Hebrew topic display names to data"""
    data = data.copy()
    
    if not topic_mappings.empty:
        topic_name_map = {}
        for _, row in topic_mappings.iterrows():
            topic_id = str(int(row['מספר נושא']))
            topic_name = row['כותרת מוצעת']
            topic_name_map[topic_id] = f"{topic_id}: {topic_name}"
        
        data['topic_display'] = data[topic_col].astype(str).map(topic_name_map)
        data['topic_display'] = data['topic_display'].fillna(data[topic_col].astype(str))
    else:
        data['topic_display'] = data[topic_col].astype(str)
    
    return data


def plot_topics_trend(yr_agg_df: pd.DataFrame, topic_mappings: pd.DataFrame, min_year: int = 1948) -> go.Figure:
    """Plot topic trends over years (percentages)"""
    filtered_data = yr_agg_df[yr_agg_df['year'] >= min_year].copy()
    filtered_data = add_topic_display_names(filtered_data, topic_mappings)
    
    # Sort topics and create colors
    unique_topics = sorted(filtered_data['strongest_topic'].unique(), key=lambda x: int(x))
    n_topics = len(unique_topics)
    distinct_colors = generate_distinct_colors(n_topics)
    
    color_mapping = {}
    for i, topic in enumerate(unique_topics):
        topic_display = filtered_data[filtered_data['strongest_topic'] == topic]['topic_display'].iloc[0]
        color_mapping[topic_display] = distinct_colors[i % len(distinct_colors)]

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
        width=1200,
        height=700,
        color_discrete_map=color_mapping
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


def plot_absolute_topics_trend(yr_agg_df: pd.DataFrame, topic_mappings: pd.DataFrame, min_year: int = 1948) -> go.Figure:
    """Plot absolute topic trends over years (number of documents)"""
    filtered_data = yr_agg_df[yr_agg_df['year'] >= min_year].copy()
    filtered_data = add_topic_display_names(filtered_data, topic_mappings)
    
    # Sort topics and create colors
    unique_topics = sorted(filtered_data['strongest_topic'].unique(), key=lambda x: int(x))
    n_topics = len(unique_topics)
    distinct_colors = generate_distinct_colors(n_topics)
    
    color_mapping = {}
    for i, topic in enumerate(unique_topics):
        topic_display = filtered_data[filtered_data['strongest_topic'] == topic]['topic_display'].iloc[0]
        color_mapping[topic_display] = distinct_colors[i % len(distinct_colors)]

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
        width=1400,
        height=800,
        color_discrete_map=color_mapping
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


def plot_topics_histogram(yr_agg_df: pd.DataFrame, topic_mappings: pd.DataFrame) -> go.Figure:
    """Plot topic histogram for all years"""
    topic_totals = yr_agg_df.groupby('strongest_topic')['verdicts'].sum().reset_index()
    topic_totals['strongest_topic'] = topic_totals['strongest_topic'].astype(str)
    topic_totals = add_topic_display_names(topic_totals, topic_mappings)
    
    # Sort by topic number for consistent order
    topic_totals['topic_num'] = topic_totals['strongest_topic'].astype(int)
    topic_totals = topic_totals.sort_values('topic_num')
    
    # Generate colors
    n_topics = len(topic_totals)
    distinct_colors = generate_distinct_colors(n_topics)
    
    color_mapping = {}
    for i, row in topic_totals.iterrows():
        topic_display = row['topic_display']
        topic_index = int(row['topic_num'])
        color_mapping[topic_display] = distinct_colors[topic_index % len(distinct_colors)]

    fig = px.bar(
        topic_totals,
        x='topic_display',
        y='verdicts',
        title='📊 Topic Distribution - All Years',
        labels={
            'verdicts': 'Number of Verdicts',
            'topic_display': 'Topic'
        },
        width=1200,
        height=600,
        color='topic_display',
        color_discrete_map=color_mapping
    )

    fig.update_layout(
        title_x=0.5,
        template='plotly_white',
        xaxis_tickangle=-45,
        showlegend=False
    )
    
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>' +
                     'Documents: %{y}<br>' +
                     '<extra></extra>'
    )

    return fig


def plot_stacked_yearly_distribution(yr_agg_df: pd.DataFrame, topic_mappings: pd.DataFrame, min_year: int = 1948) -> go.Figure:
    """Plot stacked bar chart showing topic distribution by year"""
    filtered_data = yr_agg_df[yr_agg_df['year'] >= min_year].copy()
    filtered_data = add_topic_display_names(filtered_data, topic_mappings)

    # Sort topics and create colors
    unique_topics = sorted(filtered_data['strongest_topic'].unique(), key=lambda x: int(x))
    n_topics = len(unique_topics)
    distinct_colors = generate_distinct_colors(n_topics)
    
    color_mapping = {}
    for i, topic in enumerate(unique_topics):
        topic_display = filtered_data[filtered_data['strongest_topic'] == topic]['topic_display'].iloc[0]
        color_mapping[topic_display] = distinct_colors[i % len(distinct_colors)]

    fig = go.Figure()
    years = sorted(filtered_data['year'].unique())
    topics_in_legend = set()
    
    for year in years:
        year_data = filtered_data[filtered_data['year'] == year].copy()
        year_data = year_data.sort_values('verdicts', ascending=False)
        
        cumulative_base = 0
        
        for _, row in year_data.iterrows():
            topic_display = row['topic_display']
            verdicts = row['verdicts']
            color = color_mapping[topic_display]
            
            show_in_legend = topic_display not in topics_in_legend
            if show_in_legend:
                topics_in_legend.add(topic_display)
            
            topic_num = int(row['strongest_topic'])
            
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
            
            cumulative_base += verdicts

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
            title="Topics (by number)"
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
        width=1400,
        height=800
    )

    return fig


def create_wordcloud_grid(model: LdaModel, topic_mappings: pd.DataFrame) -> plt.Figure:
    """Create a grid of word clouds for all topics"""
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