# visualization_engine.py

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class VisualizationEngine:
    """Advanced visualization engine for The Analyst"""
    
    def __init__(self):
        self.chart_suggestions = {}
        
    def analyze_data_for_charts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataframe structure to suggest appropriate visualizations"""
        analysis = {
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': [],
            'suggested_charts': []
        }
        
        # Detect datetime columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head(100))
                    analysis['datetime_columns'].append(col)
                except:
                    pass
        
        # Generate chart suggestions
        if analysis['datetime_columns'] and analysis['numeric_columns']:
            analysis['suggested_charts'].append({
                'type': 'time_series',
                'title': f"Time Series: {analysis['numeric_columns'][0]} over time",
                'x': analysis['datetime_columns'][0],
                'y': analysis['numeric_columns'][0]
            })
        
        if len(analysis['numeric_columns']) >= 2:
            analysis['suggested_charts'].append({
                'type': 'scatter',
                'title': f"Correlation: {analysis['numeric_columns'][0]} vs {analysis['numeric_columns'][1]}",
                'x': analysis['numeric_columns'][0],
                'y': analysis['numeric_columns'][1]
            })
        
        if analysis['categorical_columns'] and analysis['numeric_columns']:
            analysis['suggested_charts'].append({
                'type': 'bar',
                'title': f"Average {analysis['numeric_columns'][0]} by {analysis['categorical_columns'][0]}",
                'x': analysis['categorical_columns'][0],
                'y': analysis['numeric_columns'][0]
            })
        
        return analysis
    
    def create_time_series_chart(self, df: pd.DataFrame, x_col: str, y_col: str, title: str = None) -> go.Figure:
        """Create interactive time series chart"""
        fig = px.line(df, x=x_col, y=y_col, 
                     title=title or f"{y_col} over {x_col}",
                     template="plotly_dark")
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            hovermode='x unified'
        )
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap for numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto=True,
                       aspect="auto",
                       title="Correlation Matrix",
                       template="plotly_dark",
                       color_continuous_scale="RdBu")
        return fig
    
    def create_distribution_plot(self, df: pd.DataFrame, column: str) -> go.Figure:
        """Create distribution plot for a numeric column"""
        fig = px.histogram(df, x=column, 
                          title=f"Distribution of {column}",
                          template="plotly_dark",
                          marginal="box")
        return fig
    
    def create_bar_chart(self, df: pd.DataFrame, x_col: str, y_col: str, title: str = None) -> go.Figure:
        """Create interactive bar chart"""
        # Aggregate data if needed
        if df[x_col].dtype == 'object':
            agg_df = df.groupby(x_col)[y_col].mean().reset_index()
        else:
            agg_df = df
            
        fig = px.bar(agg_df, x=x_col, y=y_col,
                    title=title or f"Average {y_col} by {x_col}",
                    template="plotly_dark")
        return fig
    
    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, 
                          color_col: str = None, title: str = None) -> go.Figure:
        """Create interactive scatter plot"""
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                        title=title or f"{y_col} vs {x_col}",
                        template="plotly_dark",
                        trendline="ols" if len(df) > 10 else None)
        return fig
    
    def suggest_charts_for_query(self, query: str, df: pd.DataFrame) -> List[Dict]:
        """Suggest appropriate charts based on user query"""
        query_lower = query.lower()
        suggestions = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Time series keywords
        if any(word in query_lower for word in ['trend', 'over time', 'time series', 'temporal']):
            datetime_cols = []
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    datetime_cols.append(col)
            
            if datetime_cols and numeric_cols:
                suggestions.append({
                    'type': 'time_series',
                    'title': f"Time Series Analysis",
                    'description': f"Shows trends in {numeric_cols[0]} over {datetime_cols[0]}"
                })
        
        # Correlation keywords
        if any(word in query_lower for word in ['correlation', 'relationship', 'vs', 'against']):
            if len(numeric_cols) >= 2:
                suggestions.append({
                    'type': 'scatter',
                    'title': f"Correlation Analysis",
                    'description': f"Shows relationship between {numeric_cols[0]} and {numeric_cols[1]}"
                })
        
        # Distribution keywords
        if any(word in query_lower for word in ['distribution', 'histogram', 'spread']):
            if numeric_cols:
                suggestions.append({
                    'type': 'histogram',
                    'title': f"Distribution Analysis", 
                    'description': f"Shows distribution of {numeric_cols[0]}"
                })
        
        # Comparison keywords
        if any(word in query_lower for word in ['compare', 'by category', 'group by']):
            if categorical_cols and numeric_cols:
                suggestions.append({
                    'type': 'bar',
                    'title': f"Category Comparison",
                    'description': f"Compares {numeric_cols[0]} across {categorical_cols[0]}"
                })
        
        return suggestions