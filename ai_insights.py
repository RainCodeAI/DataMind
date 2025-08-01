# ai_insights.py

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
import re

class AIInsights:
    """AI-powered automatic insights and recommendations for The Analyst"""
    
    def __init__(self):
        self.insight_patterns = {
            'trend_detection': [
                'monotonic_increase', 'monotonic_decrease', 'seasonal_pattern',
                'cyclical_pattern', 'exponential_growth', 'linear_trend'
            ],
            'anomaly_patterns': [
                'sudden_spike', 'sudden_drop', 'outlier_cluster', 'missing_data_pattern'
            ],
            'correlation_patterns': [
                'strong_positive', 'strong_negative', 'unexpected_correlation',
                'weak_expected_correlation'
            ],
            'distribution_patterns': [
                'highly_skewed', 'bimodal', 'uniform', 'normal_distribution'
            ]
        }
        
        if "auto_insights" not in st.session_state:
            st.session_state.auto_insights = []
        
        if "insight_notifications" not in st.session_state:
            st.session_state.insight_notifications = []
    
    def analyze_and_generate_insights(self, df: pd.DataFrame, file_name: str = "dataset") -> List[Dict[str, Any]]:
        """Generate automatic insights from the dataset"""
        insights = []
        
        # Data quality insights
        insights.extend(self._analyze_data_quality(df))
        
        # Distribution insights
        insights.extend(self._analyze_distributions(df))
        
        # Correlation insights
        insights.extend(self._analyze_correlations(df))
        
        # Trend insights (if datetime columns exist)
        insights.extend(self._analyze_trends(df))
        
        # Anomaly insights
        insights.extend(self._detect_interesting_anomalies(df))
        
        # Business logic insights
        insights.extend(self._generate_business_insights(df))
        
        # Add metadata
        for insight in insights:
            insight.update({
                'dataset': file_name,
                'generated_at': datetime.now(),
                'confidence': insight.get('confidence', 0.7),
                'actionable': insight.get('actionable', True)
            })
        
        # Sort by importance/confidence
        insights.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Store in session state
        st.session_state.auto_insights.extend(insights)
        
        return insights[:10]  # Return top 10 insights
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate insights about data quality"""
        insights = []
        
        # Missing data insights
        missing_percentages = (df.isnull().sum() / len(df) * 100)
        high_missing = missing_percentages[missing_percentages > 20]
        
        if len(high_missing) > 0:
            insights.append({
                'type': 'data_quality',
                'category': 'missing_data',
                'title': 'High Missing Data Detected',
                'description': f"Columns {list(high_missing.index)} have >20% missing values",
                'impact': 'high',
                'recommendation': f"Consider imputation strategies or removing columns: {', '.join(high_missing.index[:3])}",
                'confidence': 0.9,
                'actionable': True,
                'details': {
                    'affected_columns': list(high_missing.index),
                    'missing_percentages': high_missing.to_dict()
                }
            })
        
        # Duplicate data insights
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(df)) * 100
            insights.append({
                'type': 'data_quality',
                'category': 'duplicates',
                'title': 'Duplicate Records Found',
                'description': f"Found {duplicate_count} duplicate rows ({duplicate_percentage:.1f}% of data)",
                'impact': 'medium' if duplicate_percentage < 10 else 'high',
                'recommendation': "Review and consider removing duplicate records",
                'confidence': 0.95,
                'actionable': True,
                'details': {
                    'duplicate_count': duplicate_count,
                    'percentage': duplicate_percentage
                }
            })
        
        # Data type insights
        object_columns = df.select_dtypes(include=['object']).columns
        for col in object_columns:
            # Check if column might be numeric
            try:
                numeric_convertible = pd.to_numeric(df[col], errors='coerce').notna().sum()
                if numeric_convertible > len(df) * 0.8:  # 80% convertible
                    insights.append({
                        'type': 'data_quality',
                        'category': 'data_types',
                        'title': f'Column "{col}" Appears Numeric',
                        'description': f"Column {col} contains mostly numeric values but is stored as text",
                        'impact': 'medium',
                        'recommendation': f"Consider converting {col} to numeric type for better analysis",
                        'confidence': 0.8,
                        'actionable': True,
                        'details': {
                            'column': col,
                            'convertible_percentage': (numeric_convertible / len(df)) * 100
                        }
                    })
            except:
                pass
        
        return insights
    
    def _analyze_distributions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate insights about data distributions"""
        insights = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            
            # Skewness analysis
            skewness = series.skew()
            if abs(skewness) > 2:
                insights.append({
                    'type': 'distribution',
                    'category': 'skewness',
                    'title': f'"{col}" is Highly Skewed',
                    'description': f"Column {col} shows {'right' if skewness > 0 else 'left'} skewness (value: {skewness:.2f})",
                    'impact': 'medium',
                    'recommendation': f"Consider log transformation or outlier removal for {col}",
                    'confidence': 0.8,
                    'actionable': True,
                    'details': {
                        'column': col,
                        'skewness': skewness,
                        'direction': 'right' if skewness > 0 else 'left'
                    }
                })
            
            # Outlier detection
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers = series[(series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)]
            
            if len(outliers) > len(series) * 0.05:  # More than 5% outliers
                insights.append({
                    'type': 'distribution',
                    'category': 'outliers',
                    'title': f'Significant Outliers in "{col}"',
                    'description': f"Column {col} has {len(outliers)} outliers ({len(outliers)/len(series)*100:.1f}% of data)",
                    'impact': 'medium',
                    'recommendation': f"Investigate outliers in {col} - they may indicate data errors or interesting patterns",
                    'confidence': 0.85,
                    'actionable': True,
                    'details': {
                        'column': col,
                        'outlier_count': len(outliers),
                        'percentage': len(outliers) / len(series) * 100,
                        'outlier_values': outliers.head().tolist()
                    }
                })
        
        return insights
    
    def _analyze_correlations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate insights about correlations"""
        insights = []
        
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return insights
        
        corr_matrix = numeric_df.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'col1': corr_matrix.columns[i],
                        'col2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        if strong_correlations:
            for corr in strong_correlations[:3]:  # Top 3 correlations
                relationship_type = "positive" if corr['correlation'] > 0 else "negative"
                insights.append({
                    'type': 'correlation',
                    'category': 'strong_correlation',
                    'title': f'Strong {relationship_type.title()} Correlation Found',
                    'description': f"{corr['col1']} and {corr['col2']} are strongly correlated (r = {corr['correlation']:.3f})",
                    'impact': 'high' if abs(corr['correlation']) > 0.8 else 'medium',
                    'recommendation': f"Explore the relationship between {corr['col1']} and {corr['col2']} further",
                    'confidence': 0.9,
                    'actionable': True,
                    'details': {
                        'column1': corr['col1'],
                        'column2': corr['col2'],
                        'correlation_value': corr['correlation'],
                        'relationship_type': relationship_type
                    }
                })
        
        # Unexpected weak correlations (columns that seem related but aren't)
        potential_related = self._find_potentially_related_columns(numeric_df.columns)
        for col1, col2 in potential_related:
            if col1 in corr_matrix.columns and col2 in corr_matrix.columns:
                corr_value = corr_matrix.loc[col1, col2]
                if abs(corr_value) < 0.3:  # Unexpectedly weak
                    insights.append({
                        'type': 'correlation',
                        'category': 'weak_correlation',
                        'title': f'Unexpectedly Weak Correlation',
                        'description': f"{col1} and {col2} show weaker correlation than expected (r = {corr_value:.3f})",
                        'impact': 'low',
                        'recommendation': f"Investigate why {col1} and {col2} aren't more strongly related",
                        'confidence': 0.6,
                        'actionable': True,
                        'details': {
                            'column1': col1,
                            'column2': col2,
                            'correlation_value': corr_value
                        }
                    })
        
        return insights
    
    def _analyze_trends(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate insights about trends in time-based data"""
        insights = []
        
        # Find potential date columns
        date_columns = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(df[col].head(100))
                    date_columns.append(col)
                except:
                    pass
        
        if not date_columns:
            return insights
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for date_col in date_columns[:2]:  # Analyze first 2 date columns
            for num_col in numeric_columns[:3]:  # Top 3 numeric columns
                try:
                    # Create time series
                    df_ts = df[[date_col, num_col]].copy()
                    df_ts[date_col] = pd.to_datetime(df_ts[date_col])
                    df_ts = df_ts.sort_values(date_col).dropna()
                    
                    if len(df_ts) < 10:
                        continue
                    
                    # Analyze trend
                    x = np.arange(len(df_ts))
                    y = df_ts[num_col].values
                    
                    # Linear regression for trend
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    if abs(r_value) > 0.5 and p_value < 0.05:  # Significant trend
                        trend_direction = "increasing" if slope > 0 else "decreasing"
                        insights.append({
                            'type': 'trend',
                            'category': 'time_series_trend',
                            'title': f'{trend_direction.title()} Trend in "{num_col}"',
                            'description': f"{num_col} shows a clear {trend_direction} trend over time (R² = {r_value**2:.3f})",
                            'impact': 'high' if abs(r_value) > 0.7 else 'medium',
                            'recommendation': f"Monitor the {trend_direction} trend in {num_col} for future planning",
                            'confidence': min(0.9, abs(r_value)),
                            'actionable': True,
                            'details': {
                                'date_column': date_col,
                                'value_column': num_col,
                                'trend_direction': trend_direction,
                                'slope': slope,
                                'r_squared': r_value ** 2,
                                'p_value': p_value
                            }
                        })
                
                except Exception:
                    continue
        
        return insights
    
    def _detect_interesting_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect and describe interesting anomalies"""
        insights = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            series = df[col].dropna()
            if len(series) < 20:
                continue
            
            # Find extreme values
            mean_val = series.mean()
            std_val = series.std()
            
            # Values more than 3 standard deviations away
            extreme_values = series[abs(series - mean_val) > 3 * std_val]
            
            if len(extreme_values) > 0 and len(extreme_values) < len(series) * 0.1:
                insights.append({
                    'type': 'anomaly',
                    'category': 'extreme_values',
                    'title': f'Extreme Values Detected in "{col}"',
                    'description': f"Found {len(extreme_values)} extreme values in {col} (>3σ from mean)",
                    'impact': 'medium',
                    'recommendation': f"Investigate extreme values in {col} - they may represent errors or special cases",
                    'confidence': 0.8,
                    'actionable': True,
                    'details': {
                        'column': col,
                        'extreme_count': len(extreme_values),
                        'extreme_values': extreme_values.head(5).tolist(),
                        'mean': mean_val,
                        'std': std_val
                    }
                })
        
        return insights
    
    def _generate_business_insights(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate business-relevant insights based on common patterns"""
        insights = []
        
        # Revenue/Sales patterns
        revenue_columns = [col for col in df.columns if any(keyword in col.lower() 
                          for keyword in ['revenue', 'sales', 'income', 'profit'])]
        
        for rev_col in revenue_columns:
            if pd.api.types.is_numeric_dtype(df[rev_col]):
                total_revenue = df[rev_col].sum()
                avg_revenue = df[rev_col].mean()
                
                # Zero revenue analysis
                zero_revenue = (df[rev_col] == 0).sum()
                if zero_revenue > len(df) * 0.1:  # More than 10% zero revenue
                    insights.append({
                        'type': 'business',
                        'category': 'revenue_analysis',
                        'title': 'High Zero Revenue Records',
                        'description': f"{zero_revenue} records ({zero_revenue/len(df)*100:.1f}%) have zero {rev_col}",
                        'impact': 'high',
                        'recommendation': f"Investigate why {zero_revenue} records have zero {rev_col}",
                        'confidence': 0.85,
                        'actionable': True,
                        'details': {
                            'column': rev_col,
                            'zero_count': zero_revenue,
                            'percentage': zero_revenue / len(df) * 100
                        }
                    })
        
        # Customer/User patterns
        customer_columns = [col for col in df.columns if any(keyword in col.lower() 
                           for keyword in ['customer', 'user', 'client'])]
        
        for cust_col in customer_columns:
            if pd.api.types.is_object_dtype(df[cust_col]):
                unique_customers = df[cust_col].nunique()
                total_records = len(df)
                
                # Customer frequency analysis
                if unique_customers < total_records * 0.8:  # Repeat customers
                    customer_frequency = df[cust_col].value_counts()
                    repeat_customers = (customer_frequency > 1).sum()
                    
                    insights.append({
                        'type': 'business',
                        'category': 'customer_analysis',
                        'title': 'Repeat Customer Pattern Detected',
                        'description': f"{repeat_customers} customers ({repeat_customers/unique_customers*100:.1f}%) have multiple records",
                        'impact': 'medium',
                        'recommendation': f"Analyze repeat customer behavior and loyalty patterns",
                        'confidence': 0.8,
                        'actionable': True,
                        'details': {
                            'column': cust_col,
                            'unique_customers': unique_customers,
                            'repeat_customers': repeat_customers,
                            'total_records': total_records
                        }
                    })
        
        return insights
    
    def _find_potentially_related_columns(self, columns: List[str]) -> List[Tuple[str, str]]:
        """Find column pairs that might be expected to correlate based on naming"""
        potentially_related = []
        
        # Common related patterns
        related_patterns = [
            (['price', 'cost'], ['quantity', 'amount', 'volume']),
            (['height', 'length'], ['width', 'diameter']),
            (['age'], ['income', 'salary', 'wage']),
            (['experience'], ['salary', 'wage', 'income']),
            (['temperature'], ['pressure', 'humidity']),
        ]
        
        for pattern_group1, pattern_group2 in related_patterns:
            cols1 = [col for col in columns if any(p in col.lower() for p in pattern_group1)]
            cols2 = [col for col in columns if any(p in col.lower() for p in pattern_group2)]
            
            for col1 in cols1:
                for col2 in cols2:
                    potentially_related.append((col1, col2))
        
        return potentially_related
    
    def suggest_follow_up_questions(self, df: pd.DataFrame, recent_queries: List[str]) -> List[str]:
        """Suggest intelligent follow-up questions based on data and conversation"""
        suggestions = []
        
        # Analyze recent queries to understand user interest
        query_themes = self._extract_query_themes(recent_queries)
        
        # Data-driven suggestions
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Correlation suggestions
        if len(numeric_columns) >= 2 and 'correlation' not in query_themes:
            suggestions.append(f"What is the correlation between {numeric_columns[0]} and {numeric_columns[1]}?")
        
        # Trend analysis suggestions
        date_candidates = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_candidates and numeric_columns and 'trend' not in query_themes:
            suggestions.append(f"Show me the trend of {numeric_columns[0]} over time")
        
        # Category analysis suggestions
        if categorical_columns and numeric_columns and 'category' not in query_themes:
            suggestions.append(f"What is the average {numeric_columns[0]} by {categorical_columns[0]}?")
        
        # Outlier analysis suggestions
        if 'outlier' not in query_themes and 'anomal' not in query_themes:
            suggestions.append(f"Are there any outliers in {numeric_columns[0]}?")
        
        # Distribution suggestions
        if 'distribution' not in query_themes and numeric_columns:
            suggestions.append(f"What does the distribution of {numeric_columns[0]} look like?")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _extract_query_themes(self, queries: List[str]) -> set:
        """Extract themes from recent queries"""
        themes = set()
        
        for query in queries:
            query_lower = query.lower()
            
            # Define theme keywords
            theme_keywords = {
                'correlation': ['correlation', 'relationship', 'related'],
                'trend': ['trend', 'over time', 'temporal', 'growth'],
                'category': ['by category', 'group by', 'compare'],
                'outlier': ['outlier', 'anomaly', 'unusual'],
                'distribution': ['distribution', 'histogram', 'spread'],
                'average': ['average', 'mean', 'typical'],
                'maximum': ['max', 'highest', 'largest', 'top'],
                'minimum': ['min', 'lowest', 'smallest', 'bottom']
            }
            
            for theme, keywords in theme_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    themes.add(theme)
        
        return themes
    
    def generate_insight_notification(self, insight: Dict[str, Any]) -> None:
        """Generate a user-friendly notification for an insight"""
        if insight['impact'] == 'high':
            notification = {
                'type': 'info' if insight['actionable'] else 'warning',
                'title': f"💡 Insight: {insight['title']}",
                'message': insight['description'],
                'recommendation': insight['recommendation'],
                'timestamp': datetime.now()
            }
            
            st.session_state.insight_notifications.append(notification)
    
    def render_insights_dashboard(self, df: Optional[pd.DataFrame] = None):
        """Render the AI insights dashboard"""
        st.subheader("🤖 AI-Powered Insights")
        
        if df is not None:
            # Generate new insights
            if st.button("🔍 Generate New Insights"):
                with st.spinner("Analyzing data and generating insights..."):
                    insights = self.analyze_and_generate_insights(df)
                    
                    if insights:
                        st.success(f"Generated {len(insights)} new insights!")
                        
                        # Show top insights
                        for insight in insights[:3]:
                            self._render_insight_card(insight)
                    else:
                        st.info("No significant insights found in the current dataset.")
        
        # Show stored insights
        if st.session_state.auto_insights:
            st.write("**Recent Insights:**")
            
            # Filter and sort insights
            recent_insights = sorted(st.session_state.auto_insights, 
                                   key=lambda x: x['generated_at'], reverse=True)
            
            # Show insights by category
            categories = set(insight['category'] for insight in recent_insights)
            
            for category in categories:
                category_insights = [i for i in recent_insights if i['category'] == category]
                
                with st.expander(f"{category.replace('_', ' ').title()} ({len(category_insights)})"):
                    for insight in category_insights[:5]:  # Show top 5 per category
                        self._render_insight_card(insight, compact=True)
        
        # Show notifications
        if st.session_state.insight_notifications:
            st.write("**Recent Notifications:**")
            
            for notification in st.session_state.insight_notifications[-5:]:
                if notification['type'] == 'info':
                    st.info(f"{notification['title']}: {notification['message']}")
                else:
                    st.warning(f"{notification['title']}: {notification['message']}")
                
                if notification.get('recommendation'):
                    st.caption(f"💡 {notification['recommendation']}")
    
    def _render_insight_card(self, insight: Dict[str, Any], compact: bool = False):
        """Render an individual insight card"""
        
        # Impact color coding
        impact_colors = {
            'high': '🔴',
            'medium': '🟡', 
            'low': '🟢'
        }
        
        impact_icon = impact_colors.get(insight['impact'], '⚪')
        
        if compact:
            st.write(f"{impact_icon} **{insight['title']}**")
            st.caption(insight['description'])
            if insight.get('recommendation'):
                st.caption(f"💡 {insight['recommendation']}")
        else:
            with st.container():
                st.write(f"{impact_icon} **{insight['title']}** (Confidence: {insight['confidence']:.0%})")
                st.write(insight['description'])
                
                if insight.get('recommendation'):
                    st.info(f"💡 **Recommendation:** {insight['recommendation']}")
                
                if insight.get('details'):
                    with st.expander("View Details"):
                        st.json(insight['details'])
                
                st.divider()