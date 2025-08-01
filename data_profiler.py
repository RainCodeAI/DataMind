# data_profiler.py

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Tuple
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

class DataProfiler:
    """Smart data profiling and quality assessment for The Analyst"""
    
    def __init__(self):
        self.profile_cache = {}
    
    def generate_data_profile(self, df: pd.DataFrame, file_name: str = "dataset") -> Dict[str, Any]:
        """Generate comprehensive data profile"""
        profile = {
            'overview': self._get_overview(df),
            'columns': self._analyze_columns(df),
            'quality': self._assess_data_quality(df),
            'patterns': self._detect_patterns(df),
            'anomalies': self._detect_anomalies(df),
            'suggestions': self._generate_suggestions(df)
        }
        
        # Cache the profile
        self.profile_cache[file_name] = profile
        return profile
    
    def _get_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Basic dataset overview"""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.value_counts().to_dict()
        }
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Detailed analysis of each column"""
        column_analysis = {}
        
        for col in df.columns:
            analysis = {
                'dtype': str(df[col].dtype),
                'non_null_count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique(),
                'unique_percentage': (df[col].nunique() / len(df)) * 100
            }
            
            # Numeric column analysis
            if pd.api.types.is_numeric_dtype(df[col]):
                analysis.update({
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'quartiles': df[col].quantile([0.25, 0.5, 0.75]).to_dict(),
                    'outliers': self._detect_outliers(df[col]),
                    'distribution': self._analyze_distribution(df[col])
                })
            
            # Categorical column analysis
            elif pd.api.types.is_object_dtype(df[col]):
                value_counts = df[col].value_counts()
                analysis.update({
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'top_values': value_counts.head(5).to_dict(),
                    'avg_length': df[col].astype(str).str.len().mean(),
                    'contains_numbers': df[col].astype(str).str.contains(r'\d').sum(),
                    'potential_datetime': self._is_potential_datetime(df[col])
                })
            
            column_analysis[col] = analysis
        
        return column_analysis
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality"""
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        
        quality_score = max(0, 100 - (missing_cells / total_cells * 100))
        
        issues = []
        if missing_cells > 0:
            issues.append(f"{missing_cells} missing values ({missing_cells/total_cells*100:.1f}%)")
        
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate rows")
        
        # Check for potential data entry errors
        for col in df.select_dtypes(include=['object']).columns:
            similar_values = self._find_similar_values(df[col])
            if similar_values:
                issues.append(f"Potential typos in '{col}': {similar_values[:3]}")
        
        return {
            'overall_score': round(quality_score, 1),
            'missing_percentage': round(missing_cells / total_cells * 100, 2),
            'duplicate_rows': duplicates,
            'issues': issues,
            'recommendations': self._generate_quality_recommendations(df, issues)
        }
    
    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect interesting patterns in the data"""
        patterns = {}
        
        # Detect time series patterns
        datetime_cols = []
        for col in df.columns:
            if self._is_potential_datetime(df[col]):
                datetime_cols.append(col)
        
        if datetime_cols:
            patterns['time_series'] = {
                'columns': datetime_cols,
                'date_range': self._get_date_range(df, datetime_cols[0]) if datetime_cols else None
            }
        
        # Detect hierarchical data
        hierarchical = self._detect_hierarchical_columns(df)
        if hierarchical:
            patterns['hierarchical'] = hierarchical
        
        # Detect key relationships
        potential_keys = []
        for col in df.columns:
            if df[col].nunique() == len(df) and df[col].isnull().sum() == 0:
                potential_keys.append(col)
        
        if potential_keys:
            patterns['unique_identifiers'] = potential_keys
        
        return patterns
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, List]:
        """Detect anomalies and outliers"""
        anomalies = {
            'statistical_outliers': [],
            'pattern_anomalies': [],
            'data_type_anomalies': []
        }
        
        # Statistical outliers for numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            outliers = self._detect_outliers(df[col])
            if len(outliers) > 0:
                anomalies['statistical_outliers'].append({
                    'column': col,
                    'count': len(outliers),
                    'percentage': len(outliers) / len(df) * 100,
                    'values': outliers[:5].tolist()  # Show first 5
                })
        
        return anomalies
    
    def _generate_suggestions(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable suggestions for data improvement"""
        suggestions = []
        
        # Missing data suggestions
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            suggestions.append(f"Consider handling missing values in: {', '.join(missing_cols[:3])}")
        
        # Data type suggestions
        for col in df.select_dtypes(include=['object']).columns:
            if self._is_potential_datetime(df[col]):
                suggestions.append(f"Column '{col}' appears to contain dates - consider converting to datetime")
        
        # Performance suggestions
        if len(df) > 100000:
            suggestions.append("Large dataset detected - consider data sampling for faster analysis")
        
        return suggestions
    
    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method"""
        if not pd.api.types.is_numeric_dtype(series):
            return pd.Series([])
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return series[(series < lower_bound) | (series > upper_bound)]
    
    def _analyze_distribution(self, series: pd.Series) -> str:
        """Analyze the distribution of a numeric series"""
        if len(series.dropna()) < 10:
            return "insufficient_data"
        
        # Test for normality
        try:
            stat, p_value = stats.normaltest(series.dropna())
            if p_value > 0.05:
                return "normal"
            else:
                skewness = stats.skew(series.dropna())
                if skewness > 1:
                    return "right_skewed"
                elif skewness < -1:
                    return "left_skewed"
                else:
                    return "moderately_skewed"
        except:
            return "unknown"
    
    def _is_potential_datetime(self, series: pd.Series) -> bool:
        """Check if a series contains potential datetime values"""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        if not pd.api.types.is_object_dtype(series):
            return False
        
        # Sample a few values and try to parse as datetime
        sample = series.dropna().head(10)
        datetime_count = 0
        
        for value in sample:
            try:
                pd.to_datetime(str(value))
                datetime_count += 1
            except:
                pass
        
        return datetime_count / len(sample) > 0.7 if len(sample) > 0 else False
    
    def _find_similar_values(self, series: pd.Series) -> List[str]:
        """Find potentially similar values that might be typos"""
        # This is a simplified version - in production, you'd use more sophisticated similarity algorithms
        value_counts = series.value_counts()
        similar_pairs = []
        
        # Look for values that are very similar (simple case-insensitive comparison)
        values = value_counts.index.tolist()[:20]  # Limit to top 20 for performance
        
        for i, val1 in enumerate(values):
            for val2 in values[i+1:]:
                if isinstance(val1, str) and isinstance(val2, str):
                    if val1.lower() == val2.lower() and val1 != val2:
                        similar_pairs.append(f"'{val1}' vs '{val2}'")
        
        return similar_pairs[:5]  # Return top 5
    
    def _generate_quality_recommendations(self, df: pd.DataFrame, issues: List[str]) -> List[str]:
        """Generate recommendations based on quality issues"""
        recommendations = []
        
        if any("missing" in issue for issue in issues):
            recommendations.append("Consider imputation strategies for missing values")
        
        if any("duplicate" in issue for issue in issues):
            recommendations.append("Remove or investigate duplicate rows")
        
        if any("typos" in issue for issue in issues):
            recommendations.append("Standardize categorical values to fix potential typos")
        
        return recommendations
    
    def _get_date_range(self, df: pd.DataFrame, date_col: str) -> Dict[str, Any]:
        """Get date range information for datetime column"""
        try:
            dates = pd.to_datetime(df[date_col])
            return {
                'start': dates.min(),
                'end': dates.max(),
                'span_days': (dates.max() - dates.min()).days
            }
        except:
            return {}
    
    def _detect_hierarchical_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect columns that might represent hierarchical data"""
        hierarchical = []
        
        for col in df.select_dtypes(include=['object']).columns:
            # Look for patterns like "Category > Subcategory" or similar
            sample_values = df[col].dropna().head(100)
            separator_counts = {}
            
            for sep in ['>', '/', '\\', '-', '|']:
                count = sum(1 for val in sample_values if sep in str(val))
                if count > len(sample_values) * 0.3:  # 30% threshold
                    hierarchical.append(col)
                    break
        
        return hierarchical

    def render_profile_dashboard(self, profile: Dict[str, Any]) -> None:
        """Render the data profiling dashboard in Streamlit"""
        st.subheader("📊 Data Profile Dashboard")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{profile['overview']['rows']:,}")
        with col2:
            st.metric("Columns", profile['overview']['columns'])
        with col3:
            st.metric("Quality Score", f"{profile['quality']['overall_score']}%")
        with col4:
            st.metric("Missing %", f"{profile['quality']['missing_percentage']}%")
        
        # Quality assessment
        if profile['quality']['issues']:
            with st.expander("⚠️ Data Quality Issues", expanded=True):
                for issue in profile['quality']['issues']:
                    st.warning(issue)
        
        # Column details
        with st.expander("📋 Column Analysis"):
            for col_name, col_info in profile['columns'].items():
                st.write(f"**{col_name}** ({col_info['dtype']})")
                
                subcol1, subcol2, subcol3 = st.columns(3)
                with subcol1:
                    st.write(f"Non-null: {col_info['non_null_count']}")
                with subcol2:
                    st.write(f"Unique: {col_info['unique_count']}")
                with subcol3:
                    st.write(f"Missing: {col_info['null_percentage']:.1f}%")
                
                st.divider()
        
        # Suggestions
        if profile['suggestions']:
            with st.expander("💡 Improvement Suggestions"):
                for suggestion in profile['suggestions']:
                    st.info(suggestion)