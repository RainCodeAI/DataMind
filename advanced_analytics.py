# advanced_analytics.py

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalytics:
    """Advanced analytics and machine learning capabilities for The Analyst"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
    
    def statistical_analysis(self, df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {
            'descriptive_stats': {},
            'correlation_analysis': {},
            'normality_tests': {},
            'hypothesis_tests': {}
        }
        
        # Descriptive statistics
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].dropna()
                
                results['descriptive_stats'][col] = {
                    'mean': series.mean(),
                    'median': series.median(),
                    'std': series.std(),
                    'skewness': stats.skew(series),
                    'kurtosis': stats.kurtosis(series),
                    'quartiles': {
                        'Q1': series.quantile(0.25),
                        'Q2': series.quantile(0.5),
                        'Q3': series.quantile(0.75)
                    }
                }
                
                # Normality test
                if len(series) > 3:
                    stat, p_value = stats.normaltest(series)
                    results['normality_tests'][col] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'is_normal': p_value > 0.05
                    }
        
        # Correlation analysis
        numeric_df = df[columns].select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            results['correlation_analysis'] = {
                'matrix': corr_matrix.to_dict(),
                'strong_correlations': self._find_strong_correlations(corr_matrix)
            }
        
        return results
    
    def time_series_analysis(self, df: pd.DataFrame, date_col: str, 
                           value_col: str) -> Dict[str, Any]:
        """Perform time series analysis and forecasting"""
        try:
            # Prepare data
            df_ts = df.copy()
            df_ts[date_col] = pd.to_datetime(df_ts[date_col])
            df_ts = df_ts.sort_values(date_col)
            df_ts.set_index(date_col, inplace=True)
            
            # Basic time series statistics
            results = {
                'trend_analysis': {},
                'seasonality': {},
                'forecast': {},
                'anomalies': []
            }
            
            # Trend analysis
            x = np.arange(len(df_ts))
            y = df_ts[value_col].values
            
            # Linear trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            results['trend_analysis'] = {
                'slope': slope,
                'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                'strength': abs(r_value),
                'significance': p_value < 0.05
            }
            
            # Simple moving averages
            df_ts['MA_7'] = df_ts[value_col].rolling(window=min(7, len(df_ts)//4)).mean()
            df_ts['MA_30'] = df_ts[value_col].rolling(window=min(30, len(df_ts)//2)).mean()
            
            # Detect anomalies using IQR method
            Q1 = df_ts[value_col].quantile(0.25)
            Q3 = df_ts[value_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            anomalies = df_ts[(df_ts[value_col] < lower_bound) | (df_ts[value_col] > upper_bound)]
            results['anomalies'] = anomalies.index.strftime('%Y-%m-%d').tolist()
            
            # Simple linear forecast (next 5 periods)
            if len(df_ts) > 5:
                last_date = df_ts.index[-1]
                future_dates = pd.date_range(start=last_date, periods=6, freq='D')[1:]
                future_x = np.arange(len(df_ts), len(df_ts) + 5)
                forecast_values = slope * future_x + intercept
                
                results['forecast'] = {
                    'dates': future_dates.strftime('%Y-%m-%d').tolist(),
                    'values': forecast_values.tolist(),
                    'confidence': 'Low' if abs(r_value) < 0.5 else 'Medium' if abs(r_value) < 0.8 else 'High'
                }
            
            return results
            
        except Exception as e:
            return {'error': f"Time series analysis failed: {str(e)}"}
    
    def predictive_modeling(self, df: pd.DataFrame, target_col: str, 
                          feature_cols: List[str] = None,
                          model_type: str = 'auto') -> Dict[str, Any]:
        """Build predictive models"""
        try:
            if target_col not in df.columns:
                return {'error': f"Target column '{target_col}' not found"}
            
            # Prepare features
            if feature_cols is None:
                feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                              if col != target_col]
            
            if not feature_cols:
                return {'error': "No suitable feature columns found"}
            
            # Clean data
            df_clean = df[feature_cols + [target_col]].dropna()
            
            if len(df_clean) < 10:
                return {'error': "Insufficient data for modeling (need at least 10 rows)"}
            
            X = df_clean[feature_cols]
            y = df_clean[target_col]
            
            # Determine problem type
            is_classification = (pd.api.types.is_object_dtype(y) or 
                               y.nunique() <= 10 and y.nunique() < len(y) * 0.1)
            
            results = {
                'problem_type': 'classification' if is_classification else 'regression',
                'features_used': feature_cols,
                'data_shape': df_clean.shape,
                'model_performance': {},
                'feature_importance': {},
                'predictions_sample': []
            }
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features for better performance
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            if is_classification:
                # Classification models
                models = {
                    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
                }
                
                # Encode target if string
                if pd.api.types.is_object_dtype(y):
                    encoder = LabelEncoder()
                    y_train_encoded = encoder.fit_transform(y_train)
                    y_test_encoded = encoder.transform(y_test)
                else:
                    y_train_encoded = y_train
                    y_test_encoded = y_test
                    encoder = None
                
                best_model = None
                best_score = 0
                
                for name, model in models.items():
                    try:
                        if name == 'Logistic Regression':
                            model.fit(X_train_scaled, y_train_encoded)
                            y_pred = model.predict(X_test_scaled)
                        else:
                            model.fit(X_train, y_train_encoded)
                            y_pred = model.predict(X_test)
                        
                        score = accuracy_score(y_test_encoded, y_pred)
                        results['model_performance'][name] = {
                            'accuracy': score,
                            'score_type': 'accuracy'
                        }
                        
                        if score > best_score:
                            best_score = score
                            best_model = model
                    except Exception as e:
                        results['model_performance'][name] = {'error': str(e)}
                
                # Feature importance from best model
                if best_model and hasattr(best_model, 'feature_importances_'):
                    importance = dict(zip(feature_cols, best_model.feature_importances_))
                    results['feature_importance'] = dict(sorted(importance.items(), 
                                                               key=lambda x: x[1], reverse=True))
            
            else:
                # Regression models
                models = {
                    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'Linear Regression': LinearRegression()
                }
                
                best_model = None
                best_score = float('-inf')
                
                for name, model in models.items():
                    try:
                        if name == 'Linear Regression':
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                        
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        results['model_performance'][name] = {
                            'r2_score': r2,
                            'mse': mse,
                            'rmse': np.sqrt(mse),
                            'score_type': 'r2'
                        }
                        
                        if r2 > best_score:
                            best_score = r2
                            best_model = model
                    except Exception as e:
                        results['model_performance'][name] = {'error': str(e)}
                
                # Feature importance from best model
                if best_model and hasattr(best_model, 'feature_importances_'):
                    importance = dict(zip(feature_cols, best_model.feature_importances_))
                    results['feature_importance'] = dict(sorted(importance.items(), 
                                                               key=lambda x: x[1], reverse=True))
            
            return results
            
        except Exception as e:
            return {'error': f"Predictive modeling failed: {str(e)}"}
    
    def clustering_analysis(self, df: pd.DataFrame, features: List[str] = None,
                          n_clusters: int = None) -> Dict[str, Any]:
        """Perform clustering analysis"""
        try:
            if features is None:
                features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not features:
                return {'error': "No numeric features found for clustering"}
            
            # Prepare data
            df_cluster = df[features].dropna()
            
            if len(df_cluster) < 5:
                return {'error': "Insufficient data for clustering (need at least 5 rows)"}
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_cluster)
            
            results = {
                'features_used': features,
                'data_shape': df_cluster.shape,
                'optimal_clusters': {},
                'cluster_analysis': {}
            }
            
            # Determine optimal number of clusters using elbow method
            if n_clusters is None:
                max_clusters = min(10, len(df_cluster) - 1)
                inertias = []
                silhouette_scores = []
                
                for k in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(X_scaled)
                    inertias.append(kmeans.inertia_)
                    
                    # Calculate silhouette score if possible
                    try:
                        from sklearn.metrics import silhouette_score
                        score = silhouette_score(X_scaled, kmeans.labels_)
                        silhouette_scores.append(score)
                    except:
                        silhouette_scores.append(0)
                
                # Find elbow point (simplified)
                if len(inertias) > 2:
                    diffs = np.diff(inertias)
                    optimal_k = np.argmax(diffs) + 2  # +2 because we start from k=2
                else:
                    optimal_k = 3
                
                results['optimal_clusters'] = {
                    'recommended': optimal_k,
                    'inertias': inertias,
                    'silhouette_scores': silhouette_scores
                }
                n_clusters = optimal_k
            
            # Perform final clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to original dataframe
            df_with_clusters = df_cluster.copy()
            df_with_clusters['Cluster'] = cluster_labels
            
            # Analyze clusters
            cluster_stats = {}
            for cluster_id in range(n_clusters):
                cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
                cluster_stats[f'Cluster_{cluster_id}'] = {
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(df_with_clusters) * 100,
                    'centers': cluster_data[features].mean().to_dict(),
                    'characteristics': self._describe_cluster(cluster_data, features)
                }
            
            results['cluster_analysis'] = {
                'n_clusters': n_clusters,
                'cluster_stats': cluster_stats,
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'labels': cluster_labels.tolist()
            }
            
            return results
            
        except Exception as e:
            return {'error': f"Clustering analysis failed: {str(e)}"}
    
    def anomaly_detection(self, df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Any]:
        """Detect anomalies in the dataset"""
        try:
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not columns:
                return {'error': "No numeric columns found for anomaly detection"}
            
            results = {
                'methods_used': [],
                'anomalies_by_method': {},
                'summary': {}
            }
            
            df_numeric = df[columns].dropna()
            
            # Method 1: IQR-based detection
            iqr_anomalies = []
            for col in columns:
                Q1 = df_numeric[col].quantile(0.25)
                Q3 = df_numeric[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                anomaly_indices = df_numeric[
                    (df_numeric[col] < lower_bound) | (df_numeric[col] > upper_bound)
                ].index.tolist()
                
                iqr_anomalies.extend(anomaly_indices)
            
            iqr_anomalies = list(set(iqr_anomalies))  # Remove duplicates
            results['anomalies_by_method']['IQR'] = iqr_anomalies
            results['methods_used'].append('IQR')
            
            # Method 2: Z-score based detection
            z_scores = np.abs(stats.zscore(df_numeric))
            z_anomalies = df_numeric[(z_scores > 3).any(axis=1)].index.tolist()
            results['anomalies_by_method']['Z-Score'] = z_anomalies
            results['methods_used'].append('Z-Score')
            
            # Summary
            all_anomalies = list(set(iqr_anomalies + z_anomalies))
            results['summary'] = {
                'total_anomalies': len(all_anomalies),
                'percentage': len(all_anomalies) / len(df) * 100,
                'consensus_anomalies': list(set(iqr_anomalies).intersection(set(z_anomalies))),
                'most_anomalous_features': self._find_most_anomalous_features(df_numeric, all_anomalies)
            }
            
            return results
            
        except Exception as e:
            return {'error': f"Anomaly detection failed: {str(e)}"}
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, 
                                threshold: float = 0.7) -> List[Dict]:
        """Find strongly correlated pairs"""
        strong_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_corr.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'Very Strong' if abs(corr_value) > 0.9 else 'Strong'
                    })
        
        return sorted(strong_corr, key=lambda x: abs(x['correlation']), reverse=True)
    
    def _describe_cluster(self, cluster_data: pd.DataFrame, features: List[str]) -> Dict[str, str]:
        """Generate human-readable cluster characteristics"""
        characteristics = {}
        
        for feature in features:
            mean_val = cluster_data[feature].mean()
            global_mean = cluster_data[feature].mean()  # This should be global mean, simplified here
            
            if mean_val > global_mean * 1.2:
                characteristics[feature] = "High"
            elif mean_val < global_mean * 0.8:
                characteristics[feature] = "Low"
            else:
                characteristics[feature] = "Average"
        
        return characteristics
    
    def _find_most_anomalous_features(self, df: pd.DataFrame, 
                                     anomaly_indices: List[int]) -> List[str]:
        """Find features that contribute most to anomalies"""
        if not anomaly_indices:
            return []
        
        anomaly_data = df.loc[anomaly_indices]
        feature_anomaly_counts = {}
        
        for col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            feature_anomalies = anomaly_data[
                (anomaly_data[col] < lower_bound) | (anomaly_data[col] > upper_bound)
            ]
            
            feature_anomaly_counts[col] = len(feature_anomalies)
        
        # Sort by anomaly count
        sorted_features = sorted(feature_anomaly_counts.items(), 
                               key=lambda x: x[1], reverse=True)
        
        return [feature for feature, count in sorted_features[:5] if count > 0]
    
    def render_analytics_dashboard(self, df: pd.DataFrame):
        """Render advanced analytics dashboard"""
        st.subheader("🧠 Advanced Analytics")
        
        if df is None or df.empty:
            st.warning("Please upload data to use advanced analytics features.")
            return
        
        # Analytics options
        analytics_type = st.selectbox(
            "Choose Analysis Type:",
            ["Statistical Analysis", "Predictive Modeling", "Clustering", 
             "Anomaly Detection", "Time Series Analysis"]
        )
        
        if analytics_type == "Statistical Analysis":
            self._render_statistical_analysis(df)
        elif analytics_type == "Predictive Modeling":
            self._render_predictive_modeling(df)
        elif analytics_type == "Clustering":
            self._render_clustering_analysis(df)
        elif analytics_type == "Anomaly Detection":
            self._render_anomaly_detection(df)
        elif analytics_type == "Time Series Analysis":
            self._render_time_series_analysis(df)
    
    def _render_statistical_analysis(self, df: pd.DataFrame):
        """Render statistical analysis interface"""
        st.write("**📊 Statistical Analysis**")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = st.multiselect("Select columns for analysis:", numeric_cols, default=numeric_cols[:3])
        
        if st.button("Run Statistical Analysis") and selected_cols:
            with st.spinner("Performing statistical analysis..."):
                results = self.statistical_analysis(df, selected_cols)
                
                # Display results
                if results.get('descriptive_stats'):
                    st.write("**Descriptive Statistics:**")
                    stats_df = pd.DataFrame(results['descriptive_stats']).T
                    st.dataframe(stats_df)
                
                if results.get('correlation_analysis', {}).get('strong_correlations'):
                    st.write("**Strong Correlations:**")
                    for corr in results['correlation_analysis']['strong_correlations']:
                        st.write(f"• {corr['feature1']} ↔ {corr['feature2']}: {corr['correlation']:.3f} ({corr['strength']})")
    
    def _render_predictive_modeling(self, df: pd.DataFrame):
        """Render predictive modeling interface"""
        st.write("**🤖 Predictive Modeling**")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = df.columns.tolist()
        
        target_col = st.selectbox("Select target variable:", all_cols)
        feature_cols = st.multiselect("Select features:", [col for col in numeric_cols if col != target_col])
        
        if st.button("Build Model") and target_col and feature_cols:
            with st.spinner("Building predictive model..."):
                results = self.predictive_modeling(df, target_col, feature_cols)
                
                if 'error' in results:
                    st.error(results['error'])
                else:
                    st.success(f"Model built successfully! Problem type: {results['problem_type']}")
                    
                    # Display performance
                    if results.get('model_performance'):
                        st.write("**Model Performance:**")
                        for model, performance in results['model_performance'].items():
                            if 'error' not in performance:
                                score_type = performance.get('score_type', 'score')
                                main_score = performance.get('accuracy') or performance.get('r2_score')
                                st.write(f"• {model}: {score_type} = {main_score:.3f}")
                    
                    # Feature importance
                    if results.get('feature_importance'):
                        st.write("**Feature Importance:**")
                        importance_df = pd.DataFrame(list(results['feature_importance'].items()), 
                                                   columns=['Feature', 'Importance'])
                        st.bar_chart(importance_df.set_index('Feature'))
    
    def _render_clustering_analysis(self, df: pd.DataFrame):
        """Render clustering analysis interface"""
        st.write("**🎯 Clustering Analysis**")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = st.multiselect("Select features for clustering:", numeric_cols, default=numeric_cols[:3])
        n_clusters = st.slider("Number of clusters (0 = auto-detect):", 0, 10, 0)
        
        if st.button("Run Clustering") and selected_features:
            with st.spinner("Performing clustering analysis..."):
                results = self.clustering_analysis(df, selected_features, n_clusters if n_clusters > 0 else None)
                
                if 'error' in results:
                    st.error(results['error'])
                else:
                    st.success("Clustering completed successfully!")
                    
                    # Display results
                    if results.get('optimal_clusters'):
                        st.write(f"**Recommended clusters:** {results['optimal_clusters']['recommended']}")
                    
                    if results.get('cluster_analysis'):
                        st.write("**Cluster Summary:**")
                        for cluster_id, stats in results['cluster_analysis']['cluster_stats'].items():
                            st.write(f"• {cluster_id}: {stats['size']} points ({stats['percentage']:.1f}%)")
    
    def _render_anomaly_detection(self, df: pd.DataFrame):
        """Render anomaly detection interface"""
        st.write("**🚨 Anomaly Detection**")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = st.multiselect("Select columns for anomaly detection:", numeric_cols, default=numeric_cols)
        
        if st.button("Detect Anomalies") and selected_cols:
            with st.spinner("Detecting anomalies..."):
                results = self.anomaly_detection(df, selected_cols)
                
                if 'error' in results:
                    st.error(results['error'])
                else:
                    summary = results['summary']
                    st.success(f"Found {summary['total_anomalies']} anomalies ({summary['percentage']:.1f}% of data)")
                    
                    if summary['total_anomalies'] > 0:
                        st.write("**Anomaly Summary:**")
                        for method, anomalies in results['anomalies_by_method'].items():
                            st.write(f"• {method} method: {len(anomalies)} anomalies")
                        
                        if summary['most_anomalous_features']:
                            st.write(f"**Most anomalous features:** {', '.join(summary['most_anomalous_features'])}")
    
    def _render_time_series_analysis(self, df: pd.DataFrame):
        """Render time series analysis interface"""
        st.write("**📈 Time Series Analysis**")
        
        # Detect potential date columns
        date_candidates = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_candidates.append(col)
        
        date_col = st.selectbox("Select date column:", date_candidates + df.columns.tolist())
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        value_col = st.selectbox("Select value column:", numeric_cols)
        
        if st.button("Analyze Time Series") and date_col and value_col:
            with st.spinner("Analyzing time series..."):
                results = self.time_series_analysis(df, date_col, value_col)
                
                if 'error' in results:
                    st.error(results['error'])
                else:
                    st.success("Time series analysis completed!")
                    
                    # Display trend analysis
                    if results.get('trend_analysis'):
                        trend = results['trend_analysis']
                        st.write(f"**Trend:** {trend['direction']} (strength: {trend['strength']:.3f})")
                    
                    # Display forecast
                    if results.get('forecast'):
                        forecast = results['forecast']
                        st.write(f"**Forecast confidence:** {forecast['confidence']}")
                        if forecast['values']:
                            st.write("**Next 5 predictions:**")
                            for date, value in zip(forecast['dates'], forecast['values']):
                                st.write(f"• {date}: {value:.2f}")
                    
                    # Display anomalies
                    if results.get('anomalies'):
                        st.write(f"**Anomalous dates:** {len(results['anomalies'])} detected")
                        if results['anomalies']:
                            st.write(f"Examples: {', '.join(results['anomalies'][:5])}")