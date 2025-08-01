# multi_file_manager.py

import pandas as pd
import streamlit as st
import os
import tempfile
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime

class MultiFileManager:
    """Manages multiple CSV files and enables cross-file analysis"""
    
    def __init__(self):
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = {}
        if "file_metadata" not in st.session_state:
            st.session_state.file_metadata = {}
        if "active_files" not in st.session_state:
            st.session_state.active_files = []
    
    def add_file(self, uploaded_file, file_name: str = None) -> str:
        """Add a new CSV file to the manager"""
        if file_name is None:
            file_name = uploaded_file.name
        
        # Create unique identifier for the file
        file_id = f"{file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name
        
        # Load and analyze the file
        try:
            df = pd.read_csv(temp_path)
            
            # Store file information
            st.session_state.uploaded_files[file_id] = {
                'name': file_name,
                'path': temp_path,
                'dataframe': df,
                'upload_time': datetime.now(),
                'rows': len(df),
                'columns': list(df.columns),
                'size_mb': os.path.getsize(temp_path) / 1024 / 1024
            }
            
            # Add to active files
            if file_id not in st.session_state.active_files:
                st.session_state.active_files.append(file_id)
            
            return file_id
            
        except Exception as e:
            st.error(f"Error loading file {file_name}: {str(e)}")
            return None
    
    def remove_file(self, file_id: str) -> bool:
        """Remove a file from the manager"""
        try:
            if file_id in st.session_state.uploaded_files:
                # Clean up temporary file
                file_info = st.session_state.uploaded_files[file_id]
                if os.path.exists(file_info['path']):
                    os.unlink(file_info['path'])
                
                # Remove from session state
                del st.session_state.uploaded_files[file_id]
                if file_id in st.session_state.active_files:
                    st.session_state.active_files.remove(file_id)
                
                return True
        except Exception as e:
            st.error(f"Error removing file: {str(e)}")
        
        return False
    
    def get_file_dataframe(self, file_id: str) -> Optional[pd.DataFrame]:
        """Get dataframe for a specific file"""
        if file_id in st.session_state.uploaded_files:
            return st.session_state.uploaded_files[file_id]['dataframe']
        return None
    
    def get_all_files(self) -> Dict[str, Dict]:
        """Get all uploaded files information"""
        return st.session_state.uploaded_files
    
    def get_active_files(self) -> List[str]:
        """Get list of active file IDs"""
        return st.session_state.active_files
    
    def compare_files(self, file_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple files and find relationships"""
        if len(file_ids) < 2:
            return {"error": "Need at least 2 files to compare"}
        
        comparison = {
            'files': [],
            'common_columns': [],
            'schema_differences': [],
            'data_overlap': {},
            'merge_suggestions': []
        }
        
        dataframes = {}
        all_columns = set()
        
        # Collect information about each file
        for file_id in file_ids:
            if file_id in st.session_state.uploaded_files:
                file_info = st.session_state.uploaded_files[file_id]
                df = file_info['dataframe']
                dataframes[file_id] = df
                
                file_summary = {
                    'id': file_id,
                    'name': file_info['name'],
                    'rows': len(df),
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.to_dict()
                }
                comparison['files'].append(file_summary)
                all_columns.update(df.columns)
        
        # Find common columns
        if len(dataframes) > 1:
            common_cols = set(list(dataframes.values())[0].columns)
            for df in list(dataframes.values())[1:]:
                common_cols = common_cols.intersection(set(df.columns))
            comparison['common_columns'] = list(common_cols)
        
        # Suggest merge operations
        merge_suggestions = self._suggest_merges(dataframes)
        comparison['merge_suggestions'] = merge_suggestions
        
        return comparison
    
    def merge_files(self, file_ids: List[str], merge_type: str = "inner", 
                   join_columns: List[str] = None) -> Optional[pd.DataFrame]:
        """Merge multiple files based on specified criteria"""
        if len(file_ids) < 2:
            return None
        
        dataframes = []
        for file_id in file_ids:
            df = self.get_file_dataframe(file_id)
            if df is not None:
                # Add file identifier column
                df_copy = df.copy()
                df_copy['_source_file'] = st.session_state.uploaded_files[file_id]['name']
                dataframes.append(df_copy)
        
        if len(dataframes) < 2:
            return None
        
        try:
            # If no join columns specified, try to auto-detect
            if not join_columns:
                join_columns = self._auto_detect_join_columns(dataframes)
            
            if not join_columns:
                # If no common columns for joining, concatenate
                return pd.concat(dataframes, ignore_index=True, sort=False)
            
            # Perform merge operation
            result = dataframes[0]
            for df in dataframes[1:]:
                result = pd.merge(result, df, on=join_columns, how=merge_type, suffixes=('', '_right'))
            
            return result
            
        except Exception as e:
            st.error(f"Error merging files: {str(e)}")
            return None
    
    def _suggest_merges(self, dataframes: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Suggest possible merge operations between files"""
        suggestions = []
        file_ids = list(dataframes.keys())
        
        for i, file_id1 in enumerate(file_ids):
            for file_id2 in file_ids[i+1:]:
                df1 = dataframes[file_id1]
                df2 = dataframes[file_id2]
                
                # Find common columns
                common_cols = set(df1.columns).intersection(set(df2.columns))
                
                # Look for potential key columns
                potential_keys = []
                for col in common_cols:
                    # Check if column could be a key (high uniqueness, similar data types)
                    if (df1[col].nunique() / len(df1) > 0.7 and 
                        df2[col].nunique() / len(df2) > 0.7):
                        potential_keys.append(col)
                
                if potential_keys:
                    file1_name = st.session_state.uploaded_files[file_id1]['name']
                    file2_name = st.session_state.uploaded_files[file_id2]['name']
                    
                    suggestions.append({
                        'file1': {'id': file_id1, 'name': file1_name},
                        'file2': {'id': file_id2, 'name': file2_name},
                        'suggested_keys': potential_keys,
                        'common_columns': list(common_cols),
                        'merge_type': 'inner'
                    })
        
        return suggestions
    
    def _auto_detect_join_columns(self, dataframes: List[pd.DataFrame]) -> List[str]:
        """Auto-detect columns suitable for joining"""
        if len(dataframes) < 2:
            return []
        
        # Find columns that exist in all dataframes
        common_columns = set(dataframes[0].columns)
        for df in dataframes[1:]:
            common_columns = common_columns.intersection(set(df.columns))
        
        # Filter to columns that could be keys
        potential_keys = []
        for col in common_columns:
            # Check if the column has reasonable uniqueness in all dataframes
            is_potential_key = True
            for df in dataframes:
                uniqueness = df[col].nunique() / len(df)
                if uniqueness < 0.1:  # Less than 10% unique values
                    is_potential_key = False
                    break
            
            if is_potential_key:
                potential_keys.append(col)
        
        return potential_keys
    
    def render_file_manager_ui(self):
        """Render the multi-file manager interface"""
        st.subheader("📁 File Manager")
        
        uploaded_files = self.get_all_files()
        
        if not uploaded_files:
            st.info("No files uploaded yet. Upload files using the sidebar.")
            return
        
        # File list with actions
        for file_id, file_info in uploaded_files.items():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.write(f"**{file_info['name']}**")
                st.caption(f"{file_info['rows']:,} rows × {len(file_info['columns'])} columns")
            
            with col2:
                st.write(f"{file_info['size_mb']:.1f} MB")
                st.caption(f"Uploaded: {file_info['upload_time'].strftime('%H:%M:%S')}")
            
            with col3:
                if st.button("Preview", key=f"preview_{file_id}"):
                    with st.expander(f"Preview: {file_info['name']}", expanded=True):
                        st.dataframe(file_info['dataframe'].head())
            
            with col4:
                if st.button("🗑️", key=f"delete_{file_id}", help="Delete file"):
                    self.remove_file(file_id)
                    st.rerun()
        
        # File comparison and merging
        if len(uploaded_files) > 1:
            st.subheader("🔄 File Operations")
            
            # File selection for comparison
            file_options = {file_id: info['name'] for file_id, info in uploaded_files.items()}
            selected_files = st.multiselect(
                "Select files to compare/merge:",
                options=list(file_options.keys()),
                format_func=lambda x: file_options[x]
            )
            
            if len(selected_files) > 1:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Compare Files"):
                        comparison = self.compare_files(selected_files)
                        self._display_comparison_results(comparison)
                
                with col2:
                    if st.button("Merge Files"):
                        merged_df = self.merge_files(selected_files)
                        if merged_df is not None:
                            st.success(f"Merged {len(selected_files)} files successfully!")
                            st.dataframe(merged_df.head())
                            
                            # Option to save merged file
                            if st.button("Add Merged File to Manager"):
                                # Create a temporary file for the merged data
                                with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
                                    merged_df.to_csv(tmp_file.name, index=False)
                                    
                                    # Add to file manager
                                    merged_file_id = f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                    st.session_state.uploaded_files[merged_file_id] = {
                                        'name': f"Merged_File_{datetime.now().strftime('%H%M%S')}",
                                        'path': tmp_file.name,
                                        'dataframe': merged_df,
                                        'upload_time': datetime.now(),
                                        'rows': len(merged_df),
                                        'columns': list(merged_df.columns),
                                        'size_mb': os.path.getsize(tmp_file.name) / 1024 / 1024
                                    }
                                    st.session_state.active_files.append(merged_file_id)
                                    st.rerun()
    
    def _display_comparison_results(self, comparison: Dict[str, Any]):
        """Display file comparison results"""
        st.subheader("📊 File Comparison Results")
        
        # Basic comparison
        if comparison.get('files'):
            st.write("**File Overview:**")
            for file_info in comparison['files']:
                st.write(f"- {file_info['name']}: {file_info['rows']:,} rows × {len(file_info['columns'])} columns")
        
        # Common columns
        if comparison.get('common_columns'):
            st.write(f"**Common Columns:** {', '.join(comparison['common_columns'])}")
        else:
            st.warning("No common columns found between files.")
        
        # Merge suggestions
        if comparison.get('merge_suggestions'):
            st.write("**Merge Suggestions:**")
            for suggestion in comparison['merge_suggestions']:
                st.info(
                    f"Join {suggestion['file1']['name']} with {suggestion['file2']['name']} "
                    f"using: {', '.join(suggestion['suggested_keys'])}"
                )