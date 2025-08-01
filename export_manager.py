# export_manager.py

import pandas as pd
import streamlit as st
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import base64
from typing import Dict, List, Any, Optional
import json

class ExportManager:
    """Handles exporting analysis results and generating reports"""
    
    def __init__(self):
        self.report_templates = {
            'executive_summary': 'Executive Summary Report',
            'detailed_analysis': 'Detailed Analysis Report', 
            'data_quality': 'Data Quality Assessment',
            'custom': 'Custom Report'
        }
    
    def generate_pdf_report(self, analysis_data: Dict[str, Any], 
                          conversation_history: List[Dict], 
                          file_info: Dict[str, Any],
                          report_type: str = 'detailed_analysis') -> bytes:
        """Generate a comprehensive PDF report"""
        
        class AnalysisReport(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 15)
                self.cell(0, 10, 'The Analyst - Data Analysis Report', 0, 1, 'C')
                self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
                self.ln(10)
            
            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
                self.cell(0, 10, 'Powered by RainCode AI', 0, 0, 'R')
        
        pdf = AnalysisReport()
        pdf.add_page()
        
        # Executive Summary
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Executive Summary', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        
        summary_text = f"""
        Dataset: {file_info.get('name', 'Unknown')}
        Records: {file_info.get('rows', 'N/A'):,}
        Columns: {len(file_info.get('columns', [])):,}
        Analysis Date: {datetime.now().strftime('%B %d, %Y')}
        Questions Analyzed: {len([msg for msg in conversation_history if msg['role'] == 'user'])}
        """
        
        for line in summary_text.strip().split('\n'):
            if line.strip():
                pdf.cell(0, 6, line.strip(), 0, 1, 'L')
        
        pdf.ln(5)
        
        # Data Overview
        if analysis_data.get('data_profile'):
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Data Quality Assessment', 0, 1, 'L')
            pdf.set_font('Arial', '', 10)
            
            profile = analysis_data['data_profile']
            quality_text = f"""
            Overall Quality Score: {profile.get('quality', {}).get('overall_score', 'N/A')}%
            Missing Data: {profile.get('quality', {}).get('missing_percentage', 'N/A')}%
            Duplicate Rows: {profile.get('quality', {}).get('duplicate_rows', 'N/A')}
            """
            
            for line in quality_text.strip().split('\n'):
                if line.strip():
                    pdf.cell(0, 6, line.strip(), 0, 1, 'L')
        
        pdf.ln(5)
        
        # Analysis History
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Analysis Questions & Insights', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        
        qa_pairs = []
        current_question = None
        
        for msg in conversation_history:
            if msg['role'] == 'user':
                current_question = msg['content']
            elif msg['role'] == 'assistant' and current_question:
                qa_pairs.append((current_question, msg['content']))
                current_question = None
        
        for i, (question, answer) in enumerate(qa_pairs[-10:], 1):  # Last 10 Q&A pairs
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 8, f'Q{i}: {question[:80]}{"..." if len(question) > 80 else ""}', 0, 1, 'L')
            pdf.set_font('Arial', '', 9)
            
            # Clean and truncate answer
            clean_answer = answer.replace('**', '').replace('*', '')[:300]
            if len(answer) > 300:
                clean_answer += "..."
            
            # Split answer into lines that fit the page
            words = clean_answer.split()
            lines = []
            current_line = ""
            
            for word in words:
                if len(current_line + word) < 80:
                    current_line += word + " "
                else:
                    lines.append(current_line.strip())
                    current_line = word + " "
            if current_line:
                lines.append(current_line.strip())
            
            for line in lines:
                pdf.cell(0, 5, line, 0, 1, 'L')
            
            pdf.ln(3)
        
        # Recommendations
        if analysis_data.get('suggestions'):
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Recommendations', 0, 1, 'L')
            pdf.set_font('Arial', '', 10)
            
            for suggestion in analysis_data['suggestions'][:10]:
                pdf.cell(0, 6, f'• {suggestion}', 0, 1, 'L')
        
        return bytes(pdf.output(dest='S'))
    
    def generate_excel_export(self, dataframes: Dict[str, pd.DataFrame], 
                            analysis_summary: Dict[str, Any]) -> bytes:
        """Generate Excel file with multiple sheets"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write each dataframe to a separate sheet
            for sheet_name, df in dataframes.items():
                # Clean sheet name for Excel compatibility
                clean_name = sheet_name.replace('/', '_').replace('\\', '_')[:31]
                df.to_excel(writer, sheet_name=clean_name, index=False)
            
            # Create summary sheet
            if analysis_summary:
                summary_df = pd.DataFrame([
                    ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    ['Total Datasets', len(dataframes)],
                    ['Generated By', 'The Analyst - RainCode AI']
                ], columns=['Metric', 'Value'])
                
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        return output.getvalue()
    
    def generate_code_export(self, conversation_history: List[Dict], 
                           analysis_type: str = 'python') -> str:
        """Generate equivalent code for the analysis performed"""
        
        if analysis_type == 'python':
            code_lines = [
                "# Generated Python code for data analysis",
                "# Powered by RainCode AI - The Analyst",
                "",
                "import pandas as pd",
                "import numpy as np",
                "import matplotlib.pyplot as plt",
                "import seaborn as sns",
                "",
                "# Load your data",
                "df = pd.read_csv('your_data.csv')",
                "",
                "# Basic data exploration",
                "print('Dataset shape:', df.shape)",
                "print('\\nColumn info:')",
                "print(df.info())",
                "print('\\nBasic statistics:')",
                "print(df.describe())",
                ""
            ]
            
            # Extract analysis commands from conversation
            user_queries = [msg['content'] for msg in conversation_history if msg['role'] == 'user']
            
            for i, query in enumerate(user_queries[-5:], 1):  # Last 5 queries
                code_lines.append(f"# Analysis {i}: {query}")
                
                # Generate relevant code based on query keywords
                query_lower = query.lower()
                
                if 'average' in query_lower or 'mean' in query_lower:
                    code_lines.append("# Calculate averages")
                    code_lines.append("numeric_columns = df.select_dtypes(include=[np.number]).columns")
                    code_lines.append("averages = df[numeric_columns].mean()")
                    code_lines.append("print(averages)")
                
                elif 'correlation' in query_lower:
                    code_lines.append("# Calculate correlation matrix")
                    code_lines.append("correlation_matrix = df.corr()")
                    code_lines.append("sns.heatmap(correlation_matrix, annot=True)")
                    code_lines.append("plt.show()")
                
                elif 'top' in query_lower or 'highest' in query_lower:
                    code_lines.append("# Find top values")
                    code_lines.append("# Replace 'column_name' with your target column")
                    code_lines.append("top_values = df.nlargest(10, 'column_name')")
                    code_lines.append("print(top_values)")
                
                elif 'missing' in query_lower or 'null' in query_lower:
                    code_lines.append("# Check for missing values")
                    code_lines.append("missing_data = df.isnull().sum()")
                    code_lines.append("print(missing_data)")
                
                code_lines.append("")
            
            return "\n".join(code_lines)
        
        return "# Code generation not supported for this analysis type"
    
    def create_shareable_link(self, analysis_session: Dict[str, Any]) -> str:
        """Create a shareable link for the analysis session"""
        # In a real implementation, this would save to a database and return a URL
        # For now, we'll create a JSON export
        
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'conversation': analysis_session.get('conversation', []),
            'file_info': analysis_session.get('file_info', {}),
            'analysis_results': analysis_session.get('analysis_results', {}),
            'powered_by': 'RainCode AI - The Analyst'
        }
        
        # Encode as base64 for sharing
        json_str = json.dumps(session_data, default=str)
        encoded = base64.b64encode(json_str.encode()).decode()
        
        return f"analyst_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{encoded[:20]}"
    
    def render_export_interface(self, conversation_history: List[Dict], 
                               file_info: Dict[str, Any],
                               analysis_data: Dict[str, Any]):
        """Render the export interface in Streamlit"""
        st.subheader("📤 Export & Reporting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Report Generation**")
            
            report_type = st.selectbox(
                "Report Type:",
                options=list(self.report_templates.keys()),
                format_func=lambda x: self.report_templates[x]
            )
            
            if st.button("Generate PDF Report"):
                try:
                    pdf_bytes = self.generate_pdf_report(
                        analysis_data, conversation_history, file_info, report_type
                    )
                    
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                    st.success("PDF report generated successfully!")
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
        
        with col2:
            st.write("**💻 Code Export**")
            
            code_type = st.selectbox("Code Type:", ["python", "sql", "r"])
            
            if st.button("Generate Code"):
                try:
                    code = self.generate_code_export(conversation_history, code_type)
                    st.code(code, language=code_type)
                    
                    st.download_button(
                        label="📝 Download Code",
                        data=code,
                        file_name=f"analysis_code_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{code_type}",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Error generating code: {str(e)}")
        
        # Excel Export
        st.write("**📊 Data Export**")
        if file_info.get('dataframe') is not None:
            if st.button("Generate Excel Export"):
                try:
                    dataframes = {'main_data': file_info['dataframe']}
                    excel_bytes = self.generate_excel_export(dataframes, analysis_data)
                    
                    st.download_button(
                        label="📊 Download Excel File",
                        data=excel_bytes,
                        file_name=f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    st.success("Excel file generated successfully!")
                except Exception as e:
                    st.error(f"Error generating Excel: {str(e)}")
        
        # Shareable Session
        st.write("**🔗 Share Analysis**")
        if st.button("Create Shareable Session"):
            try:
                session_data = {
                    'conversation': conversation_history,
                    'file_info': {k: v for k, v in file_info.items() if k != 'dataframe'},
                    'analysis_results': analysis_data
                }
                
                share_link = self.create_shareable_link(session_data)
                st.success("Shareable session created!")
                st.code(f"Session ID: {share_link}")
                st.info("Save this Session ID to restore this analysis later.")
            except Exception as e:
                st.error(f"Error creating shareable session: {str(e)}")