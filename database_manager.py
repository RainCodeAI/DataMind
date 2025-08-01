# database_manager.py

import streamlit as st
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text, inspect
from typing import Dict, List, Any, Optional, Tuple
import os
from datetime import datetime

class DatabaseManager:
    """Manages database connections and operations for The Analyst"""
    
    def __init__(self):
        self.connections = {}
        self.current_connection = None
        if "db_connections" not in st.session_state:
            st.session_state.db_connections = {}
        if "db_schemas" not in st.session_state:
            st.session_state.db_schemas = {}
    
    def create_connection(self, connection_name: str, connection_string: str) -> bool:
        """Create a new database connection"""
        try:
            engine = create_engine(connection_string)
            # Test the connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Store connection info
            st.session_state.db_connections[connection_name] = {
                'engine': engine,
                'connection_string': connection_string,
                'created_at': datetime.now(),
                'status': 'connected'
            }
            
            # Load schema information
            self._load_schema_info(connection_name, engine)
            
            return True
            
        except Exception as e:
            st.error(f"Failed to connect to database: {str(e)}")
            return False
    
    def get_default_connection(self) -> Optional[str]:
        """Get default PostgreSQL connection if available"""
        if all(key in os.environ for key in ['DATABASE_URL', 'PGHOST', 'PGPORT', 'PGUSER', 'PGPASSWORD', 'PGDATABASE']):
            connection_name = "Default PostgreSQL"
            if connection_name not in st.session_state.db_connections:
                database_url = os.environ.get('DATABASE_URL')
                if database_url and self.create_connection(connection_name, database_url):
                    return connection_name
        return None
    
    def _load_schema_info(self, connection_name: str, engine) -> None:
        """Load database schema information"""
        try:
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            
            schema_info = {
                'tables': {},
                'total_tables': len(tables)
            }
            
            # Get table details
            for table_name in tables[:20]:  # Limit to first 20 tables for performance
                try:
                    columns = inspector.get_columns(table_name)
                    column_info = [
                        {
                            'name': col['name'],
                            'type': str(col['type']),
                            'nullable': col.get('nullable', True)
                        } for col in columns
                    ]
                    
                    # Get row count
                    with engine.connect() as conn:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                        row_count = result.scalar()
                    
                    schema_info['tables'][table_name] = {
                        'columns': column_info,
                        'row_count': row_count,
                        'column_count': len(column_info)
                    }
                except Exception as e:
                    schema_info['tables'][table_name] = {
                        'error': str(e),
                        'columns': [],
                        'row_count': 0
                    }
            
            st.session_state.db_schemas[connection_name] = schema_info
            
        except Exception as e:
            st.session_state.db_schemas[connection_name] = {
                'error': str(e),
                'tables': {},
                'total_tables': 0
            }
    
    def execute_query(self, connection_name: str, query: str) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame"""
        if connection_name not in st.session_state.db_connections:
            raise ValueError(f"Connection '{connection_name}' not found")
        
        engine = st.session_state.db_connections[connection_name]['engine']
        
        try:
            # Execute query and return DataFrame
            df = pd.read_sql_query(query, engine)
            return df
            
        except Exception as e:
            raise Exception(f"Query execution failed: {str(e)}")
    
    def get_table_data(self, connection_name: str, table_name: str, limit: int = 1000) -> pd.DataFrame:
        """Get data from a specific table"""
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return self.execute_query(connection_name, query)
    
    def get_table_sample(self, connection_name: str, table_name: str, sample_size: int = 100) -> pd.DataFrame:
        """Get a sample of data from a table"""
        query = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {sample_size}"
        return self.execute_query(connection_name, query)
    
    def get_connections(self) -> Dict[str, Dict]:
        """Get all available database connections"""
        return st.session_state.db_connections
    
    def get_schema_info(self, connection_name: str) -> Dict:
        """Get schema information for a connection"""
        return st.session_state.db_schemas.get(connection_name, {})
    
    def test_connection(self, connection_name: str) -> bool:
        """Test if a connection is still working"""
        if connection_name not in st.session_state.db_connections:
            return False
        
        try:
            engine = st.session_state.db_connections[connection_name]['engine']
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except:
            return False
    
    def close_connection(self, connection_name: str) -> bool:
        """Close a database connection"""
        if connection_name in st.session_state.db_connections:
            try:
                engine = st.session_state.db_connections[connection_name]['engine']
                engine.dispose()
                del st.session_state.db_connections[connection_name]
                if connection_name in st.session_state.db_schemas:
                    del st.session_state.db_schemas[connection_name]
                return True
            except Exception as e:
                st.error(f"Error closing connection: {str(e)}")
                return False
        return False
    
    def suggest_queries(self, connection_name: str, table_name: str) -> List[str]:
        """Suggest useful queries for a table"""
        schema_info = self.get_schema_info(connection_name)
        
        if table_name not in schema_info.get('tables', {}):
            return []
        
        table_info = schema_info['tables'][table_name]
        columns = [col['name'] for col in table_info['columns']]
        
        suggestions = [
            f"SELECT * FROM {table_name} LIMIT 10",
            f"SELECT COUNT(*) as total_rows FROM {table_name}"
        ]
        
        # Add column-specific suggestions
        numeric_columns = [col['name'] for col in table_info['columns'] 
                          if any(t in str(col['type']).lower() for t in ['int', 'float', 'numeric', 'decimal'])]
        
        if numeric_columns:
            suggestions.extend([
                f"SELECT AVG({numeric_columns[0]}) as average_{numeric_columns[0]} FROM {table_name}",
                f"SELECT MIN({numeric_columns[0]}), MAX({numeric_columns[0]}) FROM {table_name}"
            ])
        
        # Date columns
        date_columns = [col['name'] for col in table_info['columns'] 
                       if any(t in str(col['type']).lower() for t in ['date', 'time', 'timestamp'])]
        
        if date_columns:
            suggestions.append(f"SELECT DATE({date_columns[0]}) as date, COUNT(*) FROM {table_name} GROUP BY DATE({date_columns[0]}) ORDER BY date DESC LIMIT 10")
        
        return suggestions[:6]  # Return top 6 suggestions
    
    def render_database_interface(self):
        """Render the database management interface"""
        st.subheader("🗄️ Database Connections")
        
        # Auto-connect to default database if available
        default_conn = self.get_default_connection()
        if default_conn and default_conn not in [conn for conn in st.session_state.db_connections.keys()]:
            st.success(f"Connected to {default_conn}")
        
        # Connection management
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**🔌 Available Connections:**")
            
            if st.session_state.db_connections:
                for conn_name, conn_info in st.session_state.db_connections.items():
                    with st.expander(f"📊 {conn_name} ({'Connected' if self.test_connection(conn_name) else 'Disconnected'})"):
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.write(f"**Created:** {conn_info['created_at'].strftime('%Y-%m-%d %H:%M')}")
                        
                        with col_b:
                            schema_info = self.get_schema_info(conn_name)
                            table_count = schema_info.get('total_tables', 0)
                            st.write(f"**Tables:** {table_count}")
                        
                        with col_c:
                            if st.button("Disconnect", key=f"disconnect_{conn_name}"):
                                if self.close_connection(conn_name):
                                    st.success(f"Disconnected from {conn_name}")
                                    st.rerun()
                        
                        # Show tables
                        if schema_info.get('tables'):
                            st.write("**Available Tables:**")
                            for table_name, table_info in schema_info['tables'].items():
                                if 'error' not in table_info:
                                    col_x, col_y = st.columns([3, 1])
                                    with col_x:
                                        st.write(f"• {table_name} ({table_info['row_count']:,} rows, {table_info['column_count']} columns)")
                                    with col_y:
                                        if st.button("Load", key=f"load_{conn_name}_{table_name}"):
                                            try:
                                                df = self.get_table_sample(conn_name, table_name, 1000)
                                                st.session_state.current_dataframe = df
                                                st.session_state.current_table_info = {
                                                    'connection': conn_name,
                                                    'table': table_name,
                                                    'source': 'database'
                                                }
                                                st.success(f"Loaded {len(df)} rows from {table_name}")
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Error loading table: {str(e)}")
            else:
                st.info("No database connections available. Add a connection to get started.")
        
        with col2:
            st.write("**➕ Add New Connection:**")
            
            conn_name = st.text_input("Connection Name:", value="My Database")
            
            # Connection type selection
            conn_type = st.selectbox("Database Type:", ["PostgreSQL", "MySQL", "SQLite"])
            
            if conn_type == "PostgreSQL":
                host = st.text_input("Host:", value="localhost")
                port = st.number_input("Port:", value=5432, min_value=1, max_value=65535)
                database = st.text_input("Database Name:")
                username = st.text_input("Username:")
                password = st.text_input("Password:", type="password")
                
                if st.button("Connect to PostgreSQL"):
                    if all([host, database, username, password]):
                        conn_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
                        if self.create_connection(conn_name, conn_string):
                            st.success(f"Successfully connected to {conn_name}!")
                            st.rerun()
                    else:
                        st.error("Please fill in all connection details.")
            
            elif conn_type == "MySQL":
                st.info("MySQL support coming soon! Currently supports PostgreSQL.")
            
            elif conn_type == "SQLite":
                st.info("SQLite support coming soon! Currently supports PostgreSQL.")
        
        # SQL Query Interface
        if st.session_state.db_connections:
            st.divider()
            st.write("**📝 SQL Query Interface:**")
            
            # Connection selector
            conn_names = list(st.session_state.db_connections.keys())
            selected_conn = st.selectbox("Select Connection:", conn_names)
            
            if selected_conn:
                # Table selector for quick queries
                schema_info = self.get_schema_info(selected_conn)
                if schema_info.get('tables'):
                    table_names = list(schema_info['tables'].keys())
                    selected_table = st.selectbox("Quick Table Access:", [""] + table_names)
                    
                    if selected_table:
                        suggestions = self.suggest_queries(selected_conn, selected_table)
                        if suggestions:
                            st.write("**💡 Suggested Queries:**")
                            for i, suggestion in enumerate(suggestions):
                                if st.button(f"📊 {suggestion}", key=f"suggestion_{i}"):
                                    st.session_state.current_query = suggestion
                
                # Custom query input
                query = st.text_area(
                    "Enter SQL Query:", 
                    value=getattr(st.session_state, 'current_query', ''),
                    height=100,
                    placeholder="SELECT * FROM your_table LIMIT 10"
                )
                
                col_query1, col_query2 = st.columns(2)
                
                with col_query1:
                    if st.button("🚀 Execute Query") and query.strip():
                        try:
                            with st.spinner("Executing query..."):
                                df = self.execute_query(selected_conn, query)
                                st.session_state.current_dataframe = df
                                st.session_state.current_table_info = {
                                    'connection': selected_conn,
                                    'query': query,
                                    'source': 'query'
                                }
                                
                                st.success(f"Query executed successfully! Returned {len(df)} rows.")
                                st.dataframe(df.head(10))
                                
                        except Exception as e:
                            st.error(f"Query execution failed: {str(e)}")
                
                with col_query2:
                    if st.button("📋 Clear Query"):
                        st.session_state.current_query = ""
                        st.rerun()
        
        # Current data info
        if hasattr(st.session_state, 'current_table_info'):
            info = st.session_state.current_table_info
            if info.get('source') == 'database':
                st.info(f"📊 Current Data: {info['table']} from {info['connection']} ({len(st.session_state.current_dataframe):,} rows)")
            elif info.get('source') == 'query':
                st.info(f"📊 Current Data: Query result from {info['connection']} ({len(st.session_state.current_dataframe):,} rows)")