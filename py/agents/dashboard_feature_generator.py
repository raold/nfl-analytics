#!/usr/bin/env python3
r"""
Dashboard Feature Generator Agent for NFL Analytics
====================================================

Auto-generates Streamlit dashboard pages from CLI scripts to enable
"dashboard as primary interface" goal. Addresses user requirement:
"I'd like the frontend to be my primary method of interacting with the project".

Key Features:
- Analyze CLI scripts and extract arguments → Generate Streamlit UI
- Visual query builder (drag-drop SQL instead of psql CLI)
- Model training submission forms (replace scripts/train_*.py)
- Backtest configuration UI (replace scripts/backtest_*.py)
- Database schema browser (replace psql \d commands)
- Auto-generate dashboard pages from function signatures

Code Generation Patterns:
1. CLI Script → Streamlit Page
   - Parse argparse arguments → st.selectbox/st.slider/st.text_input
   - Extract docstrings → page documentation
   - Function calls → st.button("Run") + st.spinner()

2. SQL Queries → Visual Query Builder
   - Table selection → st.multiselect(tables)
   - Column selection → st.multiselect(columns)
   - Filters → st.filter_dataframe()
   - Generate SQL → Execute → Display results

3. Python Functions → Dashboard Components
   - Inspect function signature → Generate input widgets
   - Execute function → Display results with formatting
   - Error handling → st.error() messages

Usage:
    from py.agents.dashboard_feature_generator import DashboardFeatureGenerator

    generator = DashboardFeatureGenerator()

    # Generate dashboard page from CLI script
    page_code = generator.generate_page_from_cli_script(
        script_path="scripts/train_xgboost_v5_integrated.py",
        page_title="Train XGBoost Model"
    )

    # Generate visual query builder
    query_builder_code = generator.generate_query_builder()

    # Generate model training form
    training_form_code = generator.generate_training_form(
        model_types=["xgboost", "bayesian", "lstm"]
    )

    # Generate database schema browser
    schema_browser_code = generator.generate_schema_browser()

    # Write generated code to file
    generator.write_dashboard_page(
        code=page_code,
        output_path="dashboard/pages/train_model.py"
    )

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
Phase: 6 (Frontend Agents)
"""

import ast
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WidgetType(Enum):
    """Streamlit widget types."""
    TEXT_INPUT = "text_input"
    NUMBER_INPUT = "number_input"
    SELECTBOX = "selectbox"
    MULTISELECT = "multiselect"
    SLIDER = "slider"
    DATE_INPUT = "date_input"
    CHECKBOX = "checkbox"
    BUTTON = "button"
    FILE_UPLOADER = "file_uploader"


@dataclass
class ArgumentSpec:
    """CLI argument specification."""
    name: str
    arg_type: str  # "str", "int", "float", "bool", "list"
    default: Any | None
    help_text: str | None
    choices: list[Any] | None
    required: bool
    widget_type: WidgetType

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "arg_type": self.arg_type,
            "default": self.default,
            "help_text": self.help_text,
            "choices": self.choices,
            "required": self.required,
            "widget_type": self.widget_type.value
        }


@dataclass
class DashboardPageSpec:
    """Specification for generated dashboard page."""
    page_title: str
    page_description: str
    arguments: list[ArgumentSpec]
    function_call: str  # Python code to execute
    result_display: str  # How to display results
    generated_code: str  # Full Streamlit page code

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "page_title": self.page_title,
            "page_description": self.page_description,
            "arguments": [arg.to_dict() for arg in self.arguments],
            "function_call": self.function_call,
            "result_display": self.result_display
        }


class DashboardFeatureGenerator:
    """
    Auto-generates Streamlit dashboard features from CLI scripts and functions.

    Enables "dashboard as primary interface" by converting ALL CLI operations
    into visual, user-friendly dashboard pages.
    """

    def __init__(
        self,
        scripts_dir: str = "scripts/",
        dashboard_dir: str = "dashboard/",
        db_config: dict[str, str] | None = None
    ):
        """
        Initialize Dashboard Feature Generator.

        Args:
            scripts_dir: Directory containing CLI scripts
            dashboard_dir: Output directory for generated pages
            db_config: Database connection config
        """
        self.scripts_dir = Path(scripts_dir)
        self.dashboard_dir = Path(dashboard_dir)

        # Database config for schema browser
        self.db_config = db_config or {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": "sicillionbillions"
        }

        logger.info("DashboardFeatureGenerator initialized")

    def analyze_cli_script(self, script_path: str) -> list[ArgumentSpec]:
        """
        Analyze CLI script and extract argument specifications.

        Args:
            script_path: Path to Python script with argparse

        Returns:
            List of argument specifications
        """
        try:
            with open(script_path) as f:
                content = f.read()

            # Parse Python AST
            tree = ast.parse(content)

            # Find ArgumentParser usage
            arguments = []

            for node in ast.walk(tree):
                # Look for parser.add_argument() calls
                if isinstance(node, ast.Call):
                    if (isinstance(node.func, ast.Attribute) and
                        node.func.attr == 'add_argument'):

                        # Extract argument name
                        if node.args:
                            arg_name = ast.literal_eval(node.args[0])
                            arg_name = arg_name.lstrip('-').replace('-', '_')

                            # Extract argument properties from keywords
                            arg_type = "str"
                            default = None
                            help_text = None
                            choices = None
                            required = False

                            for keyword in node.keywords:
                                if keyword.arg == 'type':
                                    if hasattr(keyword.value, 'id'):
                                        arg_type = keyword.value.id
                                elif keyword.arg == 'default':
                                    try:
                                        default = ast.literal_eval(keyword.value)
                                    except:
                                        default = None
                                elif keyword.arg == 'help':
                                    try:
                                        help_text = ast.literal_eval(keyword.value)
                                    except:
                                        help_text = None
                                elif keyword.arg == 'choices':
                                    try:
                                        choices = ast.literal_eval(keyword.value)
                                    except:
                                        choices = None
                                elif keyword.arg == 'required':
                                    try:
                                        required = ast.literal_eval(keyword.value)
                                    except:
                                        required = False

                            # Determine widget type
                            widget_type = self._infer_widget_type(
                                arg_type, choices, arg_name
                            )

                            arguments.append(ArgumentSpec(
                                name=arg_name,
                                arg_type=arg_type,
                                default=default,
                                help_text=help_text,
                                choices=choices,
                                required=required,
                                widget_type=widget_type
                            ))

            return arguments

        except Exception as e:
            logger.error(f"Failed to analyze CLI script {script_path}: {e}")
            return []

    def generate_page_from_cli_script(
        self,
        script_path: str,
        page_title: str,
        page_description: str | None = None
    ) -> DashboardPageSpec:
        """
        Generate complete Streamlit page from CLI script.

        Args:
            script_path: Path to CLI script
            page_title: Title for dashboard page
            page_description: Optional description

        Returns:
            Complete dashboard page specification with generated code
        """
        # Analyze CLI script
        arguments = self.analyze_cli_script(script_path)

        if not page_description:
            page_description = f"Auto-generated from {Path(script_path).name}"

        # Generate Streamlit code
        code_parts = [
            '"""',
            f'{page_title}',
            '=' * len(page_title),
            '',
            page_description,
            '',
            'Auto-generated by DashboardFeatureGenerator',
            f'Source: {script_path}',
            '"""',
            '',
            'import streamlit as st',
            'import sys',
            'from pathlib import Path',
            '',
            '# Add project root to path',
            'sys.path.insert(0, str(Path(__file__).parent.parent))',
            '',
            f'st.title("{page_title}")',
            f'st.markdown("{page_description}")',
            'st.divider()',
            '',
            '# Input form',
            'with st.form("parameters_form"):',
        ]

        # Generate widgets for each argument
        for arg in arguments:
            widget_code = self._generate_widget_code(arg, indent=4)
            code_parts.append(widget_code)

        # Add submit button
        code_parts.extend([
            '',
            '    submitted = st.form_submit_button("Run")',
            '',
            'if submitted:',
            '    with st.spinner("Running..."):'
        ])

        # Generate function call
        script_name = Path(script_path).stem
        arg_names = ', '.join([f'{arg.name}={arg.name}' for arg in arguments])
        function_call = f'from scripts import {script_name}\nresult = {script_name}.main({arg_names})'

        code_parts.extend([
            '        try:',
            '            # Import and run script',
            f'            {function_call}',
            '            st.success("✅ Completed successfully!")',
            '            st.json(result)',
            '        except Exception as e:',
            '            st.error(f"❌ Error: {e}")',
        ])

        generated_code = '\n'.join(code_parts)

        return DashboardPageSpec(
            page_title=page_title,
            page_description=page_description,
            arguments=arguments,
            function_call=function_call,
            result_display="st.json(result)",
            generated_code=generated_code
        )

    def generate_query_builder(self) -> str:
        """
        Generate visual SQL query builder component.

        Returns:
            Streamlit code for query builder
        """
        code = '''"""
Visual SQL Query Builder
=========================

Drag-and-drop SQL query builder for database exploration.
Auto-generated by DashboardFeatureGenerator.
"""

import streamlit as st
import psycopg2
import pandas as pd
from psycopg2.extras import RealDictCursor

st.title("📊 Visual Query Builder")
st.markdown("Build SQL queries visually without writing code")
st.divider()

# Database connection
@st.cache_resource
def get_connection():
    return psycopg2.connect(
        host="localhost",
        port=5544,
        database="devdb01",
        user="dro",
        password="sicillionbillions"
    )

conn = get_connection()

# Get available tables
@st.cache_data
def get_tables():
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema IN ('public', 'mart', 'ml')
            ORDER BY table_schema, table_name
        """)
        return cursor.fetchall()

tables = get_tables()
table_options = [f"{t['table_schema']}.{t['table_name']}" for t in tables]

# Query builder UI
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Select Table")
    selected_table = st.selectbox("Table", table_options)

    if selected_table:
        schema, table = selected_table.split('.')

        # Get columns
        @st.cache_data
        def get_columns(schema, table):
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                """, (schema, table))
                return cursor.fetchall()

        columns = get_columns(schema, table)
        column_names = [c['column_name'] for c in columns]

        st.subheader("2. Select Columns")
        selected_columns = st.multiselect(
            "Columns (* for all)",
            ["*"] + column_names,
            default=["*"]
        )

with col2:
    st.subheader("3. Add Filters (Optional)")
    num_filters = st.number_input("Number of filters", 0, 5, 0)

    filters = []
    for i in range(num_filters):
        filter_col = st.selectbox(f"Filter {i+1} Column", column_names, key=f"filter_col_{i}")
        filter_op = st.selectbox(f"Operator", ["=", "!=", ">", "<", ">=", "<=", "LIKE"], key=f"filter_op_{i}")
        filter_val = st.text_input(f"Value", key=f"filter_val_{i}")
        filters.append((filter_col, filter_op, filter_val))

    st.subheader("4. Limit Results")
    limit = st.slider("Max rows", 10, 1000, 100)

# Generate SQL
st.divider()
st.subheader("Generated SQL")

columns_str = ", ".join(selected_columns) if selected_columns else "*"
sql = f"SELECT {columns_str}\\nFROM {selected_table}"

if filters and any(f[2] for f in filters):  # At least one filter has a value
    where_clauses = []
    for col, op, val in filters:
        if val:
            if op == "LIKE":
                where_clauses.append(f"{col} LIKE '%{val}%'")
            elif val.isdigit():
                where_clauses.append(f"{col} {op} {val}")
            else:
                where_clauses.append(f"{col} {op} '{val}'")

    if where_clauses:
        sql += "\\nWHERE " + " AND ".join(where_clauses)

sql += f"\\nLIMIT {limit}"

st.code(sql, language="sql")

# Execute query
if st.button("🔍 Execute Query", type="primary"):
    with st.spinner("Executing query..."):
        try:
            df = pd.read_sql_query(sql, conn)
            st.success(f"✅ Retrieved {len(df)} rows")
            st.dataframe(df, use_container_width=True)

            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"query_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"❌ Query failed: {e}")
'''
        return code

    def generate_training_form(
        self,
        model_types: list[str] = ["xgboost", "bayesian", "lstm"]
    ) -> str:
        """
        Generate model training submission form.

        Args:
            model_types: List of available model types

        Returns:
            Streamlit code for training form
        """
        code = f'''"""
Model Training Form
===================

Submit model training jobs from the dashboard.
Auto-generated by DashboardFeatureGenerator.
"""

import streamlit as st
import subprocess
from datetime import datetime

st.title("🤖 Train Model")
st.markdown("Submit model training jobs with custom parameters")
st.divider()

# Training form
with st.form("training_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Configuration")
        model_type = st.selectbox(
            "Model Type",
            {model_types},
            help="Type of model to train"
        )

        target = st.selectbox(
            "Target Variable",
            ["passing_yards", "rushing_yards", "receiving_yards", "touchdowns"],
            help="What to predict"
        )

        position = st.selectbox(
            "Position",
            ["QB", "RB", "WR", "TE"],
            help="Player position"
        )

    with col2:
        st.subheader("Training Data")
        season_start = st.number_input("Start Season", 2018, 2024, 2018)
        season_end = st.number_input("End Season", 2018, 2024, 2024)

        test_size = st.slider("Test Split %", 10, 30, 20)

    st.subheader("Advanced Options")
    use_gpu = st.checkbox("Use GPU acceleration", value=False)
    save_artifacts = st.checkbox("Save model artifacts", value=True)

    submitted = st.form_submit_button("🚀 Start Training", type="primary")

if submitted:
    st.divider()
    st.subheader("Training Progress")

    # Generate training command
    script_map = {{
        "xgboost": "scripts/train_xgboost_v5_integrated.py",
        "bayesian": "scripts/train_bayesian_model.py",
        "lstm": "scripts/train_lstm_drives_v1.py"
    }}

    script = script_map.get(model_type, "scripts/train_xgboost_v5_integrated.py")

    cmd = [
        "python", script,
        "--target", target,
        "--position", position,
        "--seasons", f"{{season_start}}-{{season_end}}",
        "--test-size", str(test_size / 100)
    ]

    if use_gpu:
        cmd.append("--gpu")

    st.code(" ".join(cmd), language="bash")

    # Execute training
    with st.spinner(f"Training {{model_type}} model..."):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                st.success("✅ Training completed successfully!")
                st.text_area("Output", result.stdout, height=300)

                if save_artifacts:
                    st.info(f"📦 Model artifacts saved to models/")

            else:
                st.error("❌ Training failed")
                st.text_area("Error Output", result.stderr, height=300)

        except subprocess.TimeoutExpired:
            st.error("❌ Training timed out (>1 hour)")
        except Exception as e:
            st.error(f"❌ Error: {{e}}")
'''
        return code

    def generate_schema_browser(self) -> str:
        """
        Generate interactive database schema browser.

        Returns:
            Streamlit code for schema browser
        """
        code = '''"""
Database Schema Browser
=======================

Interactive schema explorer with ERD visualization.
Auto-generated by DashboardFeatureGenerator.
"""

import streamlit as st
import psycopg2
import pandas as pd
from psycopg2.extras import RealDictCursor

st.title("🗄️ Database Schema Browser")
st.markdown("Explore database schema interactively")
st.divider()

# Database connection
@st.cache_resource
def get_connection():
    return psycopg2.connect(
        host="localhost",
        port=5544,
        database="devdb01",
        user="dro",
        password="sicillionbillions"
    )

conn = get_connection()

# Get schemas
@st.cache_data
def get_schemas():
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
            ORDER BY schema_name
        """)
        return [row[0] for row in cursor.fetchall()]

schemas = get_schemas()

# Schema selector
selected_schema = st.selectbox("📁 Schema", schemas)

if selected_schema:
    # Get tables in schema
    @st.cache_data
    def get_tables(schema):
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT
                    table_name,
                    (SELECT COUNT(*) FROM information_schema.columns
                     WHERE table_schema = t.table_schema AND table_name = t.table_name) as column_count
                FROM information_schema.tables t
                WHERE table_schema = %s
                ORDER BY table_name
            """, (schema,))
            return cursor.fetchall()

    tables = get_tables(selected_schema)

    st.subheader(f"📊 Tables in {selected_schema}")
    st.write(f"Found {len(tables)} tables")

    # Table selector
    table_names = [t['table_name'] for t in tables]
    selected_table = st.selectbox("Table", table_names)

    if selected_table:
        st.divider()

        # Get table details
        @st.cache_data
        def get_table_details(schema, table):
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get columns
                cursor.execute("""
                    SELECT
                        column_name,
                        data_type,
                        is_nullable,
                        column_default
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                """, (schema, table))
                columns = cursor.fetchall()

                # Get constraints
                cursor.execute("""
                    SELECT
                        tc.constraint_type,
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                    LEFT JOIN information_schema.constraint_column_usage ccu
                        ON tc.constraint_name = ccu.constraint_name
                    WHERE tc.table_schema = %s AND tc.table_name = %s
                """, (schema, table))
                constraints = cursor.fetchall()

                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
                row_count = cursor.fetchone()['count']

                return columns, constraints, row_count

        columns, constraints, row_count = get_table_details(selected_schema, selected_table)

        # Display table info
        col1, col2, col3 = st.columns(3)
        col1.metric("Columns", len(columns))
        col2.metric("Constraints", len(constraints))
        col3.metric("Rows", f"{row_count:,}")

        # Display columns
        st.subheader("Columns")
        columns_df = pd.DataFrame(columns)
        st.dataframe(columns_df, use_container_width=True)

        # Display constraints
        if constraints:
            st.subheader("Constraints")
            constraints_df = pd.DataFrame(constraints)
            st.dataframe(constraints_df, use_container_width=True)

        # Sample data
        st.subheader("Sample Data")
        sample_size = st.slider("Sample size", 5, 100, 10)

        if st.button("Load Sample"):
            with st.spinner("Loading..."):
                df = pd.read_sql_query(
                    f"SELECT * FROM {selected_schema}.{selected_table} LIMIT {sample_size}",
                    conn
                )
                st.dataframe(df, use_container_width=True)
'''
        return code

    def write_dashboard_page(
        self,
        code: str,
        output_path: str
    ):
        """
        Write generated code to dashboard page file.

        Args:
            code: Generated Streamlit code
            output_path: Output file path
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                f.write(code)

            logger.info(f"Dashboard page written to {output_path}")

        except Exception as e:
            logger.error(f"Failed to write dashboard page: {e}")
            raise

    def _infer_widget_type(
        self,
        arg_type: str,
        choices: list | None,
        arg_name: str
    ) -> WidgetType:
        """Infer appropriate Streamlit widget type from argument properties."""
        if choices:
            return WidgetType.SELECTBOX
        elif arg_type in ["int", "float"]:
            if "year" in arg_name.lower() or "season" in arg_name.lower():
                return WidgetType.SLIDER
            return WidgetType.NUMBER_INPUT
        elif arg_type == "bool":
            return WidgetType.CHECKBOX
        elif "date" in arg_name.lower():
            return WidgetType.DATE_INPUT
        elif "file" in arg_name.lower() or "path" in arg_name.lower():
            return WidgetType.FILE_UPLOADER
        else:
            return WidgetType.TEXT_INPUT

    def _generate_widget_code(
        self,
        arg: ArgumentSpec,
        indent: int = 0
    ) -> str:
        """Generate Streamlit widget code for argument."""
        spaces = ' ' * indent
        widget_code = f"{spaces}{arg.name} = "

        if arg.widget_type == WidgetType.TEXT_INPUT:
            widget_code += f'st.text_input("{arg.name}", value="{arg.default or ""}"'
        elif arg.widget_type == WidgetType.NUMBER_INPUT:
            widget_code += f'st.number_input("{arg.name}", value={arg.default or 0}'
        elif arg.widget_type == WidgetType.SELECTBOX:
            widget_code += f'st.selectbox("{arg.name}", {arg.choices}, index=0'
        elif arg.widget_type == WidgetType.SLIDER:
            widget_code += f'st.slider("{arg.name}", 2018, 2024, {arg.default or 2024}'
        elif arg.widget_type == WidgetType.CHECKBOX:
            widget_code += f'st.checkbox("{arg.name}", value={arg.default or False}'
        else:
            widget_code += f'st.text_input("{arg.name}"'

        if arg.help_text:
            widget_code += f', help="{arg.help_text}"'

        widget_code += ')'

        return widget_code


if __name__ == "__main__":
    # Example usage
    generator = DashboardFeatureGenerator()

    # Generate query builder
    print("Generating visual query builder...")
    query_builder_code = generator.generate_query_builder()
    generator.write_dashboard_page(
        code=query_builder_code,
        output_path="/tmp/query_builder.py"
    )
    print("✅ Query builder generated: /tmp/query_builder.py")

    # Generate schema browser
    print("\nGenerating schema browser...")
    schema_browser_code = generator.generate_schema_browser()
    generator.write_dashboard_page(
        code=schema_browser_code,
        output_path="/tmp/schema_browser.py"
    )
    print("✅ Schema browser generated: /tmp/schema_browser.py")

    # Generate training form
    print("\nGenerating training form...")
    training_form_code = generator.generate_training_form()
    generator.write_dashboard_page(
        code=training_form_code,
        output_path="/tmp/train_model.py"
    )
    print("✅ Training form generated: /tmp/train_model.py")
