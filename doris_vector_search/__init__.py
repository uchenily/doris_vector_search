"""Doris Vector Search Python SDK"""

import concurrent.futures
import io
import logging
import math
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Self, Tuple, Union

import pandas as pd
import pyarrow as pa
import requests
from requests.auth import HTTPBasicAuth
import mysql.connector

# All supported data types for create_table/add
Block = Union[List[dict], pd.DataFrame, pa.Table, Iterable[pa.RecordBatch]]

logger = logging.getLogger(__name__)


# Configuration Options
class IndexOptions:
    """Configuration options for vector index creation."""

    def __init__(
        self,
        index_type: str = "hnsw",
        metric_type: str = "l2_distance",
        dim: int = -1,
    ):
        """Index options.

        Args:
            index_type: Type of vector index (currently only 'hnsw' is supported)
            metric_type: Distance metric ('l2_distance' or 'inner_product')
        """
        self.index_type = index_type.lower()
        self.metric_type = metric_type.lower()
        self.dim = dim

        self._validate()

    def _validate(self):
        """Validate index options."""
        supported_index_types = ["hnsw"]
        supported_distance_types = ["l2_distance", "inner_product"]

        if self.index_type not in supported_index_types:
            raise ValueError(
                f"Unsupported index_type: {self.index_type}. Currently only {supported_index_types} are supported."
            )

        if self.metric_type not in supported_distance_types:
            raise ValueError(
                f"Unsupported metric_type: {self.metric_type}. Supported: {supported_distance_types}"
            )


class AuthOptions:
    """Options for login into doris database."""

    def __init__(
        self,
        host: str = "localhost",
        query_port: int = 9030,
        http_port: int = 8030,
        user: str = "root",
        password: str = "",
    ):
        self.host = host
        self.query_port = query_port
        self.http_port = http_port
        self.user = user
        self.password = password


class LoadOptions:
    """Options for data loading."""

    def __init__(self, format: str = "arrow", batch_size: int = 10000):
        """Load options.

        Args:
            format: Format for stream loading ('csv' or 'arrow')
        """
        format = format.lower()
        if format not in ["csv", "arrow"]:
            raise ValueError(
                f"Unsupported format '{format}'. Supported formats: 'csv', 'arrow'"
            )
        self.format = format

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size


class TableOptions:
    """Configuration options for table creation."""

    def __init__(
        self,
        table_name: str,
        columns: Dict[str, str],
        key_column: str,
        vector_column: Optional[str] = None,
        vector_options: Optional[Dict[str, Any]] = None,
        table_properties: Optional[Dict[str, Any]] = None,
        num_replication: int = 1,
        num_buckets: int = 1,
    ):
        """Table options.

        Args:
            table_name: Name of the table to create
            columns: Dictionary mapping column names to their Doris types
            key_column: Name of the key column
            vector_column: Name of the vector column (optional)
            vector_options: Options for vector index creation (optional)
            table_properties: Additional table properties (optional)
            num_buckets: Number of buckets for distribution (default: 1)
        """
        self.table_name = table_name
        self.columns = columns
        self.key_column = key_column
        self.vector_column = vector_column
        self.vector_options = vector_options

        self.table_properties = table_properties or {}
        if not self.table_properties.get("replication_num"):
            self.table_properties["replication_num"] = str(num_replication)

        self.num_buckets = num_buckets

        # Validate parameters
        self._validate()

    def _validate(self):
        """Validate table options."""
        if not self.table_name:
            raise ValueError("table_name cannot be empty")

        if not self.columns:
            raise ValueError("columns cannot be empty")

        if self.key_column not in self.columns:
            raise ValueError(
                f"key_column '{self.key_column}' must be present in columns"
            )

        if self.vector_column and self.vector_column not in self.columns:
            raise ValueError(
                f"vector_column '{self.vector_column}' must be present in columns"
            )

        if self.num_buckets <= 0:
            raise ValueError("buckets must be positive")


class TableSchema:
    """Schema information for a Doris table."""

    def __init__(
        self,
        columns: Dict[str, Dict[str, Any]],
        key_column: Optional[str] = None,
        vector_column: Optional[str] = None,
        vector_dim: int = 0,
    ):
        """Initialize TableSchema.

        Args:
            columns: Dictionary mapping column names to their schema info
            key_column: Name of the key column
            vector_column: Name of the vector column
            vector_dim: Dimension of the vector column
        """
        self.columns = columns
        self.key_column = key_column
        self.vector_column = vector_column
        self.vector_dim = vector_dim


class StreamLoadFormat(ABC):
    """Abstract base class for stream load data formats."""

    @property
    @abstractmethod
    def content_type(self) -> str:
        """Return the content type for HTTP headers."""
        pass

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the format name for Doris stream load."""
        pass

    @abstractmethod
    def serialize_data(self, data: pa.Table, schema_info: TableSchema) -> bytes:
        """Serialize Arrow Table to bytes in the specific format."""
        pass

    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """Return format-specific headers for stream load."""
        pass


class CSVStreamLoadFormat(StreamLoadFormat):
    """CSV format for stream load."""

    @property
    def content_type(self) -> str:
        return "text/csv"

    @property
    def format_name(self) -> str:
        return "csv"

    def serialize_data(self, data: pa.Table, schema_info: TableSchema) -> bytes:
        """Serialize Arrow Table to CSV bytes."""
        vector_column = schema_info.vector_column
        columns = list(data.column_names)

        # Convert Arrow Table to CSV format
        csv_buffer = io.StringIO()

        for i in range(data.num_rows):
            csv_row = []
            for col in columns:
                value = data.column(col)[i].as_py()
                if col == vector_column and isinstance(value, list):
                    # Format vector as array string
                    csv_row.append(f"'[{','.join(map(str, value))}]'")
                else:
                    # Escape single quotes and wrap in single quotes
                    str_value = str(value).replace("'", "\\'")
                    csv_row.append(f"'{str_value}'")

            csv_buffer.write(",".join(csv_row))
            csv_buffer.write("\n")

        csv_data = csv_buffer.getvalue()
        return csv_data.encode("utf-8")

    def get_columns(self, data: pa.Table) -> List[str]:
        """Get column names for CSV format."""
        return list(data.column_names)

    def get_headers(self) -> Dict[str, str]:
        """Return CSV-specific headers."""
        return {
            "column_separator": ",",
            "enclose": "'",
            "trim_double_quotes": "false",
        }


class ArrowStreamLoadFormat(StreamLoadFormat):
    """Apache Arrow format for stream load."""

    @property
    def content_type(self) -> str:
        return "application/octet-stream"

    @property
    def format_name(self) -> str:
        return "arrow"

    def serialize_data(self, data: pa.Table, schema_info: TableSchema) -> bytes:
        """Serialize Arrow Table to Arrow bytes."""
        # Data is already an Arrow Table, serialize directly
        table = data

        # Serialize to Arrow IPC format (streaming format)
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write(table)

        return sink.getvalue().to_pybytes()

    def get_headers(self) -> Dict[str, str]:
        """Return Arrow-specific headers."""
        return {
            "format": "arrow",
        }


class StreamLoadFormatFactory:
    """Factory for creating stream load format handlers."""

    _formats = {
        "csv": CSVStreamLoadFormat,
        "arrow": ArrowStreamLoadFormat,
        # TODO: Add more stream load supported data types.
    }

    @classmethod
    def get_format(cls, format_name: str) -> StreamLoadFormat:
        """Get a stream load format handler by name."""
        if format_name not in cls._formats:
            available_formats = list(cls._formats.keys())
            raise ValueError(
                f"Unsupported format '{format_name}'. Available formats: {available_formats}"
            )

        return cls._formats[format_name]()


def block_to_arrow_table(block: Block) -> pa.Table:
    """Convert various data types to PyArrow Table format.

    Args:
        data: Input data in supported formats

    Returns:
        PyArrow Table representation of the data

    Raises:
        TypeError: If data type is not supported
    """
    if isinstance(block, list):
        if not block:
            raise ValueError("Cannot create table from empty list")
        # Assume list of dicts
        table = pa.Table.from_pylist(block)
    elif isinstance(block, pd.DataFrame):
        table = pa.Table.from_pandas(block, preserve_index=False)
    elif isinstance(block, pa.Table):
        table = block
    elif isinstance(block, pa.RecordBatch):
        table = pa.Table.from_batches([block])
    elif isinstance(block, Iterable):
        # Assume iterable of RecordBatch
        batches = list(block)
        if not batches:
            raise ValueError("Cannot create table from empty iterable")
        table = pa.Table.from_batches(batches)
    else:
        raise TypeError(
            f"Unsupported data type {type(block)}. Supported types: list of dicts, pandas DataFrame, pyarrow Table, pyarrow RecordBatch, or iterable of RecordBatch."
        )

    # NOTE: Cast any list[double] columns to list[float32] to ensure consistency.
    # This fixes the issue where Python list[float] becomes list[double] in Arrow.
    # Any better way?
    new_columns = []
    for col_name in table.column_names:
        col = table.column(col_name)
        if pa.types.is_list(col.type) and pa.types.is_floating(col.type.value_type):
            if col.type.value_type == pa.float64():
                # Cast float64 to float32
                new_col = col.cast(pa.list_(pa.float32()))
                new_columns.append((col_name, new_col))
            else:
                new_columns.append((col_name, col))
        else:
            new_columns.append((col_name, col))

    if new_columns:
        table = pa.Table.from_arrays(
            [col for _, col in new_columns], names=[name for name, _ in new_columns]
        )

    return table


class DorisSQLCompiler:
    """Doris-specific SQL compiler for vector search queries."""

    def compile_vector_search_query(
        self,
        table_name: str,
        query_vector: List[float],
        vector_column: str,
        limit: Optional[int] = None,
        distance_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        where_conditions: Optional[List[str]] = None,
        selected_columns: Optional[List[str]] = None,
        metric_type: str = "l2_distance",
    ) -> str:
        """Compile vector search query to Doris SQL."""
        vector_str = f"[{','.join(map(str, query_vector))}]"

        # Determine which columns to select
        if selected_columns:
            select_columns = selected_columns
        else:
            # Default: select all columns
            select_columns = ["*"]

        # Build SELECT clause
        select_clause = ", ".join(f"`{col}`" for col in select_columns)

        distance_fn = f"{metric_type}_approximate"

        # Build WHERE clause
        where_parts = []

        # Add distance range conditions
        if distance_range:
            lower_bound, upper_bound = distance_range
            if lower_bound is not None:
                where_parts.append(
                    f"{distance_fn}(`{vector_column}`, {vector_str}) >= {lower_bound}"
                )
            if upper_bound is not None:
                where_parts.append(
                    f"{distance_fn}(`{vector_column}`, {vector_str}) <= {upper_bound}"
                )

        # Add user-defined WHERE conditions
        if where_conditions:
            where_parts.extend(where_conditions)

        where_clause = ""
        if where_parts:
            where_clause = f"WHERE {' AND '.join(where_parts)}"

        # Build ORDER BY
        if metric_type == "inner_product":
            order_clause = f"ORDER BY {distance_fn}(`{vector_column}`, {vector_str}) DESC"
        else:
            order_clause = f"ORDER BY {distance_fn}(`{vector_column}`, {vector_str}) ASC"

        # Build LIMIT
        limit_clause = f"LIMIT {limit}" if limit else ""

        # Construct full SQL
        sql = f"SELECT {select_clause} FROM `{table_name}` {where_clause} {order_clause} {limit_clause}"

        return sql

    def compile_vector_search_query_prepared(
        self,
        table_name: str,
        query_vector: List[float],
        vector_column: str,
        limit: Optional[int] = None,
        distance_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        where_conditions: Optional[List[str]] = None,
        selected_columns: Optional[List[str]] = None,
        metric_type: str = "l2_distance",
    ) -> Tuple[str, List[Any]]:
        """Compile vector search query to Doris SQL with prepared statement.

        Returns:
            Tuple of (sql_string, parameters_list) for prepared statement execution
        """
        params = []

        # Use positional placeholder for vector
        vector_value = str(query_vector)

        # Determine which columns to select
        if selected_columns:
            select_columns = selected_columns
        else:
            # Default: select all columns
            select_columns = ["*"]

        # Build SELECT clause
        select_clause = ", ".join(f"`{col}`" for col in select_columns)

        distance_fn = f"{metric_type}_approximate"

        # Build WHERE clause
        where_parts = []

        # Add distance range conditions
        if distance_range:
            lower_bound, upper_bound = distance_range
            if lower_bound is not None:
                where_parts.append(f"{distance_fn}(`{vector_column}`, CAST(? AS ARRAY<FLOAT>)) >= ?")
                params.append(vector_value)
                params.append(lower_bound)
            if upper_bound is not None:
                where_parts.append(f"{distance_fn}(`{vector_column}`, CAST(? AS ARRAY<FLOAT>)) <= ?")
                params.append(vector_value)
                params.append(upper_bound)

        # Add user-defined WHERE conditions with parameterization
        if where_conditions:
            for condition in where_conditions:
                parameterized_condition, condition_params = (
                    self._parameterize_where_condition(condition, len(params))
                )
                where_parts.append(parameterized_condition)
                params.extend(condition_params)

        where_clause = ""
        if where_parts:
            where_clause = f"WHERE {' AND '.join(where_parts)}"

        # Build ORDER BY using the distance function
        if metric_type == "inner_product":
            order_clause = f"ORDER BY {distance_fn}(`{vector_column}`, CAST(? AS ARRAY<FLOAT>)) DESC"
        else:
            order_clause = f"ORDER BY {distance_fn}(`{vector_column}`, CAST(? AS ARRAY<FLOAT>)) ASC"
        params.append(vector_value)

        # Build LIMIT (inline, as Doris may not support LIMIT as prepared param)
        limit_clause = f"LIMIT {limit}" if limit is not None else ""

        sql = f"SELECT {select_clause} FROM `{table_name}` {where_clause} {order_clause} {limit_clause}"

        return sql, params

    def _parameterize_where_condition(
        self, condition: str, param_start_index: int
    ) -> Tuple[str, List[Any]]:
        """Parameterize a single WHERE condition.

        Args:
            condition: The WHERE condition string (e.g., "age > 18", "name = 'John'")
            param_start_index: Starting index for parameter placeholders

        Returns:
            Tuple of (parameterized_condition, parameters_list)
        """
        # Parse the condition: column_name operator value
        parts = condition.split()
        if len(parts) != 3:
            raise ValueError(f"Invalid WHERE condition format: {condition}")

        column_name, operator, value_str = parts

        params = []

        # Handle different value types
        if value_str.startswith(("'", '"')) and value_str.endswith(("'", '"')):
            # Quoted string value
            inner_value = value_str[1:-1]
            params.append(inner_value)
        else:
            # Numeric value
            try:
                params.append(float(value_str))
            except ValueError:
                raise ValueError(f"Invalid numeric value in condition: {condition}")

        # Create parameterized condition with positional placeholder
        parameterized_condition = f"{column_name} {operator} ?"

        return parameterized_condition, params

    def compile_set_session(self, key: str, value: Any) -> str:
        """Compile SET SESSION statement without parameters.

        Args:
            key: The session variable name
            value: The value to set

        Returns:
            sql_string for execution
        """
        if isinstance(value, str):
            sql = f"SET SESSION {key} = '{value}'"
        else:
            sql = f"SET SESSION {key} = {value}"
        return sql


class DorisDDLCompiler:
    """Doris-specific DDL compiler for table creation with vector indexes."""

    def compile_create_table(self, table_options: TableOptions) -> str:
        """Compile CREATE TABLE statement for Doris with optional vector index."""

        # Build column definitions
        column_defs = []
        for col_name, col_type in table_options.columns.items():
            nullable = "NOT NULL" if col_name == table_options.vector_column else ""
            column_defs.append(f"`{col_name}` {col_type} {nullable}".strip())
        columns_sql = ",".join(column_defs)

        # Build index clause
        index_clause = ""
        if table_options.vector_column and table_options.vector_options:
            index_clause = f""",INDEX idx_{table_options.vector_column}(`{table_options.vector_column}`) USING ANN PROPERTIES("index_type"="{table_options.vector_options['index_type']}","metric_type"="{table_options.vector_options['metric_type']}","dim"={table_options.vector_options['dim']})"""

        # Build table properties
        properties_clause = ""
        if table_options.table_properties:
            props = []
            for key, value in table_options.table_properties.items():
                if isinstance(value, str):
                    props.append(f'"{key}"="{value}"')
                else:
                    props.append(f'"{key}"={value}')
            properties_clause = f" PROPERTIES({','.join(props)})"

        # Construct full DDL
        ddl = f"""CREATE TABLE `{table_options.table_name}`({columns_sql}{index_clause}) DUPLICATE KEY(`{table_options.key_column}`) DISTRIBUTED BY HASH(`{table_options.key_column}`) BUCKETS {table_options.num_buckets}{properties_clause};"""

        return ddl

    def compile_create_index(
        self,
        table_name: str,
        vector_column: str,
        index_options: Dict[str, Any],
        dim: int,
    ) -> str:
        """Compile CREATE INDEX statement for vector index."""
        index_name = f"idx_{vector_column}"
        sql = f"""CREATE INDEX {index_name} ON {table_name}(`{vector_column}`) USING ANN PROPERTIES("index_type"="{index_options["index_type"]}","metric_type"="{index_options["metric_type"]}","dim"="{dim}")"""
        return sql

    def compile_build_index(self, table_name: str, vector_column: str) -> str:
        """Compile BUILD INDEX statement for vector index."""
        index_name = f"idx_{vector_column}"
        sql = f"BUILD INDEX {index_name} ON {table_name}"
        return sql

    def compile_drop_index(self, table_name: str, vector_column: str) -> str:
        """Compile DROP INDEX statement for vector index."""
        index_name = f"idx_{vector_column}"
        sql = f"DROP INDEX {index_name} ON {table_name}"
        return sql

    def compile_drop_table(self, table_name: str) -> str:
        """Compile DROP TABLE statement."""
        sql = f"DROP TABLE IF EXISTS `{table_name}`"
        return sql


class VectorSearchQuery:
    """Query builder for vector search."""

    def __init__(
        self,
        table: "DorisTable",
        query_vector: List[float],
        vector_column: str,
        metric_type: str = "l2_distance",
    ):
        self.table = table
        self.query_vector = query_vector
        self.vector_column = vector_column
        self.metric_type = metric_type
        self.limit_value: Optional[int] = None
        self.distance_range_upper: Optional[float] = None
        self.distance_range_lower: Optional[float] = None
        self.where_conditions: List[str] = []
        self.selected_columns: Optional[List[str]] = None
        self.compiler = DorisSQLCompiler()

    def limit(self, n: int) -> Self:
        """Set the limit for the number of results."""
        self.limit_value = n
        return self

    def distance_range(
        self, lower_bound: Optional[float] = None, upper_bound: Optional[float] = None
    ) -> Self:
        """Set distance range filter."""
        self.distance_range_lower = lower_bound
        self.distance_range_upper = upper_bound
        return self

    def where(self, condition: str) -> Self:
        """Add a WHERE condition to the query.

        Args:
            condition: WHERE condition string in the allowed format

        Raises:
            ValueError: If condition doesn't match allowed patterns
        """
        self._validate_where_condition(condition)
        self.where_conditions.append(condition)
        return self

    def _validate_where_condition(self, condition: str):
        """Validate WHERE condition to allow only safe comparisons.

        Only allows conditions in the format:
        - column_name {OP} value

        Args:
            condition: The WHERE condition string to validate

        Raises:
            ValueError: If the condition doesn't match the allowed patterns
        """
        if not condition or not condition.strip():
            raise ValueError("WHERE condition cannot be empty")

        condition = condition.strip()

        # Split condition into parts: expect exactly 3 parts (column, operator, value)
        parts = condition.split()
        if len(parts) != 3:
            raise ValueError(
                f"WHERE condition must be in format: column_name operator value. got: `{condition}`"
            )

        column_name, operator, value_str = parts

        if column_name not in self.table.column_names:
            raise ValueError(
                f"Invalid column name '{column_name}': valid columns: {self.table.column_names}"
            )

        # Validate operator
        allowed_operators = [">", "<", ">=", "<=", "=", "LIKE"]
        if operator not in allowed_operators:
            raise ValueError(
                f"Operator '{operator}' is not allowed. Use: {', '.join(allowed_operators)}"
            )

        # Validate value
        if value_str.startswith(("'", '"')) and value_str.endswith(("'", '"')):
            # Quoted string
            if len(value_str) < 2 or value_str[0] != value_str[-1]:
                raise ValueError(f"Invalid quoted string in value: {value_str}")
        else:
            # Numeric
            try:
                float(value_str)
            except ValueError:
                raise ValueError(
                    f"Value '{value_str}' is not a valid number or quoted string"
                )

        # Log the validated condition for debugging
        logger.debug(
            f"WHERE condition validated: '{condition}' (column: {column_name}, operator: {operator}, value: {value_str})"
        )

    def select(self, columns: List[str]) -> Self:
        """Specify which columns to select."""
        self.selected_columns = columns
        return self

    def _execute_and_process_query(self) -> pa.Table:
        """Execute the query and return processed results as PyArrow Table."""
        all_columns = self.table.column_names

        if self.selected_columns is not None:
            # Validate selected columns exist in table
            invalid_columns = [
                col for col in self.selected_columns if col not in all_columns
            ]
            if invalid_columns:
                raise ValueError(
                    f"Selected columns not found in table: {invalid_columns}"
                )

            select_columns = self.selected_columns
        else:
            select_columns = all_columns

        # Prepare distance range tuple
        distance_range = None
        if (
            self.distance_range_lower is not None
            or self.distance_range_upper is not None
        ):
            distance_range = (self.distance_range_lower, self.distance_range_upper)

        sql, params = self.compiler.compile_vector_search_query_prepared(
            table_name=self.table.table_name,
            query_vector=self.query_vector,
            vector_column=self.vector_column,
            limit=self.limit_value,
            distance_range=distance_range,
            where_conditions=self.where_conditions if self.where_conditions else None,
            selected_columns=select_columns,
            metric_type=self.metric_type,
        )

        logger.debug(f'generated sql: "{sql}"')
        logger.debug(f"parameters: {params}")

        # Use mysqlconnector connection with prepared cursor for vector search
        cursor = self.table._get_cursor(prepared=True)
        cursor.execute(sql, params)
        rows = cursor.fetchall()

        select_columns = self.selected_columns or all_columns
        col_data = {col: [] for col in select_columns}

        # data = [list(column) for column in zip(*rows)]
        # return pa.Table.from_arrays(data, names=columns)

        # TODO: optimize it
        for row in rows:
            # Raw cursor returns tuples, map to dict by column order
            row_dict = dict(zip(select_columns, row))

            # Decode bytes/bytearray to str and parse vector columns from Doris ARRAY<FLOAT> format
            for col_name, value in row_dict.items():
                if isinstance(value, (bytes, bytearray)):
                    value = value.decode('utf-8')
                    row_dict[col_name] = value

                if (
                    isinstance(value, str)
                    and value.startswith("[")
                    and value.endswith("]")
                ):
                    try:
                        # Parse ARRAY<FLOAT> format like: [1.0,2.0,3.0]
                        vec_str = value[1:-1]
                        if vec_str.strip():  # Not empty array
                            row_dict[col_name] = [
                                float(x.strip())
                                for x in vec_str.split(",")
                                if x.strip()
                            ]
                        else:
                            row_dict[col_name] = []
                    except (ValueError, IndexError) as e:
                        logger.warning(
                            f"Failed to parse vector column '{col_name}' with value '{value}': {e}"
                        )
                        raise e

            for col in select_columns:
                col_data[col].append(row_dict[col])

        arrays = [pa.array(col_data[col]) for col in select_columns]
        return pa.Table.from_arrays(arrays, names=select_columns)

    def to_pandas(self) -> pd.DataFrame:
        """Execute the query and return results as pandas DataFrame."""
        data = self._execute_and_process_query()
        return data.to_pandas()

    def to_arrow(self) -> pa.Table:
        """Execute the query and return results as PyArrow Table."""
        return self._execute_and_process_query()

    def to_list(self) -> List[Dict[str, Any]]:
        """Execute the query and return results as list of dictionaries."""
        data = self._execute_and_process_query()
        return data.to_pylist()


class DorisTable:
    """Represents a Doris table for vector operations."""

    def __init__(
        self,
        name: str,
        client: "DorisVectorClient",
        index_options: Optional[IndexOptions] = None,
    ):
        self.table_name = name
        self.client = client
        self._cursor: Optional[Any] = None
        self._prepared_cursor: Optional[Any] = None
        self.columns: List[Tuple[str, str]] = (
            self._get_columns()
        )  # List of (name, type)
        self.column_names: List[str] = [name for name, _ in self.columns]
        self.ddl_compiler = DorisDDLCompiler()
        self.index_options = index_options or IndexOptions()
        self._vector_column: Optional[str] = None

    def get_session(self):
        return self.client.connection

    def _get_cursor(self, prepared: bool = False):
        """Get or create a cursor, reusing if possible."""
        if prepared:
            if self._prepared_cursor is None:
                self._prepared_cursor = self.client.connection.cursor(buffered=False, prepared=True)
            return self._prepared_cursor
        else:
            if self._cursor is None:
                self._cursor = self.client.connection.cursor(buffered=False, prepared=False)
            return self._cursor

    def schema(self):
        return self.columns

    def _get_columns(self) -> List[Tuple[str, str]]:
        """Get column names and types from Doris."""
        try:
            cursor = self._get_cursor()
            cursor.execute(f"DESCRIBE {self.table_name}")
            columns = []
            for row in cursor.fetchall():
                col_name = str(row[0])  # Field name
                col_type = str(row[1])  # Type
                columns.append((col_name, col_type))
            return columns
        except Exception as e:
            logger.warning(f"Failed to get columns for table {self.table_name}: {e}")
            raise e

    def search(
        self,
        query_vector: List[float],
        vector_column: Optional[str] = None,
        metric_type: str = "l2_distance",
    ) -> VectorSearchQuery:
        """Perform vector search."""
        if vector_column is None:
            # If vector_column is not specified, try to auto detect it
            vector_column = self._detect_vector_column()

        if self._get_vector_dim(vector_column) != len(query_vector):
            raise ValueError(
                f"Invalid query vector length: vector column dim={self._get_vector_dim(vector_column)}, but query vector length={len(query_vector)}"
            )

        return VectorSearchQuery(self, query_vector, vector_column, metric_type)

    def add(self, block: Block, load_options: Optional[LoadOptions] = None):
        """Add data to the table.

        Args:
            data: Input data in supported formats (list of dicts, pandas DataFrame, pyarrow Table, etc.)
            load_options: Options for data loading (default: None, uses arrow format)

        Raises:
            ValueError: If data schema doesn't match table schema
            Exception: If data insertion fails
        """
        # Set default load options if not provided
        if not load_options:
            load_options = self.client.load_options or LoadOptions()
        if not self.client.load_options:
            self.client.load_options = load_options

        # Convert data to Arrow Table format
        arrow_table = block_to_arrow_table(block)

        # Validate input data
        if arrow_table.num_rows == 0:
            logger.warning("Input data is empty, nothing to add.")
            return

        self.client._validate_input_data(arrow_table)

        # Get table schema
        table_schema = self._get_table_schema()
        if not table_schema:
            raise ValueError(f"Could not retrieve schema for table {self.table_name}")

        # Validate data schema
        self._validate_data_schema(arrow_table, table_schema)

        # Prepare data for insertion
        vector_column = self._detect_vector_column()

        data_vec_dim = len(arrow_table.column(vector_column)[0].as_py())
        if self._get_vector_dim(vector_column) != data_vec_dim:
            raise ValueError(
                f"Inconsistent vector column dimensions: table vector column dim={self._get_vector_dim(vector_column)}, data vector column dim={data_vec_dim}"
            )

        key_column = self._detect_key_column()
        schema_info = TableSchema(
            columns=table_schema,
            key_column=key_column,
            vector_column=vector_column,
            vector_dim=0,  # Not needed for existing tables
        )

        # Insert data using stream load
        try:
            self.client._insert_data_stream_load(
                self.table_name, arrow_table, schema_info, load_options
            )
            logger.debug(
                f"Successfully added {arrow_table.num_rows} rows to table {self.table_name}"
            )
        except Exception as e:
            logger.error(f"Failed to add data to table {self.table_name}: {e}")
            raise

    def _get_table_schema(self) -> Dict[str, Dict[str, Any]]:
        """Get the table schema from Doris."""
        try:
            cursor = self._get_cursor()
            cursor.execute(f"DESCRIBE {self.table_name}")
            schema = {}

            for row in cursor.fetchall():
                # Field | Type | Null | Key | Default | Extra
                col_name, col_type, is_null, key, default, extra = row
                schema[col_name] = {
                    "doris_type": col_type,
                    "nullable": is_null.upper() == "YES",
                    "is_key": key.upper() == "YES",
                    "default": default if default != "NULL" else None,
                }

            return schema

        except Exception as e:
            logger.error(f"Failed to get schema for table {self.table_name}: {e}")
            return {}

    def _validate_data_schema(
        self, data: pa.Table, table_schema: Dict[str, Dict[str, Any]]
    ):
        """Validate that data schema matches table schema.

        Args:
            data: Input Arrow Table
            table_schema: Table schema from Doris

        Raises:
            ValueError: If schema validation fails
        """
        # Check if all data columns exist in table
        table_columns = set(table_schema.keys())
        data_columns = set(data.column_names)

        # Check for extra columns in data that don't exist in table
        extra_columns = data_columns - table_columns
        if extra_columns:
            raise ValueError(
                f"Data contains columns not in table schema: {extra_columns}"
            )

        # Check for required columns (non-nullable columns without defaults)
        required_columns = {
            col_name
            for col_name, col_info in table_schema.items()
            if not col_info["nullable"] and col_info["default"] is None
        }

        missing_required = required_columns - data_columns
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")

        # Validate data types for each column
        for col_name in data.column_names:
            if col_name not in table_schema:
                continue  # Should not happen due to previous check

            col_info = table_schema[col_name]
            doris_type = col_info["doris_type"].upper()

            # Sample first non-null value for type checking
            sample_value = None
            col = data.column(col_name)
            for i in range(min(10, data.num_rows)):  # Check first 10 rows
                val = col[i]
                if val is not None:
                    try:
                        py_val = val.as_py()
                        if py_val is not None:
                            sample_value = py_val
                            break
                    except:
                        continue

            if sample_value is None:
                # All values are null, check if column allows null
                if not col_info["nullable"]:
                    raise ValueError(
                        f"Column '{col_name}' is NOT NULL but contains null values"
                    )
                continue

            # Validate data type compatibility
            if doris_type.startswith("ARRAY<"):
                if not isinstance(sample_value, list):
                    raise ValueError(
                        f"Column '{col_name}' expects array type, got {type(sample_value)}"
                    )
                if not sample_value or not isinstance(sample_value[0], (int, float)):
                    raise ValueError(f"Column '{col_name}' expects array of numbers")
            elif "INT" in doris_type or "BIGINT" in doris_type:
                if not isinstance(sample_value, int):
                    raise ValueError(
                        f"Column '{col_name}' expects integer type, got {type(sample_value)}"
                    )
            elif "FLOAT" in doris_type or "DOUBLE" in doris_type:
                if not isinstance(sample_value, (int, float)):
                    raise ValueError(
                        f"Column '{col_name}' expects numeric type, got {type(sample_value)}"
                    )
            elif (
                "TEXT" in doris_type
                or "VARCHAR" in doris_type
                or "STRING" in doris_type
            ):
                if not isinstance(sample_value, str):
                    raise ValueError(
                        f"Column '{col_name}' expects string type, got {type(sample_value)}"
                    )

        logger.debug(f"Schema validation passed for {len(data.columns)} columns")

    def _detect_key_column(self) -> str:
        """Detect the key column from table schema."""
        try:
            table_schema = self._get_table_schema()
            for col_name, col_info in table_schema.items():
                if col_info["is_key"]:
                    return col_name

            # Fallback: we use first column as key
            if table_schema:
                return list(table_schema.keys())[0]

        except Exception as e:
            logger.warning(f"Failed to detect key column: {e}")
        raise Exception("No key column")

    def _detect_vector_column(self) -> str:
        """Auto-detect the vector column name from table schema."""
        if self._vector_column is not None:
            return self._vector_column

        # Check for ARRAY columns (vectors)
        for col_name, col_type in self.columns:
            if col_type.upper().startswith("ARRAY"):
                logger.debug(f"Auto-detected vector column: '{col_name}'")
                self._vector_column = col_name
                return col_name

        raise ValueError("No vector column")

    def _get_vector_dim(self, vector_column: str) -> int:
        """Get the dimension of the vector column."""
        # 1. If have valid index_options.dim value => use index_options.dim
        # 2. If have sample data and index_options.dim = -1 => update index_options.dim and return
        # 3. If no data and index_options.dim is invalid => raise Exception
        if self.index_options.dim != -1:
            return self.index_options.dim

        # Try to get dim from table metadata first
        try:
            dim = self._get_dim_from_table_metadata(vector_column)
            if dim != -1:
                self.index_options.dim = dim
                return self.index_options.dim
        except Exception as e:
            logger.warning(f"Failed to get vector dimension from metadata: {e}")

        try:
            # Sample a row to get vector dimension
            cursor = self._get_cursor()
            cursor.execute(f"SELECT `{vector_column}` FROM `{self.table_name}` LIMIT 1")
            row = cursor.fetchone()
            if row and row[0]:
                # Parse the vector string format like: [1.0,2.0,3.0]
                vector_str = str(row[0])
                if vector_str.startswith("[") and vector_str.endswith("]"):
                    vec_list = vector_str[1:-1].split(",")
                    # Udpate index_options.dim
                    self.index_options.dim = len(vec_list)
                    return self.index_options.dim
        except Exception as e:
            logger.warning(f"Failed to get vector dimension: {e}")
        raise ValueError("Failed to get vector dimension")

    def _get_dim_from_table_metadata(self, vector_column: str) -> int:
        """Get dimension from table's CREATE TABLE statement."""
        try:
            cursor = self._get_cursor()
            cursor.execute(f"SHOW CREATE TABLE `{self.table_name}`")
            row = cursor.fetchone()
            if row:
                create_table_sql = row[-1]
                # Handle bytes to string conversion
                if isinstance(create_table_sql, bytes):
                    create_table_sql = create_table_sql.decode('utf-8')
                # Parse the PROPERTIES in the INDEX clause
                # Looking for: USING ANN PROPERTIES("dim" = "xxx", ...)
                match = re.search(r'USING\s+ANN\s+PROPERTIES\s*\((.*?)\)', create_table_sql, re.IGNORECASE)
                if match:
                    properties_str = match.group(1)
                    # Find "dim" = "xxx"
                    dim_match = re.search(r'"dim"\s*=\s*"(\d+)"', properties_str)
                    if dim_match:
                        return int(dim_match.group(1))
        except Exception as e:
            logger.warning(f"Failed to parse dim from CREATE TABLE: {e}")
        return -1

    def add_index(self, options: IndexOptions):
        """Create and build a vector index on the table.

        Args:
            options: IndexOptions containing index configuration

        Raises:
            Exception: If index creation or building fails
        """
        vector_column = self._detect_vector_column()
        dim = self._get_vector_dim(vector_column)

        index_options_dict = {
            "index_type": options.index_type,
            "metric_type": options.metric_type,
        }

        create_sql = self.ddl_compiler.compile_create_index(
            self.table_name, vector_column, index_options_dict, dim
        )
        build_sql = self.ddl_compiler.compile_build_index(
            self.table_name, vector_column
        )

        try:
            cursor = self._get_cursor()
            # Create index
            logger.debug(f"Creating index with SQL: {create_sql}")
            cursor.execute(create_sql)
            logger.debug(f"Successfully created index idx_{vector_column}")

            # Build index
            logger.debug(f"Building index with SQL: {build_sql}")
            cursor.execute(build_sql)
            logger.debug(f"Successfully built index idx_{vector_column}")

        except Exception as e:
            logger.error(f"Failed to create/build index idx_{vector_column}: {e}")
            raise

    def drop_index(self):
        """Drop the vector index from the table.

        Raises:
            Exception: If index dropping fails
        """
        vector_column = self._detect_vector_column()

        compiler = self.ddl_compiler
        drop_sql = compiler.compile_drop_index(self.table_name, vector_column)

        logger.debug(f"Dropping index with SQL: {drop_sql}")

        try:
            cursor = self._get_cursor()
            cursor.execute(drop_sql)
            logger.debug(f"Successfully dropped index idx_{vector_column}")

        except Exception as e:
            logger.error(f"Failed to drop index idx_{vector_column}: {e}")
            raise

    def close(self):
        if self._cursor:
            self._cursor.close()
            self._cursor = None
        if self._prepared_cursor:
            self._prepared_cursor.close()
            self._prepared_cursor = None


class DorisVectorClient:
    """Client for Doris Vector Search operations."""

    def __init__(
        self,
        database: str = "default",
        auth_options: Optional[AuthOptions] = None,
        load_options: Optional[LoadOptions] = None,
    ):
        self.database = database
        self.auth_options = auth_options if auth_options else AuthOptions()
        self.load_options = load_options if load_options else LoadOptions()

        # Create direct mysql.connector connection
        self.connection = mysql.connector.connect(
            host=self.auth_options.host,
            port=self.auth_options.query_port,
            user=self.auth_options.user,
            password=self.auth_options.password,
            database=self.database
        )

        self.ddl_compiler = DorisDDLCompiler()
        self.sql_compiler = DorisSQLCompiler()

        # Set default session variables
        self.with_sessions({
            "enable_profile": "false",
            "parallel_pipeline_task_num": "1",
            "num_scanner_threads": "1",
        })

    def create_table(
        self,
        table_name: str,
        block: Block,
        create_index: bool = True,
        index_options: Optional[IndexOptions] = None,
        load_options: Optional[LoadOptions] = None,
        overwrite: bool = False,
    ) -> DorisTable:
        """Create a new table from various data formats with dynamic schema inference.

        Args:
            table_name: Name of the table to create
            block: Input data in supported formats:
                - list of dicts
                - pandas DataFrame
                - pyarrow Table
                - pyarrow RecordBatch
                - iterable of pyarrow RecordBatch
            create_index: Whether to create ANN vector index (default: True)
            index_options: Configuration options for the vector index (default: None, uses defaults)
            load_options: Options for data loading (default: None, uses arrow format)
            overwrite: Whether to drop the table if it already exists (default: False)
        """
        # Set default load options if not provided
        if not load_options:
            load_options = self.load_options or LoadOptions()
        if not self.load_options:
            self.load_options = load_options

        # Convert data to Arrow Table format
        arrow_table = block_to_arrow_table(block)

        # Validate input data
        self._validate_input_data(arrow_table)

        # Check if table already exists
        try:
            cursor = self.connection.cursor()
            try:
                cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
                if cursor.fetchone():
                    if overwrite:
                        logger.debug(
                            f"Table '{table_name}' already exists. Dropping it as requested."
                        )
                        self.drop_table(table_name)
                    else:
                        logger.debug(
                            f"Table '{table_name}' already exists. Skipping creation."
                        )
                        return DorisTable(table_name, self)
            finally:
                cursor.close()
        except Exception as e:
            if "detailMessage = Unknown table" in f"{e}":
                pass
            else:
                raise e

        # Infer schema from data
        schema_info = self._infer_schema_from_data(arrow_table)

        # Prepare column definitions
        columns = {}
        key_column = schema_info.key_column
        vector_column = schema_info.vector_column

        if not key_column:
            raise ValueError("No suitable key column found in data")

        for col_name, col_info in schema_info.columns.items():
            columns[col_name] = col_info["doris_type"]

        # Prepare vector options
        vector_options = None
        if create_index:
            if index_options is None:
                index_options = IndexOptions()

            vector_options = {
                "index_type": index_options.index_type,
                "metric_type": index_options.metric_type,
                "dim": schema_info.vector_dim,
            }

        # Determine buckets by counting alive backends
        num_buckets = self._get_alive_be_count()
        logger.debug(f"Using {num_buckets} BUCKETS according to alive backends")

        # Create TableOptions object
        table_options = TableOptions(
            table_name=table_name,
            columns=columns,
            key_column=key_column,
            vector_column=vector_column if create_index else None,
            vector_options=vector_options,
            num_buckets=num_buckets,
        )

        # Compile DDL
        ddl = self.ddl_compiler.compile_create_table(table_options)

        logger.debug(f"Creating table with DDL: {ddl}")

        # Execute DDL
        cursor = self.connection.cursor()
        try:
            cursor.execute(ddl)

            # Insert data using stream load
            self._insert_data_stream_load(
                table_name, arrow_table, schema_info, load_options
            )
        finally:
            cursor.close()

        return DorisTable(table_name, self)

    def _validate_input_data(self, data: pa.Table):
        """Validate input Arrow Table has required structure."""
        if data.num_rows == 0:
            raise ValueError("Input Arrow Table cannot be empty")

        # Check for exactly one vector column
        vector_columns = []
        non_vector_columns = []

        for col_name in data.column_names:
            col = data.column(col_name)
            # Get first non-null value for type checking
            sample_value = None
            for i in range(min(10, data.num_rows)):  # Check first 10 rows
                val = col[i]
                if val is not None:
                    try:
                        py_val = val.as_py()
                        if py_val is not None:
                            sample_value = py_val
                            break
                    except:
                        continue

            # Check if it's a list/array type (vector)
            if (
                isinstance(sample_value, list)
                and sample_value
                and isinstance(sample_value[0], (int, float))
            ):
                vector_columns.append(col_name)
            else:
                non_vector_columns.append(col_name)

        if len(vector_columns) == 0:
            raise ValueError(
                "Input Arrow Table must have exactly one vector column (list of numbers)"
            )
        elif len(vector_columns) > 1:
            raise ValueError(
                f"Input Arrow Table can have only one vector column, found: {vector_columns}"
            )

        if len(non_vector_columns) == 0:
            raise ValueError(
                "Input Arrow Table must have at least one non-vector column for the key"
            )

        logger.debug(
            f"Validated input data: key column '{non_vector_columns[0]}', vector column '{vector_columns[0]}', {len(data.columns)} total columns"
        )

    def _infer_schema_from_data(self, data: pa.Table) -> TableSchema:
        """Infer Doris schema from Arrow Table."""
        columns: Dict[str, Dict[str, Any]] = {}
        vector_column: Optional[str] = None
        vector_dim: int = 0
        key_column: Optional[str] = None

        # First pass: identify vector and non-vector columns
        vector_columns = []
        non_vector_columns = []

        for col_name in data.column_names:
            col = data.column(col_name)
            # Get first non-null value for type checking
            sample_value = None
            for i in range(min(10, data.num_rows)):  # Check first 10 rows
                val = col[i]
                if val is not None:
                    try:
                        py_val = val.as_py()
                        if py_val is not None:
                            sample_value = py_val
                            break
                    except:
                        continue

            if (
                isinstance(sample_value, list)
                and sample_value
                and isinstance(sample_value[0], (int, float))
            ):
                vector_columns.append(col_name)
            else:
                non_vector_columns.append(col_name)

        # Set key column as the first non-vector column
        if non_vector_columns:
            key_column = non_vector_columns[0]

        # Second pass: infer schema for each column
        for col_name in data.column_names:
            col = data.column(col_name)
            # Sample first non-null value
            sample_value = None
            for i in range(min(10, data.num_rows)):  # Check first 10 rows
                val = col[i]
                if val is not None:
                    try:
                        py_val = val.as_py()
                        if py_val is not None:
                            sample_value = py_val
                            break
                    except:
                        continue

            if sample_value is None:
                # Default to TEXT for empty columns
                columns[col_name] = {"doris_type": "TEXT"}
                continue

            # Infer type from sample value
            if isinstance(sample_value, list):
                # Check if it's a vector (list of numbers)
                if sample_value and isinstance(sample_value[0], (int, float)):
                    columns[col_name] = {"doris_type": "ARRAY<FLOAT>"}
                    vector_column = col_name
                    vector_dim = len(sample_value)
                else:
                    raise ValueError(
                        f"Unsupported list type in column '{col_name}': {type(sample_value[0])}"
                    )
            elif isinstance(sample_value, int):
                columns[col_name] = {"doris_type": "INT"}
            elif isinstance(sample_value, float):
                columns[col_name] = {"doris_type": "FLOAT"}
            elif isinstance(sample_value, str):
                columns[col_name] = {"doris_type": "TEXT"}  # VARCHAR?
            else:
                # Default to TEXT for unknown types
                logger.warning(
                    f"Unknown type for column '{col_name}': {type(sample_value)}, defaulting to TEXT"
                )
                columns[col_name] = {"doris_type": "TEXT"}

        return TableSchema(
            columns=columns,
            vector_column=vector_column,
            vector_dim=vector_dim,
            key_column=key_column,
        )

    def open_table(self, table_name: str) -> DorisTable:
        """Open a table for vector operations."""
        return DorisTable(table_name, self)

    def _get_stream_load_url(self, table_name: str) -> str:
        """Get the Stream Load URL for the given table."""
        return f"http://{self.auth_options.host}:{self.auth_options.http_port}/api/{self.database}/{table_name}/_stream_load"

    def _send_stream_load_request(
        self,
        table_name: str,
        data: pa.Table,
        schema_info: TableSchema,
        load_format: StreamLoadFormat,
        max_retry: int = 3,
    ):
        """Send data via Stream Load API."""
        # Serialize data

        serialized_data = load_format.serialize_data(data, schema_info)

        # Get columns for CSV format
        columns = None
        if isinstance(load_format, CSVStreamLoadFormat):
            columns = load_format.get_columns(data)

        url = self._get_stream_load_url(table_name)
        logger.debug(
            f"Stream load to {url}, data size {len(serialized_data) / 1024 / 1024:.2f} MB, format: {load_format.format_name}"
        )

        # Build Basic Auth
        auth = HTTPBasicAuth(self.auth_options.user, self.auth_options.password)

        # Build headers
        headers = {
            "Content-Type": load_format.content_type,
            "Expect": "100-continue",
        }

        # Add format-specific headers
        headers.update(load_format.get_headers())

        # Add columns header for CSV format
        if columns:
            headers["columns"] = ",".join(columns)

        # Send request with retry logic
        for attempt in range(max_retry):
            response = None
            try:
                session = requests.Session()
                session.should_strip_auth = (
                    lambda old_url, new_url: False
                )  # Don't strip auth

                response = session.put(
                    url,
                    data=serialized_data,
                    headers=headers,
                    timeout=36000,
                    auth=auth,
                )
                response.raise_for_status()

                result = response.json()
                if result.get("Status") != "Success":
                    logger.error(f"Stream load failed: {result}")
                    if result.get("Status") != "Publish Timeout":
                        raise Exception(f"Stream load failed: {result}")
                return

            except requests.exceptions.HTTPError as e:
                if response and response.status_code == 307:  # Redirect
                    url = response.headers.get("Location", url)
                    logger.debug(f"Redirect to {url}")
                    continue

                try:
                    error_result = response.json() if response else {}
                except Exception:
                    error_result = response.text if response else ""

                status_code = response.status_code if response else "unknown"
                logger.error(f"Stream load HTTP error {status_code}: {error_result}")
                raise

            except Exception as e:
                logger.exception(
                    f"Stream load request failed (attempt {attempt + 1}): {e}"
                )

        raise Exception(f"Stream load failed after {max_retry} attempts")

    def _insert_data_stream_load(
        self,
        table_name: str,
        data: pa.Table,
        schema_info: TableSchema,
        load_options: LoadOptions,
    ):
        """Insert data using Stream Load for better performance with large datasets."""
        # Get the format handler
        load_stream = StreamLoadFormatFactory.get_format(load_options.format)

        # If data is small, use single request
        if data.num_rows <= self.load_options.batch_size:
            self._send_stream_load_request(table_name, data, schema_info, load_stream)
        else:
            # Split into batches for large datasets
            self._insert_data_stream_load_batch(
                table_name, data, schema_info, load_options
            )

        logger.debug(
            f"Successfully inserted {data.num_rows} rows into {table_name} using Stream Load ({load_options.format} format)"
        )

    def _insert_data_stream_load_batch(
        self,
        table_name: str,
        data: pa.Table,
        schema_info: TableSchema,
        load_options: LoadOptions,
        num_parallel: int = 8,
    ):
        """Insert data in batches using concurrent Stream Load."""
        # Get the format handler
        load_format = StreamLoadFormatFactory.get_format(load_options.format)

        batch_size = self.load_options.batch_size
        num_batches = math.ceil(data.num_rows / batch_size)

        logger.debug(
            f"Inserting {data.num_rows} rows in {num_batches} batches of size {batch_size} ({load_format.format_name} format)"
        )

        # Create batches
        batches = []
        for i in range(0, data.num_rows, batch_size):
            end_idx = min(i + batch_size, data.num_rows)
            batch_data = data.slice(i, end_idx - i)
            batches.append(batch_data)

        # Process batches concurrently
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(num_batches, num_parallel)
        ) as executor:
            futures = []
            for batch_idx, batch_data in enumerate(batches):
                future = executor.submit(
                    self._process_batch_stream_load,
                    table_name,
                    batch_data,
                    schema_info,
                    load_options,
                    batch_idx,
                )
                futures.append(future)

            # Wait for all batches to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()

    def _process_batch_stream_load(
        self,
        table_name: str,
        batch_data: pa.Table,
        schema_info: TableSchema,
        load_options: LoadOptions,
        batch_idx: int,
    ):
        """Process a single batch for Stream Load."""
        # Get the format handler
        load_format = StreamLoadFormatFactory.get_format(load_options.format)

        self._send_stream_load_request(table_name, batch_data, schema_info, load_format)
        logger.debug(f"Batch {batch_idx} completed")

    def drop_table(self, table_name: str):
        """Drop a table from the database.

        Args:
            table_name: Name of the table to drop

        Raises:
            Exception: If table drop fails
        """
        logger.debug(f"Dropping table '{table_name}'")

        try:
            compiler = self.ddl_compiler
            drop_sql = compiler.compile_drop_table(table_name)

            logger.debug(f"Executing SQL: {drop_sql}")

            cursor = self.connection.cursor()
            try:
                cursor.execute(drop_sql)
            finally:
                cursor.close()
            logger.debug(f"Successfully dropped table '{table_name}'")
        except Exception as e:
            logger.error(f"Failed to drop table '{table_name}': {e}")
            raise

    def with_session(self, key: str, value: Any) -> Self:
        """Set a session variable in Doris.

        Args:
            key: The session variable name
            value: The value to set for the variable

        Returns:
            Self
        """
        sql = self.sql_compiler.compile_set_session(key, value)
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql)
        finally:
            cursor.close()
        logger.debug(f"Set session variable {key} = {value}")
        return self

    def with_sessions(self, variables: Dict[str, Any]):
        """Set multiple session variables in Doris.

        Args:
            variables: Dictionary of variable names to values
        """
        for key, value in variables.items():
            self.with_session(key, value)

    def _get_alive_be_count(self) -> int:
        """Get the count of alive backends."""
        try:
            cursor = self.connection.cursor()
            try:
                cursor.execute("SHOW BACKENDS")
                rows = cursor.fetchall()
                # Get column names from cursor.description
                col_names = [desc[0] for desc in cursor.description] if cursor.description else []
                alive_idx = None
                if col_names:
                    for i, n in enumerate(col_names):
                        if str(n).lower() == "alive":
                            alive_idx = i
                            break
                count = 0
                if alive_idx is None:
                    # Fallback: assume all rows are backends
                    count = len(rows) if rows else 0
                else:
                    for r in rows:
                        sval = str(r[alive_idx]).strip().lower()
                        if sval in ("true", "1", "yes", "y"):
                            count += 1
                return max(1, count)
            finally:
                cursor.close()
        except Exception as e:
            logger.warning(f"SHOW BACKENDS failed, fallback to 1 bucket: {e}")
            return 1

    def close(self):
        """Close the client connection."""
        if self.connection:
            self.connection.close()
