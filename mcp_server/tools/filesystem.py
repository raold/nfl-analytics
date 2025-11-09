"""
Filesystem operations for NFL analytics MCP server.
Provides safe read/write access to model artifacts, logs, and data files.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Allowed directories for file operations
# Auto-detect if running in Docker or locally
import os
BASE_DIR = '/app' if os.path.exists('/app/models') else os.getcwd()

ALLOWED_DIRS = {
    'models': os.path.join(BASE_DIR, 'models'),
    'logs': os.path.join(BASE_DIR, 'logs'),
    'data': os.path.join(BASE_DIR, 'data'),
    'cache': os.path.join(BASE_DIR, 'cache'),
    'results': os.path.join(BASE_DIR, 'results')
}

# File size limits (10 MB for read operations)
MAX_FILE_SIZE = 10 * 1024 * 1024


def _validate_path(directory_type: str, relative_path: str) -> Optional[Path]:
    """
    Validate and resolve file path within allowed directories.

    Args:
        directory_type: Type of directory ('models', 'logs', 'data', etc.)
        relative_path: Relative path within that directory

    Returns:
        Resolved Path object if valid, None if invalid
    """
    if directory_type not in ALLOWED_DIRS:
        logger.warning(f"Invalid directory type: {directory_type}")
        return None

    base_dir = Path(ALLOWED_DIRS[directory_type])
    try:
        # Resolve the full path and check it's within the allowed directory
        full_path = (base_dir / relative_path).resolve()

        # Security check: ensure path is within allowed directory
        if not str(full_path).startswith(str(base_dir)):
            logger.warning(f"Path traversal attempt: {relative_path}")
            return None

        return full_path

    except Exception as e:
        logger.error(f"Path validation error: {e}")
        return None


def list_files(
    directory_type: str,
    relative_path: str = "",
    pattern: str = "*"
) -> Dict[str, Any]:
    """
    List files in a directory.

    Args:
        directory_type: Type of directory ('models', 'logs', 'data', etc.)
        relative_path: Relative path within directory
        pattern: Glob pattern for filtering (e.g., "*.pkl", "*.log")

    Returns:
        Dictionary with file list or error
    """
    try:
        path = _validate_path(directory_type, relative_path)
        if not path:
            return {
                "success": False,
                "error": f"Invalid directory type or path: {directory_type}/{relative_path}"
            }

        if not path.exists():
            return {
                "success": False,
                "error": f"Path does not exist: {path}"
            }

        if not path.is_dir():
            return {
                "success": False,
                "error": f"Path is not a directory: {path}"
            }

        # List files matching pattern
        files = []
        for file_path in path.glob(pattern):
            stat = file_path.stat()
            files.append({
                "name": file_path.name,
                "path": str(file_path.relative_to(ALLOWED_DIRS[directory_type])),
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "is_dir": file_path.is_dir()
            })

        # Sort by name
        files.sort(key=lambda x: x['name'])

        logger.info(f"Listed {len(files)} files in {directory_type}/{relative_path}")

        return {
            "success": True,
            "files": files,
            "count": len(files)
        }

    except Exception as e:
        error_msg = f"Error listing files: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg
        }


def read_file(
    directory_type: str,
    relative_path: str,
    max_lines: Optional[int] = None
) -> Dict[str, Any]:
    """
    Read a text file.

    Args:
        directory_type: Type of directory
        relative_path: Path to file within directory
        max_lines: Optional limit on number of lines to read

    Returns:
        Dictionary with file contents or error
    """
    try:
        path = _validate_path(directory_type, relative_path)
        if not path:
            return {
                "success": False,
                "error": f"Invalid path: {directory_type}/{relative_path}"
            }

        if not path.exists():
            return {
                "success": False,
                "error": f"File does not exist: {path}"
            }

        # Check file size
        file_size = path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            return {
                "success": False,
                "error": f"File too large: {file_size} bytes (max: {MAX_FILE_SIZE})"
            }

        # Read file
        with open(path, 'r', encoding='utf-8') as f:
            if max_lines:
                lines = [f.readline() for _ in range(max_lines)]
                content = ''.join(lines)
                truncated = True
            else:
                content = f.read()
                truncated = False

        logger.info(f"Read file: {directory_type}/{relative_path} ({len(content)} chars)")

        return {
            "success": True,
            "content": content,
            "size": file_size,
            "truncated": truncated
        }

    except UnicodeDecodeError:
        error_msg = "File is not a text file (binary content detected)"
        logger.warning(error_msg)
        return {
            "success": False,
            "error": error_msg
        }

    except Exception as e:
        error_msg = f"Error reading file: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg
        }


def read_json(
    directory_type: str,
    relative_path: str
) -> Dict[str, Any]:
    """
    Read and parse a JSON file.

    Args:
        directory_type: Type of directory
        relative_path: Path to JSON file within directory

    Returns:
        Dictionary with parsed JSON data or error
    """
    try:
        path = _validate_path(directory_type, relative_path)
        if not path:
            return {
                "success": False,
                "error": f"Invalid path: {directory_type}/{relative_path}"
            }

        if not path.exists():
            return {
                "success": False,
                "error": f"File does not exist: {path}"
            }

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Read JSON file: {directory_type}/{relative_path}")

        return {
            "success": True,
            "data": data
        }

    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg
        }

    except Exception as e:
        error_msg = f"Error reading JSON file: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg
        }


def get_file_info(
    directory_type: str,
    relative_path: str
) -> Dict[str, Any]:
    """
    Get metadata about a file or directory.

    Args:
        directory_type: Type of directory
        relative_path: Path within directory

    Returns:
        Dictionary with file metadata or error
    """
    try:
        path = _validate_path(directory_type, relative_path)
        if not path:
            return {
                "success": False,
                "error": f"Invalid path: {directory_type}/{relative_path}"
            }

        if not path.exists():
            return {
                "success": False,
                "error": f"Path does not exist: {path}"
            }

        stat = path.stat()

        info = {
            "name": path.name,
            "path": str(path.relative_to(ALLOWED_DIRS[directory_type])),
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "created": stat.st_ctime,
            "is_dir": path.is_dir(),
            "is_file": path.is_file(),
            "extension": path.suffix
        }

        logger.info(f"Got file info: {directory_type}/{relative_path}")

        return {
            "success": True,
            "info": info
        }

    except Exception as e:
        error_msg = f"Error getting file info: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg
        }
