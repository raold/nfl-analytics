"""
GitHub operations for NFL analytics MCP server.
Provides repository management, issue tracking, and documentation automation.
"""

import os
import logging
import subprocess
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# GitHub CLI must be installed and authenticated
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
REPO = "raold/nfl-analytics"  # Default repository


def _run_gh_command(args: List[str]) -> Dict[str, Any]:
    """
    Execute a GitHub CLI command.

    Args:
        args: List of command arguments

    Returns:
        Dictionary with command output or error
    """
    try:
        result = subprocess.run(
            ['gh'] + args,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            return {
                "success": True,
                "output": result.stdout.strip()
            }
        else:
            return {
                "success": False,
                "error": result.stderr.strip()
            }

    except FileNotFoundError:
        return {
            "success": False,
            "error": "GitHub CLI (gh) not installed. Install from: https://cli.github.com"
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Command timed out after 30 seconds"
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Command execution error: {str(e)}"
        }


def create_issue(
    title: str,
    body: str,
    labels: Optional[str] = None,
    assignees: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a GitHub issue.

    Args:
        title: Issue title
        body: Issue description (markdown supported)
        labels: Comma-separated list of labels
        assignees: Comma-separated list of assignees

    Returns:
        Dictionary with issue URL or error
    """
    args = [
        'issue', 'create',
        '--repo', REPO,
        '--title', title,
        '--body', body
    ]

    if labels:
        args.extend(['--label', labels])

    if assignees:
        args.extend(['--assignee', assignees])

    logger.info(f"Creating GitHub issue: {title}")

    result = _run_gh_command(args)

    if result['success']:
        logger.info(f"Issue created: {result['output']}")

    return result


def list_recent_issues(
    limit: int = 10,
    state: str = "open",
    labels: Optional[str] = None
) -> Dict[str, Any]:
    """
    List recent GitHub issues.

    Args:
        limit: Number of issues to return
        state: Issue state ('open', 'closed', 'all')
        labels: Filter by labels

    Returns:
        Dictionary with issues list or error
    """
    args = [
        'issue', 'list',
        '--repo', REPO,
        '--limit', str(limit),
        '--state', state,
        '--json', 'number,title,state,createdAt,labels'
    ]

    if labels:
        args.extend(['--label', labels])

    logger.info(f"Listing {limit} {state} issues")

    return _run_gh_command(args)


def create_pr(
    title: str,
    body: str,
    base: str = "main",
    head: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a pull request.

    Args:
        title: PR title
        body: PR description (markdown supported)
        base: Base branch (default: main)
        head: Head branch (uses current branch if not specified)

    Returns:
        Dictionary with PR URL or error
    """
    args = [
        'pr', 'create',
        '--repo', REPO,
        '--title', title,
        '--body', body,
        '--base', base
    ]

    if head:
        args.extend(['--head', head])

    logger.info(f"Creating pull request: {title}")

    result = _run_gh_command(args)

    if result['success']:
        logger.info(f"PR created: {result['output']}")

    return result


def get_pr_status(pr_number: int) -> Dict[str, Any]:
    """
    Get status of a pull request.

    Args:
        pr_number: PR number

    Returns:
        Dictionary with PR status or error
    """
    args = [
        'pr', 'view', str(pr_number),
        '--repo', REPO,
        '--json', 'number,title,state,mergeable,reviewDecision,statusCheckRollup'
    ]

    logger.info(f"Getting PR #{pr_number} status")

    return _run_gh_command(args)


def add_comment(
    issue_number: int,
    comment: str
) -> Dict[str, Any]:
    """
    Add a comment to an issue or PR.

    Args:
        issue_number: Issue or PR number
        comment: Comment text (markdown supported)

    Returns:
        Success/error dictionary
    """
    args = [
        'issue', 'comment', str(issue_number),
        '--repo', REPO,
        '--body', comment
    ]

    logger.info(f"Adding comment to issue #{issue_number}")

    return _run_gh_command(args)


def get_repo_stats() -> Dict[str, Any]:
    """
    Get repository statistics.

    Returns:
        Dictionary with repo stats or error
    """
    args = [
        'repo', 'view', REPO,
        '--json', 'name,description,stargazerCount,forkCount,openIssuesCount,watchers'
    ]

    logger.info("Getting repository statistics")

    return _run_gh_command(args)


def search_code(
    query: str,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Search code in the repository.

    Args:
        query: Search query
        limit: Maximum number of results

    Returns:
        Dictionary with search results or error
    """
    args = [
        'search', 'code',
        query,
        '--repo', REPO,
        '--limit', str(limit),
        '--json', 'path,textMatches'
    ]

    logger.info(f"Searching code: {query}")

    return _run_gh_command(args)
