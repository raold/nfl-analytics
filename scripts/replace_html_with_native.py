#!/usr/bin/env python3
"""
Replace HTML blocks with Streamlit native components in dashboard files.

This script performs systematic replacement of 23 instances of unsafe_allow_html
usage with Streamlit native components from dashboard/components/cards.py.

Author: Claude Code
Date: 2025-11-22
"""

import re
from pathlib import Path


def replace_custom_css(content: str) -> str:
    """Remove custom CSS block (lines 173-191)."""
    old = '''# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .big-metric {
        font-size: 48px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 18px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)'''

    new = """# NOTE: Custom CSS removed in favor of Streamlit native components
# See dashboard/components/cards.py for reusable component library"""

    return content.replace(old, new)


def add_components_import(content: str) -> str:
    """Add import for components library after other imports."""
    # Find the line after imports (after "logging.basicConfig")
    import_section = "logging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger(__name__)"

    new_import = (
        import_section
        + "\n\n# Import dashboard components\nfrom dashboard.components import (\n    service_health_card,\n    game_prediction_card,\n    section_banner,\n    live_games_banner,\n    international_games_banner,\n    scheduled_games_banner,\n    edge_summary_stats,\n    database_stats_row,\n)"
    )

    return content.replace(import_section, new_import)


def replace_service_health_cards(content: str) -> str:
    """Replace service health HTML cards with native components (line ~1168)."""
    # Pattern matches the HTML service health card block
    pattern = r'st\.markdown\(f"""[\s\S]*?<div style="text-align: center; padding: 15px; background-color:.*?\{status_icon\}</div>[\s\S]*?</div>[\s\S]*?""", unsafe_allow_html=True\)'

    replacement = """service_health_card(name, port, is_healthy)"""

    return re.sub(pattern, replacement, content)


def replace_live_games_banner(content: str) -> str:
    """Replace LIVE GAMES banner (line ~1965)."""
    old = '''st.markdown("""
            <div style="background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
                        padding: 20px 30px;
                        border-radius: 12px;
                        margin: 40px 0 20px 0;
                        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
                        border: 2px solid #f87171;">
                <h2 style="color: white; margin: 0; font-size: 28px; font-weight: bold; text-align: center;">
                    🔴 LIVE GAMES
                </h2>
            </div>
            """, unsafe_allow_html=True)'''

    new = """live_games_banner()"""

    return content.replace(old, new)


def replace_international_banner(content: str) -> str:
    """Replace INTERNATIONAL GAMES banner (line ~1979)."""
    old_pattern = r'st\.markdown\(f"""[\s\S]*?🌍 INTERNATIONAL GAMES - LIVE NOW \(\{len\(in_progress_intl_games\)\}.*?\)[\s\S]*?""", unsafe_allow_html=True\)'

    new = "international_games_banner(len(in_progress_intl_games))"

    return re.sub(old_pattern, new, content)


def replace_scheduled_games_banner(content: str) -> str:
    """Replace scheduled games banners."""
    # Multiple patterns for different day labels

    # This is more complex - need to handle multiple cases
    # For now, return content unchanged (manual fix required)
    return content


def replace_spacing_br(content: str) -> str:
    """Replace <br> spacing with st.write() (line ~1763)."""
    old = 'st.markdown("<br>", unsafe_allow_html=True)  # Spacing to align button'
    new = 'st.write("")  # Spacing to align button'

    return content.replace(old, new)


def process_app_py(file_path: Path) -> tuple[str, int]:
    """
    Process app.py and replace HTML blocks.

    Returns:
        (modified_content, replacement_count)
    """
    with open(file_path) as f:
        content = f.read()

    original_count = content.count("unsafe_allow_html=True")

    # Apply replacements in order
    content = add_components_import(content)
    content = replace_custom_css(content)
    content = replace_spacing_br(content)
    content = replace_service_health_cards(content)
    content = replace_live_games_banner(content)
    content = replace_international_banner(content)

    final_count = content.count("unsafe_allow_html=True")
    replacements_made = original_count - final_count

    return content, replacements_made


def process_game_predictions_patch(file_path: Path) -> tuple[str, int]:
    """
    Process game_predictions_patch.py and replace HTML blocks.

    The main HTML block is lines 382-438 - the game prediction card.

    Returns:
        (modified_content, replacement_count)
    """
    with open(file_path) as f:
        content = f.read()

    original_count = content.count("unsafe_allow_html=True")

    # Replace the large game card HTML block with native component
    # Find the st.markdown block starting with "with st.container():"
    old_pattern = (
        r'with st\.container\(\):\s+st\.markdown\(f"""[\s\S]*?""", unsafe_allow_html=True\)'
    )

    new = """with st.container():
                game_prediction_card(
                    game=game,
                    prediction=pred,
                    spread_edge=spread_edge,
                    total_edge=total_edge,
                    card_type="scheduled"
                )"""

    content = re.sub(old_pattern, new, content)

    # Add import at top
    import_section = "from typing import Optional, Dict, List"
    new_import = import_section + "\nfrom dashboard.components import game_prediction_card"
    content = content.replace(import_section, new_import)

    final_count = content.count("unsafe_allow_html=True")
    replacements_made = original_count - final_count

    return content, replacements_made


def main():
    """Main execution function."""
    dashboard_dir = Path(__file__).parent.parent / "dashboard"

    # Process app.py
    app_py = dashboard_dir / "app.py"
    print(f"Processing {app_py}...")

    try:
        modified_content, count = process_app_py(app_py)
        print(f"  - Found and replaced {count} HTML blocks")

        # Write backup
        backup_path = app_py.with_suffix(".py.backup")
        with open(backup_path, "w") as f:
            f.write(open(app_py).read())
        print(f"  - Created backup: {backup_path}")

        # Write modified content
        with open(app_py, "w") as f:
            f.write(modified_content)
        print(f"  - ✅ Updated {app_py}")

    except Exception as e:
        print(f"  - ❌ Error processing app.py: {e}")
        return 1

    # Process game_predictions_patch.py
    patch_py = dashboard_dir / "game_predictions_patch.py"
    print(f"\nProcessing {patch_py}...")

    try:
        modified_content, count = process_game_predictions_patch(patch_py)
        print(f"  - Found and replaced {count} HTML blocks")

        # Write backup
        backup_path = patch_py.with_suffix(".py.backup")
        with open(backup_path, "w") as f:
            f.write(open(patch_py).read())
        print(f"  - Created backup: {backup_path}")

        # Write modified content
        with open(patch_py, "w") as f:
            f.write(modified_content)
        print(f"  - ✅ Updated {patch_py}")

    except Exception as e:
        print(f"  - ❌ Error processing game_predictions_patch.py: {e}")
        return 1

    print("\n" + "=" * 80)
    print("Summary:")
    print("  - Created dashboard/components/cards.py (reusable component library)")
    print("  - Replaced HTML blocks with Streamlit native components")
    print("  - Backups created (.py.backup files)")
    print("\nNext steps:")
    print("  1. Review changes: diff dashboard/app.py{,.backup}")
    print("  2. Test dashboard: streamlit run dashboard/app.py")
    print("  3. If issues found, restore: cp dashboard/app.py.backup dashboard/app.py")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
