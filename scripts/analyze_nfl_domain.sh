#!/usr/bin/env bash
#
# NFL Domain Expert - CLI Wrapper
# Usage: ./scripts/analyze_nfl_domain.sh [options]
#
# Options:
#   --injury <player_id> <type> <severity>  Analyze injury impact
#   --coaching <coach_name>                  Get coaching tendencies
#   --chemistry <qb_id> <wr_id>              Assess player chemistry
#   --news <team> <days>                     Fetch recent news
#   --context-features                       Suggest context-aware features
#
# Examples:
#   # Analyze injury
#   ./scripts/analyze_nfl_domain.sh --injury "00-0036355" "ankle_sprain" "QUESTIONABLE"
#
#   # Coaching tendencies
#   ./scripts/analyze_nfl_domain.sh --coaching "andy_reid"
#
#   # Player chemistry
#   ./scripts/analyze_nfl_domain.sh --chemistry "00-0036355" "00-0034869"
#
#   # Suggest features
#   ./scripts/analyze_nfl_domain.sh --context-features
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 22, 2025
# Phase: 6 (Frontend Agents)

set -euo pipefail

# Default values
ACTION=""
ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --injury|--coaching|--chemistry|--news|--context-features)
            ACTION="${1#--}"
            shift
            # Collect remaining args
            while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
                ARGS+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Execute based on action
case $ACTION in
    injury)
        if [ ${#ARGS[@]} -lt 3 ]; then
            echo "Usage: --injury <player_id> <injury_type> <severity>"
            exit 1
        fi
        python3 -c "
from nfl_analytics.agents.nfl_domain_expert import NFLDomainExpert
import json

expert = NFLDomainExpert()
report = expert.analyze_injury_impact(
    player_id='${ARGS[0]}',
    injury_type='${ARGS[1]}',
    severity='${ARGS[2]}'
)

print('\n=== Injury Impact Report ===')
print(f'Player: {report.player_name}')
print(f'Injury: {report.injury_type.value} ({report.severity.value})')
print(f'Expected miss: {report.expected_miss_games} games')
print(f'Performance impact: {report.performance_impact_pct:.0f}% reduction')
print(f'Replacement quality: {report.replacement_quality_pct:.0f}% of starter')
print(f'Historical recovery: {report.historical_recovery_days} days')
"
        ;;

    coaching)
        if [ ${#ARGS[@]} -lt 1 ]; then
            echo "Usage: --coaching <coach_name>"
            exit 1
        fi
        python3 -c "
from nfl_analytics.agents.nfl_domain_expert import NFLDomainExpert

expert = NFLDomainExpert()
profile = expert.get_coaching_tendencies('${ARGS[0]}')

print('\n=== Coaching Profile ===')
print(f'Coach: {profile.coach_name}')
print(f'Style: {profile.coaching_style.value}')
print(f'4th down aggression: {profile.fourth_down_aggression:.0f}%')
print(f'Timeout management: {profile.timeout_management_score:.0f}/100')
print(f'Halftime adjustments: {profile.halftime_adjustment_rating:.0f}/100')
print(f'\nPlay Calling:')
for key, value in profile.play_calling_tendencies.items():
    print(f'  {key}: {value}')
print(f'\nStrategic Patterns:')
for pattern in profile.strategic_patterns:
    print(f'  - {pattern}')
"
        ;;

    chemistry)
        if [ ${#ARGS[@]} -lt 2 ]; then
            echo "Usage: --chemistry <qb_id> <wr_id>"
            exit 1
        fi
        python3 -c "
from nfl_analytics.agents.nfl_domain_expert import NFLDomainExpert

expert = NFLDomainExpert()
chemistry = expert.assess_player_chemistry('${ARGS[0]}', '${ARGS[1]}')

print('\n=== Player Chemistry ===')
print(f'QB: {chemistry.player1_name}')
print(f'WR: {chemistry.player2_name}')
print(f'Chemistry score: {chemistry.chemistry_score:.0f}/100')
print(f'Games together: {chemistry.games_together}')
print(f'Trend: {chemistry.trend}')
"
        ;;

    context-features)
        python3 -c "
from nfl_analytics.agents.nfl_domain_expert import NFLDomainExpert

expert = NFLDomainExpert()
features = expert.suggest_context_features()

print('\n=== Suggested Context Features ===')
print(f'\nTotal suggestions: {len(features)}\n')

for i, feature in enumerate(features, 1):
    print(f'{i}. {feature[\"feature\"]}')
    print(f'   Description: {feature[\"description\"]}')
    print(f'   Expected impact: {feature[\"expected_impact\"]}')
    print(f'   Implementation: {feature[\"implementation\"]}')
    print()
"
        ;;

    *)
        echo "No action specified. Use --help for usage."
        exit 1
        ;;
esac
