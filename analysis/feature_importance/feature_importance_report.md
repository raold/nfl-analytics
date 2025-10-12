# Feature Importance Analysis Report
**Model**: models/xgboost/v3_production/model.json
**Total Features**: 63

## Summary
- **Features to Keep**: 53
- **Features to Prune**: 6
- **Pruned Importance**: 8.34%

## Performance Impact
- **Full Model Brier**: 0.2121
- **Pruned Model Brier**: 0.2140
- **Difference**: +0.0019

## Top 20 Features
51. **spread_close** - Gain: 41.7, Weight: 94
57. **venue_home_win_rate** - Gain: 10.4, Weight: 66
44. **home_success_rate_season** - Gain: 6.5, Weight: 41
52. **success_rate_l3_diff** - Gain: 6.2, Weight: 40
41. **home_rush_epa_l5** - Gain: 6.0, Weight: 40
18. **away_success_rate_l5** - Gain: 7.2, Weight: 34
2. **away_epa_home_avg** - Gain: 8.1, Weight: 39
32. **home_pass_epa_l5** - Gain: 5.7, Weight: 36
47. **pass_epa_l5_diff** - Gain: 5.8, Weight: 29
23. **epa_per_play_l3_diff** - Gain: 5.7, Weight: 33
38. **home_points_l3** - Gain: 5.3, Weight: 30
35. **home_points_against_l5** - Gain: 5.7, Weight: 19
11. **away_points_against_season** - Gain: 7.1, Weight: 23
22. **epa_per_play_l10_diff** - Gain: 7.1, Weight: 23
1. **away_epa_away_avg** - Gain: 6.4, Weight: 24
34. **home_points_against_l3** - Gain: 6.5, Weight: 27
16. **away_rush_epa_l5** - Gain: 4.8, Weight: 22
54. **total_close** - Gain: 6.1, Weight: 23
25. **home_epa_away_avg** - Gain: 7.0, Weight: 28
6. **away_losses** - Gain: 6.2, Weight: 13

## Bottom 10 Features (Candidates for Pruning)
- **home_wins** - Gain: 6.3, Composite: 0.2146
- **away_points_against_l10** - Gain: 4.9, Composite: 0.2047
- **away_epa_per_play_season** - Gain: 7.5, Composite: 0.2041
- **home_epa_per_play_season** - Gain: 5.2, Composite: 0.2033
- **week** - Gain: 5.4, Composite: 0.1956
- **away_points_l10** - Gain: 5.6, Composite: 0.1936
- **home_epa_per_play_l10** - Gain: 7.3, Composite: 0.1741
- **venue_avg_total** - Gain: 5.2, Composite: 0.1698
- **home_losses** - Gain: 6.1, Composite: 0.1511
- **home_over_rate_l10** - Gain: 4.1, Composite: 0.0649
