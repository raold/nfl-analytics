# N+1 Query Batching - Quick Reference

**Last Updated:** November 22, 2025
**Status:** ✅ Production Ready

---

## TL;DR - Use These Instead

| Old (N+1 Anti-Pattern) | New (Batched) | Speedup |
|------------------------|---------------|---------|
| Loop → `fetch_predictions(game_id)` | `predictions_batch(game_ids)` → lookup | **Nx** |
| Loop → `fetch_line_movement(player)` | `line_movement_batch(players)` → lookup | **Nx** |
| Loop → `call_game_api(game_id)` | `get_game_predictions_batch(game_ids)` | **~Nx** |

**Key Principle:** Fetch ONCE before loop, lookup MANY times inside loop.

---

## Quick Examples

### Example 1: Batch Player Predictions

**❌ BAD (N+1):**
```python
for game_id in game_ids:
    df = pd.read_sql_query("SELECT * FROM predictions WHERE game_id = %s", conn, params=[game_id])
    # Process df...
```

**✅ GOOD (Batched):**
```python
from dashboard.queries.predictions_batched import BatchedPredictionQueries, create_lookup_dict

# Fetch ONCE
df_all = BatchedPredictionQueries.predictions_batch(conn, game_ids)
pred_dict = create_lookup_dict(df_all, 'game_id')

# Lookup MANY
for game_id in game_ids:
    preds = pred_dict.get(game_id, [])
    # Process preds...
```

---

### Example 2: Batch Line Movement

**❌ BAD (N+1):**
```python
for player in players:
    df = fetch_line_movement(player)  # Query per player
    # Process df...
```

**✅ GOOD (Batched):**
```python
from dashboard.queries.predictions_batched import BatchedPredictionQueries, create_lookup_dict

# Fetch ONCE
df_all = BatchedPredictionQueries.line_movement_batch(conn, players)
movement_dict = create_lookup_dict(df_all, 'player_name')

# Lookup MANY
for player in players:
    df = movement_dict.get(player, pd.DataFrame())
    # Process df...
```

---

### Example 3: Batch API Calls (Parallel)

**❌ BAD (Sequential):**
```python
for game_id in game_ids:
    pred = call_game_api(game_id)  # Sequential (blocks 500ms each)
    # Process pred...
```

**✅ GOOD (Parallel):**
```python
from dashboard.utils.api_client_batched import get_game_predictions_batch

# Fetch ONCE (parallel execution)
pred_dict = get_game_predictions_batch(game_ids, season, week)

# Lookup MANY
for game_id in game_ids:
    pred = pred_dict.get(game_id, {})
    # Process pred...
```

---

## Available Functions

### Database Queries

```python
from dashboard.queries.predictions_batched import BatchedPredictionQueries

# Player predictions (WHERE IN clause)
df = BatchedPredictionQueries.predictions_batch(conn, game_ids, stat_type='passing_yards')

# Line movement history
df = BatchedPredictionQueries.line_movement_batch(conn, player_names, days_back=7)

# Game predictions (DISTINCT ON)
df = BatchedPredictionQueries.game_predictions_batch(conn, game_ids)

# Predictions + odds (JOIN)
df = BatchedPredictionQueries.predictions_with_odds_batch(conn, game_ids)
```

### API Calls

```python
from dashboard.utils.api_client_batched import BatchedAPIClient

client = BatchedAPIClient()

# Game predictions (parallel HTTP)
pred_dict = client.get_game_predictions_batch(game_ids, season, week)

# Player predictions (parallel HTTP)
pred_dict = client.get_player_predictions_batch(player_ids, season, week, stat_type='passing_yards')
```

### Lookup Helper

```python
from dashboard.queries.predictions_batched import create_lookup_dict

# Convert DataFrame to O(1) lookup dictionary
df = pd.DataFrame({'game_id': [...], 'value': [...]})
lookup = create_lookup_dict(df, 'game_id')

# Fast lookups
game_data = lookup.get('2024_12_BAL_LAC', [])
```

---

## Common Mistakes

### Mistake 1: Calling Batched Function Inside Loop

**❌ WRONG:**
```python
for game_id in game_ids:
    # PROBLEM: Still executes N queries!
    df = BatchedPredictionQueries.predictions_batch(conn, [game_id])
```

**✅ RIGHT:**
```python
# Call ONCE before loop
df_all = BatchedPredictionQueries.predictions_batch(conn, game_ids)
pred_dict = create_lookup_dict(df_all, 'game_id')

for game_id in game_ids:
    preds = pred_dict.get(game_id, [])
```

---

### Mistake 2: Filtering DataFrame in Loop (Slow)

**❌ SLOW (O(N²)):**
```python
df_all = BatchedPredictionQueries.predictions_batch(conn, game_ids)

for game_id in game_ids:
    # PROBLEM: O(N) filter per iteration = O(N²) total
    df_game = df_all[df_all['game_id'] == game_id]
```

**✅ FAST (O(N)):**
```python
df_all = BatchedPredictionQueries.predictions_batch(conn, game_ids)
pred_dict = create_lookup_dict(df_all, 'game_id')  # O(N) once

for game_id in game_ids:
    # O(1) lookup per iteration = O(N) total
    preds = pred_dict.get(game_id, [])
```

---

### Mistake 3: Forgetting to Convert numpy.int64 to int

**❌ FAILS:**
```python
game_ids = df['game_id'].tolist()  # Returns numpy.int64
BatchedPredictionQueries.predictions_batch(conn, game_ids)  # psycopg3 error!
```

**✅ WORKS:**
```python
game_ids = [int(gid) for gid in df['game_id'].tolist()]
BatchedPredictionQueries.predictions_batch(conn, game_ids)
```

---

## Performance Tips

### Tip 1: Always Check for Empty Lists

```python
if not game_ids:
    return pd.DataFrame()  # Early return

df = BatchedPredictionQueries.predictions_batch(conn, game_ids)
```

### Tip 2: Use Parallel API Calls (Not Sequential)

```python
# ❌ Sequential (8 seconds for 16 games)
for game_id in game_ids:
    pred = call_game_api(game_id)

# ✅ Parallel (~500ms for 16 games)
pred_dict = get_game_predictions_batch(game_ids, season, week)
```

### Tip 3: Parameterize Queries (Prevent SQL Injection)

```python
# ❌ DANGEROUS (SQL injection risk)
query = f"SELECT * FROM predictions WHERE game_id IN ({','.join(game_ids)})"

# ✅ SAFE (parameterized)
placeholders = ','.join(['%s'] * len(game_ids))
query = f"SELECT * FROM predictions WHERE game_id IN ({placeholders})"
df = pd.read_sql_query(query, conn, params=game_ids)
```

---

## Testing

### Unit Test Template

```python
@patch('dashboard.queries.predictions_batched.pd.read_sql_query')
def test_predictions_batch(mock_read_sql, mock_conn):
    """Test batched predictions query."""
    game_ids = ['game1', 'game2', 'game3']
    mock_read_sql.return_value = pd.DataFrame({'game_id': game_ids, 'value': [1, 2, 3]})

    result = BatchedPredictionQueries.predictions_batch(mock_conn, game_ids)

    # Verify query called ONCE (not 3 times!)
    assert mock_read_sql.call_count == 1

    # Verify all games returned
    assert len(result) == 3
```

---

## Migration Checklist

- [ ] Identify loops with database queries inside
- [ ] Replace with batched query before loop
- [ ] Use `create_lookup_dict()` for fast lookups
- [ ] Test with small dataset first
- [ ] Verify query count reduced (browser DevTools)
- [ ] Measure page load time improvement
- [ ] Deploy to production

---

## Full Documentation

- **Complete Guide:** [docs/deployment/N+1_QUERY_BATCHING_COMPLETE.md](../deployment/N+1_QUERY_BATCHING_COMPLETE.md)
- **Implementation:** [dashboard/queries/predictions_batched.py](../../dashboard/queries/predictions_batched.py)
- **Tests:** [tests/unit/dashboard/test_predictions_batched.py](../../tests/unit/dashboard/test_predictions_batched.py)
