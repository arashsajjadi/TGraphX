# Legal Move Fixtures

Each JSON file in this directory is a test fixture for the legal move generator.

## Schema

```json
{
  "id": "unique_fixture_name",
  "description": "Human-readable description of the position",
  "board": [0, 0, ..., 0],        // length-24 array
  "bar": [0, 0],
  "borne_off": [0, 0],
  "current_player": 0,
  "dice": [3, 5],
  "expected_num_turns": 12,        // optional: exact count of legal turns
  "expected_turns": ["13/8 6/4"],  // optional: turns that must appear
  "forbidden_turns": ["8/2 6/4"],  // optional: turns that must NOT appear
  "source": "hand-crafted",        // or "gnubg" when validated externally
  "notes": "..."
}
```

## Adding External Engine Fixtures

To validate against GNU Backgammon:
1. Run `gnubg --tty` and set up the position.
2. Use `show legal moves` to get the list.
3. Convert the moves to the notation format used by this project.
4. Save as a JSON fixture here.

The `GnuBackgammonAdapter` in `src/backgammon_rlx/engines/gnu_backgammon.py`
provides an interface for automated validation when gnubg is installed.
