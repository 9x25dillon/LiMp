## Features

- **Entropy Calculation**: Based on SHA256 hash of token values
- **Dynamic Branching**: Nodes can create new child nodes based on token state
- **Entropy Limits**: Stop processing when entropy reaches certain thresholds
- **Memory Tracking**: Complete logs of all transformations
- **Flexible Transformations**: Any function that takes (value, entropy) can be used

## Example Output

```
Initial token: <Token a1b2c3 val=hello entropy=7.23>
Final token: <Token a1b2c3 val=ollehx7 entropy=6.89>
Token summary: {'id': 'a1b2c3...', 'value': 'ollehx7', 'entropy': 6.89, 'history_len': 2}
Entropy stats: {'initial': 7.23, 'final': 6.89, 'delta': -0.34, 'steps': 2}
```

## Running the Example

```bash
python example_usage.py
```

This will demonstrate the system with various transformation functions and show how tokens evolve through the entropy engine.

