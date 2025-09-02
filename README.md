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

# Entropy Engine

A Python-based system for processing tokens through a graph of entropy-based transformations.

## Overview

The Entropy Engine processes tokens through a directed graph of transformation nodes. Each node applies a transformation function to the token's value and tracks entropy changes. The system supports:

- **Token Management**: Tokens with unique IDs, values, and entropy calculations
- **Graph Processing**: Directed acyclic graph of transformation nodes
- **Entropy Tracking**: Automatic entropy calculation and limits
- **Dynamic Branching**: Runtime creation of new processing paths
- **Memory Logging**: Complete audit trail of all transformations

## Core Components

### Token Class
Represents a data token that flows through the system:
- `value`: The actual data being processed
- `id`: Unique identifier (auto-generated UUID)
- `entropy`: Calculated entropy based on SHA256 hash
- `history`: List of previous values

### EntropyNode Class
Represents a processing node in the graph:
- `transform`: Function that modifies token values
- `entropy_limit`: Optional upper bound for processing
- `children`: List of child nodes
- `memory`: Log of all transformations performed

### EntropyEngine Class
Orchestrates the processing:
- `root`: Starting node of the processing graph
- `max_depth`: Maximum processing depth
- `token_log`: Complete processing history

## Usage Example

```python
from entropy_engine import Token, EntropyNode, EntropyEngine

# Define transformation functions
def reverse_string(value, entropy):
    return str(value)[::-1]

def add_random_char(value, entropy):
    import random
    chars = "abcdefghijklmnopqrstuvwxyz"
    return str(value) + random.choice(chars)

# Create processing graph
root = EntropyNode("reverse", reverse_string, entropy_limit=8.0)
child = EntropyNode("add_char", add_random_char, entropy_limit=9.0)
root.add_child(child)

# Create and run engine
engine = EntropyEngine(root, max_depth=3)
token = Token("hello")
engine.run(token)

print(f"Final token: {token}")
print(f"History: {token.history}")
```

## Key Features

### Entropy Calculation
Entropy is calculated using SHA256 hash of the token value:
```python
hash_val = hashlib.sha256(str(value).encode()).hexdigest()
entropy = sum(int(c, 16) for c in hash_val) / len(hash_val)
```

### Entropy Limits
Nodes can have entropy limits to prevent infinite processing:
```python
node = EntropyNode("transform", func, entropy_limit=8.0)
```

### Dynamic Branching
Nodes can create new children during processing:
```python
def dynamic_brancher(token):
    if token.entropy > 7.0:
        return [EntropyNode("extra", extra_transform)]
    return []

node = EntropyNode("main", transform, dynamic_brancher=dynamic_brancher)
```

### Memory Export
Export complete processing history:
```python
graph_data = engine.export_graph()
stats = engine.entropy_stats()
```

## Running the Example

```bash
python3 example_usage.py
```

This will demonstrate:
- Token creation and processing
- Graph traversal
- Entropy tracking
- Memory logging
- Statistics generation

## Customization

### Creating Custom Transformations
Transformation functions must accept `(value, entropy)` parameters:
```python
def my_transform(value, entropy):
    # Your transformation logic here
    return modified_value
```

### Building Complex Graphs
```python
root = EntropyNode("start", start_transform)
branch1 = EntropyNode("branch1", transform1)
branch2 = EntropyNode("branch2", transform2)
leaf1 = EntropyNode("leaf1", leaf_transform)

root.add_child(branch1)
root.add_child(branch2)
branch1.add_child(leaf1)
```

## Error Handling

The system includes several safety features:
- Entropy limits prevent infinite loops
- Maximum depth limits prevent stack overflow
- Null transformation functions are safely handled
- Child node limits prevent memory issues

## Performance Considerations

- Entropy calculation uses SHA256 (cryptographically secure but slower)
- Memory logging can grow large with many tokens
- Graph traversal is depth-first
- Consider using entropy limits for large-scale processing

