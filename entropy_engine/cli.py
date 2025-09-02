#!/usr/bin/env python3
"""
Command-line interface for the Entropy Engine
"""

import argparse
import json
import sys
import random
from .core import Token, EntropyNode, EntropyEngine

def create_builtin_transformations():
    """Create a dictionary of built-in transformation functions"""
    transformations = {
        "reverse": lambda value, entropy: str(value)[::-1],
        "uppercase": lambda value, entropy: str(value).upper(),
        "lowercase": lambda value, entropy: str(value).lower(),
        "duplicate": lambda value, entropy: str(value) * 2,
        "add_random": lambda value, entropy: str(value) + random.choice("abcdefghijklmnopqrstuvwxyz0123456789"),
        "add_entropy": lambda value, entropy: str(value) + f"*{entropy:.2f}",
        "truncate": lambda value, entropy: str(value)[:max(1, len(str(value)) // 2)],
        "multiply": lambda value, entropy: str(value) * int(entropy + 1),
    }
    return transformations

def parse_node_config(config_str):
    """Parse node configuration from string format: name:transform:limit"""
    parts = config_str.split(":")
    name = parts[0]
    transform_name = parts[1] if len(parts) > 1 else "reverse"
    limit = float(parts[2]) if len(parts) > 2 else None
    return name, transform_name, limit

def main():
    parser = argparse.ArgumentParser(
        description="Entropy Engine - Process tokens through entropy-based transformation nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with a single transformation
  entropy-engine --input "hello" --nodes "root:reverse"

  # Multiple nodes with entropy limits
  entropy-engine --input "test" --nodes "root:reverse:8.0" "child:uppercase:7.0"

  # Save results to JSON file
  entropy-engine --input "data" --nodes "root:reverse" --output results.json

  # Show available transformations
  entropy-engine --list-transforms
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input value for the token"
    )
    
    parser.add_argument(
        "--nodes", "-n",
        nargs="+",
        help="Node configurations in format 'name:transform:limit' (limit is optional)"
    )
    
    parser.add_argument(
        "--max-depth", "-d",
        type=int,
        default=5,
        help="Maximum processing depth (default: 5)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for results (JSON format)"
    )
    
    parser.add_argument(
        "--list-transforms", "-l",
        action="store_true",
        help="List available transformation functions"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # List available transformations
    if args.list_transforms:
        transforms = create_builtin_transformations()
        print("Available transformation functions:")
        for name, func in transforms.items():
            print(f"  {name}")
        return
    
    # Validate required arguments
    if not args.input:
        parser.error("--input is required")
    
    if not args.nodes:
        parser.error("--nodes is required")
    
    # Create transformation functions
    transforms = create_builtin_transformations()
    
    # Parse and create nodes
    nodes = {}
    root_node = None
    
    for node_config in args.nodes:
        name, transform_name, limit = parse_node_config(node_config)
        
        if transform_name not in transforms:
            print(f"Error: Unknown transformation '{transform_name}'", file=sys.stderr)
            print(f"Available transformations: {', '.join(transforms.keys())}", file=sys.stderr)
            sys.exit(1)
        
        node = EntropyNode(name, transforms[transform_name], entropy_limit=limit)
        nodes[name] = node
        
        if root_node is None:
            root_node = node
        else:
            # Add as child to the first node for now
            # In a more sophisticated version, you might want to specify parent-child relationships
            root_node.add_child(node)
    
    if root_node is None:
        print("Error: No valid nodes specified", file=sys.stderr)
        sys.exit(1)
    
    # Create engine and process token
    engine = EntropyEngine(root_node, max_depth=args.max_depth)
    token = Token(args.input)
    
    if args.verbose:
        print(f"Initial token: {token}")
    
    engine.run(token)
    
    if args.verbose:
        print(f"Final token: {token}")
        print(f"Token summary: {token.summary()}")
    
    # Prepare results
    results = {
        "input": args.input,
        "token_summary": token.summary(),
        "entropy_stats": engine.entropy_stats(),
        "processing_graph": engine.export_graph(),
        "configuration": {
            "max_depth": args.max_depth,
            "nodes": [node_config for node_config in args.nodes]
        }
    }
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()