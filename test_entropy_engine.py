#!/usr/bin/env python3
"""
Comprehensive test suite for the Entropy Engine
"""

from entropy_engine import Token, EntropyNode, EntropyEngine
import random

def test_basic_functionality():
    """Test basic token creation and entropy calculation"""
    print("=== Testing Basic Functionality ===")
    
    token1 = Token("hello")
    token2 = Token("world")
    
    print(f"Token 1: {token1}")
    print(f"Token 2: {token2}")
    print(f"Token 1 summary: {token1.summary()}")
    print()

def test_simple_transformations():
    """Test simple string transformations"""
    print("=== Testing Simple Transformations ===")
    
    def uppercase(value, entropy):
        return str(value).upper()
    
    def duplicate(value, entropy):
        return str(value) * 2
    
    root = EntropyNode("root", uppercase)
    root.add_child(EntropyNode("duplicate", duplicate))
    
    engine = EntropyEngine(root, max_depth=2)
    token = Token("test")
    
    print(f"Initial: {token}")
    engine.run(token)
    print(f"Final: {token}")
    print(f"History: {token.history}")
    print()

def test_entropy_limits():
    """Test entropy limit functionality"""
    print("=== Testing Entropy Limits ===")
    
    def add_noise(value, entropy):
        # Add random characters to increase entropy
        noise = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))
        return str(value) + noise
    
    # Node with entropy limit
    root = EntropyNode("root", add_noise, entropy_limit=8.0)
    root.add_child(EntropyNode("child", add_noise))
    
    engine = EntropyEngine(root, max_depth=5)
    token = Token("start")
    
    print(f"Initial entropy: {token.entropy}")
    engine.run(token)
    print(f"Final entropy: {token.entropy}")
    print(f"Final value: {token.value}")
    print()

def test_dynamic_branching():
    """Test dynamic branching based on token state"""
    print("=== Testing Dynamic Branching ===")
    
    def increase_entropy(value, entropy):
        return str(value) + str(random.randint(1000, 9999))
    
    def decrease_entropy(value, entropy):
        return str(value)[:3]  # Truncate to reduce entropy
    
    def dynamic_brancher(token):
        if token.entropy > 7.0:
            return [EntropyNode("stabilize", decrease_entropy)]
        elif token.entropy < 5.0:
            return [EntropyNode("amplify", increase_entropy)]
        return []
    
    root = EntropyNode("root", increase_entropy, dynamic_brancher=dynamic_brancher)
    engine = EntropyEngine(root, max_depth=4)
    token = Token("seed")
    
    print(f"Initial: {token}")
    engine.run(token)
    print(f"Final: {token}")
    
    # Show the graph structure
    graph = engine.export_graph()
    print(f"Number of children created: {len(graph['children'])}")
    print()

def test_numeric_transformations():
    """Test transformations with numeric values"""
    print("=== Testing Numeric Transformations ===")
    
    def square(value, entropy):
        try:
            num = float(value)
            return str(num * num)
        except ValueError:
            return str(value) + "_squared"
    
    def add_entropy(value, entropy):
        try:
            num = float(value)
            return str(num + entropy)
        except ValueError:
            return str(value) + f"+{entropy:.2f}"
    
    root = EntropyNode("root", square)
    root.add_child(EntropyNode("add_entropy", add_entropy))
    
    engine = EntropyEngine(root, max_depth=2)
    token = Token("5")
    
    print(f"Initial: {token}")
    engine.run(token)
    print(f"Final: {token}")
    print()

def test_memory_tracking():
    """Test memory tracking functionality"""
    print("=== Testing Memory Tracking ===")
    
    def transform_a(value, entropy):
        return str(value) + "_A"
    
    def transform_b(value, entropy):
        return str(value) + "_B"
    
    root = EntropyNode("root", transform_a)
    root.add_child(EntropyNode("child", transform_b))
    
    engine = EntropyEngine(root, max_depth=2)
    token = Token("base")
    
    engine.run(token)
    
    # Export and examine memory
    graph = engine.export_graph()
    print("Memory log from root node:")
    for entry in graph['log']:
        print(f"  {entry['input']} -> {entry['output']} (entropy: {entry['entropy_before']:.2f} -> {entry['entropy_after']:.2f})")
    
    print("Memory log from child node:")
    for entry in graph['children'][0]['log']:
        print(f"  {entry['input']} -> {entry['output']} (entropy: {entry['entropy_before']:.2f} -> {entry['entropy_after']:.2f})")
    print()

def test_entropy_statistics():
    """Test entropy statistics calculation"""
    print("=== Testing Entropy Statistics ===")
    
    def gradual_change(value, entropy):
        return str(value) + str(int(entropy * 10))
    
    root = EntropyNode("root", gradual_change)
    engine = EntropyEngine(root, max_depth=3)
    token = Token("initial")
    
    print(f"Starting entropy: {token.entropy}")
    engine.run(token)
    
    stats = engine.entropy_stats()
    print(f"Entropy statistics: {stats}")
    print(f"Entropy change: {stats['delta']:.4f}")
    print()

def main():
    """Run all tests"""
    print("Entropy Engine Test Suite\n")
    
    test_basic_functionality()
    test_simple_transformations()
    test_entropy_limits()
    test_dynamic_branching()
    test_numeric_transformations()
    test_memory_tracking()
    test_entropy_statistics()
    
    print("All tests completed!")

if __name__ == "__main__":
    main()