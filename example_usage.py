from entropy_engine import Token, EntropyNode, EntropyEngine
import random

# Example transformation functions
def reverse_string(value, entropy):
    """Reverse the string value"""
    return str(value)[::-1]

def add_random_char(value, entropy):
    """Add a random character to the string"""
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    return str(value) + random.choice(chars)

def multiply_by_entropy(value, entropy):
    """Multiply numeric value by entropy factor"""
    try:
        num_val = float(value)
        return str(num_val * entropy)
    except ValueError:
        return str(value) + f"*{entropy:.2f}"

def duplicate_string(value, entropy):
    """Duplicate the string based on entropy"""
    multiplier = max(1, int(entropy * 2))
    return str(value) * multiplier

# Create a simple entropy processing graph
def create_sample_graph():
    # Root node - reverses strings
    root = EntropyNode("reverse", reverse_string, entropy_limit=8.0)
    
    # Child nodes
    add_char_node = EntropyNode("add_char", add_random_char, entropy_limit=9.0)
    multiply_node = EntropyNode("multiply", multiply_by_entropy, entropy_limit=7.0)
    duplicate_node = EntropyNode("duplicate", duplicate_string, entropy_limit=6.0)
    
    # Build the graph
    root.add_child(add_char_node)
    root.add_child(multiply_node)
    multiply_node.add_child(duplicate_node)
    
    return root

def main():
    # Create the engine
    root_node = create_sample_graph()
    engine = EntropyEngine(root_node, max_depth=3)
    
    # Create some test tokens
    test_tokens = [
        Token("hello"),
        Token("123"),
        Token("test_string"),
        Token("42.5")
    ]
    
    print("=== Entropy Engine Demo ===\n")
    
    for i, token in enumerate(test_tokens, 1):
        print(f"Processing Token {i}:")
        print(f"  Initial: {token}")
        
        # Run the token through the engine
        engine.run(token)
        
        print(f"  Final: {token}")
        print(f"  History: {token.history}")
        print()
    
    # Show engine statistics
    print("=== Engine Statistics ===")
    stats = engine.entropy_stats()
    print(f"Stats: {stats}")
    
    # Export the processing graph
    print("\n=== Processing Graph ===")
    graph = engine.export_graph()
    print(f"Graph structure: {len(graph['children'])} root children")
    
    # Show detailed memory from root node
    if graph['log']:
        print(f"Root node processed {len(graph['log'])} tokens")
        for entry in graph['log'][:3]:  # Show first 3 entries
            print(f"  {entry['input']} -> {entry['output']} (entropy: {entry['entropy_before']:.2f} -> {entry['entropy_after']:.2f})")

if __name__ == "__main__":
    main()