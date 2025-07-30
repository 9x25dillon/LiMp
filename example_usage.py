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

def dynamic_brancher(token):
    """Create new child nodes based on token state"""
    if token.entropy > 7.0:
        # High entropy - create a stabilizing node
        return [EntropyNode("stabilize", lambda v, e: str(v)[:5], entropy_limit=5.0)]
    elif token.entropy < 3.0:
        # Low entropy - create an amplifying node
        return [EntropyNode("amplify", lambda v, e: str(v) * 2)]
    return []

def main():
    # Create the root node
    root = EntropyNode("root", reverse_string, dynamic_brancher=dynamic_brancher)
    
    # Add some child nodes
    root.add_child(EntropyNode("add_char", add_random_char))
    root.add_child(EntropyNode("entropy_mult", multiply_by_entropy, entropy_limit=8.0))
    
    # Create the engine
    engine = EntropyEngine(root, max_depth=3)
    
    # Create and process a token
    token = Token("hello")
    print(f"Initial token: {token}")
    
    engine.run(token)
    
    print(f"Final token: {token}")
    print(f"Token summary: {token.summary()}")
    
    # Show entropy statistics
    stats = engine.entropy_stats()
    print(f"Entropy stats: {stats}")
    
    # Show the processing graph
    graph = engine.export_graph()
    print(f"Processing graph: {graph}")

if __name__ == "__main__":
    main()