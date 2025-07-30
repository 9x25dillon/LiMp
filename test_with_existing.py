#!/usr/bin/env python3
"""
Test the same example with our existing entropy engine implementation
"""

from entropy_engine.core import Token, EntropyNode, EntropyEngine

print("🚀 Testing with Existing Entropy Engine Implementation")
print("=" * 60)

# Simple transformation that adds entropy as a string
def append_entropy(val, entropy):
    return f"{val}-{int(entropy)}"

# Optional dynamic branching example
def brancher(token):
    if "x" in token.value:
        return [EntropyNode("extra", lambda v, e: v.replace("x", "*"))]
    return []

# Root node
root = EntropyNode("root", append_entropy, entropy_limit=300, dynamic_brancher=brancher)

# Add a child
root.add_child(EntropyNode("child1", lambda v, e: v[::-1]))

# Engine and token
engine = EntropyEngine(root, max_depth=3)
t = Token("xstart")

print(f"Initial token: {t}")
print(f"Initial entropy: {t.entropy}")
print()

print("🔄 Running entropy engine...")
engine.run(t)

print("\n📊 Results:")
print("Token Summary:")
print(t.summary())

print("\nEntropy Statistics:")
print(engine.entropy_stats())

print("\nProcessing Graph:")
import json
print(json.dumps(engine.export_graph(), indent=2))

print("\n🔄 Token History:")
for i, value in enumerate(t.history):
    print(f"Step {i}: {value}")
print(f"Final: {t.value}")

print("\n✅ Test completed!")