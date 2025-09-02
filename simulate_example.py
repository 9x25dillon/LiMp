#!/usr/bin/env python3
"""
Simulation of entropy engine with sample_transform and branching_logic
"""

from entropy_engine.core import Token, EntropyNode, EntropyEngine
from pprint import pprint

print("🎲 Entropy Engine Simulation")
print("=" * 50)

def sample_transform(val, entropy):
    return val + "*" * (int(entropy) % 5)

def branching_logic(token):
    if len(token.value) % 2 == 0:
        return [EntropyNode("even_branch", lambda v, e: v.upper())]
    return []

# Set up nodes
root = EntropyNode("root", sample_transform, entropy_limit=600, dynamic_brancher=branching_logic)
root.add_child(EntropyNode("static_child", lambda v, e: v[::-1]))

# Engine
engine = EntropyEngine(root, max_depth=4)

# Token
token = Token("init")

print(f"Initial token: {token}")
print(f"Initial entropy: {token.entropy}")
print(f"Initial value length: {len(token.value)} (even: {len(token.value) % 2 == 0})")
print()

print("🔄 Running simulation...")
engine.run(token)

print("\n📊 Results:")
print("Token Summary:")
print(token.summary())

print("\nEntropy Statistics:")
print(engine.entropy_stats())

print("\nProcessing Graph:")
pprint(engine.export_graph())

print("\n🔄 Token History:")
for i, value in enumerate(token.history):
    print(f"Step {i}: '{value}' (length: {len(value)}, even: {len(value) % 2 == 0})")
print(f"Final: '{token.value}' (length: {len(token.value)}, even: {len(token.value) % 2 == 0})")

print("\n✅ Simulation completed!")