#!/usr/bin/env python3
"""
Detailed analysis of the entropy engine simulation
"""

from entropy_engine.core import Token, EntropyNode, EntropyEngine
from pprint import pprint

print("🔍 Detailed Simulation Analysis")
print("=" * 60)

def sample_transform(val, entropy):
    stars = "*" * (int(entropy) % 5)
    result = val + stars
    print(f"  sample_transform: '{val}' + '{stars}' = '{result}'")
    return result

def branching_logic(token):
    is_even = len(token.value) % 2 == 0
    print(f"  branching_logic: length={len(token.value)}, even={is_even}")
    if is_even:
        print(f"  → Creating even_branch node")
        return [EntropyNode("even_branch", lambda v, e: v.upper())]
    print(f"  → No branching")
    return []

# Set up nodes
root = EntropyNode("root", sample_transform, entropy_limit=600, dynamic_brancher=branching_logic)
root.add_child(EntropyNode("static_child", lambda v, e: v[::-1]))

# Engine
engine = EntropyEngine(root, max_depth=4)

# Token
token = Token("init")

print(f"Initial state:")
print(f"  Token: {token}")
print(f"  Value: '{token.value}'")
print(f"  Length: {len(token.value)} (even: {len(token.value) % 2 == 0})")
print(f"  Entropy: {token.entropy}")
print()

print("🔄 Step-by-step processing:")
print("-" * 40)

# Step 1: Root node processes
print("Step 1: Root node")
print(f"  Input: '{token.value}' (entropy: {token.entropy})")
entropy_int = int(token.entropy)
stars_count = entropy_int % 5
print(f"  Entropy: {token.entropy} → int: {entropy_int} → mod 5: {stars_count}")
new_value = token.value + "*" * stars_count
print(f"  Transform: '{token.value}' + '{'*' * stars_count}' = '{new_value}'")
token.mutate(sample_transform)
print(f"  Output: '{token.value}' (entropy: {token.entropy})")

# Check branching
print(f"  Branching check: length={len(token.value)} (even: {len(token.value) % 2 == 0})")
if len(token.value) % 2 == 0:
    print(f"  → Will create even_branch node")

print()

# Step 2: Static child processes
print("Step 2: Static child (reverse)")
print(f"  Input: '{token.value}' (entropy: {token.entropy})")
reversed_value = token.value[::-1]
print(f"  Transform: '{token.value}' → '{reversed_value}'")
token.mutate(lambda v, e: v[::-1])
print(f"  Output: '{token.value}' (entropy: {token.entropy})")

# Check branching again
print(f"  Branching check: length={len(token.value)} (even: {len(token.value) % 2 == 0})")
if len(token.value) % 2 == 0:
    print(f"  → Will create even_branch node")

print()

# Step 3: Dynamic branch processes (if created)
print("Step 3: Dynamic branch (uppercase)")
print(f"  Input: '{token.value}' (entropy: {token.entropy})")
uppercase_value = token.value.upper()
print(f"  Transform: '{token.value}' → '{uppercase_value}'")
token.mutate(lambda v, e: v.upper())
print(f"  Output: '{token.value}' (entropy: {token.entropy})")

print()
print("📊 Final Results:")
print("-" * 40)
print(f"Final token: {token}")
print(f"Final value: '{token.value}'")
print(f"History length: {len(token.history)}")

print("\n🔄 Complete history:")
for i, value in enumerate(token.history):
    print(f"  Step {i}: '{value}' (length: {len(value)}, even: {len(value) % 2 == 0})")
print(f"  Final: '{token.value}' (length: {len(token.value)}, even: {len(token.value) % 2 == 0})")

print("\n🎯 Key Insights:")
print("-" * 40)
print("1. The sample_transform adds asterisks based on entropy modulo 5")
print("2. The branching_logic creates 'even_branch' nodes when value length is even")
print("3. The static_child always reverses the string")
print("4. The even_branch converts to uppercase")
print("5. Dynamic branching happens at runtime based on token state")

print("\n✅ Analysis completed!")