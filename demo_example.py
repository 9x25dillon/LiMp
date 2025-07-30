#!/usr/bin/env python3
"""
Demo of the entropy engine with custom transformations and dynamic branching
"""

import random
import hashlib
import uuid

class Token:
    def __init__(self, value, id=None):
        self.id = id or str(uuid.uuid4())
        self.value = value
        self.history = []
        self.entropy = self._calculate_entropy()

    def _calculate_entropy(self):
        hash_val = hashlib.sha256(str(self.value).encode()).hexdigest()
        return sum(int(c, 16) for c in hash_val) / len(hash_val)

    def mutate(self, transformation):
        self.history.append(self.value)
        if transformation:
            self.value = transformation(self.value, self.entropy)
            self.entropy = self._calculate_entropy()

    def summary(self):
        return {
            "id": self.id,
            "value": self.value,
            "entropy": round(self.entropy, 2),
            "history_len": len(self.history)
        }

    def __repr__(self):
        return f"<Token {self.id[:6]} val={self.value} entropy={self.entropy:.2f}>"


class EntropyNode:
    def __init__(self, name, transform_function, entropy_limit=None, dynamic_brancher=None):
        self.name = name
        self.transform = transform_function
        self.children = []
        self.entropy_limit = entropy_limit
        self.dynamic_brancher = dynamic_brancher
        self.memory = []

    def process(self, token, depth, max_depth):
        if depth > max_depth or (self.entropy_limit is not None and token.entropy >= self.entropy_limit):
            return

        original_entropy = token.entropy
        original_value = token.value

        token.mutate(self.transform)

        self.memory.append({
            "token_id": token.id,
            "input": original_value,
            "output": token.value,
            "entropy_before": original_entropy,
            "entropy_after": token.entropy,
            "depth": depth
        })

        if self.dynamic_brancher:
            new_children = self.dynamic_brancher(token)
            for child in new_children:
                self.add_child(child)

        for child in self.children:
            child.process(token, depth + 1, max_depth)

    def add_child(self, child_node):
        if len(self.children) < 10:
            self.children.append(child_node)

    def export_memory(self):
        return {
            "node": self.name,
            "log": self.memory,
            "children": [child.export_memory() for child in self.children]
        }


class EntropyEngine:
    def __init__(self, root_node, max_depth=5):
        self.root = root_node
        self.max_depth = max_depth
        self.token_log = []

    def run(self, token):
        self.token_log.append((token.id, token.entropy))
        self.root.process(token, depth=0, max_depth=self.max_depth)
        self.token_log.append((token.id, token.entropy))

    def trace(self):
        return self.token_log

    def export_graph(self):
        return self.root.export_memory()

    def entropy_stats(self):
        entries = [e for _, e in self.token_log]
        delta = entries[-1] - entries[0]
        return {
            "initial": entries[0],
            "final": entries[-1],
            "delta": delta,
            "steps": len(entries)
        }

# Example usage
if __name__ == "__main__":
    print("🚀 Entropy Engine Demo")
    print("=" * 50)
    
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
    
    print("\n✅ Demo completed!")