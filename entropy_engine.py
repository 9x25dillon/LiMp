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
        if transformation is None:
            return  # Skip mutation
        self.history.append(self.value)
        # transformation function must accept (value, entropy)
        self.value = transformation(self.value, self.entropy)
        self.entropy = self._calculate_entropy()

    def __repr__(self):
        return f"<Token {self.id[:6]} val={self.value} entropy={self.entropy:.2f}>"

    def summary(self):
        return {
            "id": self.id,
            "value": self.value,
            "entropy": round(self.entropy, 2),
            "history_len": len(self.history)
        }

class EntropyNode:
    def __init__(self, name, transform_function, entropy_limit=None, dynamic_brancher=None):
        self.name = name
        self.transform = transform_function
        self.children = []
        self.entropy_limit = entropy_limit
        self.dynamic_brancher = dynamic_brancher
        self.memory = []  # Logs per-token activity

    def process(self, token, depth, max_depth):
        if depth > max_depth:
            return
        if self.entropy_limit is not None and token.entropy >= self.entropy_limit:
            return

        original_entropy = token.entropy
        original_value = token.value

        token.mutate(self.transform)

        # Log memory snapshot
        self.memory.append({
            "token_id": token.id,
            "input": original_value,
            "output": token.value,
            "entropy_before": original_entropy,
            "entropy_after": token.entropy,
            "depth": depth
        })

        # Dynamic branching
        if self.dynamic_brancher:
            new_children = self.dynamic_brancher(token)
            for child in new_children:
                self.add_child(child)

        for child in self.children:
            child.process(token, depth + 1, max_depth)

    def add_child(self, child_node):
        if len(self.children) < 10:  # Arbitrary limit
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
        if len(self.token_log) < 2:
            return {"error": "Insufficient data"}
        
        entries = [e for tid, e in self.token_log]
        delta = entries[-1] - entries[0]
        return {
            "initial": entries[0],
            "final": entries[-1],
            "delta": delta,
            "steps": len(entries)
        }