"""
Unit tests for the core entropy engine functionality
"""

import pytest
from entropy_engine.core import Token, EntropyNode, EntropyEngine

class TestToken:
    def test_token_creation(self):
        """Test basic token creation"""
        token = Token("test")
        assert token.value == "test"
        assert token.id is not None
        assert len(token.history) == 0
        assert token.entropy > 0
    
    def test_token_with_custom_id(self):
        """Test token creation with custom ID"""
        custom_id = "custom-123"
        token = Token("test", id=custom_id)
        assert token.id == custom_id
    
    def test_token_mutation(self):
        """Test token mutation"""
        token = Token("hello")
        original_entropy = token.entropy
        original_value = token.value
        
        def reverse_transform(value, entropy):
            return str(value)[::-1]
        
        token.mutate(reverse_transform)
        
        assert token.value == "olleh"
        assert len(token.history) == 1
        assert token.history[0] == original_value
        assert token.entropy != original_entropy
    
    def test_token_mutation_with_none(self):
        """Test that mutation with None transformation is skipped"""
        token = Token("test")
        original_value = token.value
        original_entropy = token.entropy
        
        token.mutate(None)
        
        assert token.value == original_value
        assert token.entropy == original_entropy
        assert len(token.history) == 0
    
    def test_token_summary(self):
        """Test token summary method"""
        token = Token("test")
        summary = token.summary()
        
        assert "id" in summary
        assert "value" in summary
        assert "entropy" in summary
        assert "history_len" in summary
        assert summary["value"] == "test"
        assert summary["history_len"] == 0

class TestEntropyNode:
    def test_node_creation(self):
        """Test basic node creation"""
        def test_transform(value, entropy):
            return str(value).upper()
        
        node = EntropyNode("test_node", test_transform)
        assert node.name == "test_node"
        assert node.transform == test_transform
        assert len(node.children) == 0
        assert node.entropy_limit is None
        assert node.dynamic_brancher is None
    
    def test_node_with_entropy_limit(self):
        """Test node creation with entropy limit"""
        def test_transform(value, entropy):
            return str(value).upper()
        
        node = EntropyNode("test_node", test_transform, entropy_limit=8.0)
        assert node.entropy_limit == 8.0
    
    def test_node_processing(self):
        """Test node processing of tokens"""
        def test_transform(value, entropy):
            return str(value).upper()
        
        node = EntropyNode("test_node", test_transform)
        token = Token("hello")
        
        node.process(token, depth=0, max_depth=5)
        
        assert token.value == "HELLO"
        assert len(node.memory) == 1
        assert node.memory[0]["input"] == "hello"
        assert node.memory[0]["output"] == "HELLO"
    
    def test_node_depth_limit(self):
        """Test that processing stops at max depth"""
        def test_transform(value, entropy):
            return str(value).upper()
        
        node = EntropyNode("test_node", test_transform)
        token = Token("hello")
        
        node.process(token, depth=6, max_depth=5)
        
        # Should not process due to depth limit
        assert token.value == "hello"
        assert len(node.memory) == 0
    
    def test_node_entropy_limit(self):
        """Test that processing stops at entropy limit"""
        def test_transform(value, entropy):
            return str(value) + "x" * 100  # High entropy transformation
        
        node = EntropyNode("test_node", test_transform, entropy_limit=8.0)
        token = Token("hello")
        
        node.process(token, depth=0, max_depth=5)
        
        # Should not process due to entropy limit
        assert token.value == "hello"
        assert len(node.memory) == 0
    
    def test_node_add_child(self):
        """Test adding children to nodes"""
        def test_transform(value, entropy):
            return str(value).upper()
        
        parent = EntropyNode("parent", test_transform)
        child = EntropyNode("child", test_transform)
        
        parent.add_child(child)
        assert len(parent.children) == 1
        assert parent.children[0] == child
    
    def test_node_export_memory(self):
        """Test memory export functionality"""
        def test_transform(value, entropy):
            return str(value).upper()
        
        node = EntropyNode("test_node", test_transform)
        token = Token("hello")
        
        node.process(token, depth=0, max_depth=5)
        memory = node.export_memory()
        
        assert memory["node"] == "test_node"
        assert len(memory["log"]) == 1
        assert memory["children"] == []

class TestEntropyEngine:
    def test_engine_creation(self):
        """Test engine creation"""
        def test_transform(value, entropy):
            return str(value).upper()
        
        root = EntropyNode("root", test_transform)
        engine = EntropyEngine(root, max_depth=3)
        
        assert engine.root == root
        assert engine.max_depth == 3
        assert len(engine.token_log) == 0
    
    def test_engine_processing(self):
        """Test engine processing of tokens"""
        def test_transform(value, entropy):
            return str(value).upper()
        
        root = EntropyNode("root", test_transform)
        engine = EntropyEngine(root, max_depth=3)
        token = Token("hello")
        
        engine.run(token)
        
        assert token.value == "HELLO"
        assert len(engine.token_log) == 2  # Initial and final
        assert engine.token_log[0][0] == token.id
        assert engine.token_log[1][0] == token.id
    
    def test_engine_trace(self):
        """Test engine trace functionality"""
        def test_transform(value, entropy):
            return str(value).upper()
        
        root = EntropyNode("root", test_transform)
        engine = EntropyEngine(root, max_depth=3)
        token = Token("hello")
        
        engine.run(token)
        trace = engine.trace()
        
        assert len(trace) == 2
        assert trace[0][0] == token.id
        assert trace[1][0] == token.id
    
    def test_engine_export_graph(self):
        """Test engine graph export"""
        def test_transform(value, entropy):
            return str(value).upper()
        
        root = EntropyNode("root", test_transform)
        engine = EntropyEngine(root, max_depth=3)
        token = Token("hello")
        
        engine.run(token)
        graph = engine.export_graph()
        
        assert graph["node"] == "root"
        assert len(graph["log"]) == 1
        assert graph["children"] == []
    
    def test_engine_entropy_stats(self):
        """Test engine entropy statistics"""
        def test_transform(value, entropy):
            return str(value).upper()
        
        root = EntropyNode("root", test_transform)
        engine = EntropyEngine(root, max_depth=3)
        token = Token("hello")
        
        engine.run(token)
        stats = engine.entropy_stats()
        
        assert "initial" in stats
        assert "final" in stats
        assert "delta" in stats
        assert "steps" in stats
        assert stats["steps"] == 2
    
    def test_engine_entropy_stats_insufficient_data(self):
        """Test entropy stats with insufficient data"""
        def test_transform(value, entropy):
            return str(value).upper()
        
        root = EntropyNode("root", test_transform)
        engine = EntropyEngine(root, max_depth=3)
        
        stats = engine.entropy_stats()
        assert "error" in stats
        assert stats["error"] == "Insufficient data"

class TestIntegration:
    def test_simple_pipeline(self):
        """Test a simple transformation pipeline"""
        def reverse_transform(value, entropy):
            return str(value)[::-1]
        
        def uppercase_transform(value, entropy):
            return str(value).upper()
        
        root = EntropyNode("root", reverse_transform)
        child = EntropyNode("child", uppercase_transform)
        root.add_child(child)
        
        engine = EntropyEngine(root, max_depth=3)
        token = Token("hello")
        
        engine.run(token)
        
        # Should be: "hello" -> "olleh" -> "OLLEH"
        assert token.value == "OLLEH"
        assert len(token.history) == 2
        assert token.history[0] == "hello"
        assert token.history[1] == "olleh"
    
    def test_entropy_based_processing(self):
        """Test processing with entropy limits"""
        def high_entropy_transform(value, entropy):
            return str(value) + "x" * 50
        
        def low_entropy_transform(value, entropy):
            return str(value)[:3]
        
        root = EntropyNode("root", high_entropy_transform, entropy_limit=8.0)
        child = EntropyNode("child", low_entropy_transform)
        root.add_child(child)
        
        engine = EntropyEngine(root, max_depth=3)
        token = Token("hello")
        
        engine.run(token)
        
        # Root should not process due to entropy limit
        # Child should process normally
        assert token.value == "hel"
        assert len(token.history) == 1