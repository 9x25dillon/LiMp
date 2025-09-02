#!/usr/bin/env python3
"""
Simple test runner for entropy engine
"""

import sys
import traceback
from entropy_engine.core import Token, EntropyNode, EntropyEngine

def test_token_creation():
    """Test basic token creation"""
    print("Testing token creation...")
    token = Token("test")
    assert token.value == "test"
    assert token.id is not None
    assert len(token.history) == 0
    assert token.entropy > 0
    print("✓ Token creation passed")

def test_token_mutation():
    """Test token mutation"""
    print("Testing token mutation...")
    token = Token("hello")
    original_entropy = token.entropy
    
    def reverse_transform(value, entropy):
        return str(value)[::-1]
    
    token.mutate(reverse_transform)
    
    assert token.value == "olleh"
    assert len(token.history) == 1
    assert token.history[0] == "hello"
    assert token.entropy != original_entropy
    print("✓ Token mutation passed")

def test_entropy_node():
    """Test entropy node functionality"""
    print("Testing entropy node...")
    
    def test_transform(value, entropy):
        return str(value).upper()
    
    node = EntropyNode("test_node", test_transform)
    token = Token("hello")
    
    node.process(token, depth=0, max_depth=5)
    
    assert token.value == "HELLO"
    assert len(node.memory) == 1
    assert node.memory[0]["input"] == "hello"
    assert node.memory[0]["output"] == "HELLO"
    print("✓ Entropy node passed")

def test_entropy_engine():
    """Test entropy engine functionality"""
    print("Testing entropy engine...")
    
    def test_transform(value, entropy):
        return str(value).upper()
    
    root = EntropyNode("root", test_transform)
    engine = EntropyEngine(root, max_depth=3)
    token = Token("hello")
    
    engine.run(token)
    
    assert token.value == "HELLO"
    assert len(engine.token_log) == 2
    assert engine.token_log[0][0] == token.id
    assert engine.token_log[1][0] == token.id
    
    stats = engine.entropy_stats()
    assert "initial" in stats
    assert "final" in stats
    assert "delta" in stats
    assert "steps" in stats
    print("✓ Entropy engine passed")

def test_integration():
    """Test integration scenario"""
    print("Testing integration...")
    
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
    print("✓ Integration test passed")

def main():
    """Run all tests"""
    tests = [
        test_token_creation,
        test_token_mutation,
        test_entropy_node,
        test_entropy_engine,
        test_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} failed: {e}")
            traceback.print_exc()
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())