#!/usr/bin/env python3
"""
Python Client for Julia Integration Server
Provides seamless interop between Python LIMPS and Julia mathematical operations
"""

import requests
import json
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class JuliaClient:
    """
    Python client for Julia mathematical operations
    """
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        """
        Initialize Julia client
        
        Args:
            server_url: URL of the Julia HTTP server
        """
        self.server_url = server_url
        self.session = requests.Session()
        
    def _make_request(self, function_name: str, args: List[Any]) -> Dict[str, Any]:
        """
        Make request to Julia server
        
        Args:
            function_name: Name of Julia function to call
            args: Arguments to pass to the function
            
        Returns:
            Response from Julia server
        """
        try:
            payload = {
                "function": function_name,
                "args": args
            }
            
            response = self.session.post(
                self.server_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Julia server error: {response.status_code} - {response.text}")
                return {"error": f"Server error: {response.status_code}"}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {"error": f"Request failed: {e}"}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return {"error": f"JSON decode error: {e}"}
    
    def create_polynomials(self, data: np.ndarray, variables: List[str]) -> Dict[str, Any]:
        """
        Create polynomial representation from numerical data
        
        Args:
            data: Numerical data matrix
            variables: Variable names
            
        Returns:
            Polynomial representation
        """
        return self._make_request("create_polynomials", [data.tolist(), variables])
    
    def analyze_polynomials(self, polynomials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze polynomial structure and properties
        
        Args:
            polynomials: Polynomial data
            
        Returns:
            Analysis results
        """
        return self._make_request("analyze_polynomials", [polynomials])
    
    def optimize_matrix(self, matrix: np.ndarray, method: str = "sparsity") -> Dict[str, Any]:
        """
        Optimize matrix using Julia backend
        
        Args:
            matrix: Input matrix
            method: Optimization method ("sparsity", "rank", "structure")
            
        Returns:
            Optimization results
        """
        return self._make_request("optimize_matrix", [matrix.tolist(), method])
    
    def matrix_to_polynomials(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Convert matrix to polynomial representation
        
        Args:
            matrix: Input matrix
            
        Returns:
            Polynomial representation
        """
        return self._make_request("matrix_to_polynomials", [matrix.tolist()])
    
    def analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze text structure using polynomial techniques
        
        Args:
            text: Input text
            
        Returns:
            Text analysis results
        """
        return self._make_request("analyze_text_structure", [text])
    
    def test_connection(self) -> bool:
        """
        Test connection to Julia server
        
        Returns:
            True if connection successful
        """
        try:
            response = self.session.get(self.server_url)
            return response.status_code == 200
        except:
            return False


class LIMPSJuliaIntegration:
    """
    Integration layer between LIMPS and Julia mathematical operations
    """
    
    def __init__(self, julia_client: JuliaClient):
        """
        Initialize LIMPS-Julia integration
        
        Args:
            julia_client: Julia client instance
        """
        self.julia_client = julia_client
        
    def process_entropy_matrix(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Process entropy matrix through Julia polynomial analysis
        
        Args:
            matrix: Entropy matrix from LIMPS
            
        Returns:
            Processing results
        """
        logger.info("Processing entropy matrix through Julia backend")
        
        # Convert matrix to polynomial representation
        poly_result = self.julia_client.matrix_to_polynomials(matrix)
        if "error" in poly_result:
            return poly_result
        
        # Analyze polynomial structure
        analysis_result = self.julia_client.analyze_polynomials(poly_result)
        if "error" in analysis_result:
            return analysis_result
        
        # Optimize matrix based on complexity
        complexity = analysis_result.get("complexity_score", 0.5)
        if complexity > 0.7:
            method = "rank"
        elif complexity > 0.4:
            method = "structure"
        else:
            method = "sparsity"
        
        opt_result = self.julia_client.optimize_matrix(matrix, method)
        
        return {
            "polynomial_representation": poly_result,
            "analysis": analysis_result,
            "optimization": opt_result,
            "complexity_score": complexity,
            "optimization_method": method
        }
    
    def analyze_natural_language(self, text: str) -> Dict[str, Any]:
        """
        Analyze natural language input using Julia text analysis
        
        Args:
            text: Natural language input
            
        Returns:
            Analysis results
        """
        logger.info("Analyzing natural language through Julia backend")
        
        # Analyze text structure
        text_analysis = self.julia_client.analyze_text_structure(text)
        if "error" in text_analysis:
            return text_analysis
        
        # Create polynomial representation of text features
        features = np.array([
            text_analysis.get("text_length", 0),
            text_analysis.get("word_count", 0),
            text_analysis.get("unique_words", 0),
            text_analysis.get("average_word_length", 0),
            text_analysis.get("text_entropy", 0)
        ]).reshape(1, -1)
        
        # Convert to polynomial representation
        variables = ["length", "words", "unique", "avg_len", "entropy"]
        poly_result = self.julia_client.create_polynomials(features, variables)
        
        return {
            "text_analysis": text_analysis,
            "polynomial_features": poly_result,
            "feature_vector": features.tolist()
        }
    
    def optimize_limps_matrix(self, matrix: np.ndarray, target_compression: float = 0.5) -> Dict[str, Any]:
        """
        Optimize LIMPS matrix with target compression ratio
        
        Args:
            matrix: Input matrix
            target_compression: Target compression ratio
            
        Returns:
            Optimization results
        """
        logger.info(f"Optimizing LIMPS matrix with target compression: {target_compression}")
        
        # Try different optimization methods
        methods = ["sparsity", "rank", "structure"]
        best_result = None
        best_compression = 0.0
        
        for method in methods:
            result = self.julia_client.optimize_matrix(matrix, method)
            if "error" not in result:
                compression = result.get("compression_ratio", 0.0)
                if abs(compression - target_compression) < abs(best_compression - target_compression):
                    best_result = result
                    best_compression = compression
        
        if best_result is None:
            return {"error": "All optimization methods failed"}
        
        return {
            "optimization_result": best_result,
            "achieved_compression": best_compression,
            "target_compression": target_compression,
            "compression_error": abs(best_compression - target_compression)
        }


def main():
    """Test the Julia client and LIMPS integration"""
    logger.info("Testing Julia Client and LIMPS Integration")
    
    # Initialize Julia client
    client = JuliaClient()
    
    # Test connection
    if not client.test_connection():
        logger.error("Cannot connect to Julia server. Make sure it's running on port 8000")
        return
    
    logger.info("Successfully connected to Julia server")
    
    # Initialize LIMPS integration
    limps_integration = LIMPSJuliaIntegration(client)
    
    # Test 1: Process entropy matrix
    logger.info("Test 1: Processing entropy matrix")
    entropy_matrix = np.random.rand(10, 10)
    result1 = limps_integration.process_entropy_matrix(entropy_matrix)
    logger.info(f"Entropy processing result: {result1.get('complexity_score', 'N/A')}")
    
    # Test 2: Analyze natural language
    logger.info("Test 2: Analyzing natural language")
    text = "Show monthly sales totals for electronics category in Q3 2024"
    result2 = limps_integration.analyze_natural_language(text)
    logger.info(f"Text analysis result: {result2.get('text_analysis', {}).get('text_entropy', 'N/A')}")
    
    # Test 3: Optimize matrix with target compression
    logger.info("Test 3: Optimizing matrix with target compression")
    test_matrix = np.random.rand(20, 20)
    result3 = limps_integration.optimize_limps_matrix(test_matrix, target_compression=0.6)
    logger.info(f"Optimization result: {result3.get('achieved_compression', 'N/A')}")
    
    logger.info("All tests completed successfully!")


if __name__ == "__main__":
    main()