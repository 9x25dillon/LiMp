#!/usr/bin/env python3
"""
Matrix Processor for 9xdSq-LIMPS-FemTO-R1C
GPU-accelerated matrix operations with polynomial optimization
"""

import torch
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import json
import time
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def setup_logging(debug: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('matrix_processor.log')
        ]
    )

class MatrixProcessor:
    """
    Advanced matrix processor with GPU acceleration and polynomial optimization
    """
    
    def __init__(self, 
                 use_gpu: bool = False,
                 precision: str = "float32",
                 max_memory_gb: float = 8.0,
                 debug: bool = False):
        """
        Initialize the matrix processor
        
        Args:
            use_gpu: Whether to use GPU acceleration
            precision: Numerical precision ("float32", "float64", "float16")
            max_memory_gb: Maximum GPU memory usage in GB
            debug: Enable debug logging
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.precision = precision
        self.max_memory_gb = max_memory_gb
        self.debug = debug
        
        # Set device and precision
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.dtype = getattr(torch, precision)
        
        # Initialize polynomial optimization parameters
        self.poly_params = self._initialize_polynomial_params()
        
        # GPU logging
        if self.use_gpu:
            logger.info(f"GPU available: {torch.cuda.get_device_name(self.device)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(self.device).total_memory / 1024**3:.1f} GB")
        else:
            logger.info("Using CPU for matrix operations")
        
        logger.info(f"Matrix Processor initialized on {self.device} with {precision} precision")
    
    def _initialize_polynomial_params(self) -> Dict[str, Any]:
        """Initialize polynomial optimization parameters"""
        return {
            "sparsity_threshold": 0.01,
            "rank_reduction_factor": 0.5,
            "compression_ratio": 0.7,
            "max_iterations": 100,
            "convergence_tolerance": 1e-6,
            "polynomial_degree": 3,
            "chebyshev_degree": 4,
            "normalization_enabled": True,
            "adaptive_thresholding": True,
            "smoothing_factor": 0.1,
            "spectrum_analysis": True,
            "validation_plots": True
        }
    
    def _calculate_compression_ratio(self, original: torch.Tensor, optimized: torch.Tensor) -> float:
        """Calculate actual compression ratio"""
        original_size = original.numel()
        optimized_size = optimized.numel()
        
        # For sparse matrices, count non-zero elements
        if optimized.is_sparse:
            optimized_size = optimized._nnz()
        
        return 1.0 - optimized_size / original_size
    
    def _calculate_error_metrics(self, original: torch.Tensor, optimized: torch.Tensor) -> Dict[str, float]:
        """Calculate error metrics between original and optimized matrices"""
        diff = original - optimized
        mse = torch.mean(diff ** 2).item()
        mae = torch.mean(torch.abs(diff)).item()
        relative_error = torch.norm(diff) / torch.norm(original)
        
        return {
            "mse": mse,
            "mae": mae,
            "relative_error": relative_error.item(),
            "max_error": torch.max(torch.abs(diff)).item()
        }
    
    def polynomial_to_matrix(self, 
                           polynomial_data: Dict[str, Any],
                           target_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Convert polynomial data to matrix format
        
        Args:
            polynomial_data: Polynomial representation data
            target_shape: Target matrix shape (rows, cols)
            
        Returns:
            Matrix as torch tensor
        """
        if "polynomial_terms" in polynomial_data:
            # Estimate matrix size from polynomial terms
            total_terms = polynomial_data["polynomial_terms"]
            if target_shape is None:
                size = int(np.sqrt(total_terms))
                target_shape = (size, size)
            
            # Create matrix from polynomial coefficients
            matrix = self._create_matrix_from_polynomial(polynomial_data, target_shape)
        else:
            # Fallback to random matrix
            if target_shape is None:
                target_shape = (10, 10)
            matrix = torch.rand(target_shape, dtype=self.dtype, device=self.device)
        
        return matrix
    
    def _create_matrix_from_polynomial(self, 
                                     poly_data: Dict[str, Any],
                                     shape: Tuple[int, int]) -> torch.Tensor:
        """Create matrix from polynomial coefficients"""
        rows, cols = shape
        total_elements = rows * cols

        # Extract polynomial coefficients if available
        if "coefficients" in poly_data:
            coeffs = poly_data["coefficients"]
            
            # Pad with zeros if not enough coefficients
            if len(coeffs) < total_elements:
                padded_coeffs = coeffs + [0.0] * (total_elements - len(coeffs))
            else:
                padded_coeffs = coeffs[:total_elements]

            # Create tensor and reshape
            matrix = torch.tensor(padded_coeffs, dtype=self.dtype, device=self.device)
            return matrix.view(rows, cols)

        # If no coefficients, return zero matrix
        return torch.zeros((rows, cols), dtype=self.dtype, device=self.device)
    
    def _generate_synthetic_matrix(self, 
                                 poly_data: Dict[str, Any],
                                 shape: Tuple[int, int]) -> torch.Tensor:
        """Generate synthetic matrix based on polynomial properties"""
        rows, cols = shape
        
        # Use polynomial properties to generate meaningful matrix
        complexity = poly_data.get("complexity_score", 0.5)
        degree = poly_data.get("degree", 2)
        
        # Create structured matrix
        if complexity > 0.8:
            # High complexity: random matrix with some structure
            matrix = torch.randn(rows, cols, dtype=self.dtype, device=self.device)
            # Add some structure based on polynomial degree
            for i in range(min(degree, rows, cols)):
                matrix[i, i] *= 2.0
        else:
            # Low complexity: structured matrix
            matrix = torch.zeros(rows, cols, dtype=self.dtype, device=self.device)
            for i in range(min(rows, cols)):
                matrix[i, i] = 1.0 + i * 0.1
        
        return matrix
    
    def optimize_matrix(self, 
                       matrix: torch.Tensor,
                       method: str = "sparsity",
                       **kwargs) -> Dict[str, Any]:
        """
        Optimize matrix using various methods
        
        Args:
            matrix: Input matrix
            method: Optimization method ("sparsity", "rank", "structure", "polynomial")
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info(f"Optimizing matrix with shape {matrix.shape} using {method} method")
        
        start_time = time.time()
        
        if method == "sparsity":
            result = self._optimize_sparsity(matrix, **kwargs)
        elif method == "rank":
            result = self._optimize_rank(matrix, **kwargs)
        elif method == "structure":
            result = self._optimize_structure(matrix, **kwargs)
        elif method == "polynomial":
            result = self._optimize_polynomial(matrix, **kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        result["optimization_time"] = time.time() - start_time
        result["method"] = method
        result["original_shape"] = matrix.shape
        
        # Add validation results
        if "optimized_matrix" in result:
            result["validation"] = self._validate_optimization(matrix, result["optimized_matrix"])
        
        return result
    
    def _optimize_sparsity(self, 
                          matrix: torch.Tensor,
                          threshold: Optional[float] = None) -> Dict[str, Any]:
        """Optimize matrix for sparsity"""
        if threshold is None:
            threshold = self.poly_params["sparsity_threshold"]
        
        # Calculate threshold based on matrix values
        abs_matrix = torch.abs(matrix)
        if threshold < 0:
            # Use percentile-based threshold
            threshold = torch.quantile(abs_matrix, -threshold)
        
        # Create sparse matrix
        mask = abs_matrix > threshold
        sparse_matrix = matrix * mask
        
        # Calculate sparsity metrics
        nnz = torch.sum(mask).item()
        total_elements = matrix.numel()
        sparsity_ratio = 1.0 - nnz / total_elements
        
        # Calculate actual compression ratio
        compression_ratio = self._calculate_compression_ratio(matrix, sparse_matrix)
        
        return {
            "optimized_matrix": sparse_matrix,
            "sparsity_ratio": sparsity_ratio,
            "non_zero_elements": nnz,
            "compression_ratio": compression_ratio,
            "threshold": threshold.item() if torch.is_tensor(threshold) else threshold,
            "parameters_used": {"threshold": threshold.item() if torch.is_tensor(threshold) else threshold}
        }
    
    def _optimize_rank(self, 
                      matrix: torch.Tensor,
                      target_rank: Optional[int] = None) -> Dict[str, Any]:
        """Optimize matrix using low-rank approximation"""
        if target_rank is None:
            target_rank = int(min(matrix.shape) * self.poly_params["rank_reduction_factor"])
        
        # Compute SVD
        U, S, Vt = torch.linalg.svd(matrix)
        
        # Keep only top singular values
        U_k = U[:, :target_rank]
        S_k = S[:target_rank]
        Vt_k = Vt[:target_rank, :]
        
        # Reconstruct low-rank matrix
        low_rank_matrix = U_k @ torch.diag(S_k) @ Vt_k
        
        # Calculate rank reduction
        original_rank = torch.linalg.matrix_rank(matrix).item()
        rank_reduction = 1.0 - target_rank / original_rank
        
        # Calculate actual compression ratio
        compression_ratio = self._calculate_compression_ratio(matrix, low_rank_matrix)
        
        return {
            "optimized_matrix": low_rank_matrix,
            "original_rank": original_rank,
            "target_rank": target_rank,
            "rank_reduction": rank_reduction,
            "compression_ratio": compression_ratio,
            "singular_values": S.cpu().numpy(),
            "parameters_used": {"target_rank": target_rank, "rank_reduction_factor": self.poly_params["rank_reduction_factor"]}
        }
    
    def _optimize_structure(self, 
                           matrix: torch.Tensor) -> Dict[str, Any]:
        """Optimize matrix structure using polynomial techniques"""
        # Detect matrix structure patterns
        structure_info = self._analyze_matrix_structure(matrix)
        
        # Apply structure-based optimization
        if structure_info["is_symmetric"]:
            optimized_matrix = self._optimize_symmetric_matrix(matrix)
        elif structure_info["is_sparse"]:
            optimized_matrix = self._optimize_sparse_structure(matrix)
        else:
            optimized_matrix = self._optimize_general_structure(matrix)
        
        # Calculate actual compression ratio
        compression_ratio = self._calculate_compression_ratio(matrix, optimized_matrix)
        
        return {
            "optimized_matrix": optimized_matrix,
            "structure_analysis": structure_info,
            "compression_ratio": compression_ratio,
            "structure_optimized": True,
            "parameters_used": {"compression_ratio": self.poly_params["compression_ratio"]}
        }
    
    def _optimize_polynomial(self, 
                            matrix: torch.Tensor,
                            degree: Optional[int] = None) -> Dict[str, Any]:
        """Optimize matrix using polynomial approximation"""
        if degree is None:
            degree = self.poly_params["polynomial_degree"]
        
        # Convert matrix to polynomial representation
        poly_coeffs = self._matrix_to_polynomial_coefficients_enhanced(matrix, degree)
        
        # Optimize polynomial coefficients
        optimized_coeffs = self._optimize_polynomial_coefficients_enhanced(poly_coeffs)
        
        # Convert back to matrix
        optimized_matrix = self._polynomial_coefficients_to_matrix(optimized_coeffs, matrix.shape)
        
        # Calculate actual compression ratio
        compression_ratio = self._calculate_compression_ratio(matrix, optimized_matrix)
        
        return {
            "optimized_matrix": optimized_matrix,
            "polynomial_degree": degree,
            "coefficients": optimized_coeffs,
            "compression_ratio": compression_ratio,
            "polynomial_optimized": True,
            "parameters_used": {
                "polynomial_degree": degree,
                "chebyshev_degree": self.poly_params["chebyshev_degree"],
                "normalization_enabled": self.poly_params["normalization_enabled"]
            }
        }
    
    def _analyze_matrix_structure(self, matrix: torch.Tensor) -> Dict[str, Any]:
        """Analyze matrix structure patterns"""
        try:
            # Check symmetry (only for square matrices)
            if matrix.shape[0] == matrix.shape[1]:
                is_symmetric = torch.allclose(matrix, matrix.T, atol=1e-6)
            else:
                is_symmetric = False
            
            # Check sparsity
            nnz = torch.count_nonzero(matrix)
            sparsity = 1.0 - nnz / matrix.numel()
            is_sparse = sparsity > 0.5
            
            # Check diagonal dominance (only for square matrices)
            if matrix.shape[0] == matrix.shape[1]:
                diag_elements = torch.diag(matrix)
                off_diag_sum = torch.sum(torch.abs(matrix), dim=1) - torch.abs(diag_elements)
                diagonal_dominance = torch.min(diag_elements / (off_diag_sum + 1e-8))
            else:
                diagonal_dominance = 0.0
            
            # Check condition number with robust handling
            try:
                condition_number = torch.linalg.cond(matrix)
                if not torch.isfinite(condition_number):
                    condition_number = float('inf')
            except:
                condition_number = float('inf')
            
            # Check rank
            try:
                matrix_rank = torch.linalg.matrix_rank(matrix)
            except:
                matrix_rank = min(matrix.shape)
            
            return {
                "is_symmetric": bool(is_symmetric),
                "is_sparse": bool(is_sparse),
                "sparsity": float(sparsity),
                "diagonal_dominance": float(diagonal_dominance),
                "condition_number": float(condition_number) if torch.isfinite(condition_number) else float('inf'),
                "rank": int(matrix_rank)
            }
        except Exception as e:
            logger.warning(f"Error in matrix structure analysis: {e}")
            return {
                "is_symmetric": False,
                "is_sparse": False,
                "sparsity": 0.0,
                "diagonal_dominance": 0.0,
                "condition_number": float('inf'),
                "rank": min(matrix.shape)
            }
    
    def _optimize_symmetric_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        """Optimize symmetric matrix"""
        try:
            # Use eigenvalue decomposition for symmetric matrices
            eigenvals, eigenvecs = torch.linalg.eigh(matrix)
            
            # Keep only significant eigenvalues
            threshold = torch.max(torch.abs(eigenvals)) * 0.01
            significant_mask = torch.abs(eigenvals) > threshold
            
            eigenvals_opt = eigenvals * significant_mask
            optimized_matrix = eigenvecs @ torch.diag(eigenvals_opt) @ eigenvecs.T
            
            return optimized_matrix
        except Exception as e:
            logger.warning(f"Error in symmetric matrix optimization: {e}")
            return matrix
    
    def _optimize_sparse_structure(self, matrix: torch.Tensor) -> torch.Tensor:
        """Optimize sparse matrix structure"""
        try:
            # Use sparse matrix operations
            sparse_matrix = matrix.to_sparse()
            
            # Apply sparse optimization
            optimized_sparse = self._apply_sparse_optimization(sparse_matrix)
            
            return optimized_sparse.to_dense()
        except Exception as e:
            logger.warning(f"Error in sparse structure optimization: {e}")
            return matrix
    
    def _optimize_general_structure(self, matrix: torch.Tensor) -> torch.Tensor:
        """Optimize general matrix structure"""
        try:
            # Use QR decomposition for general matrices
            Q, R = torch.linalg.qr(matrix)
            
            # Optimize R matrix
            R_opt = self._optimize_upper_triangular(R)
            
            return Q @ R_opt
        except Exception as e:
            logger.warning(f"Error in general structure optimization: {e}")
            return matrix
    
    def _matrix_to_polynomial_coefficients_enhanced(self, 
                                                  matrix: torch.Tensor,
                                                  degree: int) -> List[float]:
        """Convert matrix to polynomial coefficients using 2D Chebyshev fitting"""
        try:
            # Convert to numpy for scipy operations
            matrix_np = matrix.cpu().numpy()
            
            # Normalize matrix if enabled
            if self.poly_params["normalization_enabled"]:
                scaler = StandardScaler()
                matrix_np = scaler.fit_transform(matrix_np)
            
            # Create coordinate grids
            rows, cols = matrix_np.shape
            x = np.linspace(-1, 1, rows)
            y = np.linspace(-1, 1, cols)
            X, Y = np.meshgrid(x, y, indexing='ij')
            
            # Use 2D Chebyshev fitting
            cheb_degree = self.poly_params["chebyshev_degree"]
            coeffs = np.polynomial.chebyshev.chebfit2d(X, Y, matrix_np, (cheb_degree, cheb_degree))
            
            return coeffs.flatten().tolist()
        except Exception as e:
            logger.warning(f"Error in enhanced polynomial fitting: {e}")
            return self._matrix_to_polynomial_coefficients_fallback(matrix, degree)
    
    def _matrix_to_polynomial_coefficients_fallback(self, 
                                                   matrix: torch.Tensor,
                                                   degree: int) -> List[float]:
        """Fallback polynomial coefficient extraction"""
        coeffs = []
        rows, cols = matrix.shape
        
        # Create polynomial basis
        for d in range(degree + 1):
            for i in range(rows):
                for j in range(cols):
                    # Polynomial term: x^i * y^j
                    if i + j <= d:
                        coeffs.append(matrix[i, j].item())
        
        return coeffs
    
    def _optimize_polynomial_coefficients_enhanced(self, coeffs: List[float]) -> List[float]:
        """Optimize polynomial coefficients with adaptive thresholding and smoothing"""
        try:
            coeffs_array = np.array(coeffs)
            
            if self.poly_params["adaptive_thresholding"]:
                # Adaptive thresholding based on coefficient distribution
                threshold = np.std(coeffs_array) * 0.5
                mask = np.abs(coeffs_array) > threshold
                optimized_coeffs = coeffs_array * mask
            else:
                # Simple thresholding
                threshold = np.std(coeffs_array) * 0.5
                optimized_coeffs = coeffs_array * (np.abs(coeffs_array) > threshold)
            
            # Apply smoothing if enabled
            if self.poly_params["smoothing_factor"] > 0:
                smoothing_factor = self.poly_params["smoothing_factor"]
                optimized_coeffs = optimized_coeffs * (1 - smoothing_factor) + np.mean(optimized_coeffs) * smoothing_factor
            
            return optimized_coeffs.tolist()
        except Exception as e:
            logger.warning(f"Error in enhanced coefficient optimization: {e}")
            return coeffs
    
    def _polynomial_coefficients_to_matrix(self, 
                                         coeffs: List[float],
                                         shape: Tuple[int, int]) -> torch.Tensor:
        """Convert polynomial coefficients back to matrix"""
        rows, cols = shape
        
        # Reconstruct matrix from coefficients
        matrix = torch.zeros(shape, dtype=self.dtype, device=self.device)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx < len(coeffs):
                    matrix[i, j] = coeffs[idx]
                    idx += 1
        
        return matrix
    
    def _apply_sparse_optimization(self, sparse_matrix: torch.Tensor) -> torch.Tensor:
        """Apply optimization to sparse matrix"""
        try:
            # Remove small elements
            threshold = torch.std(sparse_matrix._values()) * 0.5
            mask = torch.abs(sparse_matrix._values()) > threshold
            
            # Create new sparse matrix
            indices = sparse_matrix._indices()[:, mask]
            values = sparse_matrix._values()[mask]
            
            return torch.sparse_coo_tensor(indices, values, sparse_matrix.shape, device=self.device)
        except Exception as e:
            logger.warning(f"Error in sparse optimization: {e}")
            return sparse_matrix
    
    def _optimize_upper_triangular(self, R: torch.Tensor) -> torch.Tensor:
        """Optimize upper triangular matrix"""
        try:
            # Apply thresholding to upper triangular matrix
            threshold = torch.std(R) * 0.3
            R_opt = R * (torch.abs(R) > threshold)
            
            return R_opt
        except Exception as e:
            logger.warning(f"Error in upper triangular optimization: {e}")
            return R
    
    def _validate_optimization(self, original: torch.Tensor, optimized: torch.Tensor) -> Dict[str, Any]:
        """Validate optimization results"""
        validation = {}
        
        # Calculate error metrics
        validation["error_metrics"] = self._calculate_error_metrics(original, optimized)
        
        # Analyze spectrum if enabled
        if self.poly_params["spectrum_analysis"]:
            validation["spectrum_analysis"] = self._analyze_spectrum(original, optimized)
        
        return validation
    
    def _analyze_spectrum(self, original: torch.Tensor, optimized: torch.Tensor) -> Dict[str, Any]:
        """Analyze singular value spectrum"""
        try:
            # Compute singular values
            S_orig = torch.linalg.svd(original)[1]
            S_opt = torch.linalg.svd(optimized)[1]
            
            return {
                "original_singular_values": S_orig.cpu().numpy(),
                "optimized_singular_values": S_opt.cpu().numpy(),
                "spectrum_preservation": float(torch.norm(S_opt) / torch.norm(S_orig))
            }
        except Exception as e:
            logger.warning(f"Error in spectrum analysis: {e}")
            return {"error": str(e)}
    
    def create_validation_plots(self, original: torch.Tensor, optimized: torch.Tensor, 
                               save_path: str = "validation_plots.png"):
        """Create validation plots"""
        if not self.poly_params["validation_plots"]:
            return
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original matrix heatmap
            im1 = axes[0, 0].imshow(original.cpu().numpy(), cmap='viridis')
            axes[0, 0].set_title('Original Matrix')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Optimized matrix heatmap
            im2 = axes[0, 1].imshow(optimized.cpu().numpy(), cmap='viridis')
            axes[0, 1].set_title('Optimized Matrix')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # Difference heatmap
            diff = original - optimized
            im3 = axes[0, 2].imshow(diff.cpu().numpy(), cmap='RdBu_r')
            axes[0, 2].set_title('Difference')
            plt.colorbar(im3, ax=axes[0, 2])
            
            # Sparsity comparison
            sparsity_orig = 1.0 - torch.count_nonzero(original) / original.numel()
            sparsity_opt = 1.0 - torch.count_nonzero(optimized) / optimized.numel()
            axes[1, 0].bar(['Original', 'Optimized'], [sparsity_orig, sparsity_opt])
            axes[1, 0].set_title('Sparsity Comparison')
            axes[1, 0].set_ylabel('Sparsity Ratio')
            
            # Singular values comparison
            try:
                S_orig = torch.linalg.svd(original)[1]
                S_opt = torch.linalg.svd(optimized)[1]
                axes[1, 1].semilogy(S_orig.cpu().numpy(), label='Original')
                axes[1, 1].semilogy(S_opt.cpu().numpy(), label='Optimized')
                axes[1, 1].set_title('Singular Values')
                axes[1, 1].legend()
            except:
                axes[1, 1].text(0.5, 0.5, 'SVD failed', ha='center', va='center')
            
            # Error distribution
            error_flat = diff.flatten().cpu().numpy()
            axes[1, 2].hist(error_flat, bins=50, alpha=0.7)
            axes[1, 2].set_title('Error Distribution')
            axes[1, 2].set_xlabel('Error')
            axes[1, 2].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Validation plots saved to {save_path}")
        except Exception as e:
            logger.warning(f"Error creating validation plots: {e}")
    
    def batch_optimize(self, 
                      matrices: List[torch.Tensor],
                      method: str = "sparsity",
                      **kwargs) -> List[Dict[str, Any]]:
        """Optimize multiple matrices"""
        results = []
        
        for i, matrix in enumerate(matrices):
            try:
                logger.info(f"Optimizing matrix {i+1}/{len(matrices)}")
                result = self.optimize_matrix(matrix, method, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error optimizing matrix {i+1}: {e}")
                results.append({"error": str(e), "matrix_index": i})
        
        return results
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        if self.use_gpu:
            try:
                allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
                max_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                return {
                    "gpu_allocated_gb": allocated,
                    "gpu_reserved_gb": reserved,
                    "gpu_max_memory_gb": max_memory,
                    "gpu_utilization": allocated / max_memory
                }
            except Exception as e:
                logger.warning(f"Error getting GPU memory usage: {e}")
                return {"gpu_error": str(e)}
        else:
            return {
                "cpu_memory_gb": 0.0  # Would need psutil for actual CPU memory
            }


def main():
    """Test the matrix processor"""
    parser = argparse.ArgumentParser(description="Matrix Processor for 9xdSq-LIMPS-FemTO-R1C")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--precision", default="float32", choices=["float32", "float64", "float16"], 
                       help="Numerical precision")
    parser.add_argument("--output", default="matrix_optimization_results.json", 
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    
    logger.info("Testing Enhanced Matrix Processor")
    
    # Initialize processor
    processor = MatrixProcessor(use_gpu=args.gpu, precision=args.precision, debug=args.debug)
    
    # Test matrices
    test_matrices = [
        torch.randn(10, 10),
        torch.randn(20, 20),
        torch.randn(5, 15)
    ]
    
    # Test different optimization methods
    methods = ["sparsity", "rank", "structure", "polynomial"]
    
    all_results = {}
    
    for i, matrix in enumerate(test_matrices):
        logger.info(f"\n=== Testing Matrix {i+1} (shape: {matrix.shape}) ===")
        matrix_results = {}
        
        for method in methods:
            logger.info(f"\nMethod: {method}")
            try:
                result = processor.optimize_matrix(matrix, method)
                
                logger.info(f"  Compression ratio: {result['compression_ratio']:.3f}")
                logger.info(f"  Optimization time: {result['optimization_time']:.4f}s")
                
                if "sparsity_ratio" in result:
                    logger.info(f"  Sparsity ratio: {result['sparsity_ratio']:.3f}")
                if "rank_reduction" in result:
                    logger.info(f"  Rank reduction: {result['rank_reduction']:.3f}")
                
                # Create validation plots for first matrix
                if i == 0 and "optimized_matrix" in result:
                    plot_path = f"validation_plots_matrix_{i+1}_{method}.png"
                    processor.create_validation_plots(matrix, result["optimized_matrix"], plot_path)
                
                matrix_results[method] = result
            except Exception as e:
                logger.error(f"Error in {method} optimization: {e}")
                matrix_results[method] = {"error": str(e)}
        
        all_results[f"matrix_{i+1}"] = matrix_results
    
    # Test memory usage
    memory_info = processor.get_memory_usage()
    logger.info(f"\nMemory usage: {memory_info}")
    
    # Save results
    try:
        # Convert tensors to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        json_results = convert_for_json(all_results)
        
        with open(args.output, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {args.output}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")


if __name__ == "__main__":
    main()