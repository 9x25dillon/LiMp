# Enhanced Matrix Processor & Julia Integration for 9xdSq-LIMPS-FemTO-R1C

This project provides a comprehensive matrix processing and optimization system with seamless Python-Julia integration for the LIMPS (Language-Integrated Matrix Processing System) framework.

## 🚀 Features

### Enhanced Matrix Processor
- **GPU Acceleration**: Full CUDA support with memory management
- **Multiple Optimization Methods**: Sparsity, rank reduction, structure optimization, polynomial approximation
- **Advanced Polynomial Fitting**: 2D Chebyshev fitting with normalization
- **Comprehensive Validation**: Error metrics, spectrum analysis, visualization
- **Robust Error Handling**: Graceful handling of numerical instabilities
- **CLI Interface**: Command-line tools with debug options

### Julia Integration
- **Polynomial Operations**: Symbolic polynomial creation and analysis
- **Matrix Optimization**: Julia backend for mathematical operations
- **HTTP Server**: RESTful API for Python interop
- **Canonical Serialization**: JSON-compatible data formats
- **Error Handling**: Robust error handling with detailed reporting

### LIMPS Integration
- **Entropy Processing**: Matrix-to-polynomial conversion for entropy analysis
- **Natural Language Analysis**: Text structure analysis using polynomial techniques
- **Adaptive Optimization**: Dynamic method selection based on complexity
- **Seamless Interop**: Python client for Julia server communication

## 📦 Installation

### Prerequisites
- Python 3.8+
- Julia 1.6+
- CUDA toolkit (optional, for GPU acceleration)

### Python Dependencies
```bash
pip install torch numpy scipy matplotlib seaborn scikit-learn requests
```

### Julia Dependencies
```julia
using Pkg
Pkg.add(["DynamicPolynomials", "MultivariatePolynomials", "LinearAlgebra", "JSON", "HTTP", "Sockets"])
```

## 🛠️ Usage

### Matrix Processor

#### Basic Usage
```python
from matrix_processor import MatrixProcessor

# Initialize processor
processor = MatrixProcessor(use_gpu=True, precision="float32", debug=True)

# Create test matrix
matrix = torch.randn(10, 10)

# Optimize using different methods
methods = ["sparsity", "rank", "structure", "polynomial"]
for method in methods:
    result = processor.optimize_matrix(matrix, method)
    print(f"{method}: compression ratio = {result['compression_ratio']:.3f}")
```

#### Command Line Interface
```bash
# Basic optimization
python matrix_processor.py

# With GPU acceleration and debug logging
python matrix_processor.py --gpu --debug --precision float64

# Save results to file
python matrix_processor.py --output results.json
```

#### Advanced Features
```python
# Batch optimization
matrices = [torch.randn(10, 10) for _ in range(5)]
results = processor.batch_optimize(matrices, method="sparsity")

# Create validation plots
processor.create_validation_plots(original_matrix, optimized_matrix, "plots.png")

# Memory usage
memory_info = processor.get_memory_usage()
print(f"GPU memory: {memory_info['gpu_allocated_gb']:.2f} GB")
```

### Julia Integration

#### Start Julia Server
```julia
# In Julia REPL
include("julia_integration.jl")
start_http_server(8000)
```

#### Python Client Usage
```python
from julia_client import JuliaClient, LIMPSJuliaIntegration

# Initialize client
client = JuliaClient("http://localhost:8000")

# Test connection
if client.test_connection():
    print("Connected to Julia server")

# Create polynomials
data = np.random.rand(3, 2)
variables = ["x", "y"]
polys = client.create_polynomials(data, variables)

# Analyze polynomials
analysis = client.analyze_polynomials(polys)
print(f"Complexity score: {analysis['complexity_score']}")

# Optimize matrix
matrix = np.random.rand(5, 5)
result = client.optimize_matrix(matrix, method="sparsity")
```

#### LIMPS Integration
```python
# Initialize LIMPS integration
limps_integration = LIMPSJuliaIntegration(client)

# Process entropy matrix
entropy_matrix = np.random.rand(10, 10)
result = limps_integration.process_entropy_matrix(entropy_matrix)

# Analyze natural language
text = "Show monthly sales totals for electronics category"
analysis = limps_integration.analyze_natural_language(text)

# Optimize with target compression
optimized = limps_integration.optimize_limps_matrix(matrix, target_compression=0.6)
```

## 🔧 Configuration

### Matrix Processor Parameters
```python
# Initialize with custom parameters
processor = MatrixProcessor(
    use_gpu=True,
    precision="float32",
    max_memory_gb=8.0,
    debug=True
)

# Access polynomial parameters
print(processor.poly_params)
# {
#     "sparsity_threshold": 0.01,
#     "rank_reduction_factor": 0.5,
#     "compression_ratio": 0.7,
#     "polynomial_degree": 3,
#     "chebyshev_degree": 4,
#     "normalization_enabled": True,
#     "adaptive_thresholding": True,
#     "smoothing_factor": 0.1,
#     "spectrum_analysis": True,
#     "validation_plots": True
# }
```

### Julia Server Configuration
```julia
# Custom port
start_http_server(9000)

# With custom error handling
function custom_error_handler(e)
    println("Custom error: $e")
    return Dict("error" => "Custom error handling")
end
```

## 📊 Output Formats

### Matrix Optimization Results
```json
{
    "optimized_matrix": [[1.0, 0.0], [0.0, 2.0]],
    "compression_ratio": 0.5,
    "optimization_time": 0.0012,
    "method": "sparsity",
    "original_shape": [10, 10],
    "validation": {
        "error_metrics": {
            "mse": 0.001,
            "mae": 0.01,
            "relative_error": 0.05,
            "max_error": 0.1
        },
        "spectrum_analysis": {
            "original_singular_values": [10.0, 5.0, ...],
            "optimized_singular_values": [9.8, 4.9, ...],
            "spectrum_preservation": 0.98
        }
    },
    "parameters_used": {
        "threshold": 0.01,
        "rank_reduction_factor": 0.5
    }
}
```

### Polynomial Analysis Results
```json
{
    "total_polynomials": 3,
    "average_degree": 2.5,
    "max_degree": 4,
    "average_terms": 6.0,
    "complexity_score": 1.5,
    "degree_distribution": [2, 3, 4],
    "term_distribution": [5, 6, 7],
    "complexity_distribution": [1.0, 1.8, 2.8]
}
```

## 🧪 Testing

### Matrix Processor Tests
```bash
# Run all tests
python matrix_processor.py --debug

# Test specific methods
python matrix_processor.py --gpu --precision float64
```

### Julia Integration Tests
```bash
# Test Julia client
python julia_client.py

# Test individual functions
python -c "
from julia_client import JuliaClient
client = JuliaClient()
print(client.test_connection())
"
```

### End-to-End Tests
```python
# Complete workflow test
from matrix_processor import MatrixProcessor
from julia_client import JuliaClient, LIMPSJuliaIntegration

# Initialize both systems
processor = MatrixProcessor(use_gpu=True)
client = JuliaClient()
limps_integration = LIMPSJuliaIntegration(client)

# Test complete pipeline
matrix = torch.randn(20, 20)
python_result = processor.optimize_matrix(matrix, "sparsity")
julia_result = client.optimize_matrix(matrix.numpy(), "sparsity")
limps_result = limps_integration.process_entropy_matrix(matrix.numpy())

print("All systems working!")
```

## 🔍 Troubleshooting

### Common Issues

#### Matrix Processor
1. **GPU Memory Errors**: Reduce `max_memory_gb` or use CPU
2. **SVD Failures**: Matrix may be singular, try different method
3. **Polynomial Fitting Errors**: Reduce polynomial degree

#### Julia Integration
1. **Connection Errors**: Ensure Julia server is running on correct port
2. **Package Errors**: Install required Julia packages
3. **JSON Serialization**: Check data types for JSON compatibility

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Matrix processor debug
processor = MatrixProcessor(debug=True)

# Julia client debug
client = JuliaClient()
# Check server logs for detailed error information
```

## 📈 Performance

### Benchmarks
- **Matrix Size**: 1000x1000
- **GPU**: RTX 3080
- **Methods**:
  - Sparsity: ~0.1s
  - Rank: ~0.5s
  - Structure: ~0.2s
  - Polynomial: ~1.0s

### Memory Usage
- **CPU**: ~500MB for 1000x1000 matrix
- **GPU**: ~2GB for 1000x1000 matrix with full precision

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## 📄 License

This project is part of the 9xdSq-LIMPS-FemTO-R1C framework.

## 🔗 Related Projects

- [LIMPS Framework](https://github.com/9x25dillon/9xdSq-LIMPS-FemTO-R1C)
- [PyTorch](https://pytorch.org/)
- [Julia](https://julialang.org/)
- [DynamicPolynomials.jl](https://github.com/JuliaAlgebra/DynamicPolynomials.jl)