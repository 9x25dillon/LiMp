module Config

"""
Configuration management submodule for LIMPS
"""

using JSON

export LIMPSConfig, configure_limps

"""
Configuration structure for LIMPS
"""
mutable struct LIMPSConfig
    # Server configuration
    server_port::Int
    enable_cors::Bool
    max_request_size::String
    timeout::Int
    log_level::String
    
    # Processing configuration
    default_optimization_method::String
    polynomial_degree::Int
    sparsity_threshold::Float64
    rank_reduction_factor::Float64
    
    # Performance configuration
    enable_gpu::Bool
    precision::String
    max_memory_gb::Float64
    
    # API configuration
    enable_health_check::Bool
    enable_docs::Bool
    rate_limit::Int
    
    function LIMPSConfig(;
        server_port=8000,
        enable_cors=true,
        max_request_size="10MB",
        timeout=30,
        log_level="INFO",
        default_optimization_method="sparsity",
        polynomial_degree=3,
        sparsity_threshold=0.01,
        rank_reduction_factor=0.5,
        enable_gpu=false,
        precision="float64",
        max_memory_gb=8.0,
        enable_health_check=true,
        enable_docs=true,
        rate_limit=1000
    )
        new(
            server_port,
            enable_cors,
            max_request_size,
            timeout,
            log_level,
            default_optimization_method,
            polynomial_degree,
            sparsity_threshold,
            rank_reduction_factor,
            enable_gpu,
            precision,
            max_memory_gb,
            enable_health_check,
            enable_docs,
            rate_limit
        )
    end
end

"""
Configure LIMPS with custom settings
"""
function configure_limps(config_dict::Dict{String, Any})
    """Configure LIMPS with dictionary of settings"""
    return LIMPSConfig(;
        server_port=get(config_dict, "server_port", 8000),
        enable_cors=get(config_dict, "enable_cors", true),
        max_request_size=get(config_dict, "max_request_size", "10MB"),
        timeout=get(config_dict, "timeout", 30),
        log_level=get(config_dict, "log_level", "INFO"),
        default_optimization_method=get(config_dict, "default_optimization_method", "sparsity"),
        polynomial_degree=get(config_dict, "polynomial_degree", 3),
        sparsity_threshold=get(config_dict, "sparsity_threshold", 0.01),
        rank_reduction_factor=get(config_dict, "rank_reduction_factor", 0.5),
        enable_gpu=get(config_dict, "enable_gpu", false),
        precision=get(config_dict, "precision", "float64"),
        max_memory_gb=get(config_dict, "max_memory_gb", 8.0),
        enable_health_check=get(config_dict, "enable_health_check", true),
        enable_docs=get(config_dict, "enable_docs", true),
        rate_limit=get(config_dict, "rate_limit", 1000)
    )
end

"""
Load configuration from JSON file
"""
function load_config_from_file(filepath::String)
    """Load LIMPS configuration from JSON file"""
    try
        config_data = JSON.parsefile(filepath)
        return configure_limps(config_data)
    catch e
        error("Failed to load configuration from $filepath: $e")
    end
end

"""
Save configuration to JSON file
"""
function save_config_to_file(config::LIMPSConfig, filepath::String)
    """Save LIMPS configuration to JSON file"""
    try
        config_dict = Dict(
            "server_port" => config.server_port,
            "enable_cors" => config.enable_cors,
            "max_request_size" => config.max_request_size,
            "timeout" => config.timeout,
            "log_level" => config.log_level,
            "default_optimization_method" => config.default_optimization_method,
            "polynomial_degree" => config.polynomial_degree,
            "sparsity_threshold" => config.sparsity_threshold,
            "rank_reduction_factor" => config.rank_reduction_factor,
            "enable_gpu" => config.enable_gpu,
            "precision" => config.precision,
            "max_memory_gb" => config.max_memory_gb,
            "enable_health_check" => config.enable_health_check,
            "enable_docs" => config.enable_docs,
            "rate_limit" => config.rate_limit
        )
        
        open(filepath, "w") do f
            JSON.print(f, config_dict, 2)
        end
        
        println("Configuration saved to $filepath")
    catch e
        error("Failed to save configuration to $filepath: $e")
    end
end

"""
Get default configuration
"""
function get_default_config()
    """Get default LIMPS configuration"""
    return LIMPSConfig()
end

"""
Validate configuration
"""
function validate_config(config::LIMPSConfig)
    """Validate LIMPS configuration"""
    errors = String[]
    
    # Validate server settings
    if config.server_port < 1 || config.server_port > 65535
        push!(errors, "Invalid server port: $(config.server_port)")
    end
    
    if config.timeout < 1
        push!(errors, "Invalid timeout: $(config.timeout)")
    end
    
    # Validate processing settings
    if config.polynomial_degree < 1
        push!(errors, "Invalid polynomial degree: $(config.polynomial_degree)")
    end
    
    if config.sparsity_threshold < 0 || config.sparsity_threshold > 1
        push!(errors, "Invalid sparsity threshold: $(config.sparsity_threshold)")
    end
    
    if config.rank_reduction_factor < 0 || config.rank_reduction_factor > 1
        push!(errors, "Invalid rank reduction factor: $(config.rank_reduction_factor)")
    end
    
    # Validate performance settings
    if config.max_memory_gb <= 0
        push!(errors, "Invalid max memory: $(config.max_memory_gb)")
    end
    
    if !(config.precision in ["float32", "float64"])
        push!(errors, "Invalid precision: $(config.precision)")
    end
    
    # Validate API settings
    if config.rate_limit < 1
        push!(errors, "Invalid rate limit: $(config.rate_limit)")
    end
    
    return errors
end

"""
Create configuration dictionary for server
"""
function get_server_config(config::LIMPSConfig)
    """Convert LIMPSConfig to server configuration dictionary"""
    return Dict(
        "enable_cors" => config.enable_cors,
        "max_request_size" => config.max_request_size,
        "timeout" => config.timeout,
        "log_level" => config.log_level
    )
end

"""
Create configuration dictionary for processing
"""
function get_processing_config(config::LIMPSConfig)
    """Convert LIMPSConfig to processing configuration dictionary"""
    return Dict(
        "default_optimization_method" => config.default_optimization_method,
        "polynomial_degree" => config.polynomial_degree,
        "sparsity_threshold" => config.sparsity_threshold,
        "rank_reduction_factor" => config.rank_reduction_factor,
        "enable_gpu" => config.enable_gpu,
        "precision" => config.precision,
        "max_memory_gb" => config.max_memory_gb
    )
end

end # module Config