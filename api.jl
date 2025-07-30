module API

"""
REST API submodule for LIMPS microservice
"""

using HTTP
using Sockets
using JSON
using Logging

export start_http_server, start_limps_server

"""
Convert any Julia object to JSON-serializable format
"""
function to_json(obj::Any)
    try
        if obj isa Dict
            # Handle nested dictionaries
            result = Dict{String, Any}()
            for (k, v) in obj
                result[string(k)] = to_json(v)
            end
            return result
        elseif obj isa Array
            # Handle arrays
            return [to_json(item) for item in obj]
        elseif obj isa Number
            # Handle numbers
            return obj
        elseif obj isa String
            # Handle strings
            return obj
        elseif obj isa Bool
            # Handle booleans
            return obj
        else
            # Convert other types to string
            return string(obj)
        end
    catch e
        return Dict{String, Any}("error" => "JSON conversion failed: $(e)")
    end
end

"""
Start basic HTTP server for Julia functions
"""
function start_http_server(port::Int=8000)
    println("Starting Julia HTTP server on port $port")
    
    HTTP.serve(port) do req::HTTP.Request
        try
            if req.method == "POST"
                body = JSON.parse(String(req.body))
                
                if haskey(body, "function")
                    func_name = body["function"]
                    args = get(body, "args", [])
                    
                    if func_name == "create_polynomials"
                        data = Matrix{Float64}(args[1])
                        variables = Vector{String}(args[2])
                        result = create_polynomials(data, variables)
                        return HTTP.Response(200, JSON.json(to_json(result)))
                        
                    elseif func_name == "analyze_polynomials"
                        polys = Dict{String, Any}(args[1])
                        result = analyze_polynomials(polys)
                        return HTTP.Response(200, JSON.json(to_json(result)))
                        
                    elseif func_name == "optimize_matrix"
                        matrix = Matrix{Float64}(args[1])
                        method = get(args, 2, "sparsity")
                        result = optimize_matrix(matrix, method)
                        return HTTP.Response(200, JSON.json(to_json(result)))
                        
                    elseif func_name == "matrix_to_polynomials"
                        matrix = Matrix{Float64}(args[1])
                        result = matrix_to_polynomials(matrix)
                        return HTTP.Response(200, JSON.json(to_json(result)))
                        
                    elseif func_name == "analyze_text_structure"
                        text = String(args[1])
                        result = analyze_text_structure(text)
                        return HTTP.Response(200, JSON.json(to_json(result)))
                        
                    else
                        return HTTP.Response(400, JSON.json(Dict("error" => "Unknown function: $func_name")))
                    end
                else
                    return HTTP.Response(400, JSON.json(Dict("error" => "No function specified")))
                end
            else
                return HTTP.Response(405, JSON.json(Dict("error" => "Method not allowed")))
            end
        catch e
            return HTTP.Response(500, JSON.json(Dict("error" => "Server error: $(e)")))
        end
    end
end

"""
Start comprehensive LIMPS REST API server
"""
function start_limps_server(port::Int=8000; config::Dict{String, Any}=Dict())
    println("Starting LIMPS REST API server on port $port")
    
    # Default configuration
    default_config = Dict(
        "enable_cors" => true,
        "max_request_size" => "10MB",
        "timeout" => 30,
        "log_level" => "INFO"
    )
    
    # Merge with provided config
    server_config = merge(default_config, config)
    
    # CORS headers
    cors_headers = server_config["enable_cors"] ? [
        "Access-Control-Allow-Origin" => "*",
        "Access-Control-Allow-Methods" => "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers" => "Content-Type, Authorization"
    ] : []
    
    HTTP.serve(port) do req::HTTP.Request
        try
            # Handle CORS preflight
            if req.method == "OPTIONS"
                return HTTP.Response(200, cors_headers)
            end
            
            # Parse request
            if req.method == "GET"
                return handle_get_request(req, cors_headers)
            elseif req.method == "POST"
                return handle_post_request(req, cors_headers)
            else
                return HTTP.Response(405, cors_headers, JSON.json(Dict("error" => "Method not allowed")))
            end
            
        catch e
            @error "Server error" exception=(e, catch_backtrace())
            return HTTP.Response(500, cors_headers, JSON.json(Dict("error" => "Internal server error: $(e)")))
        end
    end
end

"""
Handle GET requests
"""
function handle_get_request(req::HTTP.Request, headers::Vector{Pair{String, String}})
    path = HTTP.URIs.splitpath(req.target)
    
    if path[1] == "health"
        return HTTP.Response(200, headers, JSON.json(health_check()))
    elseif path[1] == "status"
        return HTTP.Response(200, headers, JSON.json(get_server_status()))
    elseif path[1] == "docs"
        return HTTP.Response(200, headers, JSON.json(get_api_docs()))
    else
        return HTTP.Response(404, headers, JSON.json(Dict("error" => "Endpoint not found")))
    end
end

"""
Handle POST requests
"""
function handle_post_request(req::HTTP.Request, headers::Vector{Pair{String, String}})
    path = HTTP.URIs.splitpath(req.target)
    
    if path[1] == "process"
        return handle_process_request(req, headers)
    elseif path[1] == "batch"
        return handle_batch_request(req, headers)
    elseif path[1] == "optimize"
        return handle_optimize_request(req, headers)
    elseif path[1] == "analyze"
        return handle_analyze_request(req, headers)
    else
        return HTTP.Response(404, headers, JSON.json(Dict("error" => "Endpoint not found")))
    end
end

"""
Handle /process endpoint
"""
function handle_process_request(req::HTTP.Request, headers::Vector{Pair{String, String}})
    body = JSON.parse(String(req.body))
    
    if !haskey(body, "data") || !haskey(body, "type")
        return HTTP.Response(400, headers, JSON.json(Dict("error" => "Missing 'data' or 'type' field")))
    end
    
    data = body["data"]
    data_type = body["type"]
    
    try
        if data_type == "matrix"
            matrix = Matrix{Float64}(data)
            result = process_limps_data(matrix, "matrix")
        elseif data_type == "text"
            text = String(data)
            result = process_limps_data(text, "text")
        else
            return HTTP.Response(400, headers, JSON.json(Dict("error" => "Unsupported data type: $data_type")))
        end
        
        return HTTP.Response(200, headers, JSON.json(to_json(result)))
    catch e
        return HTTP.Response(500, headers, JSON.json(Dict("error" => "Processing failed: $(e)")))
    end
end

"""
Handle /batch endpoint
"""
function handle_batch_request(req::HTTP.Request, headers::Vector{Pair{String, String}})
    body = JSON.parse(String(req.body))
    
    if !haskey(body, "data_list") || !haskey(body, "types")
        return HTTP.Response(400, headers, JSON.json(Dict("error" => "Missing 'data_list' or 'types' field")))
    end
    
    data_list = body["data_list"]
    types = body["types"]
    
    try
        result = batch_process_limps(data_list, types)
        return HTTP.Response(200, headers, JSON.json(to_json(result)))
    catch e
        return HTTP.Response(500, headers, JSON.json(Dict("error" => "Batch processing failed: $(e)")))
    end
end

"""
Handle /optimize endpoint
"""
function handle_optimize_request(req::HTTP.Request, headers::Vector{Pair{String, String}})
    body = JSON.parse(String(req.body))
    
    if !haskey(body, "matrix")
        return HTTP.Response(400, headers, JSON.json(Dict("error" => "Missing 'matrix' field")))
    end
    
    matrix = Matrix{Float64}(body["matrix"])
    method = get(body, "method", "sparsity")
    
    try
        result = optimize_matrix(matrix, method)
        return HTTP.Response(200, headers, JSON.json(to_json(result)))
    catch e
        return HTTP.Response(500, headers, JSON.json(Dict("error" => "Optimization failed: $(e)")))
    end
end

"""
Handle /analyze endpoint
"""
function handle_analyze_request(req::HTTP.Request, headers::Vector{Pair{String, String}})
    body = JSON.parse(String(req.body))
    
    if !haskey(body, "text")
        return HTTP.Response(400, headers, JSON.json(Dict("error" => "Missing 'text' field")))
    end
    
    text = String(body["text"])
    
    try
        result = analyze_text_structure(text)
        return HTTP.Response(200, headers, JSON.json(to_json(result)))
    catch e
        return HTTP.Response(500, headers, JSON.json(Dict("error" => "Analysis failed: $(e)")))
    end
end

"""
Get server status
"""
function get_server_status()
    return Dict(
        "status" => "running",
        "timestamp" => string(now()),
        "version" => "1.0.0",
        "uptime" => "0s",  # Would need to track actual uptime
        "memory_usage" => "N/A"  # Would need to implement memory tracking
    )
end

"""
Get API documentation
"""
function get_api_docs()
    return Dict(
        "endpoints" => Dict(
            "GET /health" => "Health check endpoint",
            "GET /status" => "Server status information",
            "GET /docs" => "API documentation",
            "POST /process" => "Process single dataset",
            "POST /batch" => "Process multiple datasets",
            "POST /optimize" => "Optimize matrix",
            "POST /analyze" => "Analyze text structure"
        ),
        "examples" => Dict(
            "process_matrix" => Dict(
                "url" => "/process",
                "method" => "POST",
                "body" => Dict(
                    "data" => [[1.0, 2.0], [3.0, 4.0]],
                    "type" => "matrix"
                )
            ),
            "analyze_text" => Dict(
                "url" => "/analyze",
                "method" => "POST",
                "body" => Dict(
                    "text" => "Sample text for analysis"
                )
            )
        )
    )
end

end # module API