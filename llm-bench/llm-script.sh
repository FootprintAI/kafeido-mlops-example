#!/bin/bash

# Quick Maximum Tokens/Second Test for GPT-OSS 20B
# This script runs a fast test to find peak tokens/second performance
#
# Usage:
#   ./llm-script.sh                              # Normal run (3 rounds each test)
#   DEBUG=true ./llm-script.sh                   # Debug mode with full request/response details
#   VERBOSE=true ./llm-script.sh                 # Show request/response for all tests
#   ROUNDS=5 ./llm-script.sh                     # Run 5 rounds per test
#   WARMUP_ROUNDS=2 ROUNDS=5 ./llm-script.sh     # 2 warmup + 5 measured rounds
#   PREDICT_TOKENS=800 CONTEXT_SIZE=16384 ./llm-script.sh  # Larger limits for complex prompts
#   MAX_RETRIES=3 REQUEST_TIMEOUT=180 ./llm-script.sh       # More retries, longer timeout
#
# Requirements:
#   - curl, jq, bc utilities
#   - Running Ollama instance with gpt-oss:20b model

MODEL="gpt-oss:20b"
HOST="localhost:11434"
MAX_TOKENS_PER_SECOND=0
BEST_RUN=""
DEBUG=${DEBUG:-false}
VERBOSE=${VERBOSE:-false}
ROUNDS=${ROUNDS:-3}
WARMUP_ROUNDS=${WARMUP_ROUNDS:-1}
MAX_RETRIES=${MAX_RETRIES:-2}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-120}
CONTEXT_SIZE=${CONTEXT_SIZE:-8192}
PREDICT_TOKENS=${PREDICT_TOKENS:-600}

echo "🎯 Quick Maximum Tokens/Second Test for $MODEL"
echo "=============================================="
echo "Host: $HOST"
echo "Rounds per test: $ROUNDS (+ $WARMUP_ROUNDS warmup)"
echo "Config: ${PREDICT_TOKENS} tokens, ${CONTEXT_SIZE} context, ${REQUEST_TIMEOUT}s timeout, ${MAX_RETRIES} retries"
echo "Started: $(date)"
echo ""

# Test connection and model availability
echo "🔍 Testing connection and model availability..."
test_response=$(curl -s --max-time 10 -w "%{http_code}" http://$HOST/api/generate \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"Test\", \"stream\": false, \"options\": {\"num_predict\": 1}}")

test_http_code="${test_response: -3}"
test_response_body="${test_response%???}"

if [ "$test_http_code" != "200" ]; then
    echo "❌ Connection failed. HTTP status: $test_http_code"
    echo "Response: $test_response_body"
    echo "Please check:"
    echo "  - Ollama is running (ollama serve)"
    echo "  - Model is available (ollama list)"
    echo "  - Host/port is correct: $HOST"
    exit 1
fi

test_error=$(echo "$test_response_body" | jq -r '.error // empty')
if [ ! -z "$test_error" ]; then
    echo "❌ Model error: $test_error"
    echo "Please check:"
    echo "  - Model '$MODEL' is installed (ollama pull $MODEL)"
    echo "  - Model name is correct"
    exit 1
fi

echo "✅ Connection successful, model responding"
echo ""

# Function to run a single request (internal use)
run_single_request() {
    local prompt="$1"
    local max_tokens="$2"
    local temp="$3"
    local is_warmup="$4"
    
    # Use configurable values or override with test-specific values
    local actual_predict_tokens=${max_tokens:-$PREDICT_TOKENS}
    local actual_context_size=${CONTEXT_SIZE}
    
    # Escape the prompt for JSON
    local escaped_prompt=$(echo "$prompt" | jq -R -s '.')
    
    # Prepare request payload using jq to ensure proper JSON formatting
    request_payload=$(jq -n \
        --arg model "$MODEL" \
        --argjson prompt "$escaped_prompt" \
        --arg temp "$temp" \
        --arg predict "$actual_predict_tokens" \
        --arg context "$actual_context_size" \
        '{
            model: $model,
            prompt: $prompt,
            stream: false,
            options: {
                num_predict: ($predict | tonumber),
                temperature: ($temp | tonumber),
                top_p: 0.9,
                top_k: 40,
                num_ctx: ($context | tonumber),
                stop: ["<|thinking|>"],
                repeat_penalty: 1.1
            }
        }')
    
    if [ "$is_warmup" != "true" ] && ([ "$DEBUG" = "true" ] || [ "$VERBOSE" = "true" ]); then
        echo "    📤 Request payload:" >&2
        echo "$request_payload" | jq '.' >&2
        echo "" >&2
    fi
    
    # Make request and measure time
    start_time=$(date +%s.%N)
    
    response=$(curl -s --max-time $REQUEST_TIMEOUT -w "%{http_code}" http://$HOST/api/generate \
        -H "Content-Type: application/json" \
        -d "$request_payload")
    
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc -l)
    
    # Extract HTTP status and response body
    http_code="${response: -3}"
    response_body="${response%???}"
    
    if [ "$is_warmup" != "true" ] && ([ "$DEBUG" = "true" ] || [ "$VERBOSE" = "true" ]); then
        echo "    📥 Response HTTP Code: $http_code" >&2
        echo "    📥 Full Response (formatted):" >&2
        echo "$response_body" | jq '.' 2>/dev/null || echo "$response_body" >&2
        echo "" >&2
    fi
    
    # Check for HTTP errors
    if [ "$http_code" != "200" ]; then
        echo "error:HTTP_Error_$http_code:$response_body"
        return 1
    fi
    
    # Extract response and check for errors
    error_msg=$(echo "$response_body" | jq -r '.error // empty')
    if [ ! -z "$error_msg" ] && [ "$error_msg" != "empty" ]; then
        echo "error:API_Error:$error_msg"
        return 1
    fi
    
    response_text=$(echo "$response_body" | jq -r '.response // empty')
    thinking_text=$(echo "$response_body" | jq -r '.thinking // empty')
    
    # Only accept proper response content, reject thinking-only responses
    content_source="response"
    if [ -z "$response_text" ] || [ "$response_text" = "null" ] || [ "$response_text" = "empty" ]; then
        echo "error:No_Response:Model_returned_only_thinking_or_empty_response"
        return 1
    fi
    
    if [ ! -z "$response_text" ] && [ "$response_text" != "null" ] && [ "$response_text" != "empty" ]; then
        token_count=$(echo "$response_text" | wc -w)
        char_count=$(echo "$response_text" | wc -c)
        tokens_per_second=$(echo "scale=2; $token_count / $duration" | bc -l)
        chars_per_second=$(echo "scale=0; $char_count / $duration" | bc -l)
        
        # Check for potential truncation (content ends abruptly without punctuation)
        last_chars=$(echo "$response_text" | tail -c 50 | tr -d '\n\r ')
        if [[ ! "$last_chars" =~ [.!?]$ ]] && [ ${#response_text} -gt 100 ]; then
            content_source="${content_source}_truncated"
        fi
        
        # Return results in a parseable format
        echo "success:$duration:$token_count:$tokens_per_second:$chars_per_second:$content_source"
        return 0
    else
        echo "error:Empty_Response:No_content_in_response_or_thinking_fields"
        return 1
    fi
}

# Helper function to extract response text for context building
get_response_text() {
    local prompt="$1"
    local max_tokens="$2"
    local temp="$3"
    
    # Make a quick request just to get the response text
    local actual_predict_tokens=${max_tokens:-$PREDICT_TOKENS}
    local actual_context_size=${CONTEXT_SIZE}
    
    # Escape the prompt for JSON
    local escaped_prompt=$(echo "$prompt" | jq -R -s '.')
    
    # Prepare request payload using jq to ensure proper JSON formatting
    request_payload=$(jq -n \
        --arg model "$MODEL" \
        --argjson prompt "$escaped_prompt" \
        --arg temp "$temp" \
        --arg predict "$actual_predict_tokens" \
        --arg context "$actual_context_size" \
        '{
            model: $model,
            prompt: $prompt,
            stream: false,
            options: {
                num_predict: ($predict | tonumber),
                temperature: ($temp | tonumber),
                top_p: 0.9,
                top_k: 40,
                num_ctx: ($context | tonumber),
                stop: ["<|thinking|>"],
                repeat_penalty: 1.1
            }
        }')
    
    response=$(curl -s --max-time $REQUEST_TIMEOUT http://$HOST/api/generate \
        -H "Content-Type: application/json" \
        -d "$request_payload")
    
    response_text=$(echo "$response" | jq -r '.response // empty')
    
    # Only return actual response content, not thinking
    if [ ! -z "$response_text" ] && [ "$response_text" != "null" ] && [ "$response_text" != "empty" ]; then
        echo "$response_text"
    else
        echo ""
    fi
}

# Function to run request with retry logic
run_request_with_retry() {
    local prompt="$1"
    local max_tokens="$2"
    local temp="$3"
    local is_warmup="$4"
    local retry_count=0
    
    while [ $retry_count -le $MAX_RETRIES ]; do
        result=$(run_single_request "$prompt" "$max_tokens" "$temp" "$is_warmup")
        
        if [[ $result == success:* ]]; then
            echo "$result"
            return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -le $MAX_RETRIES ]; then
                if [ "$is_warmup" != "true" ]; then
                    echo "      🔄 Retry $retry_count/$MAX_RETRIES after failure..." >&2
                fi
                sleep 1  # Brief pause before retry
            fi
        fi
    done
    
    # All retries failed, return the last error
    echo "$result"
    return 1
}

# Function to test tokens/second with chained prompts
test_tokens_per_second() {
    local initial_prompt="$1"
    local max_tokens="$2"
    local temp="$3"
    local description="$4"
    
    echo "Testing: $description"
    echo "Max tokens: $max_tokens, Temperature: $temp, Rounds: $ROUNDS (+ $WARMUP_ROUNDS warmup) - CHAINED"
    
    # Arrays to store results
    local durations=()
    local token_counts=()
    local tokens_per_seconds=()
    local chars_per_seconds=()
    local successful_runs=0
    local content_sources=()
    
    # Conversation context - starts with initial prompt
    local conversation_context="$initial_prompt"
    local previous_response=""
    
    # Run warmup rounds
    if [ "$WARMUP_ROUNDS" -gt 0 ]; then
        echo "  🔥 Running $WARMUP_ROUNDS warmup round(s)..."
        for i in $(seq 1 $WARMUP_ROUNDS); do
            # For warmup, use the initial prompt 
            result=$(run_request_with_retry "$conversation_context" "$max_tokens" "$temp" "true")
            if [[ $result == success:* ]]; then
                echo "    Warmup $i: ✅"
                # Extract response for context building (but don't count in stats)
                IFS=':' read -r status duration tokens tokens_per_sec chars_per_sec source <<< "$result"
                # Get the actual response text for context building
                warmup_response=$(get_response_text "$conversation_context" "$max_tokens" "$temp")
                if [ ! -z "$warmup_response" ]; then
                    previous_response="$warmup_response"
                fi
            else
                echo "    Warmup $i: ❌"
            fi
        done
        echo ""
    fi
    
    # Follow-up prompts for chaining
    local follow_up_prompts=(
        "Continue with more details and examples."
        "Expand on the previous points with practical applications."
        "Provide additional context and use cases."
        "Add more technical depth to the explanation."
        "Include implementation considerations and best practices."
        "Discuss potential challenges and solutions."
        "Compare different approaches mentioned earlier."
        "Summarize the key points and provide recommendations."
    )
    
    # Run measured rounds
    echo "  📊 Running $ROUNDS measured round(s)..."
    for i in $(seq 1 $ROUNDS); do
        echo "    Round $i/$ROUNDS:"
        
        # Build the chained prompt
        local current_prompt=""
        if [ "$i" -eq 1 ] && [ -z "$previous_response" ]; then
            # First round, use initial prompt
            current_prompt="$conversation_context"
        else
            # Subsequent rounds: previous context + follow-up
            local follow_up_index=$(( (i - 1) % ${#follow_up_prompts[@]} ))
            local follow_up="${follow_up_prompts[$follow_up_index]}"
            
            if [ ! -z "$previous_response" ]; then
                current_prompt="Previous: $previous_response

$follow_up"
            else
                current_prompt="$conversation_context

$follow_up"
            fi
        fi
        
        result=$(run_request_with_retry "$current_prompt" "$max_tokens" "$temp" "false")
        
        if [[ $result == success:* ]]; then
            # Parse result: success:duration:tokens:tokens_per_sec:chars_per_sec:source
            IFS=':' read -r status duration tokens tokens_per_sec chars_per_sec source <<< "$result"
            
            durations+=("$duration")
            token_counts+=("$tokens")
            tokens_per_seconds+=("$tokens_per_sec")
            chars_per_seconds+=("$chars_per_sec")
            content_sources+=("$source")
            successful_runs=$((successful_runs + 1))
            
            # Show truncation warning if detected
            truncation_warning=""
            if [[ "$source" == *"_truncated" ]]; then
                truncation_warning=" ⚠️ TRUNCATED"
            fi
            
            echo "      Duration: ${duration}s, Tokens: $tokens, Tokens/s: $tokens_per_sec (from ${source%_truncated})$truncation_warning"
            
            # Update conversation context for next round
            current_response=$(get_response_text "$current_prompt" "$max_tokens" "$temp")
            if [ ! -z "$current_response" ]; then
                previous_response="$current_response"
            fi
        else
            # Parse error: error:type:message
            IFS=':' read -r status error_type error_msg <<< "$result"
            echo "      ❌ $error_type: $error_msg"
            
            # Show raw result for debugging if needed
            if [ "$DEBUG" = "true" ]; then
                echo "      Debug - Raw result: '$result'"
            fi
        fi
    done
    
    echo ""
    
    # Calculate statistics if we have successful runs
    if [ "$successful_runs" -gt 0 ]; then
        # Calculate averages
        local avg_duration=$(echo "${durations[@]}" | tr ' ' '\n' | awk '{sum+=$1} END {printf "%.3f", sum/NR}')
        local avg_tokens=$(echo "${token_counts[@]}" | tr ' ' '\n' | awk '{sum+=$1} END {printf "%.0f", sum/NR}')
        local avg_tokens_per_sec=$(echo "${tokens_per_seconds[@]}" | tr ' ' '\n' | awk '{sum+=$1} END {printf "%.2f", sum/NR}')
        local avg_chars_per_sec=$(echo "${chars_per_seconds[@]}" | tr ' ' '\n' | awk '{sum+=$1} END {printf "%.0f", sum/NR}')
        
        # Find max tokens/second
        local max_tokens_per_sec=$(echo "${tokens_per_seconds[@]}" | tr ' ' '\n' | sort -n | tail -1)
        
        # Calculate standard deviation for tokens/second
        local std_dev=0
        if [ "$successful_runs" -gt 1 ]; then
            std_dev=$(echo "${tokens_per_seconds[@]}" | tr ' ' '\n' | awk -v avg="$avg_tokens_per_sec" '{sum += ($1 - avg)^2} END {printf "%.2f", sqrt(sum/(NR-1))}')
        fi
        
        echo "  📈 Results Summary ($successful_runs/$ROUNDS successful):"
        echo "    Average: ${avg_tokens_per_sec} tokens/s (±${std_dev})"
        echo "    Maximum: ${max_tokens_per_sec} tokens/s"
        echo "    Avg tokens: $avg_tokens, Avg duration: ${avg_duration}s"
        # Check for truncation issues
        truncated_count=$(printf '%s\n' "${content_sources[@]}" | grep -c "_truncated" || true)
        if [ "$truncated_count" -gt 0 ]; then
            echo "    Content source: ${content_sources[0]%_truncated} (⚠️ $truncated_count/$successful_runs truncated)"
        else
            echo "    Content source: ${content_sources[0]}"
        fi
        
        # Check if this is the best result (using max, not average)
        if (( $(echo "$max_tokens_per_sec > $MAX_TOKENS_PER_SECOND" | bc -l) )); then
            MAX_TOKENS_PER_SECOND=$max_tokens_per_sec
            BEST_RUN="$description (${max_tokens_per_sec} tokens/s max, ${avg_tokens_per_sec}±${std_dev} avg, $successful_runs/$ROUNDS success)"
        fi
        
        echo "  ✅ Test completed successfully"
    else
        echo "  ❌ All rounds failed"
        echo "  Possible causes:"
        echo "    - Model stopped generating (context limit reached)"
        echo "    - Prompt too complex for requested token count"
        echo "    - Model overloaded or temperature issues"
        echo "    - Connection issues"
    fi
    echo ""
}

# Test 1: Technical Discussion Chain
test_tokens_per_second \
    "Explain machine learning algorithms and their applications in modern software development." \
    "" \
    0.7 \
    "ML Technical Discussion"

# Test 2: Creative Content Chain  
test_tokens_per_second \
    "Write a story about a software engineer discovering an AI that can predict the future." \
    "" \
    0.8 \
    "Creative AI Story"

# Test 3: Problem-Solving Chain
test_tokens_per_second \
    "Help me design a scalable web application architecture for handling millions of users." \
    "" \
    0.6 \
    "Architecture Design Discussion"

# Results Summary
echo "================================================"
echo "🏆 MAXIMUM PERFORMANCE RESULTS"
echo "================================================"
echo "Maximum tokens/second achieved: $MAX_TOKENS_PER_SECOND"
echo "Best run: $BEST_RUN"
echo ""
echo "Completed: $(date)"

# Save results to file
results_file="quick_max_tokens_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "Maximum Tokens/Second Test Results"
    echo "================================="
    echo "Model: $MODEL"
    echo "Host: $HOST"
    echo "Test Date: $(date)"
    echo ""
    echo "Maximum tokens/second: $MAX_TOKENS_PER_SECOND"
    echo "Best configuration: $BEST_RUN"
} > "$results_file"

echo "💾 Results saved to: $results_file"

# Performance classification
if (( $(echo "$MAX_TOKENS_PER_SECOND >= 50" | bc -l) )); then
    echo "🚀 EXCELLENT: >50 tokens/second"
elif (( $(echo "$MAX_TOKENS_PER_SECOND >= 30" | bc -l) )); then
    echo "✅ GOOD: 30-50 tokens/second"
elif (( $(echo "$MAX_TOKENS_PER_SECOND >= 15" | bc -l) )); then
    echo "⚠️ FAIR: 15-30 tokens/second"
elif (( $(echo "$MAX_TOKENS_PER_SECOND >= 5" | bc -l) )); then
    echo "🐌 SLOW: 5-15 tokens/second"
else
    echo "❌ VERY SLOW: <5 tokens/second"
fi
