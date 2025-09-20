# LLM Benchmark Suite

A comprehensive benchmarking suite for testing the maximum performance (tokens/second) of Large Language Model APIs, specifically designed for GPT-OSS 20B model.

## Overview

This directory contains two complementary benchmarking tools:

1. **Python Benchmark** (`llm-bench.py`) - Comprehensive performance analysis
2. **Shell Script** (`llm-script.sh`) - Quick performance test

## Files

### llm-bench.py

A comprehensive Python-based benchmark that performs detailed performance analysis including:

- **Optimal prompt length testing** - Tests different prompt types to find optimal generation patterns
- **Max tokens configuration** - Tests various token limits (100-1500) to find peak performance  
- **Temperature optimization** - Tests different temperature settings (0.1-0.9) for performance impact
- **Stress testing** - Long-form content generation with up to 2000 tokens
- **Concurrent testing** - Multi-threaded performance analysis

**Features:**
- Detailed metrics collection (tokens/second, characters/second, latency)
- JSON result output with timestamps
- Statistical analysis across multiple runs
- Automatic tracking of best configurations

### llm-script.sh

A lightweight bash script for quick performance testing:

- **6 different test scenarios** with varying content types and token limits
- **Performance classification** (Excellent >50 t/s, Good 30-50 t/s, etc.)
- **Quick results** - Fast execution for immediate feedback
- **Text file output** with timestamp

## Usage

### Python Benchmark (Detailed Analysis)

```bash
python3 llm-bench.py
```

**Requirements:**
- Python 3.x
- `requests`, `psutil` libraries
- Running Ollama instance with GPT-OSS 20B model

### Shell Script (Quick Test)

```bash
./llm-script.sh
```

**Requirements:**
- `curl`, `jq`, `bc` utilities
- Running Ollama instance with GPT-OSS 20B model

## Configuration

Both tools are configured for:
- **Host:** localhost:11434 (Ollama default)
- **Model:** gpt-oss:20b
- **Timeout:** 300 seconds (Python version)

## Output

### Python Benchmark
- Console output with detailed progress
- JSON file: `max_tokens_benchmark_YYYYMMDD_HHMMSS.json`

### Shell Script  
- Console output with real-time results
- Text file: `quick_max_tokens_YYYYMMDD_HHMMSS.txt`

## Performance Metrics

The benchmarks measure:
- **Tokens per second** - Primary performance metric
- **Characters per second** - Secondary throughput metric  
- **Latency** - Response time
- **Token count** - Actual tokens generated
- **Success rate** - Request completion rate

## Performance Classification

- 🚀 **Excellent:** >50 tokens/second
- ✅ **Good:** 30-50 tokens/second  
- ⚠️ **Fair:** 15-30 tokens/second
- 🐌 **Slow:** 5-15 tokens/second
- ❌ **Very Slow:** <5 tokens/second

## Use Cases

- **Model performance evaluation** before production deployment
- **Hardware optimization** testing for inference servers
- **Configuration tuning** for optimal throughput
- **Baseline establishment** for performance monitoring
- **Comparative analysis** across different setups