#!/usr/bin/env python3
"""
Maximum Tokens/Second Benchmark for GPT-OSS 20B
This script finds the peak performance (tokens/second) the model can achieve
Supports both Ollama and OpenAI API formats
"""

import requests
import time
import json
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import psutil
import sys
from abc import ABC, abstractmethod
from enum import Enum

class APIType(Enum):
    """Supported API types"""
    OLLAMA = "ollama"
    OPENAI = "openai"  # Standard OpenAI format (chat completions)

class APIAdapter(ABC):
    """Abstract base class for API adapters"""

    @abstractmethod
    def build_request(self, model, prompt, max_tokens, temperature, top_p, top_k):
        """Build the request payload for the specific API"""
        pass

    @abstractmethod
    def get_endpoint(self, base_url):
        """Get the API endpoint"""
        pass

    @abstractmethod
    def extract_response(self, response_json):
        """Extract the generated text from the response"""
        pass

class OllamaAdapter(APIAdapter):
    """Adapter for Ollama API format"""

    def build_request(self, model, prompt, max_tokens, temperature, top_p, top_k):
        options = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }

        if max_tokens:
            options["num_predict"] = max_tokens

        return {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options
        }

    def get_endpoint(self, base_url):
        return f"{base_url}/api/generate"

    def extract_response(self, response_json):
        return response_json.get('response', '')

class OpenAIAdapter(APIAdapter):
    """Adapter for standard OpenAI Chat Completions API format (/v1/chat/completions)"""

    def build_request(self, model, prompt, max_tokens, temperature, top_p, top_k):
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "temperature": temperature,
            "top_p": top_p,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        return payload

    def get_endpoint(self, base_url):
        return f"{base_url}/v1/chat/completions"

    def extract_response(self, response_json):
        """
        Extract response from standard OpenAI format.
        Supports:
        - choices[0].message.content (standard)
        - choices[0].message.reasoning_content (for reasoning models)
        """
        choices = response_json.get('choices', [])
        if choices:
            message = choices[0].get('message', {})

            # Try content first (standard field)
            content = message.get('content', '')
            if content:
                return content

            # Fallback to reasoning_content (for models with reasoning)
            reasoning_content = message.get('reasoning_content', '')
            if reasoning_content:
                return reasoning_content

        return ''

class MaxTokensBenchmark:
    def __init__(self, host="localhost", port=11434, model="gpt-oss:20b",
                 api_type=APIType.OLLAMA, verbose=False, max_concurrency=32):
        """
        Initialize benchmark with configurable API type

        Args:
            host: API host
            port: API port
            model: Model name/identifier
            api_type: APIType enum (OLLAMA or OPENAI)
            verbose: Enable verbose debug output
            max_concurrency: Maximum concurrency level to test
        """
        self.base_url = f"http://{host}:{port}"
        self.model = model
        self.max_tokens_per_second = 0
        self.best_config = {}
        self.api_type = api_type
        self.verbose = verbose
        self.max_concurrency = max_concurrency

        # Track throughput metrics
        self.max_throughput = 0
        self.best_throughput_config = {}

        # Select appropriate adapter
        if api_type == APIType.OLLAMA:
            self.adapter = OllamaAdapter()
        elif api_type == APIType.OPENAI:
            self.adapter = OpenAIAdapter()
        else:
            raise ValueError(f"Unsupported API type: {api_type}")

        print(f"Using {api_type.value} API format")
        print(f"Endpoint: {self.adapter.get_endpoint(self.base_url)}")

    def generate_long_content_request(self, prompt, max_tokens=None, temperature=0.7):
        """Make a request optimized for maximum token generation"""
        url = self.adapter.get_endpoint(self.base_url)

        payload = self.adapter.build_request(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=40
        )

        if self.verbose:
            print(f"\n📤 Request to: {url}")
            print(f"📤 Payload: {json.dumps(payload, indent=2)[:500]}")

        start_time = time.time()

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()

            end_time = time.time()

            # Debug: Print raw response for troubleshooting
            raw_text = response.text

            if self.verbose:
                print(f"📥 Response status: {response.status_code}")
                print(f"📥 Response (first 500 chars): {raw_text[:500]}")

            # Try to parse JSON
            try:
                result = response.json()
            except json.JSONDecodeError as json_err:
                return {
                    'success': False,
                    'error': f'JSON decode error: {json_err}. Raw response (first 500 chars): {raw_text[:500]}',
                    'latency': time.time() - start_time
                }

            # Calculate metrics
            latency = end_time - start_time
            response_text = self.adapter.extract_response(result)

            if not response_text:
                return {
                    'success': False,
                    'error': f'Empty response from API. Full response: {json.dumps(result, indent=2)[:500]}',
                    'latency': latency
                }

            tokens_generated = len(response_text.split())
            char_count = len(response_text)
            tokens_per_second = tokens_generated / latency if latency > 0 else 0

            return {
                'success': True,
                'latency': latency,
                'tokens_generated': tokens_generated,
                'tokens_per_second': tokens_per_second,
                'char_count': char_count,
                'chars_per_second': char_count / latency if latency > 0 else 0,
                'response_text': response_text,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'prompt_length': len(prompt.split())
            }

        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'Request error: {str(e)}',
                'latency': time.time() - start_time
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}',
                'latency': time.time() - start_time
            }
    
    def test_optimal_prompt_length(self):
        """Find the optimal prompt length for maximum token generation"""
        print("🎯 Testing optimal prompt length for maximum tokens/second...")
        
        base_prompt = "Write a detailed, comprehensive explanation about"
        topics = [
            " artificial intelligence and machine learning",
            " quantum computing and its applications in modern technology", 
            " the history and evolution of computer programming languages",
            " climate change and renewable energy solutions",
            " space exploration and the future of human civilization"
        ]
        
        results = []
        
        for i, topic in enumerate(topics):
            prompt = base_prompt + topic + ". Please provide extensive details, examples, and analysis."
            
            print(f"  Testing prompt {i+1}/5: {topic[:50]}...")
            
            # Test with high token count to maximize generation
            result = self.generate_long_content_request(prompt, max_tokens=500, temperature=0.7)
            
            if result['success']:
                results.append(result)
                print(f"    Tokens/s: {result['tokens_per_second']:.2f}, Total tokens: {result['tokens_generated']}")
                
                # Track maximum
                if result['tokens_per_second'] > self.max_tokens_per_second:
                    self.max_tokens_per_second = result['tokens_per_second']
                    self.best_config = {
                        'prompt': prompt,
                        'max_tokens': 500,
                        'temperature': 0.7,
                        'result': result
                    }
            else:
                print(f"    FAILED: {result['error']}")
        
        return results
    
    def test_different_max_tokens(self):
        """Test different max_tokens settings to find optimal generation length"""
        print("📏 Testing different max_tokens settings...")
        
        optimal_prompt = "Write a comprehensive technical explanation about neural networks, including their architecture, training process, applications, and future developments. Provide detailed examples and analysis."
        
        token_limits = [100, 200, 500, 1000, 1500]
        results = []
        
        for max_tokens in token_limits:
            print(f"  Testing max_tokens: {max_tokens}")
            
            # Run multiple times for consistency
            runs = []
            for run in range(3):
                result = self.generate_long_content_request(optimal_prompt, max_tokens=max_tokens)
                if result['success']:
                    runs.append(result)
                    print(f"    Run {run+1}: {result['tokens_per_second']:.2f} tokens/s")
            
            if runs:
                avg_tokens_per_second = statistics.mean([r['tokens_per_second'] for r in runs])
                avg_tokens_generated = statistics.mean([r['tokens_generated'] for r in runs])
                
                result_summary = {
                    'max_tokens': max_tokens,
                    'avg_tokens_per_second': avg_tokens_per_second,
                    'avg_tokens_generated': avg_tokens_generated,
                    'runs': len(runs)
                }
                results.append(result_summary)
                
                print(f"    Average: {avg_tokens_per_second:.2f} tokens/s ({avg_tokens_generated:.0f} tokens)")
                
                # Track maximum
                if avg_tokens_per_second > self.max_tokens_per_second:
                    self.max_tokens_per_second = avg_tokens_per_second
                    self.best_config = {
                        'max_tokens': max_tokens,
                        'avg_tokens_per_second': avg_tokens_per_second,
                        'type': 'max_tokens_test'
                    }
        
        return results
    
    def test_temperature_settings(self):
        """Test different temperature settings for optimal performance"""
        print("🌡️ Testing temperature settings...")
        
        optimal_prompt = "Provide a detailed analysis of machine learning algorithms including supervised learning, unsupervised learning, and reinforcement learning. Include mathematical foundations, practical applications, and implementation considerations."
        
        temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]
        results = []
        
        for temp in temperatures:
            print(f"  Testing temperature: {temp}")
            
            result = self.generate_long_content_request(optimal_prompt, max_tokens=400, temperature=temp)
            
            if result['success']:
                results.append(result)
                print(f"    Tokens/s: {result['tokens_per_second']:.2f}, Tokens: {result['tokens_generated']}")
                
                # Track maximum
                if result['tokens_per_second'] > self.max_tokens_per_second:
                    self.max_tokens_per_second = result['tokens_per_second']
                    self.best_config = {
                        'temperature': temp,
                        'tokens_per_second': result['tokens_per_second'],
                        'type': 'temperature_test',
                        'result': result
                    }
            else:
                print(f"    FAILED: {result['error']}")
        
        return results
    
    def stress_test_maximum_output(self):
        """Stress test with very long generation to find absolute maximum"""
        print("🚀 Stress testing for absolute maximum tokens/second...")
        
        # Use prompts designed to generate very long responses
        long_prompts = [
            "Write a complete technical tutorial on building a web application from scratch. Include setup, frontend development, backend development, database design, testing, deployment, and maintenance. Provide code examples, best practices, and detailed explanations for each step.",
            
            "Create a comprehensive business plan for a technology startup. Include executive summary, market analysis, competitive analysis, product development strategy, marketing plan, financial projections, risk assessment, and implementation timeline. Provide detailed analysis and recommendations.",
            
            "Explain the complete process of software development lifecycle including requirements gathering, system design, coding, testing, deployment, and maintenance. Provide examples, methodologies, tools, and best practices for each phase."
        ]
        
        results = []
        
        for i, prompt in enumerate(long_prompts):
            print(f"  Stress test {i+1}/3...")
            
            # Test with maximum tokens
            result = self.generate_long_content_request(prompt, max_tokens=2000, temperature=0.6)
            
            if result['success']:
                results.append(result)
                print(f"    Duration: {result['latency']:.1f}s")
                print(f"    Tokens generated: {result['tokens_generated']}")
                print(f"    Tokens/second: {result['tokens_per_second']:.2f}")
                print(f"    Characters/second: {result['chars_per_second']:.0f}")
                
                # Track maximum
                if result['tokens_per_second'] > self.max_tokens_per_second:
                    self.max_tokens_per_second = result['tokens_per_second']
                    self.best_config = {
                        'type': 'stress_test',
                        'prompt': prompt[:100] + "...",
                        'result': result
                    }
            else:
                print(f"    FAILED: {result['error']}")
        
        return results
    
    def concurrent_generation_test(self):
        """Test concurrent requests to measure system throughput"""
        print("=" * 70)
        print("⚡ CONCURRENT THROUGHPUT BENCHMARK")
        print("=" * 70)
        print(f"Testing concurrency levels from 1 to {self.max_concurrency}")
        print("Goal: Find optimal concurrency for maximum system throughput")
        print()

        prompt = "Write a detailed explanation about cloud computing, including IaaS, PaaS, SaaS, benefits, challenges, and future trends."

        def single_request():
            return self.generate_long_content_request(prompt, max_tokens=300, temperature=0.7)

        # Generate concurrency levels: 1, 2, 4, 8, 16, 32, ... up to max_concurrency
        concurrency_levels = []
        level = 1
        while level <= self.max_concurrency:
            concurrency_levels.append(level)
            if level == 1:
                level = 2
            elif level == 2:
                level = 4
            else:
                level *= 2

        # If max_concurrency is not a power of 2, add it
        if concurrency_levels[-1] != self.max_concurrency:
            concurrency_levels.append(self.max_concurrency)

        results_summary = []

        for concurrency in concurrency_levels:
            print(f"📊 Testing {concurrency} concurrent request(s)...")

            start_time = time.time()

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(single_request) for _ in range(concurrency)]
                results = [future.result() for future in as_completed(futures)]

            total_time = time.time() - start_time
            successful = [r for r in results if r['success']]
            failed_count = len(results) - len(successful)

            if successful:
                total_tokens = sum(r['tokens_generated'] for r in successful)
                overall_throughput = total_tokens / total_time  # System throughput
                avg_individual_tokens_per_second = statistics.mean([r['tokens_per_second'] for r in successful])
                avg_latency = statistics.mean([r['latency'] for r in successful])
                min_latency = min([r['latency'] for r in successful])
                max_latency = max([r['latency'] for r in successful])

                result_data = {
                    'concurrency': concurrency,
                    'successful': len(successful),
                    'failed': failed_count,
                    'total_tokens': total_tokens,
                    'total_time': total_time,
                    'throughput': overall_throughput,
                    'avg_tokens_per_sec': avg_individual_tokens_per_second,
                    'avg_latency': avg_latency,
                    'min_latency': min_latency,
                    'max_latency': max_latency
                }
                results_summary.append(result_data)

                print(f"  ✅ Success: {len(successful)}/{concurrency} requests")
                if failed_count > 0:
                    print(f"  ❌ Failed: {failed_count}")
                print(f"  📈 System Throughput: {overall_throughput:.2f} tokens/s")
                print(f"  ⚡ Avg Request Speed: {avg_individual_tokens_per_second:.2f} tokens/s")
                print(f"  ⏱️  Latency: avg={avg_latency:.2f}s, min={min_latency:.2f}s, max={max_latency:.2f}s")
                print()

                # Track maximum throughput
                if overall_throughput > self.max_throughput:
                    self.max_throughput = overall_throughput
                    self.best_throughput_config = result_data

                # Also track if individual speed is best
                if avg_individual_tokens_per_second > self.max_tokens_per_second:
                    self.max_tokens_per_second = avg_individual_tokens_per_second
                    self.best_config = {
                        'type': 'concurrent_test',
                        'concurrency': concurrency,
                        'individual_tokens_per_second': avg_individual_tokens_per_second,
                        'overall_tokens_per_second': overall_throughput
                    }
            else:
                print(f"  ❌ All {concurrency} requests failed")
                print()

        # Print summary
        if results_summary:
            print("=" * 70)
            print("📊 THROUGHPUT SUMMARY")
            print("=" * 70)
            print(f"{'Concurrency':<12} {'Throughput':<15} {'Avg Latency':<15} {'Success Rate':<15}")
            print("-" * 70)
            for result in results_summary:
                success_rate = f"{result['successful']}/{result['concurrency']}"
                print(f"{result['concurrency']:<12} {result['throughput']:<15.2f} {result['avg_latency']:<15.2f} {success_rate:<15}")

            print()
            print("🏆 BEST THROUGHPUT:")
            best = self.best_throughput_config
            print(f"  Concurrency: {best['concurrency']}")
            print(f"  System Throughput: {best['throughput']:.2f} tokens/s")
            print(f"  Avg Request Latency: {best['avg_latency']:.2f}s")
            print(f"  Success Rate: {best['successful']}/{best['concurrency']}")
            print()

        return results_summary
    
    def run_maximum_performance_benchmark(self):
        """Run comprehensive benchmark to find maximum tokens/second"""
        print("=" * 70)
        print(f"🎯 MAXIMUM TOKENS/SECOND BENCHMARK FOR {self.model}")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Verify model is available
        try:
            test_result = self.generate_long_content_request("Test", max_tokens=10)
            if not test_result['success']:
                print(f"❌ Model {self.model} not available: {test_result.get('error')}")
                return None
            print(f"✅ Model {self.model} is available and responding")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return None
        
        print()
        
        # Run all tests
        self.test_optimal_prompt_length()
        print()
        
        self.test_different_max_tokens()
        print()
        
        self.test_temperature_settings()
        print()
        
        self.stress_test_maximum_output()
        print()
        
        self.concurrent_generation_test()
        print()
        
        # Final results
        print("=" * 70)
        print("🏆 MAXIMUM PERFORMANCE RESULTS")
        print("=" * 70)

        print("\n📊 Single Request Performance:")
        print(f"  Maximum tokens/second: {self.max_tokens_per_second:.2f}")
        print(f"  Best configuration: {self.best_config.get('type', 'N/A')}")

        if self.best_throughput_config:
            print("\n🚀 System Throughput Performance:")
            best = self.best_throughput_config
            print(f"  Maximum throughput: {best['throughput']:.2f} tokens/s")
            print(f"  Optimal concurrency: {best['concurrency']}")
            print(f"  Avg request latency: {best['avg_latency']:.2f}s")
            print(f"  Success rate: {best['successful']}/{best['concurrency']}")

        if 'result' in self.best_config:
            result = self.best_config['result']
            print(f"\n📝 Best Single Request Details:")
            print(f"  Latency: {result['latency']:.2f}s")
            print(f"  Tokens generated: {result['tokens_generated']}")
            print(f"  Characters generated: {result['char_count']}")
            print(f"  Characters/second: {result['chars_per_second']:.0f}")

        print(f"\nBenchmark completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return {
            'max_tokens_per_second': self.max_tokens_per_second,
            'max_throughput': self.max_throughput,
            'best_config': self.best_config,
            'best_throughput_config': self.best_throughput_config,
            'model': self.model
        }

def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Maximum Tokens/Second Benchmark - Supports Ollama and OpenAI API formats'
    )
    parser.add_argument(
        '--api-type',
        type=str,
        choices=['ollama', 'openai'],
        default='ollama',
        help='API format to use: "ollama" for Ollama API, "openai" for standard OpenAI /v1/chat/completions (default: ollama)'
    )
    parser.add_argument('--host', type=str, default='localhost', help='API host')
    parser.add_argument('--port', type=int, default=11434, help='API port (11434 for Ollama, 8080 for custom servers)')
    parser.add_argument('--model', type=str, default='gpt-oss:20b', help='Model name')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose debug output')
    parser.add_argument('--max-concurrency', type=int, default=32, help='Maximum concurrency level to test (default: 32)')

    args = parser.parse_args()

    # Convert string to APIType enum
    api_type_map = {
        'ollama': APIType.OLLAMA,
        'openai': APIType.OPENAI
    }
    api_type = api_type_map[args.api_type]

    print(f"Maximum Tokens/Second Benchmark")
    print(f"API Type: {args.api_type}")
    print(f"Model: {args.model}")
    print(f"Endpoint: {args.host}:{args.port}")
    print(f"Max Concurrency: {args.max_concurrency}")
    print()

    # Initialize benchmark
    benchmark = MaxTokensBenchmark(
        args.host,
        args.port,
        args.model,
        api_type,
        verbose=args.verbose,
        max_concurrency=args.max_concurrency
    )

    # Run benchmark
    results = benchmark.run_maximum_performance_benchmark()

    if results:
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"max_tokens_benchmark_{args.api_type}_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {filename}")

        # Quick summary
        print(f"\n📋 QUICK SUMMARY:")
        print(f"🎯 Maximum tokens/second: {results['max_tokens_per_second']:.2f}")
        print(f"🏆 This is the peak performance for {args.model}")

if __name__ == "__main__":
    main()
