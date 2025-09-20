#!/usr/bin/env python3
"""
Maximum Tokens/Second Benchmark for GPT-OSS 20B
This script finds the peak performance (tokens/second) the model can achieve
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

class MaxTokensBenchmark:
    def __init__(self, host="localhost", port=11434, model="gpt-oss:20b"):
        self.base_url = f"http://{host}:{port}"
        self.model = model
        self.max_tokens_per_second = 0
        self.best_config = {}
        
    def generate_long_content_request(self, prompt, max_tokens=None, temperature=0.7):
        """Make a request optimized for maximum token generation"""
        url = f"{self.base_url}/api/generate"
        
        options = {
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
        }
        
        if max_tokens:
            options["num_predict"] = max_tokens
            
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            
            end_time = time.time()
            result = response.json()
            
            # Calculate metrics
            latency = end_time - start_time
            response_text = result.get('response', '')
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
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
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
        """Test if concurrent requests can increase overall throughput"""
        print("⚡ Testing concurrent generation (may reduce individual speed but increase total throughput)...")
        
        prompt = "Write a detailed explanation about cloud computing, including IaaS, PaaS, SaaS, benefits, challenges, and future trends."
        
        def single_request():
            return self.generate_long_content_request(prompt, max_tokens=300, temperature=0.7)
        
        # Test different concurrency levels
        concurrency_levels = [1, 2, 3]
        
        for concurrency in concurrency_levels:
            print(f"  Testing {concurrency} concurrent requests...")
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(single_request) for _ in range(concurrency)]
                results = [future.result() for future in as_completed(futures)]
            
            total_time = time.time() - start_time
            successful = [r for r in results if r['success']]
            
            if successful:
                total_tokens = sum(r['tokens_generated'] for r in successful)
                overall_tokens_per_second = total_tokens / total_time
                avg_individual_tokens_per_second = statistics.mean([r['tokens_per_second'] for r in successful])
                
                print(f"    Total tokens: {total_tokens}")
                print(f"    Total time: {total_time:.2f}s") 
                print(f"    Overall tokens/s: {overall_tokens_per_second:.2f}")
                print(f"    Avg individual tokens/s: {avg_individual_tokens_per_second:.2f}")
                
                # For concurrent tests, we care about individual speed
                if avg_individual_tokens_per_second > self.max_tokens_per_second:
                    self.max_tokens_per_second = avg_individual_tokens_per_second
                    self.best_config = {
                        'type': 'concurrent_test',
                        'concurrency': concurrency,
                        'individual_tokens_per_second': avg_individual_tokens_per_second,
                        'overall_tokens_per_second': overall_tokens_per_second
                    }
    
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
        print(f"Maximum tokens/second achieved: {self.max_tokens_per_second:.2f}")
        print()
        print("Best configuration:")
        for key, value in self.best_config.items():
            if key != 'result':
                print(f"  {key}: {value}")
        
        if 'result' in self.best_config:
            result = self.best_config['result']
            print(f"\nBest run details:")
            print(f"  Latency: {result['latency']:.2f}s")
            print(f"  Tokens generated: {result['tokens_generated']}")
            print(f"  Characters generated: {result['char_count']}")
            print(f"  Characters/second: {result['chars_per_second']:.0f}")
        
        print(f"\nBenchmark completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return {
            'max_tokens_per_second': self.max_tokens_per_second,
            'best_config': self.best_config,
            'model': self.model
        }

def main():
    # Configuration
    HOST = "localhost"
    PORT = 11434
    MODEL = "gpt-oss:20b"
    
    print(f"Maximum Tokens/Second Benchmark")
    print(f"Model: {MODEL}")
    print(f"Endpoint: {HOST}:{PORT}")
    print()
    
    # Initialize benchmark
    benchmark = MaxTokensBenchmark(HOST, PORT, MODEL)
    
    # Run benchmark
    results = benchmark.run_maximum_performance_benchmark()
    
    if results:
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"max_tokens_benchmark_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filename}")
        
        # Quick summary
        print(f"\n📋 QUICK SUMMARY:")
        print(f"🎯 Maximum tokens/second: {results['max_tokens_per_second']:.2f}")
        print(f"🏆 This is the peak performance for {MODEL}")

if __name__ == "__main__":
    main()
