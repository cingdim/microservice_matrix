import requests
import numpy as np
import time
import uuid
import sys
import io
import os
import psutil
import threading
import csv
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Set matplotlib backend for Docker
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- service URLs ---
SPLITTER_URL = "http://splitter:8000"
AGGREGATOR_URL = "http://aggregator:8002"

# Create results directory
os.makedirs("/app/results", exist_ok=True)

class ResourceMonitor:
    """Real-time CPU and memory monitoring with data export"""
    def __init__(self, interval=1.0, save_to_file=True):
        self.interval = interval
        self.save_to_file = save_to_file
        self.monitoring = False
        self.thread = None
        self.start_time = None
        self.stats = {
            'cpu_percent': [],
            'memory_percent': [],
            'memory_mb': [],
            'timestamps': [],
            'elapsed_sec': []
        }
        
    def start(self):
        """Start monitoring in background thread"""
        self.monitoring = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        print("📊 Resource monitoring started (press Ctrl+C to stop)")
        
    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.thread:
            self.thread.join()
        print("\n📊 Resource monitoring stopped")
        
    def _monitor(self):
        """Background monitoring loop"""
        process = psutil.Process()
        
        # Get baseline system-wide stats
        system_cpu_count = psutil.cpu_count(logical=True)
        
        while self.monitoring:
            try:
                # Get CPU percentage (per-process)
                cpu = process.cpu_percent(interval=0.1)
                
                # Get memory info
                mem_info = process.memory_info()
                mem_mb = mem_info.rss / (1024 * 1024)  # Convert to MB
                mem_percent = process.memory_percent()
                
                # Calculate elapsed time
                elapsed = time.time() - self.start_time
                
                # Record stats
                self.stats['cpu_percent'].append(cpu)
                self.stats['memory_percent'].append(mem_percent)
                self.stats['memory_mb'].append(mem_mb)
                self.stats['timestamps'].append(time.time())
                self.stats['elapsed_sec'].append(elapsed)
                
                # Print real-time stats (overwrite line)
                print(f"\r⚡ CPU: {cpu:5.1f}% | 💾 Memory: {mem_mb:7.1f} MB ({mem_percent:4.1f}%) | ⏱️  {elapsed:6.1f}s", end='', flush=True)
                
                time.sleep(self.interval)
            except Exception as e:
                print(f"\n⚠️ Monitoring error: {e}")
                break
                
    def get_summary(self):
        """Get summary statistics"""
        if not self.stats['cpu_percent']:
            return None
            
        return {
            'cpu_avg': np.mean(self.stats['cpu_percent']),
            'cpu_max': np.max(self.stats['cpu_percent']),
            'cpu_min': np.min(self.stats['cpu_percent']),
            'cpu_median': np.median(self.stats['cpu_percent']),
            'memory_avg_mb': np.mean(self.stats['memory_mb']),
            'memory_max_mb': np.max(self.stats['memory_mb']),
            'memory_min_mb': np.min(self.stats['memory_mb']),
            'memory_median_mb': np.median(self.stats['memory_mb']),
            'memory_avg_percent': np.mean(self.stats['memory_percent']),
            'memory_max_percent': np.max(self.stats['memory_percent']),
            'duration_sec': self.stats['timestamps'][-1] - self.stats['timestamps'][0] if len(self.stats['timestamps']) > 1 else 0,
            'samples_collected': len(self.stats['cpu_percent'])
        }
        
    def save_data(self, filename):
        """Save monitoring data to CSV file"""
        if not self.stats['cpu_percent']:
            print("No data to save")
            return
            
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Elapsed_Sec', 'CPU_Percent', 'Memory_MB', 'Memory_Percent', 'Timestamp'])
                
                for i in range(len(self.stats['cpu_percent'])):
                    writer.writerow([
                        f"{self.stats['elapsed_sec'][i]:.2f}",
                        f"{self.stats['cpu_percent'][i]:.2f}",
                        f"{self.stats['memory_mb'][i]:.2f}",
                        f"{self.stats['memory_percent'][i]:.2f}",
                        self.stats['timestamps'][i]
                    ])
            print(f"✅ Monitoring data saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save monitoring data: {e}")
        
    def print_summary(self):
        """Print summary statistics"""
        summary = self.get_summary()
        if not summary:
            print("No monitoring data collected")
            return
            
        print("\n\n" + "="*70)
        print("📊 RESOURCE USAGE SUMMARY")
        print("="*70)
        print(f"⚡ CPU Usage:")
        print(f"   Average:  {summary['cpu_avg']:6.1f}%")
        print(f"   Median:   {summary['cpu_median']:6.1f}%")
        print(f"   Maximum:  {summary['cpu_max']:6.1f}%")
        print(f"   Minimum:  {summary['cpu_min']:6.1f}%")
        print(f"\n💾 Memory Usage:")
        print(f"   Average:  {summary['memory_avg_mb']:7.1f} MB ({summary['memory_avg_percent']:.1f}%)")
        print(f"   Median:   {summary['memory_median_mb']:7.1f} MB")
        print(f"   Maximum:  {summary['memory_max_mb']:7.1f} MB ({summary['memory_max_percent']:.1f}%)")
        print(f"   Minimum:  {summary['memory_min_mb']:7.1f} MB")
        print(f"\n⏱️  Monitoring Stats:")
        print(f"   Duration: {summary['duration_sec']:.1f}s")
        print(f"   Samples:  {summary['samples_collected']}")
        print(f"   Rate:     {summary['samples_collected']/summary['duration_sec']:.2f} samples/sec")
        print("="*70)

def create_matrix(n, identity=False):
    dtype = np.float32
    return np.eye(n, dtype=dtype) if identity else np.arange(1, n * n + 1, dtype=dtype).reshape(n, n)

def run_pipeline(n=10, block_size=500, job_id=None, splitter_url=SPLITTER_URL):
    job_label = f"[Job-{job_id[:8]}]" if job_id is not None else ""
    print(f"\n{job_label} ⚙️ Creating matrices A({n}x{n}) and B({n}x{n})")

    # --- Create test matrices ---
    A = create_matrix(n, identity=False)
    B = create_matrix(n, identity=True)

    bufA, bufB = io.BytesIO(), io.BytesIO()
    np.save(bufA, A)
    np.save(bufB, B)
    bufA.seek(0)
    bufB.seek(0)

    print(f"{job_label} 🟢 Sending job to splitter {splitter_url}...")
    files = {
        "A_file": ("A.npy", bufA, "application/octet-stream"),
        "B_file": ("B.npy", bufB, "application/octet-stream")
    }
    
    data = {
        "block_size": str(block_size),
        "worker_url": "http://worker:8001",
        "aggregator_url": AGGREGATOR_URL,
        "job_id": job_id
    }

    try:
        resp = requests.post(f"{SPLITTER_URL}/split", files=files, data=data, timeout=1200)
        resp.raise_for_status()
    except Exception as e:
        print(f"\n{job_label} ❌ Splitter request failed: {e}")
        return None

    job_info = resp.json()
    print(f"{job_label} ✅ Splitter accepted job {job_id[:8]}, dispatched {job_info.get('blocks_dispatched', '?')} blocks")

    # --- Poll aggregator for completion ---
    result_url = f"{AGGREGATOR_URL}/aggregate/final_result/{job_id}"
    print(f"{job_label} ⏳ Waiting for aggregator result...")

    for attempt in range(900):
        time.sleep(3)
        try:
            r = requests.get(result_url, timeout=1000)
            if r.status_code == 404:
                if attempt % 20 == 0 and attempt > 0:
                    print(f"\n{job_label} ⏳ Still waiting... ({attempt * 3}s elapsed)")
                continue
            
            data = r.json()
            if data.get("message") == "Aggregation complete":
                print(f"\n{job_label} 🏁 Final result ready! Shape: {data['shape']}")
                
                if n <= 1000:
                    if isinstance(data.get("final_result"), list):
                        print(f"{job_label} 🔍 Verifying correctness...")
                        final = np.array(data["final_result"], dtype=np.float32)
                        expected = A @ B
                        if np.allclose(final, expected, rtol=1e-3, atol=1e-4):
                            print(f"{job_label} ✅ Correct final matrix.")
                        else:
                            print(f"{job_label} ❌ Incorrect matrix result!")
                    elif isinstance(data.get("final_result"), str):
                        print(f"{job_label} ℹ️ Large matrix — correctness check skipped.")
                        print(f"{job_label} Summary: {data['final_result']}")
                        if "result_summary" in data:
                            print(f"{job_label} Stats: {data['result_summary']}")
                else:
                    print(f"{job_label} ✅ Job completed (verification skipped for n > 1000)")
                    if isinstance(data.get("final_result"), str):
                        print(f"{job_label} Result: {data['final_result']}")
                        if "result_summary" in data:
                            print(f"{job_label} Stats: {data['result_summary']}")
                
                if "worker_time_total" in data:
                    print(f"{job_label} 📊 Worker time: {data['worker_time_total']:.2f}s")
                if "aggregation_time_sec" in data:
                    print(f"{job_label} 📊 Aggregation time: {data['aggregation_time_sec']:.2f}s")

                return data
        except Exception as e:
            if attempt % 20 == 0 and attempt > 0:
                print(f"\n{job_label} ⚠️ Poll error (attempt {attempt}): {e}")
            continue
    
    print(f"\n{job_label} ❌ Aggregator timeout after 45 minutes.")
    return None

if __name__ == "__main__":
    time.sleep(15)  # wait for containers to boot
    
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    block_size = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    num_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    print("="*70)
    print(f"🚀 Distributed Matrix Multiplication - Real-Time Resource Monitor")
    print(f"   Matrix size: {n}×{n}")
    print(f"   Block size: {block_size}×{block_size}")
    print(f"   Concurrent jobs: {num_jobs}")
    print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Get system info
    print(f"\n💻 System Information:")
    print(f"   CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    mem = psutil.virtual_memory()
    print(f"   Total RAM: {mem.total / (1024**3):.1f} GB")
    print(f"   Available RAM: {mem.available / (1024**3):.1f} GB ({mem.percent}% used)")

    # Start resource monitoring
    monitor = ResourceMonitor(interval=1.0, save_to_file=True)
    monitor.start()
    
    start = time.time()
    
    try:
        # Run jobs
        if num_jobs == 1:
            print("\n🔧 Running single job (optimized mode)\n")
            job_id = str(uuid.uuid4())
            result = run_pipeline(n, block_size, job_id, SPLITTER_URL)
            results = [result] if result else []
        else:
            print(f"\n🔧 Running {num_jobs} concurrent jobs\n")
            with ProcessPoolExecutor(max_workers=num_jobs) as executor:
                futures = [
                    executor.submit(run_pipeline, n, block_size, str(uuid.uuid4()), SPLITTER_URL)
                    for i in range(num_jobs)
                ]
                
                results = []
                for f in as_completed(futures):
                    result = f.result()
                    if result:
                        results.append(result)
        
        end = time.time()
        
    finally:
        # Stop monitoring
        monitor.stop()
        
        # Save data to CSV in /app/results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f'/app/results/resource_monitor_{n}x{n}_{timestamp}.csv'
        monitor.save_data(csv_filename)
        
        # Print summary
        monitor.print_summary()
        
        # Automatically generate visualization in /app/results
        print("\n📊 Generating visualization...")
        try:
            df = pd.read_csv(csv_filename)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            fig.suptitle(f'Resource Usage: {n}×{n} Matrix Multiplication', fontsize=16, fontweight='bold')
            
            # Plot 1: CPU Usage
            ax1 = axes[0]
            ax1.plot(df['Elapsed_Sec'], df['CPU_Percent'], color='#3498db', linewidth=1.5)
            ax1.fill_between(df['Elapsed_Sec'], df['CPU_Percent'], alpha=0.3, color='#3498db')
            ax1.set_xlabel('Time (seconds)', fontsize=12)
            ax1.set_ylabel('CPU Usage (%)', fontsize=12)
            ax1.set_title('CPU Usage Over Time', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            cpu_mean = df['CPU_Percent'].mean()
            cpu_max = df['CPU_Percent'].max()
            cpu_min = df['CPU_Percent'].min()
            ax1.axhline(y=cpu_mean, color='green', linestyle='--', linewidth=1, alpha=0.7)
            ax1.text(0.02, 0.98, f'Max: {cpu_max:.1f}%\nMean: {cpu_mean:.1f}%\nMin: {cpu_min:.1f}%', 
                     transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Plot 2: Memory Usage
            ax2 = axes[1]
            ax2.plot(df['Elapsed_Sec'], df['Memory_MB'], color='#e74c3c', linewidth=1.5)
            ax2.fill_between(df['Elapsed_Sec'], df['Memory_MB'], alpha=0.3, color='#e74c3c')
            ax2.set_xlabel('Time (seconds)', fontsize=12)
            ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
            ax2.set_title('Memory Usage Over Time', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            mem_mean = df['Memory_MB'].mean()
            mem_max = df['Memory_MB'].max()
            mem_min = df['Memory_MB'].min()
            ax2.axhline(y=mem_mean, color='green', linestyle='--', linewidth=1, alpha=0.7)
            ax2.text(0.02, 0.98, f'Max: {mem_max:.1f} MB\nMean: {mem_mean:.1f} MB\nMin: {mem_min:.1f} MB', 
                     transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            plt.tight_layout()
            
            # Save the plot to /app/results
            plot_filename = f'/app/results/resource_plot_{n}x{n}_{timestamp}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            
            if os.path.exists(plot_filename):
                size_kb = os.path.getsize(plot_filename) / 1024
                print(f"✅ Graph saved: {plot_filename}")
                print(f"   Size: {size_kb:.1f} KB")
            else:
                print(f"❌ Failed to save graph")
            
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Could not generate graph: {e}")
            print("   Make sure pandas and matplotlib are installed")
    
    print("\n" + "="*70)
    print(f"✅ COMPLETED: {len(results)}/{num_jobs} jobs successful")
    print(f"⏱️  Total wall time: {end - start:.2f}s")
    print(f"   End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if len(results) > 0:
        avg_worker_time = np.mean([r.get("worker_time_total", 0) for r in results if "worker_time_total" in r])
        avg_agg_time = np.mean([r.get("aggregation_time_sec", 0) for r in results if "aggregation_time_sec" in r])
        
        if avg_worker_time > 0:
            print(f"📊 Avg worker time: {avg_worker_time:.2f}s")
        if avg_agg_time > 0:
            print(f"📊 Avg aggregation time: {avg_agg_time:.2f}s")
    
    if len(results) < num_jobs:
        print(f"⚠️  Warning: {num_jobs - len(results)} jobs failed")
    
    print("\n" + "="*70)
    print("📁 All files saved to: /app/results/")
    print("   • resource_plot_*.png (Graphs)")
    print("   • resource_monitor_*.csv (Raw data)")
    print("="*70)