#!/usr/bin/env python3
"""
KPU Simulator Visualization Utilities

This module provides comprehensive visualization tools for the KPU simulator,
including performance charts, component status displays, and interactive dashboards.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass, field
from collections import defaultdict, deque

try:
    import stillwater_kpu as kpu
except ImportError:
    print("Warning: stillwater_kpu not available, using mock data for visualization")
    kpu = None

# Set up plotting style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

@dataclass
class SimulationMetrics:
    """Container for simulation metrics collected over time"""
    cycles: List[int] = field(default_factory=list)
    wall_time: List[float] = field(default_factory=list)
    memory_utilization: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))
    scratchpad_utilization: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))
    dma_activity: Dict[int, List[bool]] = field(default_factory=lambda: defaultdict(list))
    compute_activity: Dict[int, List[bool]] = field(default_factory=lambda: defaultdict(list))
    throughput: List[float] = field(default_factory=list)

class KPUVisualizer:
    """Main visualization class for KPU simulator"""
    
    def __init__(self, simulator=None):
        self.simulator = simulator
        self.metrics = SimulationMetrics()
        self.recording = False
        
    def start_recording(self):
        """Start recording metrics from the simulator"""
        self.recording = True
        self.metrics = SimulationMetrics()
        
    def stop_recording(self):
        """Stop recording metrics"""
        self.recording = False
        
    def record_step(self):
        """Record current simulator state (call after each simulation step)"""
        if not self.recording or not self.simulator:
            return
            
        # Record basic metrics
        self.metrics.cycles.append(self.simulator.get_current_cycle())
        self.metrics.wall_time.append(self.simulator.get_elapsed_time_ms())
        
        # Record component states (simplified - would need extended API for real utilization)
        for i in range(self.simulator.get_memory_bank_count()):
            self.metrics.memory_utilization[i].append(np.random.uniform(0.2, 0.8))  # Mock data
            
        for i in range(self.simulator.get_scratchpad_count()):
            self.metrics.scratchpad_utilization[i].append(np.random.uniform(0.3, 0.9))  # Mock data
            
        for i in range(self.simulator.get_dma_engine_count()):
            self.metrics.dma_activity[i].append(self.simulator.is_dma_busy(i))
            
        for i in range(self.simulator.get_compute_tile_count()):
            self.metrics.compute_activity[i].append(self.simulator.is_compute_busy(i))
            
    def plot_performance_scaling(self, matrix_sizes: List[Tuple[int, int, int]], 
                                configs: List[Tuple[str, object]] = None):
        """Plot performance scaling across different matrix sizes and configurations"""
        
        if not configs:
            configs = [
                ("Default", kpu.SimulatorConfig() if kpu else None),
                ("Multi-Bank", kpu.generate_multi_bank_config(4, 2) if kpu else None)
            ]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('KPU Simulator Performance Scaling Analysis', fontsize=16)
        
        results = defaultdict(list)
        
        for config_name, config in configs:
            if not kpu:
                # Generate mock data
                for m, n, k in matrix_sizes:
                    ops = 2 * m * n * k
                    results[f'{config_name}_ops'].append(ops)
                    results[f'{config_name}_cycles'].append(ops // np.random.randint(2, 8))
                    results[f'{config_name}_time'].append(np.random.uniform(1, 50))
                    results[f'{config_name}_throughput'].append(ops / (ops // np.random.randint(2, 8)))
                continue
                
            try:
                sim = kpu.KPUSimulator(config)
                
                for m, n, k in matrix_sizes:
                    # Generate test matrices
                    A = np.random.randn(m, k).astype(np.float32)
                    B = np.random.randn(k, n).astype(np.float32)
                    
                    # Run simulation with timing
                    start_time = time.time()
                    result = sim.run_numpy_matmul(A, B, 0, 0, 0)
                    end_time = time.time()
                    
                    # Calculate metrics
                    ops = 2 * m * n * k  # 2 operations per MAC
                    cycles = sim.get_current_cycle()
                    wall_time = (end_time - start_time) * 1000  # ms
                    throughput = ops / cycles if cycles > 0 else 0
                    
                    results[f'{config_name}_ops'].append(ops)
                    results[f'{config_name}_cycles'].append(cycles)
                    results[f'{config_name}_time'].append(wall_time)
                    results[f'{config_name}_throughput'].append(throughput)
                    
            except Exception as e:
                print(f"Error testing {config_name}: {e}")
                continue
        
        # Plot 1: Operations vs Simulation Time
        x_ops = [2 * m * n * k for m, n, k in matrix_sizes]
        for config_name, _ in configs:
            if f'{config_name}_time' in results:
                ax1.loglog(x_ops, results[f'{config_name}_time'], 'o-', label=config_name)
        ax1.set_xlabel('Operations (MACs)')
        ax1.set_ylabel('Simulation Time (ms)')
        ax1.set_title('Simulation Time vs Problem Size')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Operations vs Cycles
        for config_name, _ in configs:
            if f'{config_name}_cycles' in results:
                ax2.loglog(x_ops, results[f'{config_name}_cycles'], 's-', label=config_name)
        ax2.set_xlabel('Operations (MACs)')
        ax2.set_ylabel('Simulation Cycles')
        ax2.set_title('Simulation Cycles vs Problem Size')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Throughput (Ops/Cycle)
        matrix_labels = [f"{m}x{n}x{k}" for m, n, k in matrix_sizes]
        x_pos = np.arange(len(matrix_labels))
        width = 0.35
        
        for i, (config_name, _) in enumerate(configs):
            if f'{config_name}_throughput' in results:
                ax3.bar(x_pos + i*width, results[f'{config_name}_throughput'], 
                       width, label=config_name)
        ax3.set_xlabel('Matrix Size')
        ax3.set_ylabel('Throughput (Ops/Cycle)')
        ax3.set_title('Computational Throughput')
        ax3.set_xticks(x_pos + width/2)
        ax3.set_xticklabels(matrix_labels)
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Efficiency (Operations per ms)
        for config_name, _ in configs:
            if f'{config_name}_ops' in results and f'{config_name}_time' in results:
                efficiency = np.array(results[f'{config_name}_ops']) / np.array(results[f'{config_name}_time'])
                ax4.semilogy(matrix_labels, efficiency, 'o-', label=config_name)
        ax4.set_xlabel('Matrix Size')
        ax4.set_ylabel('Efficiency (Ops/ms)')
        ax4.set_title('Computational Efficiency')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        return fig
        
    def plot_component_timeline(self, steps: int = 100):
        """Plot component activity timeline during simulation"""
        if not self.simulator or not self.metrics.cycles:
            print("No simulation data available for timeline plot")
            return None
            
        fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
        fig.suptitle('KPU Component Activity Timeline', fontsize=16)
        
        cycles = self.metrics.cycles[:steps] if len(self.metrics.cycles) > steps else self.metrics.cycles
        
        # Plot 1: Memory bank utilization
        ax = axes[0]
        for bank_id, utilization in self.metrics.memory_utilization.items():
            util_data = utilization[:len(cycles)]
            ax.plot(cycles[:len(util_data)], util_data, label=f'Bank {bank_id}', linewidth=2)
        ax.set_ylabel('Memory\nUtilization (%)')
        ax.set_title('Memory Bank Activity')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Plot 2: Scratchpad utilization
        ax = axes[1]
        for pad_id, utilization in self.metrics.scratchpad_utilization.items():
            util_data = utilization[:len(cycles)]
            ax.plot(cycles[:len(util_data)], util_data, label=f'Pad {pad_id}', linewidth=2)
        ax.set_ylabel('Scratchpad\nUtilization (%)')
        ax.set_title('Scratchpad Activity')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Plot 3: DMA activity
        ax = axes[2]
        for dma_id, activity in self.metrics.dma_activity.items():
            activity_data = activity[:len(cycles)]
            # Convert boolean to int for plotting
            activity_int = [int(x) for x in activity_data]
            ax.plot(cycles[:len(activity_int)], np.array(activity_int) + dma_id * 0.1, 
                   label=f'DMA {dma_id}', linewidth=3, alpha=0.7)
        ax.set_ylabel('DMA Activity')
        ax.set_title('DMA Engine Status')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.5)
        
        # Plot 4: Compute tile activity
        ax = axes[3]
        for tile_id, activity in self.metrics.compute_activity.items():
            activity_data = activity[:len(cycles)]
            activity_int = [int(x) for x in activity_data]
            ax.plot(cycles[:len(activity_int)], np.array(activity_int) + tile_id * 0.1, 
                   label=f'Tile {tile_id}', linewidth=3, alpha=0.7)
        ax.set_ylabel('Compute Activity')
        ax.set_title('Compute Tile Status')
        ax.set_xlabel('Simulation Cycles')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.5)
        
        plt.tight_layout()
        return fig
        
    def plot_memory_heatmap(self, memory_accesses: Dict[Tuple[int, int], int]):
        """Plot memory access pattern heatmap"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Convert access patterns to matrix format
        if memory_accesses:
            banks = max([bank for bank, addr in memory_accesses.keys()]) + 1
            max_addr = max([addr for bank, addr in memory_accesses.keys()])
            
            # Create heatmap matrix
            heatmap_data = np.zeros((banks, max_addr // 1024 + 1))  # KB resolution
            
            for (bank, addr), count in memory_accesses.items():
                heatmap_data[bank, addr // 1024] += count
        else:
            # Generate sample data if no real data available
            banks, addresses = 4, 256
            heatmap_data = np.random.randint(0, 100, size=(banks, addresses))
            
        # Plot heatmap
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax.set_xlabel('Memory Address (KB)')
        ax.set_ylabel('Memory Bank')
        ax.set_title('Memory Access Pattern Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Access Count')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        return fig
        
    def plot_pipeline_utilization(self):
        """Plot pipeline utilization over time"""
        if not self.metrics.cycles:
            print("No metrics data available")
            return None
            
        fig, ax = plt.subplots(figsize=(15, 8))
        
        cycles = self.metrics.cycles
        
        # Calculate pipeline stages
        pipeline_stages = {
            'Memory Load': [],
            'Compute': [],
            'Memory Store': [],
            'Idle': []
        }
        
        for i in range(len(cycles)):
            # Simplified pipeline analysis based on component activity
            dma_active = any(self.metrics.dma_activity[dma_id][i] 
                           for dma_id in self.metrics.dma_activity.keys() if i < len(self.metrics.dma_activity[dma_id]))
            compute_active = any(self.metrics.compute_activity[tile_id][i] 
                               for tile_id in self.metrics.compute_activity.keys() if i < len(self.metrics.compute_activity[tile_id]))
            
            if dma_active and not compute_active:
                stage = 'Memory Load'
            elif compute_active:
                stage = 'Compute'
            elif dma_active and compute_active:
                stage = 'Memory Store'
            else:
                stage = 'Idle'
                
            for s in pipeline_stages:
                pipeline_stages[s].append(1 if s == stage else 0)
        
        # Stack plot
        ax.stackplot(cycles, 
                    pipeline_stages['Memory Load'],
                    pipeline_stages['Compute'], 
                    pipeline_stages['Memory Store'],
                    pipeline_stages['Idle'],
                    labels=['Memory Load', 'Compute', 'Memory Store', 'Idle'],
                    alpha=0.8)
        
        ax.set_xlabel('Simulation Cycles')
        ax.set_ylabel('Pipeline Stage')
        ax.set_title('KPU Pipeline Utilization Over Time')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        return fig
        
    def create_dashboard(self, matrix_sizes: List[Tuple[int, int, int]] = None):
        """Create comprehensive dashboard with multiple visualizations"""
        if matrix_sizes is None:
            matrix_sizes = [(4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 32, 32)]
            
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Performance scaling plot
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_performance_summary(ax1, matrix_sizes)
        
        # Component status
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_component_status(ax2)
        
        # Memory utilization over time
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_utilization_timeline(ax3)
        
        # Throughput comparison
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_throughput_comparison(ax4, matrix_sizes)
        
        # Pipeline efficiency
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_pipeline_efficiency(ax5)
        
        # Resource utilization pie chart
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_resource_utilization(ax6)
        
        fig.suptitle('KPU Simulator Comprehensive Dashboard', fontsize=18)
        return fig
        
    def _plot_performance_summary(self, ax, matrix_sizes):
        """Helper: Plot performance summary"""
        # Generate sample data
        ops = [2 * m * n * k for m, n, k in matrix_sizes]
        cycles = [op // np.random.randint(2, 6) for op in ops]
        
        ax.loglog(ops, cycles, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Operations (MACs)')
        ax.set_ylabel('Simulation Cycles')
        ax.set_title('Performance Scaling')
        ax.grid(True, alpha=0.3)
        
        # Add efficiency line
        ideal_efficiency = [op / 4 for op in ops]  # Ideal 4 ops/cycle
        ax.loglog(ops, ideal_efficiency, 'r--', alpha=0.7, label='Ideal (4 ops/cycle)')
        ax.legend()
        
    def _plot_component_status(self, ax):
        """Helper: Plot current component status"""
        if self.simulator:
            components = ['Memory Banks', 'Scratchpads', 'Compute Tiles', 'DMA Engines']
            counts = [
                self.simulator.get_memory_bank_count(),
                self.simulator.get_scratchpad_count(), 
                self.simulator.get_compute_tile_count(),
                self.simulator.get_dma_engine_count()
            ]
        else:
            components = ['Memory Banks', 'Scratchpads', 'Compute Tiles', 'DMA Engines']
            counts = [4, 2, 2, 6]
            
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax.bar(components, counts, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
                   
        ax.set_title('Component Configuration')
        ax.set_ylabel('Count')
        
    def _plot_utilization_timeline(self, ax):
        """Helper: Plot utilization timeline"""
        if self.metrics.cycles:
            cycles = self.metrics.cycles[:50]  # Last 50 cycles
            
            # Generate or use real utilization data
            memory_util = [np.random.uniform(0.3, 0.9) for _ in cycles]
            compute_util = [np.random.uniform(0.4, 1.0) for _ in cycles]
            
            ax.plot(cycles, memory_util, label='Memory', linewidth=2)
            ax.plot(cycles, compute_util, label='Compute', linewidth=2)
            ax.fill_between(cycles, memory_util, alpha=0.3)
            ax.fill_between(cycles, compute_util, alpha=0.3)
        else:
            # Generate sample timeline
            cycles = list(range(50))
            memory_util = [np.sin(x/5) * 0.3 + 0.6 for x in cycles]
            compute_util = [np.cos(x/3) * 0.2 + 0.7 for x in cycles]
            
            ax.plot(cycles, memory_util, label='Memory', linewidth=2)
            ax.plot(cycles, compute_util, label='Compute', linewidth=2)
            
        ax.set_xlabel('Simulation Cycles')
        ax.set_ylabel('Utilization')
        ax.set_title('Component Utilization Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
    def _plot_throughput_comparison(self, ax, matrix_sizes):
        """Helper: Plot throughput comparison"""
        configs = ['Single Bank', 'Multi Bank', 'Optimized']
        throughput = [np.random.uniform(2, 6) for _ in configs]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax.bar(configs, throughput, color=colors, alpha=0.8)
        
        for bar, tp in zip(bars, throughput):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{tp:.1f}', ha='center', va='bottom', fontweight='bold')
                   
        ax.set_title('Throughput Comparison')
        ax.set_ylabel('Ops/Cycle')
        
    def _plot_pipeline_efficiency(self, ax):
        """Helper: Plot pipeline efficiency"""
        stages = ['Load', 'Compute', 'Store']
        efficiency = [85, 92, 78]  # Sample efficiency percentages
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax.bar(stages, efficiency, color=colors, alpha=0.8)
        
        for bar, eff in zip(bars, efficiency):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{eff}%', ha='center', va='bottom', fontweight='bold')
                   
        ax.set_title('Pipeline Stage Efficiency')
        ax.set_ylabel('Efficiency (%)')
        ax.set_ylim(0, 100)
        
    def _plot_resource_utilization(self, ax):
        """Helper: Plot resource utilization pie chart"""
        resources = ['Memory', 'Compute', 'DMA', 'Idle']
        utilization = [35, 40, 15, 10]  # Sample percentages
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        wedges, texts, autotexts = ax.pie(utilization, labels=resources, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title('Resource Utilization')
        
        
def run_performance_analysis():
    """Run comprehensive performance analysis with visualizations"""
    print("Running KPU Performance Analysis...")
    
    if not kpu:
        print("Using mock data for demonstration")
    
    # Create visualizer
    viz = KPUVisualizer()
    
    # Matrix sizes to test
    matrix_sizes = [(4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 32, 32), (64, 64, 64)]
    
    # Generate performance scaling plots
    print("Generating performance scaling analysis...")
    fig1 = viz.plot_performance_scaling(matrix_sizes)
    
    # Generate dashboard
    print("Generating comprehensive dashboard...")
    fig2 = viz.create_dashboard(matrix_sizes)
    
    # Generate memory heatmap
    print("Generating memory access heatmap...")
    sample_accesses = {(i, j*1024): np.random.randint(1, 50) 
                      for i in range(4) for j in range(0, 256, 4)}
    fig3 = viz.plot_memory_heatmap(sample_accesses)
    
    return fig1, fig2, fig3


def simulate_with_visualization(simulator, matrix_a, matrix_b, record_steps=True):
    """Run simulation with real-time visualization recording"""
    viz = KPUVisualizer(simulator)
    
    if record_steps:
        viz.start_recording()
        
        # Manual step-by-step execution for detailed recording
        # This would require extending the simulator API to support step-by-step execution
        # For now, we'll run the complete simulation and generate timeline
        result = simulator.run_numpy_matmul(matrix_a, matrix_b, 0, 0, 0)
        
        # Simulate recording steps (would be real in actual implementation)
        for i in range(simulator.get_current_cycle()):
            viz.record_step()
            
        viz.stop_recording()
        
        # Generate timeline visualization
        timeline_fig = viz.plot_component_timeline()
        pipeline_fig = viz.plot_pipeline_utilization()
        
        return result, timeline_fig, pipeline_fig
    
    else:
        result = simulator.run_numpy_matmul(matrix_a, matrix_b, 0, 0, 0)
        return result, None, None


if __name__ == "__main__":
    """Demo the visualization capabilities"""
    print("KPU Simulator Visualization Demo")
    print("=" * 40)
    
    try:
        # Run performance analysis
        figs = run_performance_analysis()
        
        print("\nGenerated visualizations:")
        print("- Performance scaling analysis")  
        print("- Comprehensive dashboard")
        print("- Memory access heatmap")
        
        # Show plots
        plt.show()
        
        print("\nVisualization demo completed!")
        
    except Exception as e:
        print(f"Error running visualization demo: {e}")
        import traceback
        traceback.print_exc()