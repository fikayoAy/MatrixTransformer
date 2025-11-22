"""
Validation script for hyperdimensional connection metrics.
Performs geometric consistency checks, curvature validation, and statistical analysis.
"""

import json
import numpy as np
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsValidator:
    """Validates geometric and statistical properties of hyperdimensional connections."""
    
    def __init__(self, connections_file, expected_radius=7.0):
        """
        Args:
            connections_file: Path to hyperdimensional_connections_output.json
            expected_radius: Expected radius of the hypersphere (default: 7.0)
        """
        self.connections_file = connections_file
        self.expected_radius = expected_radius
        self.connections = None
        self.validation_results = {}
        
    def load_connections(self):
        """Load connection data from JSON file."""
        logger.info(f"Loading connections from {self.connections_file}")
        with open(self.connections_file, 'r') as f:
            data = json.load(f)
        
        # Extract all connections into flat list
        self.connections = []

        # New format: top-level 'datasets' -> <dataset_id> -> 'connections'
        if isinstance(data, dict) and 'datasets' in data and isinstance(data['datasets'], dict):
            logger.debug("Found 'datasets' structure in connections file")
            for dsid, dsobj in data['datasets'].items():
                conns = dsobj.get('connections', {}) if isinstance(dsobj, dict) else {}
                if not isinstance(conns, dict):
                    continue
                for source_idx_str, conn_list in conns.items():
                    # Skip non-numeric keys
                    try:
                        source_idx = int(source_idx_str)
                    except Exception:
                        logger.debug(f"Skipping non-numeric source key: {source_idx_str} in dataset {dsid}")
                        continue
                    # conn_list expected to be a list of connection dicts
                    if isinstance(conn_list, list):
                        for conn in conn_list:
                            if not isinstance(conn, dict) or not conn:
                                continue
                            c = dict(conn)
                            c['source_idx'] = source_idx
                            c['source_dataset'] = dsid
                            self.connections.append(c)
                    elif isinstance(conn_list, dict) and 'connections' in conn_list:
                        for conn in conn_list.get('connections', []):
                            if not isinstance(conn, dict) or not conn:
                                continue
                            c = dict(conn)
                            c['source_idx'] = source_idx
                            c['source_dataset'] = dsid
                            self.connections.append(c)
                    else:
                        logger.debug(f"Unexpected connection list type for source {source_idx} in dataset {dsid}: {type(conn_list)}")

        else:
            # Backwards-compatible handling: top-level 'connections' mapping or flat mapping
            logger.debug("Falling back to legacy connections layout")
            connections_data = {}
            if isinstance(data, dict) and 'connections' in data and isinstance(data['connections'], dict):
                connections_data = data['connections']
            elif isinstance(data, dict):
                connections_data = data
            for source_idx, conn_list in connections_data.items():
                if source_idx in ['summary', 'metadata', 'variance_metrics']:
                    continue
                try:
                    source = int(source_idx)
                except Exception:
                    logger.debug(f"Skipping non-numeric key: {source_idx}")
                    continue
                if isinstance(conn_list, list):
                    for conn in conn_list:
                        if not isinstance(conn, dict) or not conn:
                            continue
                        c = dict(conn)
                        c['source_idx'] = source
                        self.connections.append(c)
                elif isinstance(conn_list, dict) and 'connections' in conn_list:
                    for conn in conn_list['connections']:
                        if not isinstance(conn, dict) or not conn:
                            continue
                        c = dict(conn)
                        c['source_idx'] = source
                        self.connections.append(c)
                else:
                    logger.debug(f"Unexpected structure for source {source_idx}: {type(conn_list)}")

        logger.info(f"Loaded {len(self.connections)} connections")
        return self.connections
    
    def validate_log_map_consistency(self):
        """Check if log_map_norm equals physical_dist (Critical Issue #1)."""
        logger.info("\n=== Validating Log Map Norm vs Physical Distance ===")
        
        discrepancies = []
        for conn in self.connections:
            log_norm = conn.get('log_map_norm', 0)
            phys_dist = conn.get('physical_dist', 0)
            
            # They should be equal within numerical precision
            diff = abs(log_norm - phys_dist)
            
            # Use adaptive tolerance: absolute error for tiny distances, relative for larger
            # This handles floating point precision limits at very small scales
            if phys_dist < 1e-5:
                # For very small distances, use absolute error threshold
                tolerance_failed = diff > 1e-8
                rel_error = diff / phys_dist if phys_dist > 0 else float('inf')
            else:
                # For normal distances, use relative error
                rel_error = diff / phys_dist if phys_dist > 0 else float('inf')
                tolerance_failed = rel_error > 1e-4  # 0.01% tolerance (relaxed from 0.0001%)
            
            if tolerance_failed:
                discrepancies.append({
                    'source': conn['source_idx'],
                    'target': conn['target_idx'],
                    'log_map_norm': log_norm,
                    'physical_dist': phys_dist,
                    'difference': diff,
                    'relative_error': rel_error
                })
        
        if discrepancies:
            logger.error(f"FAILED: Found {len(discrepancies)} connections where log_map_norm ≠ physical_dist")
            logger.error(f"Sample discrepancies (first 5):")
            for disc in discrepancies[:5]:
                logger.error(f"  {disc['source']}->{disc['target']}: "
                           f"log_norm={disc['log_map_norm']:.6e}, "
                           f"phys_dist={disc['physical_dist']:.6f}, "
                           f"rel_error={disc['relative_error']:.2%}")
            self.validation_results['log_map_consistency'] = 'FAILED'
            self.validation_results['log_map_discrepancies'] = len(discrepancies)
        else:
            logger.info("PASSED: log_map_norm equals physical_dist for all connections")
            self.validation_results['log_map_consistency'] = 'PASSED'
        
        return discrepancies
    
    def validate_projection_norms(self):
        """Check if projection norms match expected radius (Critical Issue #2)."""
        logger.info("\n=== Validating Projection Norms ===")
        
        source_norms = [conn.get('source_projection_norm', 0) for conn in self.connections]
        target_norms = [conn.get('target_projection_norm', 0) for conn in self.connections]
        
        all_norms = source_norms + target_norms
        mean_norm = np.mean(all_norms)
        std_norm = np.std(all_norms)
        min_norm = np.min(all_norms)
        max_norm = np.max(all_norms)
        
        logger.info(f"Projection norm statistics:")
        logger.info(f"  Mean: {mean_norm:.6f}")
        logger.info(f"  Std:  {std_norm:.6f}")
        logger.info(f"  Min:  {min_norm:.6f}")
        logger.info(f"  Max:  {max_norm:.6f}")
        logger.info(f"  Expected radius: {self.expected_radius:.6f}")
        
        # Check if norms are close to expected radius
        diff_from_expected = abs(mean_norm - self.expected_radius)
        
        if diff_from_expected > 0.1 * self.expected_radius:
            logger.warning(f"WARNING: Mean projection norm ({mean_norm:.2f}) differs from "
                         f"expected radius ({self.expected_radius}) by {diff_from_expected:.2f}")
            logger.warning(f"  This is {diff_from_expected/self.expected_radius:.1%} of expected radius")
            logger.warning(f"  Ratio: {mean_norm / self.expected_radius:.4f}")
            self.validation_results['projection_norm_check'] = 'WARNING'
        else:
            logger.info(f"PASSED: Projection norms close to expected radius")
            self.validation_results['projection_norm_check'] = 'PASSED'
        
        self.validation_results['mean_projection_norm'] = mean_norm
        self.validation_results['projection_norm_std'] = std_norm
        
        return {'mean': mean_norm, 'std': std_norm, 'min': min_norm, 'max': max_norm}
    
    def validate_curvature(self):
        """Check if local curvature matches expected value for sphere (Critical Issue #3)."""
        logger.info("\n=== Validating Local Curvature ===")
        
        curvatures = [conn.get('local_curvature', 0) for conn in self.connections]
        mean_curv = np.mean(curvatures)
        
        # For a sphere of radius R, curvature = -1/R²
        expected_curvature = -1.0 / (self.expected_radius ** 2)
        
        logger.info(f"Curvature statistics:")
        logger.info(f"  Mean observed: {mean_curv:.10f}")
        logger.info(f"  Expected for radius {self.expected_radius}: {expected_curvature:.10f}")
        
        # Check which radius would give the observed curvature
        if mean_curv < 0:
            inferred_radius = np.sqrt(-1.0 / mean_curv)
            logger.info(f"  Inferred sphere radius from curvature: {inferred_radius:.6f}")
            
            if abs(inferred_radius - 1.0) < 0.01:
                logger.warning(f"WARNING: Curvature suggests unit sphere (radius=1.0), not radius={self.expected_radius}")
                self.validation_results['curvature_check'] = 'FAILED'
            elif abs(inferred_radius - self.expected_radius) > 0.1:
                logger.warning(f"WARNING: Inferred radius ({inferred_radius:.2f}) differs from expected ({self.expected_radius})")
                self.validation_results['curvature_check'] = 'WARNING'
            else:
                logger.info(f"PASSED: Curvature consistent with radius {self.expected_radius}")
                self.validation_results['curvature_check'] = 'PASSED'
        
        self.validation_results['mean_curvature'] = mean_curv
        self.validation_results['expected_curvature'] = expected_curvature
        
        return mean_curv
    
    def validate_energy_gradient(self):
        """Verify energy_gradient = target_energy - local_energy."""
        logger.info("\n=== Validating Energy Gradient ===")
        
        errors = []
        for conn in self.connections:
            local_e = conn.get('local_energy', 0)
            target_e = conn.get('target_energy', 0)
            gradient = conn.get('energy_gradient', 0)
            
            expected_gradient = target_e - local_e
            error = abs(gradient - expected_gradient)
            
            if error > 1e-10:
                errors.append({
                    'source': conn['source_idx'],
                    'target': conn['target_idx'],
                    'error': error
                })
        
        if errors:
            logger.warning(f"WARNING: {len(errors)} connections have energy gradient errors > 1e-10")
            self.validation_results['energy_gradient_check'] = 'WARNING'
        else:
            logger.info(f"PASSED: All energy gradients computed correctly")
            self.validation_results['energy_gradient_check'] = 'PASSED'
        
        return errors
    
    def validate_geodesic_error(self):
        """Check if geodesic_error is near zero (should be 0 since phys_dist = vnorm)."""
        logger.info("\n=== Validating Geodesic Error ===")
        
        geodesic_errors = [conn.get('geodesic_error', 0) for conn in self.connections]
        
        mean_error = np.mean(geodesic_errors)
        max_error = np.max(geodesic_errors)
        nonzero_errors = [e for e in geodesic_errors if e > 1e-10]
        
        logger.info(f"Geodesic error statistics:")
        logger.info(f"  Mean: {mean_error:.10e}")
        logger.info(f"  Max:  {max_error:.10e}")
        logger.info(f"  Connections with error > 1e-10: {len(nonzero_errors)}")
        
        if mean_error > 1e-8 or len(nonzero_errors) > 0:
            logger.warning(f"WARNING: Geodesic errors detected (should be ~0 since phys_dist=vnorm)")
            logger.warning(f"  This suggests phys_dist and log_map_norm diverged during calculation")
            self.validation_results['geodesic_error_check'] = 'WARNING'
        else:
            logger.info(f"PASSED: Geodesic errors are negligible (phys_dist = log_map_norm)")
            self.validation_results['geodesic_error_check'] = 'PASSED'
        
        return {'mean': mean_error, 'max': max_error, 'nonzero_count': len(nonzero_errors)}
    
    def validate_energy_values(self):
        """Check if energy values are reasonable and non-zero."""
        logger.info("\n=== Validating Energy Values ===")
        
        local_energies = [conn.get('local_energy', 0) for conn in self.connections]
        target_energies = [conn.get('target_energy', 0) for conn in self.connections]
        
        local_mean = np.mean(local_energies)
        local_std = np.std(local_energies)
        target_mean = np.mean(target_energies)
        target_std = np.std(target_energies)
        
        zero_local = sum(1 for e in local_energies if abs(e) < 1e-10)
        zero_target = sum(1 for e in target_energies if abs(e) < 1e-10)
        
        logger.info(f"Local energy statistics:")
        logger.info(f"  Mean: {local_mean:.6e}")
        logger.info(f"  Std:  {local_std:.6e}")
        logger.info(f"  Zero values: {zero_local}/{len(local_energies)}")
        
        logger.info(f"Target energy statistics:")
        logger.info(f"  Mean: {target_mean:.6e}")
        logger.info(f"  Std:  {target_std:.6e}")
        logger.info(f"  Zero values: {zero_target}/{len(target_energies)}")
        
        # Check if all energies are zero (problematic)
        if zero_local == len(local_energies) or zero_target == len(target_energies):
            logger.error(f"FAILED: All energy values are zero - projection distances not calculated")
            self.validation_results['energy_values_check'] = 'FAILED'
        elif zero_local > len(local_energies) * 0.5 or zero_target > len(target_energies) * 0.5:
            logger.warning(f"WARNING: >50% of energy values are zero")
            self.validation_results['energy_values_check'] = 'WARNING'
        else:
            logger.info(f"PASSED: Energy values are reasonable")
            self.validation_results['energy_values_check'] = 'PASSED'
        
        return {
            'local_mean': local_mean,
            'target_mean': target_mean,
            'zero_local': zero_local,
            'zero_target': zero_target
        }
    
    def validate_norm_variance_values(self):
        """Check if norm_variance is near zero (should be 0 since all norms = 7.0)."""
        logger.info("\n=== Validating Norm Variance Values ===")
        
        norm_variances = [conn.get('norm_variance', 0) for conn in self.connections]
        norm_var_rel = [conn.get('norm_variance_relative', 0) for conn in self.connections]
        
        mean_var = np.mean(norm_variances)
        max_var = np.max(norm_variances)
        nonzero_var = [v for v in norm_variances if v > 1e-10]
        
        mean_rel = np.mean(norm_var_rel)
        max_rel = np.max(norm_var_rel)
        nonzero_rel = [v for v in norm_var_rel if v > 1e-10]
        
        logger.info(f"Norm variance (absolute) statistics:")
        logger.info(f"  Mean: {mean_var:.10e}")
        logger.info(f"  Max:  {max_var:.10e}")
        logger.info(f"  Connections with variance > 1e-10: {len(nonzero_var)}")
        
        logger.info(f"Norm variance (relative) statistics:")
        logger.info(f"  Mean: {mean_rel:.10e}")
        logger.info(f"  Max:  {max_rel:.10e}")
        logger.info(f"  Connections with rel variance > 1e-10: {len(nonzero_rel)}")
        
        # Since all projection_norms are exactly 7.0, variance should be 0
        if mean_var > 1e-8 or len(nonzero_var) > 0:
            logger.warning(f"WARNING: Norm variances detected (should be 0 since all norms = 7.0)")
            logger.warning(f"  This suggests projection_norms are not all equal")
            self.validation_results['norm_variance_values_check'] = 'WARNING'
        else:
            logger.info(f"PASSED: Norm variances are zero (all projections have same radius)")
            self.validation_results['norm_variance_values_check'] = 'PASSED'
        
        return {
            'mean_absolute': mean_var,
            'max_absolute': max_var,
            'nonzero_count': len(nonzero_var)
        }
    
    def validate_reciprocal_angle(self):
        """Check if reciprocal angles are in valid range [0, π]."""
        logger.info("\n=== Validating Reciprocal Angles ===")
        
        angles = [conn.get('reciprocal_angle', 0) for conn in self.connections]
        
        min_angle = np.min(angles)
        max_angle = np.max(angles)
        mean_angle = np.mean(angles)
        
        logger.info(f"Reciprocal angle statistics:")
        logger.info(f"  Min:  {min_angle:.6f}")
        logger.info(f"  Max:  {max_angle:.6f}")
        logger.info(f"  Mean: {mean_angle:.6f}")
        logger.info(f"  π:    {np.pi:.6f}")
        
        # Check if angles are in valid range
        invalid = [a for a in angles if a < 0 or a > np.pi + 1e-6]
        
        if invalid:
            logger.error(f"FAILED: {len(invalid)} angles outside valid range [0, π]")
            self.validation_results['reciprocal_angle_check'] = 'FAILED'
        else:
            logger.info(f"PASSED: All reciprocal angles in valid range")
            self.validation_results['reciprocal_angle_check'] = 'PASSED'
        
        # Check distribution
        near_zero = sum(1 for a in angles if a < 0.1)
        near_pi = sum(1 for a in angles if a > np.pi - 0.1)
        
        logger.info(f"  Near 0 (same direction): {near_zero}")
        logger.info(f"  Near π (opposite direction): {near_pi}")
        
        return {'min': min_angle, 'max': max_angle, 'mean': mean_angle}
    
    def validate_norm_variance(self):
        """Check if norm variance values are reasonable."""
        logger.info("\n=== Validating Norm Variance ===")
        
        variances = [conn.get('norm_variance', 0) for conn in self.connections]
        rel_variances = [conn.get('norm_variance_relative', 0) for conn in self.connections]
        
        mean_var = np.mean(variances)
        max_var = np.max(variances)
        mean_rel = np.mean(rel_variances)
        max_rel = np.max(rel_variances)
        
        logger.info(f"Norm variance statistics:")
        logger.info(f"  Mean absolute: {mean_var:.6e}")
        logger.info(f"  Max absolute:  {max_var:.6e}")
        logger.info(f"  Mean relative: {mean_rel:.6e}")
        logger.info(f"  Max relative:  {max_rel:.6e}")
        
        # Check if relative variance is small (< 0.01%)
        high_variance = [v for v in rel_variances if v > 1e-4]
        
        if high_variance:
            logger.warning(f"WARNING: {len(high_variance)} connections have relative norm variance > 0.01%")
            self.validation_results['norm_variance_check'] = 'WARNING'
        else:
            logger.info(f"PASSED: All norm variances are small (good projection stability)")
            self.validation_results['norm_variance_check'] = 'PASSED'
        
        return {'mean': mean_var, 'max': max_var, 'mean_relative': mean_rel}
    
    def validate_ratio_consistency(self):
        """Verify ratio = physical_dist / high_dim_dist."""
        logger.info("\n=== Validating Ratio Calculation ===")
        
        errors = []
        for conn in self.connections:
            ratio = conn.get('ratio', 0)
            phys_dist = conn.get('physical_dist', 0)
            hd_dist = conn.get('high_dim_dist', 1e-10)
            
            if hd_dist > 0:
                expected_ratio = phys_dist / hd_dist
                error = abs(ratio - expected_ratio) / expected_ratio if expected_ratio > 0 else 0
                
                if error > 1e-6:
                    errors.append({
                        'source': conn['source_idx'],
                        'target': conn['target_idx'],
                        'error': error
                    })
        
        if errors:
            logger.warning(f"WARNING: {len(errors)} connections have ratio calculation errors")
            self.validation_results['ratio_check'] = 'WARNING'
        else:
            logger.info(f"PASSED: All ratios computed correctly")
            self.validation_results['ratio_check'] = 'PASSED'
        
        return errors
    
    def plot_distributions(self, output_dir='diagnostics'):
        """Generate distribution plots for key metrics."""
        logger.info("\n=== Generating Distribution Plots ===")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Extract metrics
        physical_dists = [conn.get('physical_dist', 0) for conn in self.connections]
        log_map_norms = [conn.get('log_map_norm', 0) for conn in self.connections]
        energy_gradients = [conn.get('energy_gradient', 0) for conn in self.connections]
        norm_variances = [conn.get('norm_variance_relative', 0) for conn in self.connections]
        ratios = [conn.get('ratio', 0) for conn in self.connections]
        strengths = [conn.get('strength', 0) for conn in self.connections]
        
        # Create 2x3 subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Hyperdimensional Connection Metrics Distributions', fontsize=16)
        
        # Plot 1: Physical Distance
        axes[0, 0].hist(physical_dists, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Physical Distance')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'Physical Distance (mean={np.mean(physical_dists):.4f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Log Map Norm vs Physical Distance
        axes[0, 1].scatter(physical_dists, log_map_norms, alpha=0.5, s=10)
        axes[0, 1].plot([0, max(physical_dists)], [0, max(physical_dists)], 'r--', label='y=x')
        axes[0, 1].set_xlabel('Physical Distance')
        axes[0, 1].set_ylabel('Log Map Norm')
        axes[0, 1].set_title('Log Map Norm vs Physical Distance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Energy Gradient
        axes[0, 2].hist(energy_gradients, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 2].set_xlabel('Energy Gradient')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title(f'Energy Gradient (mean={np.mean(energy_gradients):.2e})')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Norm Variance (log scale)
        axes[1, 0].hist(np.log10(np.array(norm_variances) + 1e-20), bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Log10(Relative Norm Variance)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Relative Norm Variance Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Ratio distribution (filter out zeros to avoid log10(0) = -inf)
        ratios_nonzero = [r for r in ratios if r > 0]
        if ratios_nonzero:
            axes[1, 1].hist(np.log10(ratios_nonzero), bins=50, edgecolor='black', alpha=0.7)
            axes[1, 1].set_xlabel('Log10(Ratio)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title(f'Ratio Distribution (median={np.median(ratios_nonzero):.2e})')
        else:
            axes[1, 1].text(0.5, 0.5, 'No non-zero ratios', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_xlabel('Log10(Ratio)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Ratio Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Strength distribution
        axes[1, 2].hist(strengths, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 2].set_xlabel('Connection Strength')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title(f'Strength Distribution (mean={np.mean(strengths):.2f})')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = Path(output_dir) / 'metrics_distributions.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        logger.info(f"Saved distribution plots to {plot_file}")
        plt.close()
    
    def generate_report(self, output_file='validation_report.txt'):
        """Generate comprehensive validation report."""
        logger.info(f"\n=== Generating Validation Report ===")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("HYPERDIMENSIONAL CONNECTIONS VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total connections validated: {len(self.connections)}\n")
            f.write(f"Expected sphere radius: {self.expected_radius}\n\n")
            
            f.write("VALIDATION RESULTS:\n")
            f.write("-" * 80 + "\n")
            for check, result in self.validation_results.items():
                f.write(f"{check:.<50} {result}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("CRITICAL ISSUES:\n")
            f.write("=" * 80 + "\n\n")
            
            # Issue 1: Log map vs physical distance
            if self.validation_results.get('log_map_consistency') == 'FAILED':
                f.write("❌ ISSUE 1: log_map_norm ≠ physical_dist\n")
                f.write(f"   Found {self.validation_results.get('log_map_discrepancies', 0)} discrepancies\n")
                f.write("   These should be identical - indicates calculation error\n\n")
            
            # Issue 2: Projection norms
            mean_norm = self.validation_results.get('mean_projection_norm', 0)
            if abs(mean_norm - self.expected_radius) > 0.1 * self.expected_radius:
                f.write("⚠️  ISSUE 2: Projection norms don't match expected radius\n")
                f.write(f"   Mean projection norm: {mean_norm:.6f}\n")
                f.write(f"   Expected radius: {self.expected_radius}\n")
                f.write(f"   Ratio: {mean_norm / self.expected_radius:.4f}\n\n")
            
            # Issue 3: Curvature
            if self.validation_results.get('curvature_check') in ['FAILED', 'WARNING']:
                f.write("⚠️  ISSUE 3: Local curvature inconsistent with sphere radius\n")
                f.write(f"   Mean curvature: {self.validation_results.get('mean_curvature', 0):.10f}\n")
                f.write(f"   Expected: {self.validation_results.get('expected_curvature', 0):.10f}\n\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("See console output for detailed statistics\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Validation report saved to {output_file}")
    
    def run_all_validations(self):
        """Run all validation checks and generate reports."""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING COMPREHENSIVE VALIDATION")
        logger.info("=" * 80)
        
        # Load data
        self.load_connections()
        
        # Run all validations
        self.validate_log_map_consistency()
        self.validate_projection_norms()
        self.validate_curvature()
        self.validate_geodesic_error()
        self.validate_energy_values()
        self.validate_energy_gradient()
        self.validate_reciprocal_angle()
        self.validate_norm_variance()
        self.validate_norm_variance_values()
        self.validate_ratio_consistency()
        
        # Generate visualizations
        self.plot_distributions()
        
        # Generate report
        self.generate_report()
        
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION COMPLETE")
        logger.info("=" * 80)
        
        # Summary
        passed = sum(1 for r in self.validation_results.values() if r == 'PASSED')
        warnings = sum(1 for r in self.validation_results.values() if r == 'WARNING')
        failed = sum(1 for r in self.validation_results.values() if r == 'FAILED')
        
        logger.info(f"\nSummary: {passed} PASSED, {warnings} WARNINGS, {failed} FAILED")
        
        return self.validation_results


if __name__ == '__main__':
    # Run validation
    validator = MetricsValidator(
        connections_file='hyperdimensional_connections_output.json',
        expected_radius=7.0
    )
    
    results = validator.run_all_validations()
    
    # Print final status
    print("\n" + "=" * 80)
    print("Validation complete! Check:")
    print("  - validation_report.txt for detailed findings")
    print("  - diagnostics/metrics_distributions.png for visualizations")
    print("=" * 80)
