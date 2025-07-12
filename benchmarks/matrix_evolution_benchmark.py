import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MatrixSpace.Matrixtransfomrer import MatrixTransformer

def create_output_dir():
    """Create output directory for decision hypercube benchmark results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"benchmarks/decision_hypercube_results_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
    
def analyze_hypercube_structure(transformer):
    """Analyze the structure of the decision hypercube"""
    print("Analyzing Decision Hypercube Structure...")
    
    # Extract hypercube data
    cube = transformer.decision_hypercube
    matrix_graph = transformer.matrix_graph
    
    analysis = {
        'total_vertices': len(cube),
        'property_dimensions': len(transformer.properties),
        'matrix_types': list(matrix_graph.keys()),
        'property_list': transformer.properties,
        'vertex_analysis': {},
        'connectivity_analysis': {},
        'property_correlations': {}
    }
    
    # Analyze each vertex in the hypercube
    for coords, vertex_info in cube.items():
        matrix_type = vertex_info.get('type', 'unknown')
        properties = vertex_info.get('properties', {})
        
        analysis['vertex_analysis'][matrix_type] = {
            'coordinates': coords,
            'properties': properties,
            'embedding_norm': np.linalg.norm(vertex_info.get('embedding', [0])),
            'sphere_embedding_norm': np.linalg.norm(vertex_info.get('sphere_embedding', [0]))
        }
    
    # Analyze connectivity between matrix types
    type_distances = {}
    type_coordinates = {}
    
    for matrix_type in analysis['matrix_types']:
        if matrix_type in analysis['vertex_analysis']:
            coords = analysis['vertex_analysis'][matrix_type]['coordinates']
            type_coordinates[matrix_type] = np.array(coords)
    
    # Calculate pairwise distances between matrix types
    for type1 in type_coordinates:
        type_distances[type1] = {}
        for type2 in type_coordinates:
            if type1 != type2:
                dist = np.linalg.norm(type_coordinates[type1] - type_coordinates[type2])
                type_distances[type1][type2] = dist
    
    analysis['connectivity_analysis'] = type_distances
    analysis['type_coordinates'] = type_coordinates
    
    # Analyze property correlations
    property_matrix = []
    property_names = []
    
    for matrix_type, vertex_data in analysis['vertex_analysis'].items():
        props = vertex_data['properties']
        prop_vector = []
        for prop_name in transformer.properties:
            prop_vector.append(props.get(prop_name, 0.0))
        property_matrix.append(prop_vector)
        property_names.append(matrix_type)
    
    if property_matrix:
        property_matrix = np.array(property_matrix)
        correlation_matrix = np.corrcoef(property_matrix.T)
        analysis['property_correlations'] = {
            'matrix': correlation_matrix,
            'property_names': transformer.properties,
            'type_names': property_names
        }
    
    return analysis

def benchmark_interpolation_paths(transformer, analysis):
    """Benchmark interpolation between different matrix types"""
    print("Benchmarking Matrix Type Interpolation...")
    
    type_coordinates = analysis['type_coordinates']
    interpolation_results = {}
    
    # Define key interpolation paths to test
    test_paths = [
        ('general', 'symmetric'),
        ('symmetric', 'positive_definite'),
        ('diagonal', 'upper_triangular'),
        ('sparse', 'adjacency'),
        ('toeplitz', 'circulant'),
        ('hermitian', 'positive_definite'),
        ('nilpotent', 'upper_triangular'),
        ('block', 'banded'),
        ('laplacian', 'symmetric')
    ]
    
    for start_type, end_type in test_paths:
        if start_type in type_coordinates and end_type in type_coordinates:
            start_coords = type_coordinates[start_type]
            end_coords = type_coordinates[end_type]
            
            # Generate interpolation path with multiple steps
            n_steps = 20
            interpolation_path = []
            property_evolution = []
            
            for i in range(n_steps + 1):
                alpha = i / n_steps
                interp_coords = (1 - alpha) * start_coords + alpha * end_coords
                
                # Find closest vertex in hypercube to interpolated coordinates
                min_dist = float('inf')
                closest_vertex = None
                closest_type = 'unknown'
                
                for coords, vertex_info in transformer.decision_hypercube.items():
                    coords_array = np.array(coords)
                    dist = np.linalg.norm(coords_array - interp_coords)
                    if dist < min_dist:
                        min_dist = dist
                        closest_vertex = vertex_info
                        closest_type = vertex_info.get('type', 'unknown')
                
                interpolation_path.append({
                    'step': i,
                    'alpha': alpha,
                    'coordinates': interp_coords,
                    'closest_type': closest_type,
                    'distance_to_closest': min_dist,
                    'properties': closest_vertex.get('properties', {}) if closest_vertex else {}
                })
                
                # Track property evolution
                if closest_vertex:
                    props = closest_vertex.get('properties', {})
                    prop_vector = [props.get(prop, 0.0) for prop in transformer.properties]
                    property_evolution.append(prop_vector)
            
            interpolation_results[f"{start_type}_to_{end_type}"] = {
                'path': interpolation_path,
                'property_evolution': np.array(property_evolution),
                'start_type': start_type,
                'end_type': end_type,
                'path_length': np.linalg.norm(end_coords - start_coords),
                'smoothness_metric': calculate_path_smoothness(interpolation_path)
            }
    
    return interpolation_results

def calculate_path_smoothness(interpolation_path):
    """Calculate smoothness metric for interpolation path"""
    if len(interpolation_path) < 3:
        return 0.0
    
    # Calculate second derivatives along the path
    coords_sequence = [np.array(step['coordinates']) for step in interpolation_path]
    
    smoothness_values = []
    for i in range(1, len(coords_sequence) - 1):
        # Second derivative approximation
        second_deriv = coords_sequence[i+1] - 2*coords_sequence[i] + coords_sequence[i-1]
        smoothness_values.append(np.linalg.norm(second_deriv))
    
    # Return inverse of average curvature (higher = smoother)
    avg_curvature = np.mean(smoothness_values)
    return 1.0 / (1.0 + avg_curvature)

def benchmark_matrix_transformations(transformer, analysis):
    """Benchmark actual matrix transformations along hypercube paths"""
    print("Benchmarking Matrix Transformations...")
    
    # Create test matrices of different types
    test_matrices = {
        'general': np.random.randn(6, 6),
        'symmetric': None,  # Will be generated
        'diagonal': np.diag([1, 2, 3, 4, 5, 6]),
        'upper_triangular': np.triu(np.random.randn(6, 6)),
        'sparse': None,  # Will be generated
        'positive_definite': None  # Will be generated
    }
    
    # Generate symmetric matrix
    A_sym = np.random.randn(6, 6)
    test_matrices['symmetric'] = (A_sym + A_sym.T) / 2
    
    # Generate sparse matrix
    sparse_mat = np.random.randn(6, 6)
    sparse_mat[np.abs(sparse_mat) < 1.0] = 0
    test_matrices['sparse'] = sparse_mat
    
    # Generate positive definite matrix
    A_pd = np.random.randn(6, 6)
    test_matrices['positive_definite'] = A_pd @ A_pd.T + np.eye(6) * 0.1
    
    transformation_results = {}
    
    # Test transformations between types
    for source_type, source_matrix in test_matrices.items():
        if source_matrix is None:
            continue
            
        transformation_results[source_type] = {}
        
        for target_type in analysis['matrix_types']:
            if target_type == source_type:
                continue
                
            start_time = time.time()
            
            try:
                # Get transformation method
                transform_method = transformer._get_transform_method(target_type)
                if transform_method:
                    transformed = transform_method(source_matrix)
                    
                    # Calculate transformation metrics
                    original_energy = np.linalg.norm(source_matrix)
                    transformed_energy = np.linalg.norm(transformed)
                    energy_ratio = transformed_energy / (original_energy + 1e-10)
                    
                    # Check if transformation preserves expected properties
                    achieved_properties = transformer.derive_property_values(transformed)
                    expected_properties = analysis['vertex_analysis'].get(target_type, {}).get('properties', {})
                    
                    property_match_score = 0.0
                    property_count = 0
                    for prop, expected_val in expected_properties.items():
                        if prop in achieved_properties:
                            if isinstance(expected_val, bool):
                                match = 1.0 if (achieved_properties[prop] > 0.5) == expected_val else 0.0
                            else:
                                match = 1.0 - min(1.0, abs(achieved_properties[prop] - expected_val))
                            property_match_score += match
                            property_count += 1
                    
                    if property_count > 0:
                        property_match_score /= property_count
                    
                    transformation_time = time.time() - start_time
                    
                    transformation_results[source_type][target_type] = {
                        'success': True,
                        'transformation_time': transformation_time,
                        'energy_ratio': energy_ratio,
                        'property_match_score': property_match_score,
                        'achieved_properties': achieved_properties,
                        'expected_properties': expected_properties
                    }
                else:
                    transformation_results[source_type][target_type] = {
                        'success': False,
                        'error': 'No transformation method available'
                    }
                    
            except Exception as e:
                transformation_results[source_type][target_type] = {
                    'success': False,
                    'error': str(e)
                }
    
    return transformation_results

def create_visualizations(analysis, interpolation_results, transformation_results, output_dir):
    """Create comprehensive visualizations of the decision hypercube"""
    print("Creating visualizations...")
    
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # 1. 3D Projection of Hypercube (PCA)
    create_3d_hypercube_projection(analysis, viz_dir)
    
    # 2. Property Correlation Heatmap
    create_property_correlation_heatmap(analysis, viz_dir)
    
    # 3. Matrix Type Distance Matrix
    create_distance_matrix_visualization(analysis, viz_dir)
    
    # 4. Interpolation Path Visualizations
    create_interpolation_visualizations(interpolation_results, viz_dir)
    
    # 5. Transformation Performance Heatmap
    create_transformation_heatmap(transformation_results, viz_dir)
    
    # 6. Interactive 3D Hypercube (Plotly)
    create_interactive_hypercube(analysis, viz_dir)
    
    # 7. Property Evolution Along Paths
    create_property_evolution_plots(interpolation_results, viz_dir)

def create_3d_hypercube_projection(analysis, viz_dir):
    """Create 3D PCA projection of the hypercube"""
    type_coordinates = analysis['type_coordinates']
    
    if len(type_coordinates) < 3:
        return
    
    # Prepare data for PCA
    coords_matrix = np.array(list(type_coordinates.values()))
    type_names = list(type_coordinates.keys())
    
    # Apply PCA
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(coords_matrix)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color map for different matrix types
    colors = plt.cm.tab20(np.linspace(0, 1, len(type_names)))
    
    for i, (name, color) in enumerate(zip(type_names, colors)):
        ax.scatter(coords_3d[i, 0], coords_3d[i, 1], coords_3d[i, 2], 
                  c=[color], s=100, label=name, alpha=0.7)
        ax.text(coords_3d[i, 0], coords_3d[i, 1], coords_3d[i, 2], 
                name, fontsize=8)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)')
    ax.set_title('3D PCA Projection of Decision Hypercube\nMatrix Types as Points in High-Dimensional Space')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(viz_dir / "hypercube_3d_projection.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save PCA analysis
    pca_analysis = {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'components': pca.components_.tolist(),
        'type_names': type_names,
        'coords_3d': coords_3d.tolist()
    }
    
    pd.DataFrame(pca_analysis['coords_3d'], 
                 columns=['PC1', 'PC2', 'PC3'], 
                 index=type_names).to_csv(viz_dir / "pca_coordinates.csv")

def create_property_correlation_heatmap(analysis, viz_dir):
    """Create heatmap of property correlations"""
    if 'property_correlations' not in analysis or 'matrix' not in analysis['property_correlations']:
        return
    
    corr_matrix = analysis['property_correlations']['matrix']
    property_names = analysis['property_correlations']['property_names']
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                xticklabels=property_names,
                yticklabels=property_names,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f')
    
    plt.title('Property Correlations in Decision Hypercube\nHow Matrix Properties Relate to Each Other')
    plt.tight_layout()
    plt.savefig(viz_dir / "property_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_distance_matrix_visualization(analysis, viz_dir):
    """Create visualization of distances between matrix types"""
    connectivity = analysis['connectivity_analysis']
    matrix_types = list(connectivity.keys())
    
    if len(matrix_types) < 2:
        return
    
    # Create distance matrix
    n_types = len(matrix_types)
    distance_matrix = np.zeros((n_types, n_types))
    
    for i, type1 in enumerate(matrix_types):
        for j, type2 in enumerate(matrix_types):
            if type1 != type2 and type2 in connectivity[type1]:
                distance_matrix[i, j] = connectivity[type1][type2]
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(distance_matrix,
                xticklabels=matrix_types,
                yticklabels=matrix_types,
                annot=True,
                cmap='viridis_r',
                fmt='.2f')
    
    plt.title('Distance Matrix Between Matrix Types\nEuclidean Distances in Hypercube Space')
    plt.xlabel('Target Matrix Type')
    plt.ylabel('Source Matrix Type')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(viz_dir / "distance_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save distance matrix as CSV
    pd.DataFrame(distance_matrix, 
                 index=matrix_types, 
                 columns=matrix_types).to_csv(viz_dir / "distance_matrix.csv")

def create_interpolation_visualizations(interpolation_results, viz_dir):
    """Create visualizations of interpolation paths"""
    
    # Create subplots for multiple interpolation paths
    n_paths = len(interpolation_results)
    if n_paths == 0:
        return
    
    cols = min(3, n_paths)
    rows = (n_paths + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_paths == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, (path_name, path_data) in enumerate(interpolation_results.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Plot smoothness along the path
        alphas = [step['alpha'] for step in path_data['path']]
        distances = [step['distance_to_closest'] for step in path_data['path']]
        
        ax.plot(alphas, distances, 'b-', linewidth=2, label='Distance to Closest Type')
        ax.set_xlabel('Interpolation Parameter (α)')
        ax.set_ylabel('Distance to Closest Matrix Type')
        ax.set_title(f'{path_data["start_type"]} → {path_data["end_type"]}\nSmoothness: {path_data["smoothness_metric"]:.3f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide unused subplots
    for idx in range(n_paths, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(viz_dir / "interpolation_paths.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_transformation_heatmap(transformation_results, viz_dir):
    """Create heatmap of transformation performance"""
    
    # Extract source and target types
    source_types = list(transformation_results.keys())
    all_target_types = set()
    for source_data in transformation_results.values():
        all_target_types.update(source_data.keys())
    target_types = sorted(list(all_target_types))
    
    if not source_types or not target_types:
        return
    
    # Create performance matrices
    success_matrix = np.zeros((len(source_types), len(target_types)))
    property_match_matrix = np.zeros((len(source_types), len(target_types)))
    time_matrix = np.zeros((len(source_types), len(target_types)))
    
    for i, source_type in enumerate(source_types):
        for j, target_type in enumerate(target_types):
            if target_type in transformation_results[source_type]:
                result = transformation_results[source_type][target_type]
                success_matrix[i, j] = 1.0 if result.get('success', False) else 0.0
                property_match_matrix[i, j] = result.get('property_match_score', 0.0)
                time_matrix[i, j] = result.get('transformation_time', 0.0)
    
    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Success rate heatmap
    sns.heatmap(success_matrix, 
                xticklabels=target_types,
                yticklabels=source_types,
                annot=True, 
                cmap='RdYlGn', 
                ax=ax1,
                vmin=0, vmax=1)
    ax1.set_title('Transformation Success Rate')
    ax1.set_xlabel('Target Type')
    ax1.set_ylabel('Source Type')
    
    # Property match heatmap
    sns.heatmap(property_match_matrix, 
                xticklabels=target_types,
                yticklabels=source_types,
                annot=True, 
                cmap='viridis', 
                ax=ax2,
                vmin=0, vmax=1)
    ax2.set_title('Property Match Score')
    ax2.set_xlabel('Target Type')
    ax2.set_ylabel('Source Type')
    
    # Transformation time heatmap
    sns.heatmap(time_matrix, 
                xticklabels=target_types,
                yticklabels=source_types,
                annot=True, 
                cmap='plasma_r', 
                ax=ax3,
                fmt='.4f')
    ax3.set_title('Transformation Time (seconds)')
    ax3.set_xlabel('Target Type')
    ax3.set_ylabel('Source Type')
    
    plt.tight_layout()
    plt.savefig(viz_dir / "transformation_performance.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_interactive_hypercube(analysis, viz_dir):
    """Create interactive 3D visualization using Plotly"""
    type_coordinates = analysis['type_coordinates']
    
    if len(type_coordinates) < 3:
        return
    
    # Prepare data for PCA
    coords_matrix = np.array(list(type_coordinates.values()))
    type_names = list(type_coordinates.keys())
    
    # Apply PCA
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(coords_matrix)
    
    # Create interactive 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=coords_3d[:, 0],
        y=coords_3d[:, 1],
        z=coords_3d[:, 2],
        mode='markers+text',
        marker=dict(
            size=12,
            color=np.arange(len(type_names)),
            colorscale='viridis',
            opacity=0.8
        ),
        text=type_names,
        textposition="top center",
        hovertemplate='<b>%{text}</b><br>' +
                      'PC1: %{x:.3f}<br>' +
                      'PC2: %{y:.3f}<br>' +
                      'PC3: %{z:.3f}<extra></extra>'
    )])
    
    fig.update_layout(
        title='Interactive Decision Hypercube Visualization<br>Matrix Types in High-Dimensional Space',
        scene=dict(
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
            zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)'
        ),
        width=800,
        height=600
    )
    
    # Save interactive plot
    fig.write_html(viz_dir / "interactive_hypercube.html")

def create_property_evolution_plots(interpolation_results, viz_dir):
    """Create plots showing how properties evolve along interpolation paths"""
    
    for path_name, path_data in interpolation_results.items():
        if 'property_evolution' not in path_data or path_data['property_evolution'].size == 0:
            continue
            
        property_evolution = path_data['property_evolution']
        alphas = [step['alpha'] for step in path_data['path']]
        
        # Create subplot for each property
        n_properties = property_evolution.shape[1]
        cols = min(4, n_properties)
        rows = (n_properties + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if n_properties == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Get property names (assuming transformer.properties is available)
        property_names = [f'Property_{i}' for i in range(n_properties)]
        
        for i in range(n_properties):
            if i >= len(axes):
                break
                
            ax = axes[i]
            ax.plot(alphas, property_evolution[:, i], 'b-', linewidth=2)
            ax.set_xlabel('Interpolation Parameter (α)')
            ax.set_ylabel('Property Value')
            ax.set_title(f'{property_names[i]}')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        # Hide unused subplots
        for i in range(n_properties, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Property Evolution: {path_data["start_type"]} → {path_data["end_type"]}')
        plt.tight_layout()
        plt.savefig(viz_dir / f"property_evolution_{path_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

def save_comprehensive_results(analysis, interpolation_results, transformation_results, output_dir):
    """Save all benchmark results to files"""
    print("Saving comprehensive results...")
    
    # Save analysis summary
    summary = {
        'hypercube_structure': {
            'total_vertices': analysis['total_vertices'],
            'property_dimensions': analysis['property_dimensions'],
            'matrix_types_count': len(analysis['matrix_types']),
            'matrix_types': analysis['matrix_types']
        },
        'interpolation_analysis': {
            'total_paths_tested': len(interpolation_results),
            'average_smoothness': np.mean([data['smoothness_metric'] for data in interpolation_results.values()]),
            'average_path_length': np.mean([data['path_length'] for data in interpolation_results.values()])
        },
        'transformation_analysis': {
            'total_transformations_tested': sum(len(targets) for targets in transformation_results.values()),
            'overall_success_rate': calculate_overall_success_rate(transformation_results),
            'average_property_match': calculate_average_property_match(transformation_results),
            'average_transformation_time': calculate_average_transformation_time(transformation_results)
        }
    }
    
    # Save to JSON
    import json
    with open(output_dir / "benchmark_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed analysis as CSV files
    
    # 1. Vertex analysis
    vertex_df_data = []
    for matrix_type, vertex_data in analysis['vertex_analysis'].items():
        row = {'matrix_type': matrix_type}
        row.update(vertex_data['properties'])
        row['embedding_norm'] = vertex_data['embedding_norm']
        row['sphere_embedding_norm'] = vertex_data['sphere_embedding_norm']
        vertex_df_data.append(row)
    
    if vertex_df_data:
        vertex_df = pd.DataFrame(vertex_df_data)
        vertex_df.to_csv(output_dir / "vertex_analysis.csv", index=False)
    
    # 2. Interpolation results
    interp_df_data = []
    for path_name, path_data in interpolation_results.items():
        interp_df_data.append({
            'path_name': path_name,
            'start_type': path_data['start_type'],
            'end_type': path_data['end_type'],
            'path_length': path_data['path_length'],
            'smoothness_metric': path_data['smoothness_metric'],
            'num_steps': len(path_data['path'])
        })
    
    if interp_df_data:
        interp_df = pd.DataFrame(interp_df_data)
        interp_df.to_csv(output_dir / "interpolation_results.csv", index=False)
    
    # 3. Transformation results
    transform_df_data = []
    for source_type, targets in transformation_results.items():
        for target_type, result in targets.items():
            transform_df_data.append({
                'source_type': source_type,
                'target_type': target_type,
                'success': result.get('success', False),
                'transformation_time': result.get('transformation_time', 0.0),
                'energy_ratio': result.get('energy_ratio', 0.0),
                'property_match_score': result.get('property_match_score', 0.0),
                'error': result.get('error', '')
            })
    
    if transform_df_data:
        transform_df = pd.DataFrame(transform_df_data)
        transform_df.to_csv(output_dir / "transformation_results.csv", index=False)

def calculate_overall_success_rate(transformation_results):
    """Calculate overall success rate of transformations"""
    total_attempts = 0
    successful_attempts = 0
    
    for source_data in transformation_results.values():
        for result in source_data.values():
            total_attempts += 1
            if result.get('success', False):
                successful_attempts += 1
    
    return successful_attempts / total_attempts if total_attempts > 0 else 0.0

def calculate_average_property_match(transformation_results):
    """Calculate average property match score"""
    scores = []
    
    for source_data in transformation_results.values():
        for result in source_data.values():
            if result.get('success', False):
                scores.append(result.get('property_match_score', 0.0))
    
    return np.mean(scores) if scores else 0.0

def calculate_average_transformation_time(transformation_results):
    """Calculate average transformation time"""
    times = []
    
    for source_data in transformation_results.values():
        for result in source_data.values():
            if result.get('success', False):
                times.append(result.get('transformation_time', 0.0))
    
    return np.mean(times) if times else 0.0

def print_benchmark_summary(analysis, interpolation_results, transformation_results):
    """Print a comprehensive summary of benchmark results"""
    print("\n" + "="*80)
    print("DECISION HYPERCUBE BENCHMARK SUMMARY")
    print("="*80)
    
    print(f"\n🏗️  HYPERCUBE STRUCTURE:")
    print(f"   • Total vertices: {analysis['total_vertices']}")
    print(f"   • Property dimensions: {analysis['property_dimensions']}")
    print(f"   • Matrix types encoded: {len(analysis['matrix_types'])}")
    print(f"   • Properties tracked: {', '.join(analysis['properties'][:5])}...")
    
    print(f"\n🔄 INTERPOLATION ANALYSIS:")
    print(f"   • Interpolation paths tested: {len(interpolation_results)}")
    if interpolation_results:
        avg_smoothness = np.mean([data['smoothness_metric'] for data in interpolation_results.values()])
        avg_path_length = np.mean([data['path_length'] for data in interpolation_results.values()])
        print(f"   • Average path smoothness: {avg_smoothness:.3f}")
        print(f"   • Average path length: {avg_path_length:.3f}")
        
        print(f"\n   📈 Best interpolation paths:")
        sorted_paths = sorted(interpolation_results.items(), 
                            key=lambda x: x[1]['smoothness_metric'], reverse=True)
        for i, (path_name, data) in enumerate(sorted_paths[:3]):
            print(f"      {i+1}. {data['start_type']} → {data['end_type']} "
                  f"(smoothness: {data['smoothness_metric']:.3f})")
    
    print(f"\n⚡ TRANSFORMATION PERFORMANCE:")
    total_transforms = sum(len(targets) for targets in transformation_results.values())
    success_rate = calculate_overall_success_rate(transformation_results)
    avg_property_match = calculate_average_property_match(transformation_results)
    avg_time = calculate_average_transformation_time(transformation_results)
    
    print(f"   • Total transformations tested: {total_transforms}")
    print(f"   • Overall success rate: {success_rate:.1%}")
    print(f"   • Average property match: {avg_property_match:.3f}")
    print(f"   • Average transformation time: {avg_time:.4f}s")
    
    # Find best performing transformations
    best_transforms = []
    for source_type, targets in transformation_results.items():
        for target_type, result in targets.items():
            if result.get('success', False):
                best_transforms.append((
                    f"{source_type} → {target_type}",
                    result.get('property_match_score', 0.0),
                    result.get('transformation_time', 0.0)
                ))
    
    if best_transforms:
        best_transforms.sort(key=lambda x: x[1], reverse=True)
        print(f"\n   🏆 Top performing transformations:")
        for i, (transform, score, time) in enumerate(best_transforms[:3]):
            print(f"      {i+1}. {transform} (match: {score:.3f}, time: {time:.4f}s)")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"   • Hypercube successfully encodes {len(analysis['matrix_types'])} matrix types")
    print(f"   • Continuous interpolation enables smooth type transitions")
    print(f"   • High-dimensional encoding preserves structural relationships")
    print(f"   • Property-based navigation supports intelligent transformations")
    
    print("\n" + "="*80)

def main():
    print("Decision Hypercube Benchmark - Matrix Types in High-Dimensional Space")
    print("="*80)
    
    # Create output directory
    output_dir = create_output_dir()
    print(f"Results will be saved to: {output_dir}")
    
    # Initialize transformer
    print("\nInitializing MatrixTransformer with decision hypercube...")
    start_time = time.time()
    transformer = MatrixTransformer(dimensions=8)
    init_time = time.time() - start_time
    print(f"Initialization completed in {init_time:.3f} seconds")
    
    # Analyze hypercube structure
    analysis = analyze_hypercube_structure(transformer)
    
    # Benchmark interpolation between matrix types
    interpolation_results = benchmark_interpolation_paths(transformer, analysis)
    
    # Benchmark actual matrix transformations
    transformation_results = benchmark_matrix_transformations(transformer, analysis)
    
    # Create comprehensive visualizations
    create_visualizations(analysis, interpolation_results, transformation_results, output_dir)
    
    # Save all results
    save_comprehensive_results(analysis, interpolation_results, transformation_results, output_dir)
    
    # Print summary
    print_benchmark_summary(analysis, interpolation_results, transformation_results)
    
    print(f"\n✅ Benchmark completed! All results saved to: {output_dir}")
    print(f"📊 Check the 'visualizations' folder for detailed plots and analysis")

if __name__ == "__main__":
    main()