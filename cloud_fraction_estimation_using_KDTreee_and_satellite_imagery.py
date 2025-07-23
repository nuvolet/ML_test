#!/usr/bin/env python3
"""
Cloud Fraction Estimation using KDTree and Satellite Imagery
===========================================================

This script demonstrates using sklearn.neighbors.KDTree to estimate cloud fraction
values for new pixels based on similar spectral characteristics from satellite data.

The approach uses k-nearest neighbors in spectral space to interpolate cloud fraction
values based on known reference points.

Author: Satellite Data Processing Team
Date: 2025-06-24

Key Features of this Implementation:

1. Spectral Feature Engineering:
Multiple satellite bands (visible, NIR, SWIR, thermal)
Calculated spectral indices (NDVI, NDSI, cloud indices)
Feature scaling for improved distance calculations
2. KDTree Applications:
Distance-based similarity: Finding spectrally similar pixels
Multiple estimation methods: Weighted average, simple average, median
Configurable k-values: Testing different numbers of neighbors
3. Cloud-Specific Considerations:
Physical relationships: Using known cloud spectral signatures
Spatial context: Maintaining image structure
Validation metrics: Cloud-relevant evaluation measures
4. Real-World Applicability:
Modular design: Easy integration with real satellite data
Scalable approach: Efficient for large imagery
Quality assessment: Comprehensive evaluation framework
This approach is particularly useful for:

Gap-filling in cloud products
Sensor fusion between different satellites
Quality improvement of existing cloud fraction products
Rapid processing of new satellite imagery

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import rasterio
from rasterio.plot import show
import warnings
warnings.filterwarnings('ignore')

class SatelliteCloudProcessor:
    """Process satellite imagery for cloud fraction estimation"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.kdtree = None
        self.reference_cloud_fractions = None
        
    def generate_synthetic_satellite_data(self, width=100, height=100, n_bands=6):
        """
        Generate synthetic satellite imagery with multiple spectral bands
        Simulates MODIS-like data with visible, near-infrared, and thermal bands
        """
        np.random.seed(42)
        
        # Create coordinate grids
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Generate synthetic spectral bands
        bands = {}
        
        # Band 1: Red (0.620-0.670 μm) - sensitive to clouds
        bands['red'] = (
            0.3 + 0.4 * np.sin(5 * X) * np.cos(5 * Y) + 
            0.2 * np.random.random((height, width))
        )
        
        # Band 2: Near-Infrared (0.841-0.876 μm) - vegetation/cloud discrimination
        bands['nir'] = (
            0.4 + 0.3 * np.cos(4 * X) * np.sin(4 * Y) + 
            0.15 * np.random.random((height, width))
        )
        
        # Band 3: Blue (0.459-0.479 μm) - atmospheric scattering
        bands['blue'] = (
            0.2 + 0.3 * (X + Y) / 2 + 
            0.1 * np.random.random((height, width))
        )
        
        # Band 4: Green (0.545-0.565 μm)
        bands['green'] = (
            0.25 + 0.35 * np.sin(3 * X) + 
            0.1 * np.random.random((height, width))
        )
        
        # Band 5: SWIR (1.628-1.652 μm) - cloud ice/water discrimination
        bands['swir1'] = (
            0.15 + 0.25 * np.cos(6 * Y) + 
            0.1 * np.random.random((height, width))
        )
        
        # Band 6: Thermal (10.78-11.28 μm) - cloud top temperature
        bands['thermal'] = (
            0.8 - 0.4 * (X**2 + Y**2) + 
            0.1 * np.random.random((height, width))
        )
        
        # Normalize bands to 0-1
        for band_name in bands:
            bands[band_name] = np.clip(bands[band_name], 0, 1)
        
        # Generate cloud fraction based on spectral characteristics
        # High reflectance in visible + low thermal = high cloud fraction
        cloud_fraction = (
            0.6 * (bands['red'] + bands['blue']) / 2 +  # High visible reflectance
            0.3 * (1 - bands['thermal']) +              # Low thermal (cold clouds)
            0.1 * bands['swir1'] +                       # Ice cloud signature
            0.1 * np.random.random((height, width))     # Noise
        )
        cloud_fraction = np.clip(cloud_fraction, 0, 1)
        
        return bands, cloud_fraction
    
    def calculate_spectral_indices(self, bands):
        """
        Calculate spectral indices useful for cloud detection
        """
        indices = {}
        
        # Normalized Difference Vegetation Index (NDVI)
        indices['ndvi'] = (bands['nir'] - bands['red']) / (bands['nir'] + bands['red'] + 1e-8)
        
        # Normalized Difference Snow Index (NDSI) - also detects bright clouds
        indices['ndsi'] = (bands['green'] - bands['swir1']) / (bands['green'] + bands['swir1'] + 1e-8)
        
        # Brightness Temperature Difference (BTD)
        indices['btd'] = bands['thermal'] - 0.5  # Simplified BTD
        
        # Simple Cloud Index (high visible reflectance)
        indices['cloud_index'] = (bands['red'] + bands['blue'] + bands['green']) / 3
        
        # Cirrus Detection (SWIR-based)
        indices['cirrus'] = bands['swir1'] - bands['nir']
        
        return indices
    
    def prepare_feature_vectors(self, bands, indices):
        """
        Prepare feature vectors for KDTree from spectral bands and indices
        """
        height, width = bands['red'].shape
        
        # Flatten all bands and indices
        features = []
        
        # Add spectral bands
        for band_name, band_data in bands.items():
            features.append(band_data.flatten())
        
        # Add spectral indices
        for index_name, index_data in indices.items():
            features.append(index_data.flatten())
        
        # Stack features (n_pixels, n_features)
        feature_matrix = np.column_stack(features)
        
        return feature_matrix
    
    def create_reference_dataset(self, bands, cloud_fraction, sample_fraction=0.1):
        """
        Create reference dataset with known cloud fraction values
        Simulates ground truth from higher quality retrievals or manual analysis
        """
        height, width = bands['red'].shape
        total_pixels = height * width
        n_samples = int(total_pixels * sample_fraction)
        
        # Randomly sample pixels for reference dataset
        np.random.seed(123)
        sample_indices = np.random.choice(total_pixels, n_samples, replace=False)
        
        # Calculate spectral indices
        indices = self.calculate_spectral_indices(bands)
        
        # Prepare feature vectors
        feature_matrix = self.prepare_feature_vectors(bands, indices)
        
        # Extract reference features and cloud fractions
        reference_features = feature_matrix[sample_indices]
        reference_cloud_fractions = cloud_fraction.flatten()[sample_indices]
        
        return reference_features, reference_cloud_fractions, sample_indices
    
    def build_kdtree_model(self, reference_features, reference_cloud_fractions):
        """
        Build KDTree model for cloud fraction estimation
        """
        # Scale features for better distance calculations
        reference_features_scaled = self.scaler.fit_transform(reference_features)
        
        # Build KDTree
        self.kdtree = KDTree(reference_features_scaled, metric='euclidean')
        self.reference_cloud_fractions = reference_cloud_fractions
        
        print(f"KDTree built with {len(reference_features)} reference points")
        print(f"Feature dimensions: {reference_features_scaled.shape[1]}")
        
    def estimate_cloud_fraction(self, query_features, k=5, method='weighted_average'):
        """
        Estimate cloud fraction for new pixels using KDTree
        
        Parameters:
        -----------
        query_features : array-like, shape (n_pixels, n_features)
            Features for pixels where cloud fraction is to be estimated
        k : int, default=5
            Number of nearest neighbors to use
        method : str, default='weighted_average'
            Method for combining neighbor values ('weighted_average', 'simple_average', 'median')
        """
        if self.kdtree is None:
            raise ValueError("KDTree model not built. Call build_kdtree_model first.")
        
        # Scale query features using the same scaler
        query_features_scaled = self.scaler.transform(query_features)
        
        # Find k nearest neighbors
        distances, indices = self.kdtree.query(query_features_scaled, k=k)
        
        # Get cloud fractions of nearest neighbors
        neighbor_cloud_fractions = self.reference_cloud_fractions[indices]
        
        # Estimate cloud fraction based on method
        if method == 'weighted_average':
            # Use inverse distance weighting
            weights = 1 / (distances + 1e-8)  # Add small epsilon to avoid division by zero
            weights = weights / weights.sum(axis=1, keepdims=True)  # Normalize weights
            estimated_cloud_fraction = np.sum(neighbor_cloud_fractions * weights, axis=1)
            
        elif method == 'simple_average':
            estimated_cloud_fraction = np.mean(neighbor_cloud_fractions, axis=1)
            
        elif method == 'median':
            estimated_cloud_fraction = np.median(neighbor_cloud_fractions, axis=1)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return estimated_cloud_fraction, distances, indices

class CloudFractionEvaluator:
    """Evaluate cloud fraction estimation performance"""
    
    def __init__(self):
        pass
    
    def calculate_metrics(self, true_values, predicted_values):
        """Calculate evaluation metrics"""
        mse = mean_squared_error(true_values, predicted_values)
        mae = mean_absolute_error(true_values, predicted_values)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_values, predicted_values)
        
        # Cloud-specific metrics
        # Bias in cloud fraction
        bias = np.mean(predicted_values - true_values)
        
        # Correlation coefficient
        correlation = np.corrcoef(true_values, predicted_values)[0, 1]
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'Bias': bias,
            'Correlation': correlation
        }
        
        return metrics
    
    def plot_results(self, true_values, predicted_values, metrics, method_name="KDTree"):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Scatter plot
        axes[0, 0].scatter(true_values, predicted_values, alpha=0.6, s=1)
        axes[0, 0].plot([0, 1], [0, 1], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Cloud Fraction')
        axes[0, 0].set_ylabel('Predicted Cloud Fraction')
        axes[0, 0].set_title(f'{method_name} - Scatter Plot')
        axes[0, 0].grid(True)
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].set_ylim(0, 1)
        
        # Residuals
        residuals = predicted_values - true_values
        axes[0, 1].scatter(true_values, residuals, alpha=0.6, s=1)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('True Cloud Fraction')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title(f'{method_name} - Residuals')
        axes[0, 1].grid(True)
        
        # Histograms
        axes[1, 0].hist(true_values, bins=50, alpha=0.7, label='True', density=True)
        axes[1, 0].hist(predicted_values, bins=50, alpha=0.7, label='Predicted', density=True)
        axes[1, 0].set_xlabel('Cloud Fraction')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Distribution Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Metrics text
        axes[1, 1].axis('off')
        metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        axes[1, 1].text(0.1, 0.7, f'{method_name} Metrics:\n\n{metrics_text}', 
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    def plot_spatial_comparison(self, true_image, predicted_image, title="Cloud Fraction Comparison"):
        """Plot spatial comparison of true vs predicted cloud fraction"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # True cloud fraction
        im1 = axes[0].imshow(true_image, cmap='Blues', vmin=0, vmax=1)
        axes[0].set_title('True Cloud Fraction')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Predicted cloud fraction
        im2 = axes[1].imshow(predicted_image, cmap='Blues', vmin=0, vmax=1)
        axes[1].set_title('Predicted Cloud Fraction')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Difference
        diff = predicted_image - true_image
        im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        axes[2].set_title('Difference (Pred - True)')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()

def demonstrate_kdtree_cloud_estimation():
    """
    Main demonstration function
    """
    print("🛰️  Cloud Fraction Estimation using KDTree")
    print("=" * 50)
    
    # Initialize processor
    processor = SatelliteCloudProcessor()
    evaluator = CloudFractionEvaluator()
    
    # Generate synthetic satellite data
    print("📡 Generating synthetic satellite imagery...")
    bands, true_cloud_fraction = processor.generate_synthetic_satellite_data(
        width=150, height=150, n_bands=6
    )
    
    print(f"Image dimensions: {true_cloud_fraction.shape}")
    print(f"Cloud fraction range: {true_cloud_fraction.min():.3f} - {true_cloud_fraction.max():.3f}")
    print(f"Mean cloud fraction: {true_cloud_fraction.mean():.3f}")
    
    # Display sample bands
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    band_names = list(bands.keys())
    
    for i, (band_name, band_data) in enumerate(bands.items()):
        row, col = i // 3, i % 3
        im = axes[row, col].imshow(band_data, cmap='viridis')
        axes[row, col].set_title(f'{band_name.upper()} Band')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    # Create reference dataset (simulate ground truth samples)
    print("\n🎯 Creating reference dataset...")
    reference_features, reference_cloud_fractions, sample_indices = processor.create_reference_dataset(
        bands, true_cloud_fraction, sample_fraction=0.15
    )
    
    print(f"Reference dataset size: {len(reference_features)} pixels")
    print(f"Feature dimensions: {reference_features.shape[1]}")
    
    # Build KDTree model
    print("\n🌳 Building KDTree model...")
    processor.build_kdtree_model(reference_features, reference_cloud_fractions)
    
    # Prepare query dataset (all remaining pixels)
    indices = self.calculate_spectral_indices(bands)
    all_features = processor.prepare_feature_vectors(bands, indices)
    
    # Create mask for non-reference pixels
    all_indices = np.arange(len(all_features))
    query_mask = ~np.isin(all_indices, sample_indices)
    query_features = all_features[query_mask]
    query_true_values = true_cloud_fraction.flatten()[query_mask]
    
    print(f"Query dataset size: {len(query_features)} pixels")
    
    # Test different methods and k values
    methods = ['weighted_average', 'simple_average', 'median']
    k_values = [3, 5, 7, 10]
    
    results = {}
    
    print("\n🔍 Testing different KDTree configurations...")
    
    for method in methods:
        for k in k_values:
            print(f"  Testing {method} with k={k}...")
            
            # Estimate cloud fraction
            predicted_values, distances, neighbor_indices = processor.estimate_cloud_fraction(
                query_features, k=k, method=method
            )
            
            # Calculate metrics
            metrics = evaluator.calculate_metrics(query_true_values, predicted_values)
            
            # Store results
            config_name = f"{method}_k{k}"
            results[config_name] = {
                'predicted': predicted_values,
                'metrics': metrics,
                'method': method,
                'k': k
            }
            
            print(f"    R² = {metrics['R²']:.4f}, RMSE = {metrics['RMSE']:.4f}")
    
    # Find best configuration
    best_config = max(results.keys(), key=lambda x: results[x]['metrics']['R²'])
    best_result = results[best_config]
    
    print(f"\n🏆 Best configuration: {best_config}")
    print(f"   R² = {best_result['metrics']['R²']:.4f}")
    print(f"   RMSE = {best_result['metrics']['RMSE']:.4f}")
    
    # Plot results for best configuration
    evaluator.plot_results(
        query_true_values, 
        best_result['predicted'], 
        best_result['metrics'],
        f"KDTree ({best_config})"
    )
    
    # Create full prediction image
    print("\n🗺️  Creating full prediction map...")
    full_predicted = np.zeros(len(all_features))
    full_predicted[sample_indices] = reference_cloud_fractions  # Use reference values
    full_predicted[query_mask] = best_result['predicted']  # Use predicted values
    
    # Reshape to image dimensions
    height, width = true_cloud_fraction.shape
    predicted_image = full_predicted.reshape(height, width)
    
    # Plot spatial comparison
    evaluator.plot_spatial_comparison(
        true_cloud_fraction, 
        predicted_image,
        f"KDTree Cloud Fraction Estimation ({best_config})"
    )
    
    # Compare all methods
    print("\n📊 Method Comparison:")
    print("-" * 60)
    comparison_data = []
    
    for config_name, result in results.items():
        comparison_data.append({
            'Configuration': config_name,
            'R²': result['metrics']['R²'],
            'RMSE': result['metrics']['RMSE'],
            'MAE': result['metrics']['MAE'],
            'Bias': result['metrics']['Bias']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('R²', ascending=False)
    print(comparison_df.round(4))
    
    # Analyze neighbor distances
    print(f"\n📏 Analyzing neighbor distances for best method...")
    _, distances, _ = processor.estimate_cloud_fraction(
        query_features[:1000],  # Sample for analysis
        k=best_result['k'], 
        method=best_result['method']
    )
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(distances.flatten(), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Distance to Nearest Neighbors')
    plt.ylabel('Frequency')
    plt.title('Distribution of Neighbor Distances')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([distances[:, i] for i in range(distances.shape[1])], 
                labels=[f'k={i+1}' for i in range(distances.shape[1])])
    plt.xlabel('Neighbor Rank')
    plt.ylabel('Distance')
    plt.title('Distance by Neighbor Rank')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\n✅ Cloud fraction estimation complete!")
    print(f"   Best method achieved R² = {best_result['metrics']['R²']:.4f}")
    print(f"   Mean absolute error: {best_result['metrics']['MAE']:.4f}")

if __name__ == "__main__":
    # Run the demonstration
    demonstrate_kdtree_cloud_estimation()