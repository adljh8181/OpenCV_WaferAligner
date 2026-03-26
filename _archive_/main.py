"""
================================================================================
WAFER EDGE DETECTION PIPELINE - MAIN ENTRY POINT
================================================================================
Run this file to execute the complete pipeline:
1. FOV Classification
2. Edge Line Finding (if Edge FOV detected)

Usage:
    python main.py                          # Use default image
    python main.py path/to/image.png        # Use custom image

Author: Auto-generated for Wafer Alignment System
================================================================================
"""

import sys
import os

from fov_classifier import FOVClassifier, ClassificationConfig
from edge_finder import EdgeLineFinder, EdgeFinderConfig


def run_pipeline(image_path=None):
    """
    Run the complete wafer edge detection pipeline.

    Args:
        image_path: Path to image file. If None, uses default from config.

    Returns:
        dict with combined results from classification and edge finding
    """
    # Create configuration
    config = EdgeFinderConfig()

    # Use default image if not provided
    if image_path is None:
        image_path = os.path.join(config.IMAGE_FOLDER, config.DEFAULT_IMAGE)

    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        return None

    print("=" * 70)
    print("     WAFER EDGE DETECTION PIPELINE")
    print("=" * 70)
    print(f"\n📁 Image: {image_path}\n")

    # === PHASE 1: FOV Classification ===
    print("-" * 70)
    print("PHASE 1: FOV Classification")
    print("-" * 70)

    classifier = FOVClassifier(config)
    classification = classifier.classify(image_path)

    print(f"   FOV Type:    {classification['fov_type']}")
    print(f"   Edge Type:   {classification['edge']['edge_type']}")
    print(f"   Wafer Side:  {classification['edge']['wafer_side']}")
    print(f"   Method:      {classification['edge']['detection_method']}")

    # === PHASE 2: Edge Finding (if Edge FOV) ===
    edge_result = None
    if classification['fov_type'] == 'EDGE_FOV':
        print("\n" + "-" * 70)
        print("PHASE 2: Edge Line Finding")
        print("-" * 70)

        finder = EdgeLineFinder(config)
        edge_result = finder.find_edge(
            classification['image'],
            edge_info=classification['edge']
        )

        if edge_result['success']:
            endpoints = edge_result['line_endpoints']
            line_params = edge_result['line_params']
            import numpy as np
            angle = np.degrees(np.arctan2(line_params['vx'], line_params['vy']))

            print(f"   ✓ Edge line found!")
            print(f"   Points:    {edge_result['num_points']} detected, {edge_result['num_inliers']} inliers")
            print(f"   Line X:    {endpoints['x_top']} (top) → {endpoints['x_bot']} (bottom)")
            print(f"   Angle:     {angle:.2f}°")
        else:
            print(f"   ✗ Edge finding failed: {edge_result.get('reason', 'Unknown')}")
    else:
        print(f"\n   ℹ️  Skipping edge finding (not an Edge FOV)")

    # === Summary ===
    print("\n" + "=" * 70)
    if classification['fov_type'] == 'EDGE_FOV' and edge_result and edge_result['success']:
        print("   ✓ PIPELINE COMPLETE - Edge detected successfully!")
    elif classification['fov_type'] != 'EDGE_FOV':
        print(f"   ℹ️  PIPELINE COMPLETE - FOV classified as {classification['fov_type']}")
    else:
        print("   ⚠️  PIPELINE COMPLETE - Edge FOV detected but line fitting failed")
    print("=" * 70)

    # Combine results
    result = {
        'classification': classification,
        'edge_result': edge_result,
        'image_path': image_path
    }

    return result


def visualize_results(result):
    """Show visualization of results"""
    if result is None:
        return

    classification = result['classification']
    edge_result = result['edge_result']

    if edge_result:
        # Use edge finder visualization (more comprehensive)
        from edge_finder import EdgeLineFinder, EdgeFinderConfig
        finder = EdgeLineFinder(EdgeFinderConfig())
        finder.visualize(edge_result)
    else:
        # Use classifier visualization
        from fov_classifier import FOVClassifier, ClassificationConfig
        classifier = FOVClassifier(ClassificationConfig())
        classifier.visualize(classification)


def main():
    """Main entry point"""
    # Check for command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = None

    # Run pipeline
    result = run_pipeline(image_path)

    # Visualize
    if result:
        visualize_results(result)

    return result


if __name__ == "__main__":
    result = main()
