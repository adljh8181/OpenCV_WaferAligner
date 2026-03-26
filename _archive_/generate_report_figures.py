"""Generate comparison figures for the IMRAD report.
Runs old method and new method on selected images, saves comparison plots."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_compare import OldEdgeDetector
from edge_finder import EdgeLineFinder, EdgeFinderConfig
from fov_classifier import FOVClassifier, ClassificationConfig, preprocess_image

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report_figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_FOLDER = r'C:\Users\adrain.lim\OneDrive - QES (Asia-Pacific) Sdn Bhd\Desktop\EmguFindEdge\Images\LEFT_EDGE_PNG'

def save_old_method_failures():
    """Generate figures showing old method failures."""
    old = OldEdgeDetector()
    
    # === Figure 1: Threshold sensitivity ===
    # Show how different thresholds give different binarizations
    img_path = os.path.join(IMAGE_FOLDER, 'KW_Wafer4.png')
    if not os.path.exists(img_path):
        img_path = os.path.join(IMAGE_FOLDER, 'Copy of LeftEdge.png')
    
    img_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_raw is None:
        print(f"Cannot load {img_path}")
        return
    
    # Resize for consistency
    img, _, scale = preprocess_image(img_path, 1024)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Issue 1: Threshold Sensitivity in Old Method (Binarization)', 
                 fontsize=14, fontweight='bold')
    
    thresholds = [80, 120, 150, 180, 200, 230]
    for idx, thresh in enumerate(thresholds):
        row, col = idx // 3, idx % 3
        result = old.detect_edge(img, threshold=thresh, direction="LEFT")
        
        # Show binary image with detected line
        binary = result['binary']
        display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        if result['success']:
            ep = result['line_endpoints']
            h = img.shape[0]
            cv2.line(display, (ep['x_top'], 0), (ep['x_bot'], h), (0, 255, 0), 2)
            title = f"Threshold={thresh}\nLine: {ep['x_top']}→{ep['x_bot']}"
        else:
            title = f"Threshold={thresh}\nFAILED: {result['reason']}"
        
        axes[row][col].imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        axes[row][col].set_title(title, fontsize=10)
        axes[row][col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_threshold_sensitivity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig1_threshold_sensitivity.png")
    
    # === Figure 2: Outlier sensitivity (dust/pattern) ===
    # Show old method vs new method on a noisy/patterned image
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Issue 2: Outlier Sensitivity — Old Method Uses ALL Edge Points', 
                 fontsize=14, fontweight='bold')
    
    result_old = old.detect_edge(img, threshold=150, direction="LEFT")
    
    # Show: original, canny edges (all points), result line
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image', fontsize=11)
    axes[0].axis('off')
    
    if result_old['success']:
        edges = result_old['edges']
        axes[1].imshow(edges, cmap='gray')
        axes[1].set_title(f"Canny Edges\n({result_old['num_points']} points — ALL used for fitting)", fontsize=10)
        axes[1].axis('off')
        
        display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ep = result_old['line_endpoints']
        h = img.shape[0]
        cv2.line(display, (ep['x_top'], 0), (ep['x_bot'], h), (0, 255, 0), 2)
        
        # Mark ALL edge points
        for pt in result_old['detected_points'][:500]:
            cv2.circle(display, (int(pt[0]), int(pt[1])), 1, (0, 0, 255), -1)
        
        axes[2].imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"fitLine Result (Least Squares)\nLine pulled by ALL {result_old['num_points']} points", fontsize=10)
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_outlier_sensitivity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig2_outlier_sensitivity.png")
    

def save_new_method_comparison():
    """Generate figures showing new method advantages."""
    old = OldEdgeDetector()
    
    # === Figure 3: RANSAC vs fitLine ===
    img_path = os.path.join(IMAGE_FOLDER, 'Copy of LeftEdge.png')
    if not os.path.exists(img_path):
        img_path = os.path.join(IMAGE_FOLDER, 'KW_Wafer2.png')
    
    img, _, scale = preprocess_image(img_path, 1024)
    
    # Old method
    result_old = old.detect_edge(img, threshold=150, direction="LEFT")
    
    # New method 
    cfg = EdgeFinderConfig()
    cfg.SCAN_DIRECTION = "LEFT"
    finder = EdgeLineFinder(cfg)
    result_new = finder.find_edge(img, skip_classification=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('New Method: Gradient Peak Detection + RANSAC Line Fitting', 
                 fontsize=14, fontweight='bold')
    
    # Original
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image', fontsize=11)
    axes[0].axis('off')
    
    # Old method result
    if result_old['success']:
        display_old = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ep = result_old['line_endpoints']
        h = img.shape[0]
        cv2.line(display_old, (ep['x_top'], 0), (ep['x_bot'], h), (0, 0, 255), 2)
        for pt in result_old['detected_points'][:500]:
            cv2.circle(display_old, (int(pt[0]), int(pt[1])), 1, (0, 0, 255), -1)
        axes[1].imshow(cv2.cvtColor(display_old, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Old: Canny + fitLine\n{result_old['num_points']} pts, angle={result_old['angle']:.2f}°", fontsize=10)
    axes[1].axis('off')
    
    # New method result
    if result_new['success']:
        display_new = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ep = result_new['line_endpoints']
        h = img.shape[0]
        cv2.line(display_new, (ep['x_top'], 0), (ep['x_bot'], h), (0, 255, 0), 2)
        for pt in result_new['detected_points']:
            cv2.circle(display_new, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
        for pt in result_new['inliers']:
            cv2.circle(display_new, (int(pt[0]), int(pt[1])), 5, (255, 255, 0), 2)
        axes[2].imshow(cv2.cvtColor(display_new, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"New: Gradient + RANSAC\n{result_new['num_inliers']}/{result_new['num_points']} inliers", fontsize=10)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_new_method_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig3_new_method_comparison.png")


def save_fov_classification():
    """Generate figure showing FOV classification results."""
    classifier = FOVClassifier(ClassificationConfig())
    
    # Test images: die vs edge
    test_cases = [
        (os.path.join(IMAGE_FOLDER, 'Copy of sample_wafer.jpg'), 'Die Image (Pure Die)'),
        (os.path.join(IMAGE_FOLDER, 'Copy of LeftEdge.png'), 'Edge Image (Die + Edge)'),
        (os.path.join(IMAGE_FOLDER, 'KW_Wafer2.png'), 'Edge Image (Wafer + Edge)'),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('FOV Classification: Differentiating Die Region vs Wafer Edge', 
                 fontsize=14, fontweight='bold')
    
    for idx, (path, label) in enumerate(test_cases):
        if not os.path.exists(path):
            continue
        img, _, _ = preprocess_image(path, 1024)
        result = classifier.classify(img)
        
        display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        fov = result['fov_type']
        conf = result['confidence']
        
        if fov == 'DIE_FOV':
            color = (0, 0, 255)  # Red border
            cv2.rectangle(display, (5, 5), (img.shape[1]-5, img.shape[0]-5), color, 4)
        else:
            color = (0, 255, 0)  # Green border
            cv2.rectangle(display, (5, 5), (img.shape[1]-5, img.shape[0]-5), color, 4)
        
        axes[idx].imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(f"{label}\n→ {fov} ({conf:.0%})", fontsize=10, 
                           color='red' if fov == 'DIE_FOV' else 'green',
                           fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_fov_classification.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig4_fov_classification.png")


def save_wafer_centering_impact():
    """Generate figure showing how edge mis-detection affects centering."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Impact on Wafer Centering: Incorrect Edge → Offset Error', 
                 fontsize=14, fontweight='bold')
    
    # Diagram 1: Correct edge → correct center
    ax = axes[0]
    circle = plt.Circle((0.5, 0.5), 0.4, fill=False, color='blue', linewidth=2)
    ax.add_patch(circle)
    ax.axvline(x=0.12, color='green', linewidth=3, label='Correct Edge')
    ax.plot(0.5, 0.5, 'g+', markersize=20, markeredgewidth=3, label='Correct Center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('✓ Correct Edge Detection\n→ Accurate Wafer Center', fontsize=11, color='green')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    # Diagram 2: Wrong edge → wrong center
    ax = axes[1]
    circle = plt.Circle((0.5, 0.5), 0.4, fill=False, color='blue', linewidth=2)
    ax.add_patch(circle)
    ax.axvline(x=0.12, color='green', linewidth=2, linestyle='--', alpha=0.5, label='True Edge')
    ax.axvline(x=0.25, color='red', linewidth=3, label='Wrong Edge (offset)')
    ax.plot(0.5, 0.5, 'g+', markersize=15, markeredgewidth=2, alpha=0.5, label='True Center')
    ax.plot(0.57, 0.5, 'rx', markersize=20, markeredgewidth=3, label='Wrong Center')
    
    # Error arrow
    ax.annotate('', xy=(0.57, 0.45), xytext=(0.5, 0.45),
               arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(0.535, 0.42, 'Error', fontsize=10, color='red', ha='center', fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('✗ Incorrect Edge Detection\n→ Wafer Center Offset', fontsize=11, color='red')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_centering_impact.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig5_centering_impact.png")


if __name__ == '__main__':
    print("Generating report figures...")
    save_old_method_failures()
    save_new_method_comparison()
    save_fov_classification()
    save_wafer_centering_impact()
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
