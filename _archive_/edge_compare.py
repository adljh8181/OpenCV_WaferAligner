"""
================================================================================
EDGE DETECTION METHOD COMPARISON TOOL
================================================================================
Compare OLD (Threshold + Canny + fitLine) vs NEW (Gradient + RANSAC) methods
for wafer edge detection side by side.

Usage:
    python edge_compare.py                    # Use default image
    python edge_compare.py path/to/image.png  # Use custom image

Author: Adrain Lim
================================================================================
"""

import sys
import os

# Add parent directory to path to allow importing from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import tkinter as tk
from tkinter import filedialog

from app.services.fov_classifier import FOVClassifier, ClassificationConfig, create_gradient_kernel, preprocess_image
from app.services.edge_finder import EdgeLineFinder, EdgeFinderConfig


# ==============================================================================
#                    OLD METHOD: Threshold + Canny + fitLine
# ==============================================================================

class OldEdgeDetector:
    """
    OLD edge detection method using:
    1. Gaussian blur
    2. Binary threshold
    3. Morphological closing
    4. Canny edge detection
    5. cv2.fitLine (least squares)
    """
    
    def detect_edge(self, image, threshold=150, direction="LEFT"):
        """
        Detect edge using old pipeline.
        
        Args:
            image: Grayscale image (numpy array)
            threshold: Binary threshold value
            direction: "LEFT", "RIGHT", "TOP", "BOTTOM"
            
        Returns:
            dict with detection results
        """
        h, w = image.shape[:2]
        is_vertical = direction in ["LEFT", "RIGHT"]
        
        # Step 1: Gaussian smoothing (kernel size = 3% of width)
        gauss_k = int(0.03 * w)
        if gauss_k % 2 == 0:
            gauss_k += 1
        gauss_k = max(3, gauss_k)
        blurred = cv2.GaussianBlur(image, (gauss_k, gauss_k), 0)
        
        # Step 2: Binary threshold
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        
        # Step 3: Morphological closing
        close_k = int(0.048 * w)
        close_k = max(3, close_k)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Step 4: Canny edge detection
        edges = cv2.Canny(morphed, 30, 100)
        
        # Step 5: Extract edge points
        y_coords, x_coords = np.where(edges == 255)
        
        if len(x_coords) < 3:
            return {
                'success': False,
                'reason': 'Not enough edge points',
                'num_points': len(x_coords),
                'edges': edges,
                'binary': binary,
                'morphed': morphed
            }
        
        edge_points = np.column_stack((x_coords, y_coords)).astype(np.float32)
        
        # Step 6: Fit line using cv2.fitLine (least squares)
        edge_points_reshaped = edge_points.reshape(-1, 1, 2)
        line_params = cv2.fitLine(edge_points_reshaped, cv2.DIST_L2, 0, 0.01, 0.01)
        
        vx = float(line_params[0][0])
        vy = float(line_params[1][0])
        x0 = float(line_params[2][0])
        y0 = float(line_params[3][0])
        
        # Calculate endpoints
        if is_vertical:
            if abs(vy) < 1e-5:
                x_top = int(x0)
                x_bot = int(x0)
            else:
                x_top = int(x0 + (0 - y0) * (vx / vy))
                x_bot = int(x0 + (h - y0) * (vx / vy))
            
            angle = np.degrees(np.arctan2(vx, vy))
            endpoints = {'x_top': x_top, 'x_bot': x_bot}
        else:
            if abs(vx) < 1e-5:
                y_left = int(y0)
                y_right = int(y0)
            else:
                y_left = int(y0 + (0 - x0) * (vy / vx))
                y_right = int(y0 + (w - x0) * (vy / vx))
            
            angle = np.degrees(np.arctan2(vy, vx))
            endpoints = {'y_left': y_left, 'y_right': y_right}
        
        return {
            'success': True,
            'line_params': {'vx': vx, 'vy': vy, 'x0': x0, 'y0': y0},
            'line_endpoints': endpoints,
            'detected_points': list(zip(x_coords.tolist(), y_coords.tolist())),
            'num_points': len(x_coords),
            'angle': angle,
            'edges': edges,
            'binary': binary,
            'morphed': morphed,
            'is_vertical_edge': is_vertical
        }


# ==============================================================================
#                    COMPARISON GUI
# ==============================================================================

class EdgeMethodComparator:
    """
    Interactive GUI to compare OLD vs NEW edge detection methods.
    Toggle between methods while keeping the same sliders for tuning.
    """
    
    def __init__(self, image_path=None):
        # Default config
        self.config = EdgeFinderConfig()
        
        # Get image path
        if image_path is None:
            image_path = os.path.join(self.config.IMAGE_FOLDER, 'KW_Wafer4.png')
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        self.image_path = image_path
        
        # Load and preprocess image
        self.img, self.original_img, self.scale = preprocess_image(
            image_path, 
            self.config.TARGET_PROCESS_DIM
        )
        self.h, self.w = self.img.shape
        
        # Old method detector
        self.old_detector = OldEdgeDetector()
        
        # Setup UI
        self.setup_ui()
        
        # Initial update
        self.update(None)
    
    def setup_ui(self):
        """Setup the matplotlib UI"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Edge Detection Method Comparison', fontsize=14, fontweight='bold')
        
        # ==================== LEFT COLUMN: IMAGES + PROFILE ====================
        # Top-left: Detection result image
        self.ax_img = self.fig.add_axes([0.03, 0.55, 0.45, 0.38])
        self.ax_img.set_title("Detection Result")
        self.ax_img.axis('off')
        
        # Middle-left: Pipeline visualization (edge/gradient)
        self.ax_pipeline = self.fig.add_axes([0.03, 0.30, 0.45, 0.22])
        self.ax_pipeline.set_title("Pipeline Visualization")
        self.ax_pipeline.axis('off')
        
        # Bottom-left: Gradient/Edge profile
        self.ax_profile = self.fig.add_axes([0.03, 0.06, 0.45, 0.18])
        self.ax_profile.set_title("Profile")
        
        # ==================== RIGHT COLUMN: CONTROLS + STATUS ====================
        # Status panel (top right)
        self.ax_status = self.fig.add_axes([0.55, 0.55, 0.40, 0.38])
        self.ax_status.axis('off')
        
        # ==================== METHOD TOGGLE ====================
        self.fig.text(0.64, 0.52, 'METHOD', fontsize=10, fontweight='bold', ha='center',
                     color='darkblue')
        method_ax = self.fig.add_axes([0.55, 0.44, 0.18, 0.08])
        method_ax.set_facecolor('#e8e8e8')
        self.method_radio = RadioButtons(method_ax, 
                                          ('NEW (Gradient+RANSAC)', 'OLD (Canny+fitLine)'), 
                                          active=0)
        self.method_radio.on_clicked(self.update)
        
        # ==================== DIRECTION TOGGLE ====================
        self.fig.text(0.86, 0.52, 'DIRECTION', fontsize=10, fontweight='bold', ha='center',
                     color='darkblue')
        dir_ax = self.fig.add_axes([0.78, 0.44, 0.17, 0.08])
        dir_ax.set_facecolor('#e8e8e8')
        self.direction_radio = RadioButtons(dir_ax, ('LEFT', 'RIGHT', 'TOP', 'BOTTOM'), active=0)
        self.direction_radio.on_clicked(self.update)
        
        # ==================== SLIDERS (right column) ====================
        slider_left = 0.62
        slider_width = 0.32
        slider_height = 0.025
        
        self.fig.text(slider_left + 0.16, 0.40, 'PARAMETERS', 
                     fontsize=10, fontweight='bold', ha='center', color='darkgreen')
        
        sliders_config = [
            ('THRESHOLD', 5, 20000, 150, 1, 'Threshold'),
            ('KERNEL_SIZE', 3, 200, self.config.KERNEL_SIZE, 2, 'Kernel Size'),
            ('NUM_REGIONS', 10, 80, self.config.NUM_REGIONS, 5, 'Scan Regions'),
            ('RANSAC_THRESHOLD', 1.0, 20.0, self.config.RANSAC_THRESHOLD, 0.5, 'RANSAC Thresh'),
        ]
        
        self.sliders = {}
        for i, (name, vmin, vmax, vinit, vstep, label) in enumerate(sliders_config):
            y_pos = 0.34 - i * 0.06
            ax = self.fig.add_axes([slider_left, y_pos, slider_width, slider_height])
            color = 'lightblue' if i == 0 else 'lightgreen'
            slider = Slider(ax, label, vmin, vmax, valinit=vinit, valstep=vstep, color=color)
            slider.on_changed(self.update)
            self.sliders[name] = slider
        
        # ==================== BUTTONS ====================
        load_ax = self.fig.add_axes([slider_left, 0.06, 0.12, 0.04])
        self.load_btn = Button(load_ax, 'Load Image', color='lightblue', hovercolor='skyblue')
        self.load_btn.on_clicked(self.load_image)
        
        reset_ax = self.fig.add_axes([slider_left + 0.16, 0.06, 0.12, 0.04])
        self.reset_btn = Button(reset_ax, 'Reset', color='lightcoral', hovercolor='salmon')
        self.reset_btn.on_clicked(self.reset)
    
    def update(self, val):
        """Update visualization based on current settings"""
        method = self.method_radio.value_selected
        direction = self.direction_radio.value_selected
        is_vertical = direction in ["LEFT", "RIGHT"]
        
        # Clear all axes
        self.ax_img.clear()
        self.ax_pipeline.clear()
        self.ax_profile.clear()
        self.ax_status.clear()
        self.ax_status.axis('off')
        
        if 'OLD' in method:
            self._run_old_method(direction, is_vertical)
        else:
            self._run_new_method(direction, is_vertical)
        
        self.ax_img.axis('off')
        self.ax_pipeline.axis('off')
        self.fig.canvas.draw_idle()
    
    def _run_new_method(self, direction, is_vertical):
        """Run the NEW gradient + RANSAC method"""
        kernel_size = int(self.sliders['KERNEL_SIZE'].val)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Build config from sliders
        class TunerConfig(EdgeFinderConfig):
            KERNEL_SIZE = kernel_size
            EDGE_THRESHOLD = int(self.sliders['THRESHOLD'].val)
            NUM_REGIONS = int(self.sliders['NUM_REGIONS'].val)
            RANSAC_THRESHOLD = self.sliders['RANSAC_THRESHOLD'].val
            SCAN_DIRECTION = direction
        
        config = TunerConfig()
        
        # Run edge finder
        finder = EdgeLineFinder(config)
        result = finder.find_edge(self.img, skip_classification=True)
        
        # ---- Draw result image ----
        result_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        
        if result['success']:
            endpoints = result['line_endpoints']
            
            if is_vertical:
                cv2.line(result_img, (endpoints['x_top'], 0), (endpoints['x_bot'], self.h),
                        (0, 255, 0), 3)
            else:
                cv2.line(result_img, (0, endpoints['y_left']), (self.w, endpoints['y_right']),
                        (0, 255, 0), 3)
            
            # Draw detected points (red) and inliers (cyan)
            for point in result['detected_points']:
                cv2.circle(result_img, (int(point[0]), int(point[1])), 4, (0, 0, 255), -1)
            for point in result['inliers']:
                cv2.circle(result_img, (int(point[0]), int(point[1])), 6, (255, 255, 0), 2)
        
        self.ax_img.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        
        if result['success']:
            self.ax_img.set_title(
                f"NEW METHOD | ✓ {direction} EDGE | {result['num_inliers']}/{result['num_points']} inliers",
                fontsize=11, fontweight='bold', color='green')
        else:
            self.ax_img.set_title(
                f"NEW METHOD | ✗ FAILED: {result.get('reason', 'Unknown')}",
                fontsize=11, fontweight='bold', color='red')
        
        # ---- Draw gradient visualization ----
        # Show gradient magnitude image
        kernel = create_gradient_kernel(kernel_size)
        if is_vertical:
            profile = np.median(self.img, axis=0).astype(np.float64)
        else:
            profile = np.median(self.img, axis=1).astype(np.float64)
        
        gradient = np.convolve(profile, kernel, mode='same')
        abs_gradient = np.abs(gradient)
        
        # Create gradient image for display
        grad_x = cv2.filter2D(self.img.astype(np.float64), -1, kernel.reshape(1, -1) if is_vertical else kernel.reshape(-1, 1))
        grad_display = np.abs(grad_x)
        grad_display = (grad_display / grad_display.max() * 255).astype(np.uint8) if grad_display.max() > 0 else grad_display.astype(np.uint8)
        self.ax_pipeline.imshow(grad_display, cmap='hot')
        self.ax_pipeline.set_title("Gradient Magnitude (NEW)", fontsize=11, fontweight='bold')
        
        # Clean borders (ignore image edges where gradient is artificially high due to padding)
        border_ignore = int(len(abs_gradient) * config.BORDER_IGNORE_PCT) + config.KERNEL_SIZE
        abs_gradient[:border_ignore] = 0
        abs_gradient[-border_ignore:] = 0
        
        # ---- Draw profile ----
        self.ax_profile.plot(abs_gradient, color='green', linewidth=1.5, label='Gradient')
        self.ax_profile.fill_between(range(len(abs_gradient)), abs_gradient, alpha=0.3, color='green')
        self.ax_profile.axhline(config.EDGE_THRESHOLD, color='orange', linestyle='--', linewidth=2,
                                label=f'Threshold = {config.EDGE_THRESHOLD}')
        self.ax_profile.legend(loc='upper right', fontsize=8)
        self.ax_profile.set_xlabel("Position")
        self.ax_profile.set_ylabel("Gradient Magnitude")
        self.ax_profile.set_title("Gradient Profile (median across image)", fontsize=10)
        self.ax_profile.grid(True, alpha=0.3)
        
        # ---- Status ----
        if result['success']:
            lp = result['line_params']
            angle = np.degrees(np.arctan2(lp['vx'], lp['vy']))
            summary = (f"METHOD: NEW (Gradient + RANSAC)\n"
                      f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                      f"Status: ✓ SUCCESS\n"
                      f"Direction: {direction}\n"
                      f"Points: {result['num_points']}\n"
                      f"Inliers: {result['num_inliers']}\n"
                      f"Angle: {angle:.3f}°\n"
                      f"Sub-pixel: YES")
            color = 'lightgreen'
        else:
            summary = (f"METHOD: NEW (Gradient + RANSAC)\n"
                      f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                      f"Status: ✗ FAILED\n"
                      f"Reason: {result.get('reason', 'Unknown')}\n"
                      f"Points: {result.get('num_points', 0)}")
            color = '#ffcccc'
        
        self.ax_status.text(0.5, 0.5, summary, transform=self.ax_status.transAxes,
                           fontsize=10, fontfamily='monospace',
                           verticalalignment='center', horizontalalignment='center',
                           bbox=dict(boxstyle='round', facecolor=color, alpha=0.9))
    
    def _run_old_method(self, direction, is_vertical):
        """Run the OLD threshold + Canny + fitLine method"""
        threshold = int(self.sliders['THRESHOLD'].val)
        
        # Run old detector
        result = self.old_detector.detect_edge(self.img, threshold=threshold, direction=direction)
        
        # ---- Draw result image ----
        result_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        
        if result['success']:
            endpoints = result['line_endpoints']
            
            if is_vertical:
                cv2.line(result_img, (endpoints['x_top'], 0), (endpoints['x_bot'], self.h),
                        (0, 255, 0), 3)
            else:
                cv2.line(result_img, (0, endpoints['y_left']), (self.w, endpoints['y_right']),
                        (0, 255, 0), 3)
            
            # Draw a sample of detected edge points (max 500 to avoid slowdown)
            sample_points = result['detected_points']
            if len(sample_points) > 500:
                indices = np.random.choice(len(sample_points), 500, replace=False)
                sample_points = [sample_points[i] for i in indices]
            
            for x, y in sample_points:
                cv2.circle(result_img, (int(x), int(y)), 1, (0, 0, 255), -1)
        
        self.ax_img.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        
        if result['success']:
            self.ax_img.set_title(
                f"OLD METHOD | ✓ {direction} EDGE | {result['num_points']} edge pixels",
                fontsize=11, fontweight='bold', color='green')
        else:
            self.ax_img.set_title(
                f"OLD METHOD | ✗ FAILED: {result.get('reason', 'Unknown')}",
                fontsize=11, fontweight='bold', color='red')
        
        # ---- Draw pipeline stages ----
        # Show binary threshold image (adjustable via Threshold slider)
        if result.get('binary') is not None:
            self.ax_pipeline.imshow(result['binary'], cmap='gray')
            self.ax_pipeline.set_title(f"Binary Threshold (OLD) | Threshold={threshold}", 
                                       fontsize=11, fontweight='bold')
        
        # ---- Draw edge pixel profile ----
        if result.get('edges') is not None:
            edges = result['edges']
            if is_vertical:
                edge_density = np.sum(edges > 0, axis=0)
                self.ax_profile.plot(edge_density, color='blue', linewidth=1.5, label='Edge Density')
                self.ax_profile.fill_between(range(len(edge_density)), edge_density, alpha=0.3, color='blue')
                self.ax_profile.set_xlabel("X Position")
            else:
                edge_density = np.sum(edges > 0, axis=1)
                self.ax_profile.plot(edge_density, color='blue', linewidth=1.5, label='Edge Density')
                self.ax_profile.fill_between(range(len(edge_density)), edge_density, alpha=0.3, color='blue')
                self.ax_profile.set_xlabel("Y Position")
            
            self.ax_profile.set_ylabel("Edge Pixel Count")
            self.ax_profile.set_title("Edge Pixel Distribution (OLD)", fontsize=10)
            self.ax_profile.legend(loc='upper right', fontsize=8)
            self.ax_profile.grid(True, alpha=0.3)
        
        # ---- Status ----
        if result['success']:
            summary = (f"METHOD: OLD (Canny + fitLine)\n"
                      f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                      f"Status: ✓ SUCCESS\n"
                      f"Direction: {direction}\n"
                      f"Edge Pixels: {result['num_points']}\n"
                      f"Angle: {result['angle']:.3f}°\n"
                      f"Sub-pixel: NO (integer only)\n"
                      f"Line Fit: Least Squares (no RANSAC)")
            color = 'lightyellow'
        else:
            summary = (f"METHOD: OLD (Canny + fitLine)\n"
                      f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                      f"Status: ✗ FAILED\n"
                      f"Reason: {result.get('reason', 'Unknown')}\n"
                      f"Edge Pixels: {result.get('num_points', 0)}")
            color = '#ffcccc'
        
        self.ax_status.text(0.5, 0.5, summary, transform=self.ax_status.transAxes,
                           fontsize=10, fontfamily='monospace',
                           verticalalignment='center', horizontalalignment='center',
                           bbox=dict(boxstyle='round', facecolor=color, alpha=0.9))
    
    def load_image(self, event):
        """Open file dialog to load a new image"""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_path = filedialog.askopenfilename(
            title="Select Wafer Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()
        
        if file_path:
            try:
                self.image_path = file_path
                self.img, self.original_img, self.scale = preprocess_image(
                    file_path, 
                    self.config.TARGET_PROCESS_DIM
                )
                self.h, self.w = self.img.shape
                
                filename = os.path.basename(file_path)
                self.fig.suptitle(f'Edge Detection Comparison - {filename}', 
                                fontsize=14, fontweight='bold')
                
                self.update(None)
                print(f"Loaded image: {file_path}")
            except Exception as e:
                print(f"Error loading image: {e}")
    
    def reset(self, event):
        """Reset all sliders to defaults"""
        default_config = EdgeFinderConfig()
        self.sliders['THRESHOLD'].set_val(150)
        self.sliders['KERNEL_SIZE'].set_val(default_config.KERNEL_SIZE)
        self.sliders['NUM_REGIONS'].set_val(default_config.NUM_REGIONS)
        self.sliders['RANSAC_THRESHOLD'].set_val(default_config.RANSAC_THRESHOLD)
    
    def show(self):
        """Display the interactive tuner"""
        plt.show()


# ==============================================================================
#                           MAIN
# ==============================================================================

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = None
    
    print("=" * 60)
    print("  EDGE DETECTION METHOD COMPARISON TOOL")
    print("=" * 60)
    print()
    print("  Compare two edge detection approaches:")
    print()
    print("  OLD (Canny + fitLine):")
    print("    • Gaussian blur → Binary threshold → Morph close")
    print("    • Canny edge detection → cv2.fitLine (least squares)")
    print("    • Integer pixel accuracy only")
    print()
    print("  NEW (Gradient + RANSAC):")
    print("    • Custom gradient kernel convolution")
    print("    • Region-by-region peak detection + sub-pixel refinement")
    print("    • RANSAC line fitting (robust to outliers)")
    print("    • Sub-pixel accuracy")
    print()
    print("  Toggle between methods using the radio buttons!")
    print("=" * 60)
    
    comparator = EdgeMethodComparator(image_path)
    comparator.show()


if __name__ == "__main__":
    main()
