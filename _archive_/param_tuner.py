"""
================================================================================
INTERACTIVE PARAMETER TUNER v1.2 (Simplified)
================================================================================
Allows real-time adjustment of edge detection parameters with live preview.
Focuses on parameters that have immediate visual impact.

Usage:
    python param_tuner.py                    # Use default image
    python param_tuner.py path/to/image.png  # Use custom image

Author: Auto-generated for Wafer Alignment System
================================================================================
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import tkinter as tk
from tkinter import filedialog

from fov_classifier import FOVClassifier, ClassificationConfig, create_gradient_kernel, preprocess_image
from edge_finder import EdgeLineFinder, EdgeFinderConfig


class InteractiveParameterTuner:
    """
    Interactive GUI for tuning edge detection parameters with live preview.
    Simplified to show only parameters with immediate visual impact.
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
        
        # Create figure
        self.setup_ui()
        
        # Initial update
        self.update(None)
    
    def setup_ui(self):
        """Setup the matplotlib UI with sliders and image display"""
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.suptitle('Wafer Edge Detection - Parameter Tuner', fontsize=14, fontweight='bold')
        
        # Image display area (left)
        self.ax_img = self.fig.add_axes([0.05, 0.30, 0.55, 0.62])
        self.ax_img.set_title("Edge Detection Result")
        self.ax_img.axis('off')
        
        # Gradient profile (bottom left)
        self.ax_gradient = self.fig.add_axes([0.05, 0.06, 0.55, 0.18])
        self.ax_gradient.set_title("Gradient Profile")
        
        # Status summary (top right)
        self.ax_status = self.fig.add_axes([0.65, 0.75, 0.30, 0.17])
        self.ax_status.axis('off')
        
        # ==================== SLIDERS ====================
        slider_height = 0.025
        slider_left = 0.65
        slider_width = 0.28
        
        # Core parameters that affect edge detection visually
        sliders_config = [
            ('KERNEL_SIZE', 3, 20, self.config.KERNEL_SIZE, 2, 'Kernel Size'),
            ('EDGE_THRESHOLD', 5, 500, self.config.EDGE_THRESHOLD, 1, 'Edge Threshold'),
            ('NUM_REGIONS', 10, 80, self.config.NUM_REGIONS, 5, 'Scan Regions'),
            ('BORDER_IGNORE_PCT', 0.01, 0.15, self.config.BORDER_IGNORE_PCT, 0.01, 'Border %'),
            ('RANSAC_THRESHOLD', 1.0, 20.0, self.config.RANSAC_THRESHOLD, 0.5, 'RANSAC Thresh'),
        ]
        
        self.sliders = {}
        
        # Label for parameters
        self.fig.text(slider_left + 0.14, 0.70, 'EDGE FINDER PARAMETERS', 
                     fontsize=10, fontweight='bold', ha='center', color='darkgreen')
        
        for i, (name, vmin, vmax, vinit, vstep, label) in enumerate(sliders_config):
            y_pos = 0.64 - i * 0.08
            ax = self.fig.add_axes([slider_left, y_pos, slider_width, slider_height])
            slider = Slider(ax, label, vmin, vmax, valinit=vinit, valstep=vstep, color='lightgreen')
            slider.on_changed(self.update)
            self.sliders[name] = slider
        
        # Direction radio buttons
        self.fig.text(slider_left + 0.14, 0.23, 'EDGE DIRECTION', 
                     fontsize=10, fontweight='bold', ha='center', color='darkblue')
        radio_ax = self.fig.add_axes([slider_left + 0.03, 0.06, 0.22, 0.16])
        radio_ax.set_facecolor('lightgray')
        self.direction_radio = RadioButtons(radio_ax, ('LEFT', 'RIGHT', 'TOP', 'BOTTOM'), active=0)
        self.direction_radio.on_clicked(self.update)
        
        # Load Image button
        load_ax = self.fig.add_axes([slider_left, 0.01, 0.12, 0.04])
        self.load_button = Button(load_ax, 'Load Image', color='lightblue', hovercolor='skyblue')
        self.load_button.on_clicked(self.load_image)
        
        # Reset button
        reset_ax = self.fig.add_axes([slider_left + 0.16, 0.01, 0.12, 0.04])
        self.reset_button = Button(reset_ax, 'Reset', color='lightcoral', hovercolor='salmon')
        self.reset_button.on_clicked(self.reset)
    
    def update(self, val):
        """Update the visualization when a slider changes"""
        # Update config from sliders
        kernel_size = int(self.sliders['KERNEL_SIZE'].val)
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            self.sliders['KERNEL_SIZE'].set_val(kernel_size)
            return
        
        # Get direction from radio buttons
        direction = self.direction_radio.value_selected
        
        # Create custom config with current slider values
        class TunerConfig(EdgeFinderConfig):
            KERNEL_SIZE = kernel_size
            EDGE_THRESHOLD = int(self.sliders['EDGE_THRESHOLD'].val)
            NUM_REGIONS = int(self.sliders['NUM_REGIONS'].val)
            BORDER_IGNORE_PCT = self.sliders['BORDER_IGNORE_PCT'].val
            RANSAC_THRESHOLD = self.sliders['RANSAC_THRESHOLD'].val
            SCAN_DIRECTION = direction
        
        config = TunerConfig()
        
        # Run FOV classification first
        classifier = FOVClassifier(config)
        classification = classifier.classify(self.img)
        self.fov_type = classification['fov_type']
        self.edge_type = classification.get('edge', {}).get('edge_type', 'N/A')
        self.wafer_side = classification.get('edge', {}).get('wafer_side', 'N/A')
        
        # Clear axes
        self.ax_img.clear()
        self.ax_gradient.clear()
        self.ax_status.clear()
        self.ax_status.axis('off')
        
        # If DIE_FOV, show error and skip edge detection
        if self.fov_type != 'EDGE_FOV':
            result_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
            # Draw red border to indicate error
            cv2.rectangle(result_img, (5, 5), (self.w - 5, self.h - 5), (0, 0, 255), 4)
            self.ax_img.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            self.ax_img.set_title(f"✗ ERROR: {self.fov_type} detected — No wafer edge found!", 
                                fontsize=12, fontweight='bold', color='red')
            self.ax_img.axis('off')
            
            self.ax_gradient.text(0.5, 0.5, "No gradient data — not an Edge FOV",
                                transform=self.ax_gradient.transAxes, fontsize=12,
                                ha='center', va='center', color='red')
            self.ax_gradient.set_xlabel("X Position")
            self.ax_gradient.set_ylabel("Gradient")
            self.ax_gradient.grid(True, alpha=0.3)
            
            summary = (f"STATUS: ✗ ERROR\n\n"
                      f"FOV Type: {self.fov_type}\n\n"
                      f"This image does not\n"
                      f"contain a wafer edge.\n\n"
                      f"Please load an image\n"
                      f"with a visible wafer edge.")
            self.ax_status.text(0.5, 0.5, summary, transform=self.ax_status.transAxes,
                               fontsize=10, fontfamily='monospace',
                               verticalalignment='center', horizontalalignment='center',
                               bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.9, pad=0.5))
            
            self.fig.canvas.draw_idle()
            return
        
        # Run edge finder with skip_classification for tuning mode
        finder = EdgeLineFinder(config)
        result = finder.find_edge(self.img, skip_classification=True)
        
        # Draw result image
        result_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        
        # Determine if vertical or horizontal edge
        is_vertical = result.get('is_vertical_edge', True)
        
        if result['success']:
            endpoints = result['line_endpoints']
            
            if is_vertical:
                # Draw vertical edge line (green) - LEFT/RIGHT
                cv2.line(result_img, (endpoints['x_top'], 0), (endpoints['x_bot'], self.h),
                        (0, 255, 0), 3)
            else:
                # Draw horizontal edge line (green) - TOP/BOTTOM
                cv2.line(result_img, (0, endpoints['y_left']), (self.w, endpoints['y_right']),
                        (0, 255, 0), 3)
            
            # Draw all detected points (red)
            for point in result['detected_points']:
                cv2.circle(result_img, (int(point[0]), int(point[1])), 4, (0, 0, 255), -1)
            
            # Draw inliers (cyan circles)
            for point in result['inliers']:
                cv2.circle(result_img, (int(point[0]), int(point[1])), 6, (255, 255, 0), 2)
        
        # Draw border ignore zones (gray lines)
        border = int(self.w * config.BORDER_IGNORE_PCT) + config.KERNEL_SIZE
        if is_vertical:
            # Vertical borders for LEFT/RIGHT
            cv2.line(result_img, (border, 0), (border, self.h), (100, 100, 100), 1)
            cv2.line(result_img, (self.w - border, 0), (self.w - border, self.h), (100, 100, 100), 1)
        else:
            # Horizontal borders for TOP/BOTTOM
            border_h = int(self.h * config.BORDER_IGNORE_PCT) + config.KERNEL_SIZE
            cv2.line(result_img, (0, border_h), (self.w, border_h), (100, 100, 100), 1)
            cv2.line(result_img, (0, self.h - border_h), (self.w, self.h - border_h), (100, 100, 100), 1)
        
        self.ax_img.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        
        # Set title with status
        fov_info = f"EDGE_FOV ({self.edge_type}, {self.wafer_side})"
        if result['success']:
            title = f"✓ {direction} EDGE | {result['num_inliers']}/{result['num_points']} inliers | {fov_info}"
            self.ax_img.set_title(title, fontsize=11, fontweight='bold', color='green')
        else:
            title = f"✗ FAILED: {result.get('reason', 'Unknown')} | {fov_info}"
            self.ax_img.set_title(title, fontsize=11, fontweight='bold', color='red')
        self.ax_img.axis('off')
        
        # Draw gradient profile
        if result.get('region_data'):
            mid_idx = len(result['region_data']) // 2
            mid_region = result['region_data'][mid_idx]
            gradient = mid_region['gradient']
            
            self.ax_gradient.plot(gradient, color='green', linewidth=1.5)
            self.ax_gradient.fill_between(range(len(gradient)), gradient, alpha=0.3, color='green')
            self.ax_gradient.axhline(config.EDGE_THRESHOLD, color='orange', linestyle='--', linewidth=2,
                                     label=f'Threshold = {config.EDGE_THRESHOLD}')
            
            if mid_region['edge_point']['found']:
                # Handle both horizontal (x_int) and vertical (y_int) edge points
                x = mid_region['edge_point'].get('x_int') or mid_region['edge_point'].get('y_int')
                if x is not None:
                    self.ax_gradient.axvline(x, color='red', linewidth=2, linestyle='--',
                                            label=f'Edge @ {x}')
            
            self.ax_gradient.legend(loc='upper right', fontsize=9)
        
        self.ax_gradient.set_xlabel("X Position")
        self.ax_gradient.set_ylabel("Gradient")
        self.ax_gradient.grid(True, alpha=0.3)
        
        # Draw status summary
        fov_label = f"FOV: EDGE_FOV ({self.edge_type}, {self.wafer_side})"
        
        if result['success']:
            endpoints = result['line_endpoints']
            line_params = result['line_params']
            angle = np.degrees(np.arctan2(line_params['vx'], line_params['vy']))
            
            if is_vertical:
                line_str = f"Line: {endpoints.get('x_top', '?')} → {endpoints.get('x_bot', '?')}"
            else:
                line_str = f"Line: {endpoints.get('y_left', '?')} → {endpoints.get('y_right', '?')}"
            
            summary = (f"STATUS: ✓ SUCCESS\n\n"
                      f"{fov_label}\n"
                      f"Points Detected: {result['num_points']}\n"
                      f"Inliers (used): {result['num_inliers']}\n"
                      f"{line_str}\n"
                      f"Angle: {angle:.2f}°")
            color = 'lightgreen'
        else:
            summary = (f"STATUS: ✗ FAILED\n\n"
                      f"{fov_label}\n"
                      f"Points Detected: {result.get('num_points', 0)}\n"
                      f"Reason: {result.get('reason', 'Unknown')}")
            color = '#ffcccc'
        
        self.ax_status.text(0.5, 0.5, summary, transform=self.ax_status.transAxes,
                           fontsize=10, fontfamily='monospace',
                           verticalalignment='center', horizontalalignment='center',
                           bbox=dict(boxstyle='round', facecolor=color, alpha=0.9, pad=0.5))
        
        self.fig.canvas.draw_idle()
    
    def load_image(self, event):
        """Open file dialog to load a new image"""
        # Hide matplotlib window temporarily to show dialog on top
        root = tk.Tk()
        root.withdraw()  # Hide the tkinter root window
        root.attributes('-topmost', True)  # Make dialog appear on top
        
        file_path = filedialog.askopenfilename(
            title="Select Wafer Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()
        
        if file_path:
            try:
                # Load and preprocess new image
                self.image_path = file_path
                self.img, self.original_img, self.scale = preprocess_image(
                    file_path, 
                    self.config.TARGET_PROCESS_DIM
                )
                self.h, self.w = self.img.shape
                
                # Update the figure title with new filename
                filename = os.path.basename(file_path)
                self.fig.suptitle(f'Wafer Edge Detection - {filename}', fontsize=14, fontweight='bold')
                
                # Trigger update
                self.update(None)
                print(f"Loaded image: {file_path}")
            except Exception as e:
                print(f"Error loading image: {e}")
    
    def reset(self, event):
        """Reset all sliders to default values"""
        default_config = EdgeFinderConfig()
        self.sliders['KERNEL_SIZE'].set_val(default_config.KERNEL_SIZE)
        self.sliders['EDGE_THRESHOLD'].set_val(default_config.EDGE_THRESHOLD)
        self.sliders['NUM_REGIONS'].set_val(default_config.NUM_REGIONS)
        self.sliders['BORDER_IGNORE_PCT'].set_val(default_config.BORDER_IGNORE_PCT)
        self.sliders['RANSAC_THRESHOLD'].set_val(default_config.RANSAC_THRESHOLD)
    
    def show(self):
        """Display the interactive tuner"""
        plt.show()


def main():
    """Main entry point"""
    # Check for command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = None
    
    print("=" * 50)
    print("  PARAMETER TUNER v1.2 (Simplified)")
    print("=" * 50)
    print("\n  Key parameters for edge detection:")
    print("  • Kernel Size - Gradient window width")
    print("  • Edge Threshold - Min gradient for edge")
    print("  • Scan Regions - Vertical sampling density")
    print("  • Border % - Edge of image to ignore")
    print("  • RANSAC Thresh - Line fitting tolerance")
    print("\n  Adjust sliders to see live changes!")
    print("=" * 50)
    
    tuner = InteractiveParameterTuner(image_path)
    tuner.show()


if __name__ == "__main__":
    main()
