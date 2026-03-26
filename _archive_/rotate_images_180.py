"""
================================================================================
IMAGE ROTATION UTILITY - 180 DEGREE ROTATION
================================================================================
Rotates all images in a selected folder by 180 degrees and saves them to
an output folder.

Usage:
    python rotate_images_180.py

Author: Adrain Lim
================================================================================
"""

import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path


def select_folder(title="Select Folder"):
    """Open folder selection dialog"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    folder_path = filedialog.askdirectory(title=title)
    root.destroy()
    
    return folder_path


def rotate_images_180(input_folder, output_folder):
    """
    Rotate all images in input_folder by 180 degrees and save to output_folder.
    
    Args:
        input_folder: Path to folder containing images
        output_folder: Path to folder where rotated images will be saved
        
    Returns:
        Tuple of (success_count, error_count, error_list)
    """
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif'}
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_files = []
    for file in os.listdir(input_folder):
        if Path(file).suffix.lower() in image_extensions:
            image_files.append(file)
    
    if len(image_files) == 0:
        return 0, 0, ["No image files found in the input folder"]
    
    success_count = 0
    error_count = 0
    error_list = []
    
    print("\n" + "=" * 60)
    print(f"  ROTATING {len(image_files)} IMAGES BY 180 DEGREES")
    print("=" * 60)
    print(f"\nInput folder:  {input_folder}")
    print(f"Output folder: {output_folder}\n")
    
    for i, filename in enumerate(image_files, 1):
        try:
            # Read image
            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                raise ValueError(f"Could not read image: {filename}")
            
            # Rotate 180 degrees
            rotated = cv2.rotate(img, cv2.ROTATE_180)
            
            # Save rotated image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, rotated)
            
            success_count += 1
            print(f"[{i}/{len(image_files)}] ✓ {filename}")
            
        except Exception as e:
            error_count += 1
            error_msg = f"{filename}: {str(e)}"
            error_list.append(error_msg)
            print(f"[{i}/{len(image_files)}] ✗ {filename} - Error: {str(e)}")
    
    return success_count, error_count, error_list


def main():
    """Main entry point with GUI folder selection"""
    
    print("=" * 60)
    print("  IMAGE ROTATION UTILITY - 180 DEGREE ROTATION")
    print("=" * 60)
    print("\nThis tool will rotate all images in a folder by 180 degrees.")
    print()
    
    # Select input folder
    print("Step 1: Select INPUT folder (containing images to rotate)...")
    input_folder = select_folder("Select INPUT Folder (Images to Rotate)")
    
    if not input_folder:
        print("\n✗ No input folder selected. Exiting.")
        return
    
    print(f"✓ Input folder: {input_folder}")
    
    # Select output folder
    print("\nStep 2: Select OUTPUT folder (where rotated images will be saved)...")
    output_folder = select_folder("Select OUTPUT Folder (Save Rotated Images)")
    
    if not output_folder:
        print("\n✗ No output folder selected. Exiting.")
        return
    
    print(f"✓ Output folder: {output_folder}")
    
    # Confirm if output folder is the same as input
    if os.path.normpath(input_folder) == os.path.normpath(output_folder):
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        result = messagebox.askyesno(
            "Warning: Same Folder",
            "Input and output folders are the same.\n"
            "This will OVERWRITE the original images!\n\n"
            "Do you want to continue?"
        )
        root.destroy()
        
        if not result:
            print("\n✗ Operation cancelled by user.")
            return
    
    # Process images
    success, errors, error_list = rotate_images_180(input_folder, output_folder)
    
    # Print summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"✓ Successfully rotated: {success} images")
    print(f"✗ Errors: {errors} images")
    
    if errors > 0:
        print("\nError details:")
        for err in error_list:
            print(f"  - {err}")
    
    print("\n" + "=" * 60)
    
    # Show completion message box
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    if errors == 0:
        messagebox.showinfo(
            "Success",
            f"Successfully rotated {success} images!\n\n"
            f"Output folder:\n{output_folder}"
        )
    else:
        messagebox.showwarning(
            "Completed with Errors",
            f"Successfully rotated: {success} images\n"
            f"Errors: {errors} images\n\n"
            f"Check console for error details."
        )
    
    root.destroy()


if __name__ == "__main__":
    main()
