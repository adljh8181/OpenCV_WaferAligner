import os
import glob
import io
from PIL import Image

def convert_to_target_size(input_bmp, output_png, target_kb=1000):
    """
    Iteratively downsizes an image until its PNG file size is under the target KB.
    """
    target_bytes = target_kb * 1024
    
    try:
        with Image.open(input_bmp) as img:
            # Since 76MB -> 1MB is a huge jump, we start the scale at 35% 
            # to save processing time, rather than starting at 100%.
            scale = 0.35 
            
            while scale > 0.05: # Safety limit so it doesn't shrink to 0
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                
                # Resize using the high-quality Lanczos filter
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Save to a temporary virtual memory buffer (not the hard drive)
                buffer = io.BytesIO()
                resized_img.save(buffer, format="PNG", optimize=True)
                
                # Check the file size of the compressed PNG in bytes
                size_in_bytes = buffer.tell()
                
                if size_in_bytes <= target_bytes:
                    # Target achieved! Write the buffer to the actual file
                    with open(output_png, "wb") as f:
                        f.write(buffer.getvalue())
                        
                    final_kb = size_in_bytes / 1024
                    return True, final_kb, new_width, new_height
                
                # If the file is still larger than 1000 KB, shrink it by another 5% and repeat
                scale *= 0.95
                
            print(f"⚠️ Could not reach target size for {os.path.basename(input_bmp)}")
            return False, 0, 0, 0
            
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(input_bmp)}: {e}")
        return False, 0, 0, 0


def main():
    target_folder = r"C:\Users\adrain.lim\Downloads\OneDrive_1_26-02-2026"
    target_max_kb = 1000  # Set your target size here!
    
    if not os.path.isdir(target_folder):
        print(f"❌ Error: Folder '{target_folder}' not found.")
        return

    bmp_files = glob.glob(os.path.join(target_folder, "*.bmp"))
    
    if not bmp_files:
        print(f"⚠️ No BMP files found in '{target_folder}'.")
        return

    print(f"Found {len(bmp_files)} BMP files. Targeting ~{target_max_kb} KB per file...\n")
    
    success_count = 0
    
    for index, bmp_path in enumerate(bmp_files, 1):
        base_name = os.path.splitext(bmp_path)[0]
        png_path = f"{base_name}_small.png"
        
        print(f"[{index}/{len(bmp_files)}] Processing {os.path.basename(bmp_path)}...")
        
        success, final_kb, final_w, final_h = convert_to_target_size(bmp_path, png_path, target_kb=target_max_kb)
        
        if success:
            print(f"   ✅ Saved! Size: {final_kb:.1f} KB | Dimensions: {final_w}x{final_h}")
            success_count += 1

    print(f"\n🎉 Batch complete! {success_count}/{len(bmp_files)} files successfully reduced to under {target_max_kb} KB.")

if __name__ == "__main__":
    main()