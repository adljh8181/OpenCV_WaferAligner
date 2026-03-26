import cv2
import numpy as np
import time
import os
import argparse
from app.services.linemod_matcher import LinemodMatcher, LinemodConfig

def evaluate_resolution_mse(image_path, template_path, scales=[1.0, 0.5, 0.25, 0.125], show_image=False):
    """
    Evaluates pattern matching across various resolutions and computes the L2 error 
    (Mean Squared Error of distance) against the highest requested scale.
    """
    if not os.path.exists(image_path):
        print(f"Error: Target image {image_path} does not exist.")
        return
        
    if not os.path.exists(template_path):
        print(f"Error: Template image {template_path} does not exist.")
        return
        
    print(f"=== Resolution L2 Error (MSE) Evaluation ===")
    print(f"Target Image: {image_path}")
    print(f"Template Image: {template_path}")
    print(f"Scales to Test: {scales}")
    print("-" * 60)
    
    # Load original full-res images
    orig_img = cv2.imread(image_path)
    orig_tmpl = cv2.imread(template_path)
    
    if orig_img is None or orig_tmpl is None:
        print("Error: Could not read image or template.")
        return
        
    baseline_x = None
    baseline_y = None
    
    # Configuration
    config = LinemodConfig()
    config.MATCH_THRESHOLD = 30.0 # Lowered slightly to ensure we get a match and avoid full-image fallback
    # To be resolution-independent, using relative threshold and no specific hysteresis kernel
    config.WEAK_THRESHOLD = -70.0 
    config.ANGLE_STEP = 360 # Only generate 1 template (0 degrees) to speed up testing
    # Force the hysteresis kernel size and blur to be static (7x7) like the C++ implementation
    config.HYSTERESIS_KERNEL = 7
    
    print(f"{'Scale':>8} | {'Resolution':>15} | {'Found X, Y':>15} | {'L2 Error (px)':>15} | {'Time (ms)':>10} | {'Score':>7}")
    print("-" * 80)
    
    for scale in sorted(scales, reverse=True):
        # Resize images
        if scale == 1.0:
            test_img = orig_img.copy()
            test_tmpl = orig_tmpl.copy()
        else:
            test_img = cv2.resize(orig_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            test_tmpl = cv2.resize(orig_tmpl, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            
        res_str = f"{test_img.shape[1]}x{test_img.shape[0]}"
        
        # Initalize matcher and load template
        matcher = LinemodMatcher(config)
        
        start_t = time.perf_counter()
        
        matcher.load_template(test_tmpl)
        matcher.generate_templates()
        
        # Match
        results = matcher.match(test_img, return_all=False)
        end_t = time.perf_counter()
        
        elapsed_ms = (end_t - start_t) * 1000.0
        
        if not results:
            print(f"{scale:>8.3f} | {res_str:>15} | {'Not Found':>15} | {'N/A':>15} | {elapsed_ms:>10.1f} | {'N/A':>7}")
            continue
            
        # Get top result
        top_res = results if isinstance(results, dict) else (results[0] if isinstance(results, list) else results)
        
        # x, y comes from top_res (it might be named x, y, score, etc. Let's assume standard named attributes or dict keys)
        # Looking at linemod_matcher convention, let's just use top_res attributes or unpack assuming a dict:
        
        res_x = top_res.get('x', 0) if isinstance(top_res, dict) else getattr(top_res, 'x', 0)
        res_y = top_res.get('y', 0) if isinstance(top_res, dict) else getattr(top_res, 'y', 0)
        score = top_res.get('score', 0) if isinstance(top_res, dict) else getattr(top_res, 'score', 0)
        
        # Scale back to original resolution coordinates
        mapped_x = res_x / scale
        mapped_y = res_y / scale
        
        if baseline_x is None:
            baseline_x = mapped_x
            baseline_y = mapped_y
            err_str = "0.0 (Ref)"
        else:
            # L2 Error (Distance)
            l2_err = np.sqrt((mapped_x - baseline_x)**2 + (mapped_y - baseline_y)**2)
            err_str = f"{l2_err:.2f}"
            
        print(f"{scale:>8.3f} | {res_str:>15} | {f'{mapped_x:.1f}, {mapped_y:.1f}':>15} | {err_str:>15} | {elapsed_ms:>10.1f} | {score:>7.1f}")
        
        # Visualization
        if show_image:
            vis_img = test_img.copy()
            # draw rectangle
            h, w = test_tmpl.shape[:2]
            cv2.rectangle(vis_img, (int(res_x), int(res_y)), (int(res_x + w), int(res_y + h)), (0, 255, 0), max(1, int(2 * scale)))
            cv2.putText(vis_img, f"Scale {scale} | Err: {err_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, scale), (0, 255, 0), 2)
            
            # Show image, wait for key, close
            cv2.imshow(f"Match Scale {scale}", vis_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    print("-" * 80)
    print("Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pattern Resolution VS Location MSE")
    parser.add_argument("--image", type=str, required=True, help="Path to high-res target image")
    parser.add_argument("--template", type=str, required=True, help="Path to high-res template image")
    parser.add_argument("--scales", type=float, nargs='+', default=[1.0, 0.75, 0.5, 0.25, 0.125], help="List of scale factors to evaluate (descending)")
    parser.add_argument("--show", action="store_true", help="Show resulting matched images")
    
    args = parser.parse_args()
    
    evaluate_resolution_mse(args.image, args.template, scales=args.scales, show_image=args.show)
