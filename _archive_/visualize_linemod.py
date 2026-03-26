import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from linemod_matcher import LinemodMatcher, LinemodConfig

def visualize_pipeline_steps():
    print("Loading images for visualization...")
    search_img_path = r"Images\TOP_EDGE_PNG\Copy of TopEdge.png"
    search_img = cv2.imread(search_img_path)
    
    if search_img is None:
        print("Test image not found.")
        return
        
    # Crop a small region as a template
    h, w = search_img.shape[:2]
    cx, cy = w // 2, h // 2
    template_img = search_img[cy-100:cy+100, cx-100:cx+100].copy()
    
    # 1. Show Original Images
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle('Step 1: Original Inputs', fontsize=16)
    
    plt.subplot(121)
    plt.title('Search Image')
    plt.imshow(cv2.cvtColor(search_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(122)
    plt.title('Template Image (Cropped Fiducial)')
    plt.imshow(cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualize_step1_inputs.png')
    plt.close()
    
    # Initialize matcher
    config = LinemodConfig()
    config.ANGLE_STEP = 360
    config.SCALE_MIN = 1.0
    config.SCALE_MAX = 1.0
    matcher = LinemodMatcher(config)
    
    # 2. Extract and Visualize Template Gradients
    gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    t_grad = matcher._compute_gradient_features(gray_template, use_mask=True)
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Step 2: Template Edge Orientation Extraction (Ignoring Texture)', fontsize=16)
    
    plt.subplot(221)
    plt.title('Grayscale Template')
    plt.imshow(gray_template, cmap='gray')
    plt.axis('off')
    
    plt.subplot(222)
    plt.title('Gradient Magnitude (Strength of Edges)')
    plt.imshow(t_grad['magnitude'], cmap='hot')
    plt.axis('off')
    
    plt.subplot(223)
    plt.title('DX Component (Horizontal Edges) - MASKED')
    # Use coolwarm to show negative, zero, and positive directions
    plt.imshow(t_grad['dx'], cmap='coolwarm', vmin=-1, vmax=1)
    plt.axis('off')
    
    plt.subplot(224)
    plt.title('DY Component (Vertical Edges) - MASKED')
    plt.imshow(t_grad['dy'], cmap='coolwarm', vmin=-1, vmax=1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualize_step2_template_gradients.png')
    plt.close()
    
    # 3. Extract and Visualize Search Image Gradients
    gray_search = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
    s_grad = matcher._compute_gradient_features(gray_search, use_mask=False)
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Step 3: Search Image Edge Extractions (Unmasked)', fontsize=16)
    
    plt.subplot(221)
    plt.title('Grayscale Search Image')
    plt.imshow(gray_search, cmap='gray')
    plt.axis('off')
    
    plt.subplot(222)
    plt.title('Gradient Magnitude')
    plt.imshow(s_grad['magnitude'], cmap='hot')
    plt.axis('off')
    
    # Plot a specific cropped region of the search image to see vectors clearly
    s_dx_crop = s_grad['dx'][cy-150:cy+150, cx-150:cx+150]
    s_dy_crop = s_grad['dy'][cy-150:cy+150, cx-150:cx+150]
    
    plt.subplot(223)
    plt.title('Search DX Component (Zoomed In)')
    plt.imshow(s_dx_crop, cmap='coolwarm', vmin=-1, vmax=1)
    plt.axis('off')
    
    plt.subplot(224)
    plt.title('Search DY Component (Zoomed In)')
    plt.imshow(s_dy_crop, cmap='coolwarm', vmin=-1, vmax=1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualize_step3_search_gradients.png')
    plt.close()
    
    # 4. Perform Convolution Match and show Score Map
    t_dx = t_grad['dx']
    t_dy = t_grad['dy']
    active_pixels = np.sum(t_grad['mask'])
    
    res_x = cv2.matchTemplate(s_grad['dx'], t_dx, cv2.TM_CCORR)
    res_y = cv2.matchTemplate(s_grad['dy'], t_dy, cv2.TM_CCORR)
    result = (res_x + res_y) / active_pixels
    
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle('Step 4: Vector Dot-Product Convolution Result (Similarity Map)', fontsize=16)
    
    plt.subplot(121)
    plt.title('Overall Search Result - Hotter = Better Match')
    plt.imshow(result, cmap='hot')
    plt.colorbar()
    plt.axis('off')
    
    # Highlight the peak
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    plt.subplot(122)
    plt.title(f'Peak Detection ({max_val*100:.2f}%)')
    res_vis = cv2.cvtColor(search_img, cv2.COLOR_BGR2RGB)
    
    # Draw match
    tw, th = t_dx.shape[1], t_dx.shape[0]
    pt = max_loc
    cv2.rectangle(res_vis, pt, (pt[0]+tw, pt[1]+th), (0, 255, 0), 5)
    
    # Zoom in to the result
    zoom_res = res_vis[max(0, pt[1]-500):min(res_vis.shape[0], pt[1]+th+500), 
                       max(0, pt[0]-500):min(res_vis.shape[1], pt[0]+tw+500)]
    plt.imshow(zoom_res)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualize_step4_result.png')
    plt.close()
    
    print("Saved visualization images: 'visualize_step1_inputs.png' to 'visualize_step4_result.png'")

if __name__ == "__main__":
    visualize_pipeline_steps()
