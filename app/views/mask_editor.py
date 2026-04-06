"""
================================================================================
MASK EDITOR — Interactive Brush ROI Drawing
================================================================================
Opens an OpenCV window on a template image and lets the user PAINT the
detection region with a circular brush.

Controls:
  - Left-click + drag:  paint detection region (white)
  - Right-click + drag: erase (remove painted region)
  - Mouse wheel:        increase / decrease brush size
  - 'r':                reset (clear all paint)
  - [Save] button or Enter:  confirm mask
  - [Close] button or Escape: cancel (returns None)

Returns a binary mask (uint8, 255=painted region, 0=outside).
================================================================================
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Shared helper: draw a clickable button on an OpenCV image
# ---------------------------------------------------------------------------
def _draw_button(img, text, x, y, w, h, color=(60, 60, 60),
                 text_color=(255, 255, 255), hover=False):
    """Draw a rounded-rect button. Returns (x, y, w, h) hit-box."""
    fill = tuple(min(255, c + 40) for c in color) if hover else color
    cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.45, h / 40)
    (tw, th_t), _ = cv2.getTextSize(text, font, fs, 1)
    tx = x + (w - tw) // 2
    ty = y + (h + th_t) // 2
    cv2.putText(img, text, (tx, ty), font, fs, text_color, 1, cv2.LINE_AA)
    return (x, y, w, h)


def _point_in_rect(px, py, rect):
    """Check if point is inside (x, y, w, h) rectangle."""
    rx, ry, rw, rh = rect
    return rx <= px <= rx + rw and ry <= py <= ry + rh


def draw_detection_mask(template_img, window_title="Paint Detection Region"):
    """
    Open an interactive OpenCV window to paint a mask on the template
    using a circular brush.

    Args:
        template_img: Template image (grayscale or BGR).
        window_title: Window title.

    Returns:
        np.ndarray: Binary mask (uint8, 255/0), same size as template_img,
                    or None if the user cancels (Escape).
    """
    if len(template_img.shape) == 2:
        base_img = cv2.cvtColor(template_img, cv2.COLOR_GRAY2BGR)
    else:
        base_img = template_img.copy()

    h, w = base_img.shape[:2]

    # Scale display to fit screen
    try:
        import ctypes
        user32 = ctypes.windll.user32
        screen_w = int(user32.GetSystemMetrics(0) * 0.80)
        screen_h = int(user32.GetSystemMetrics(1) * 0.80)
    except Exception:
        screen_w, screen_h = 1280, 900

    disp_scale = min(screen_w / w, screen_h / h, 1.0)
    if disp_scale < 1.0:
        disp_w = int(w * disp_scale)
        disp_h = int(h * disp_scale)
    else:
        disp_w, disp_h = w, h
        disp_scale = 1.0

    # Pre-compute resized background (expensive, done ONCE)
    if disp_scale < 1.0:
        bg_display = cv2.resize(base_img, (disp_w, disp_h),
                                interpolation=cv2.INTER_AREA)
    else:
        bg_display = base_img.copy()

    # Mask at DISPLAY resolution (fast painting — upscale on confirm)
    disp_mask = np.zeros((disp_h, disp_w), dtype=np.uint8)

    # Green overlay layer (pre-allocated)
    green_layer = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
    green_layer[:, :] = (0, 130, 0)

    # Brush settings (in display coordinates)
    brush_radius = max(3, min(disp_w, disp_h) // 30)
    min_brush = 2
    max_brush = max(30, min(disp_w, disp_h) // 5)
    brush_step = max(1, brush_radius // 5)

    # Button dimensions
    btn_w, btn_h = 90, 30
    btn_gap = 10
    btn_save_rect = (disp_w - 2 * btn_w - btn_gap - 10, disp_h - btn_h - 10,
                     btn_w, btn_h)
    btn_close_rect = (disp_w - btn_w - 10, disp_h - btn_h - 10,
                      btn_w, btn_h)

    painting = False
    erasing = False
    cursor_pos = (disp_w // 2, disp_h // 2)
    needs_redraw = True
    action = None  # 'save' or 'close'

    def _redraw():
        """Composite background + mask overlay + cursor + buttons."""
        vis = bg_display.copy()

        # Blend green where mask is painted
        mask_bool = disp_mask > 0
        if np.any(mask_bool):
            vis[mask_bool] = (
                vis[mask_bool].astype(np.int16) * 5 // 10 +
                green_layer[mask_bool].astype(np.int16) * 5 // 10
            ).clip(0, 255).astype(np.uint8)

        # Draw brush cursor
        cx, cy = cursor_pos
        cv2.circle(vis, (cx, cy), brush_radius, (0, 200, 255), 1)

        # Instructions (small, top-left)
        font_scale = max(0.35, 0.45 * (disp_w / 800))
        lines = [
            "L-drag: paint | R-drag: erase | Scroll: brush size | R: reset",
        ]
        for i, txt in enumerate(lines):
            y = 16 + i * 16
            cv2.putText(vis, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 0), 1, cv2.LINE_AA)

        # Buttons (bottom-right)
        hover_save = _point_in_rect(cx, cy, btn_save_rect)
        hover_close = _point_in_rect(cx, cy, btn_close_rect)
        _draw_button(vis, "Save", *btn_save_rect,
                     color=(40, 120, 40), hover=hover_save)
        _draw_button(vis, "Close", *btn_close_rect,
                     color=(40, 40, 140), hover=hover_close)

        cv2.imshow(window_title, vis)

    def _paint_at(mx, my, value=255):
        """Paint or erase on the display-resolution mask."""
        cv2.circle(disp_mask, (mx, my), brush_radius, int(value), -1)

    prev_pos = None

    def _mouse_cb(event, mx, my, flags, param):
        nonlocal painting, erasing, cursor_pos, brush_radius, needs_redraw
        nonlocal prev_pos, action

        cursor_pos = (mx, my)

        if event == cv2.EVENT_LBUTTONDOWN:
            # Check button clicks first
            if _point_in_rect(mx, my, btn_save_rect):
                action = 'save'
                return
            if _point_in_rect(mx, my, btn_close_rect):
                action = 'close'
                return
            painting = True
            prev_pos = (mx, my)
            _paint_at(mx, my, 255)
            needs_redraw = True

        elif event == cv2.EVENT_LBUTTONUP:
            painting = False
            prev_pos = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            erasing = True
            prev_pos = (mx, my)
            _paint_at(mx, my, 0)
            needs_redraw = True

        elif event == cv2.EVENT_RBUTTONUP:
            erasing = False
            prev_pos = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if painting:
                if prev_pos is not None:
                    cv2.line(disp_mask, prev_pos, (mx, my), 255,
                             brush_radius * 2)
                _paint_at(mx, my, 255)
                prev_pos = (mx, my)
                needs_redraw = True
            elif erasing:
                if prev_pos is not None:
                    cv2.line(disp_mask, prev_pos, (mx, my), 0,
                             brush_radius * 2)
                _paint_at(mx, my, 0)
                prev_pos = (mx, my)
                needs_redraw = True
            else:
                needs_redraw = True  # cursor / hover only

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                brush_radius = min(max_brush, brush_radius + brush_step)
            else:
                brush_radius = max(min_brush, brush_radius - brush_step)
            needs_redraw = True

    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_title, _mouse_cb)
    _redraw()

    result_mask = None

    while True:
        key = cv2.waitKey(16) & 0xFF  # ~60 FPS cap

        # Detect window closed via X button
        try:
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        # Button clicks
        if action == 'save':
            key = 13  # treat as Enter
            action = None
        elif action == 'close':
            key = 27  # treat as Escape
            action = None

        if key == 27:  # Escape → cancel
            result_mask = None
            break

        elif key == ord('r') or key == ord('R'):  # Reset
            disp_mask[:] = 0
            needs_redraw = True

        elif key in (13, 10):  # Enter → confirm
            if np.any(disp_mask > 0):
                if disp_scale < 1.0:
                    result_mask = cv2.resize(disp_mask, (w, h),
                                            interpolation=cv2.INTER_NEAREST)
                else:
                    result_mask = disp_mask.copy()
                result_mask = (result_mask > 127).astype(np.uint8) * 255
            else:
                result_mask = None
            break

        if needs_redraw:
            _redraw()
            needs_redraw = False

    cv2.destroyWindow(window_title)
    cv2.waitKey(1)
    return result_mask
