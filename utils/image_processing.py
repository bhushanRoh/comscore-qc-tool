import cv2
import numpy as np


def process_image(file_bytes):
    """
    Process uploaded image with multiple strategies for optimal OCR on
    blurry text, logo text, and normal text.

    Args:
        file_bytes: Bytes of the uploaded image file.
    Returns:
        variants: List of processed images (numpy arrays) for multi-pass OCR.
        original_image: Original BGR image for display.
    """
    # Decode raw bytes
    nparr = np.frombuffer(file_bytes, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if original_image is None:
        raise ValueError("Could not decode image")

    variants = []

    # --- Strategy 1: CLAHE Enhanced Grayscale ---
    # Best for: blurry text, low-contrast images
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    variants.append(sharpened)

    # --- Strategy 2: Upscaled (2x) ---
    # Best for: small text in logos, distant text
    h, w = gray.shape[:2]
    upscaled = cv2.resize(enhanced, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    variants.append(upscaled)

    # --- Strategy 3: Adaptive Threshold (Binary) ---
    # Best for: text on colored/gradient backgrounds (logos, badges)
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    variants.append(binary)

    # --- Strategy 4: Original Color Image ---
    # Best for: colored text that disappears in grayscale conversion
    variants.append(original_image.copy())

    return variants, original_image


def create_tiles(image, grid=(2, 2), overlap=0.15):
    """
    Split an image into overlapping tiles for better OCR on composite boards.
    
    Args:
        image: Original BGR or grayscale image.
        grid: (rows, cols) number of tiles.
        overlap: Fraction of overlap between adjacent tiles (0.0 - 0.5).
    Returns:
        tiles: List of (tile_image, x_offset, y_offset) tuples.
    """
    h, w = image.shape[:2]
    rows, cols = grid
    
    # Calculate tile dimensions with overlap
    tile_h = int(h / rows * (1 + overlap))
    tile_w = int(w / cols * (1 + overlap))
    
    # Step size (how far to move for next tile)
    step_h = int(h / rows)
    step_w = int(w / cols)
    
    tiles = []
    for r in range(rows):
        for c in range(cols):
            y_start = max(0, r * step_h - int(tile_h * overlap / 2))
            x_start = max(0, c * step_w - int(tile_w * overlap / 2))
            
            # Ensure we don't go past image boundaries
            y_end = min(h, y_start + tile_h)
            x_end = min(w, x_start + tile_w)
            
            # Adjust start if end was clamped
            y_start = max(0, y_end - tile_h)
            x_start = max(0, x_end - tile_w)
            
            tile = image[y_start:y_end, x_start:x_end]
            tiles.append((tile, x_start, y_start))
    
    return tiles
