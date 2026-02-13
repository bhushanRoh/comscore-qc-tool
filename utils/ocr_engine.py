import easyocr
import numpy as np


class OCREngine:
    def __init__(self, languages=['en']):
        self.reader = easyocr.Reader(languages)

    def extract_text(self, image):
        """
        Extract text from a single image at word level.
        """
        results = self.reader.readtext(
            image,
            low_text=0.3,
            text_threshold=0.5,
            link_threshold=0.3,
            width_ths=0.1,
            paragraph=False,
        )
        return results

    def extract_text_multi(self, image_variants, tiles=None):
        """
        Run OCR on multiple preprocessed variants AND optional tiles.
        Merges and deduplicates all results by spatial overlap.

        Args:
            image_variants: List of full-image numpy arrays.
            tiles: Optional list of (tile_image, x_offset, y_offset) tuples.
        Returns:
            merged: Deduplicated list of (bbox, text, confidence).
        """
        all_results = []

        # Pass 1: Full image variants
        for idx, img in enumerate(image_variants):
            results = self.extract_text(img)

            # Variant at index 1 is 2x upscaled — rescale bboxes
            if idx == 1:
                results = self._rescale_bboxes(results, scale=0.5)

            all_results.extend(results)

        # Pass 2: Tile-based scanning (shifted coordinates back to original)
        if tiles:
            for tile_img, x_off, y_off in tiles:
                tile_results = self.extract_text(tile_img)
                # Shift bboxes by the tile's offset in the original image
                shifted = self._shift_bboxes(tile_results, x_off, y_off)
                all_results.extend(shifted)

        # Deduplicate by spatial overlap
        merged = self._deduplicate_spatial(all_results)
        return merged

    def _rescale_bboxes(self, results, scale):
        """Rescale bounding box coordinates by a factor."""
        rescaled = []
        for bbox, text, conf in results:
            new_bbox = [[int(pt[0] * scale), int(pt[1] * scale)] for pt in bbox]
            rescaled.append((new_bbox, text, conf))
        return rescaled

    def _shift_bboxes(self, results, x_offset, y_offset):
        """Shift bounding box coordinates by an offset (for tile-based scanning)."""
        shifted = []
        for bbox, text, conf in results:
            new_bbox = [[pt[0] + x_offset, pt[1] + y_offset] for pt in bbox]
            shifted.append((new_bbox, text, conf))
        return shifted

    def _deduplicate_spatial(self, results):
        """
        Deduplicate OCR results by spatial overlap (IoU).
        When two bboxes overlap significantly, keep the one with higher confidence.
        Preserves the same word at different locations.
        """
        if not results:
            return []

        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
        kept = []

        for bbox, text, conf in sorted_results:
            text = text.strip()
            if not text:
                continue

            is_duplicate = False
            for kept_bbox, kept_text, kept_conf in kept:
                iou = self._compute_iou(bbox, kept_bbox)
                if iou > 0.3:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append((bbox, text, conf))

        return kept

    def _compute_iou(self, bbox1, bbox2):
        """Compute Intersection over Union between two bounding boxes."""
        x1_a, y1_a = min(p[0] for p in bbox1), min(p[1] for p in bbox1)
        x2_a, y2_a = max(p[0] for p in bbox1), max(p[1] for p in bbox1)
        x1_b, y1_b = min(p[0] for p in bbox2), min(p[1] for p in bbox2)
        x2_b, y2_b = max(p[0] for p in bbox2), max(p[1] for p in bbox2)

        x1_i = max(x1_a, x1_b)
        y1_i = max(y1_a, y1_b)
        x2_i = min(x2_a, x2_b)
        y2_i = min(y2_a, y2_b)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)
        union = area_a + area_b - intersection

        if union == 0:
            return 0.0
        return intersection / union
