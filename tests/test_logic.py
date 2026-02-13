import cv2
import numpy as np
import os
from utils.ocr_engine import OCREngine
from utils.blacklist_manager import check_compliance, add_term, load_blacklist


def create_test_image(text, filename="test_image.png", blur_strength=5):
    """Create a test image with text and configurable blur."""
    img = np.ones((200, 600, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (50, 100), font, 2, (0, 0, 0), 3, cv2.LINE_AA)

    # Apply blur to simulate real-world conditions
    if blur_strength > 0:
        ksize = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    success = cv2.imwrite(filename, img)
    return filename if success else None


def test_workflow():
    print("Starting verification...\n")

    # 1. Setup Blacklist
    test_term = "ForbiddenPretendDrug"
    print(f"Adding '{test_term}' to blacklist...")
    add_term(test_term)
    current_blacklist = load_blacklist()
    assert test_term in current_blacklist, "Blacklist add failed"
    print("Blacklist update verified.\n")

    # 2. Test with normal image
    print("=" * 50)
    print("TEST 1: Normal text (light blur)")
    print("=" * 50)
    run_test(test_term, blur_strength=5, current_blacklist=current_blacklist)

    # 3. Test with heavily blurred image
    print("\n" + "=" * 50)
    print("TEST 2: Heavily blurred text")
    print("=" * 50)
    run_test(test_term, blur_strength=11, current_blacklist=current_blacklist)

    print("\nAll verification complete.")


def run_test(test_term, blur_strength, current_blacklist):
    base_filename = f"test_blur_{blur_strength}.png"
    text_on_image = f"Buy {test_term} Now"
    print(f"Creating image with text: '{text_on_image}' (blur={blur_strength})")
    create_test_image(text_on_image, base_filename, blur_strength=blur_strength)

    # Read image and create multi-strategy variants
    from utils.image_processing import process_image
    with open(base_filename, "rb") as f:
        image_variants, original_img = process_image(f.read())

    print(f"Generated {len(image_variants)} preprocessed variants")

    # OCR multi-pass
    print("Initializing OCR...")
    ocr = OCREngine()

    print("Running multi-pass OCR...")
    results = ocr.extract_text_multi(image_variants)
    print(f"Merged OCR Results ({len(results)} unique):")
    for bbox, text, conf in results:
        print(f"  Detected: '{text}' (conf: {conf:.2f})")

    # Check Compliance
    violations = []
    found_text = False
    for bbox, text, conf in results:
        if test_term.lower() in text.lower():
            found_text = True
        v = check_compliance(text, current_blacklist)
        if v:
            violations.extend(v)

    if test_term in violations:
        print(f"✅ SUCCESS: Violation '{test_term}' correctly detected.")
    else:
        print(f"❌ FAILURE: Violation '{test_term}' not detected.")
        if not found_text:
            print("  Reason: OCR did not read the text correctly.")
        else:
            print("  Reason: compliance check logic failed.")

    # Cleanup
    if os.path.exists(base_filename):
        os.remove(base_filename)


if __name__ == "__main__":
    test_workflow()
