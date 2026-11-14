
"""
Daily Life Accessibility Companion
OCR Text Recognition Module

Features:
- Extract text from camera images
- Image preprocessing
- Grayscale and threshold enhancement
- Tesseract OCR
- Clear installation error messages
"""

import cv2
import pytesseract


def extract_text(frame):

    """
    Extracts readable text from an OpenCV image.

    Returns extracted text as a string.
    """

    if frame is None or frame.size == 0:
        return ""

    try:

        # Convert to grayscale.
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Improve contrast.
        gray = cv2.convertScaleAbs(
            gray,
            alpha=1.5,
            beta=10
        )

        # Reduce noise.
        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        # Adaptive thresholding.
        processed = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )

        # OCR.
        text = pytesseract.image_to_string(
            processed,
            config="--oem 3 --psm 6"
        )

        return text.strip()

    except pytesseract.TesseractNotFoundError:

        print(
            "Tesseract executable not found. "
            "Install Tesseract OCR and configure its path."
        )

        return (
            "Tesseract OCR is not installed or "
            "its executable path is not configured."
        )

    except Exception as error:

        print(f"OCR error: {error}")

        return f"OCR failed: {error}"
