from pathlib import Path

import cv2
import numpy as np

from tests.test_detector import _candled_egg


FILES = {
    "original": Path(
        r"C:\Users\Davie\AppData\Local\Temp\codex-clipboard-7203239d-7c30-41ec-83ad-f081701c2ce6.png"
    ).read_bytes(),
    "latest_overlay": Path(
        r"C:\Users\Davie\AppData\Local\Temp\codex-clipboard-0e8e72b6-9822-4fbd-b154-965d45026ec6.png"
    ).read_bytes(),
    "clean": _candled_egg(),
    "dark": _candled_egg("dark"),
    "subtle": _candled_egg("subtle"),
    "bright": _candled_egg("bright"),
    "texture": _candled_egg(textured=True),
    "texture_subtle": _candled_egg("subtle", textured=True),
}


def remove_overlay(image: np.ndarray) -> np.ndarray:
    red = (
        (image[:, :, 2] > 180)
        & (image[:, :, 2] > image[:, :, 1] * 1.5)
        & (image[:, :, 2] > image[:, :, 0] * 1.5)
    )
    green = (
        (image[:, :, 1] > 130)
        & (image[:, :, 1] > image[:, :, 2] * 1.45)
        & (image[:, :, 1] > image[:, :, 0] * 1.45)
    )
    mask = np.where(red | green, 255, 0).astype(np.uint8)
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def pdf_pipeline(data: bytes, canny_low: int, canny_high: int) -> dict:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image.shape[1] in {638, 755}:
        image = remove_overlay(image)
    resized = cv2.resize(image, (1147, 633), interpolation=cv2.INTER_AREA)
    red = resized[:, :, 2]
    green = resized[:, :, 1]
    red_blur = cv2.GaussianBlur(red, (11, 11), 0)
    threshold, binary = cv2.threshold(
        red_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    if cv2.countNonZero(binary) > binary.size * 0.5:
        binary = cv2.bitwise_not(binary)
    green_roi = cv2.multiply(green, binary, scale=1.0 / 255.0)
    edges = cv2.Canny(green_roi, canny_low, canny_high, L2gradient=True)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    morphology = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    subtraction = cv2.subtract(binary, morphology)
    binary_inverse = cv2.bitwise_not(binary)
    subtraction_inverse = cv2.bitwise_not(subtraction)
    crack = cv2.subtract(subtraction_inverse, binary_inverse)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(crack, 8)
    components = []
    for index in range(1, count):
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        w = int(stats[index, cv2.CC_STAT_WIDTH])
        h = int(stats[index, cv2.CC_STAT_HEIGHT])
        pixels = int(stats[index, cv2.CC_STAT_AREA])
        component = np.where(
            labels[y:y + h, x:x + w] == index, 255, 0,
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
        )
        area = float(sum(cv2.contourArea(contour) for contour in contours))
        length = float(sum(cv2.arcLength(contour, False) for contour in contours))
        components.append({
            "box": (x, y, w, h),
            "pixels": pixels,
            "area": round(area, 1),
            "length": round(length, 1),
            "span": round(float(np.hypot(w, h)), 1),
        })
    components.sort(key=lambda item: item["length"], reverse=True)
    return {
        "otsu": round(float(threshold), 1),
        "binary_pixels": cv2.countNonZero(binary),
        "edge_pixels": cv2.countNonZero(edges),
        "crack_pixels": cv2.countNonZero(crack),
        "components": components[:12],
    }


for low, high in ((30, 90), (50, 150), (75, 175), (100, 200)):
    print("CANNY", low, high)
    for name, data in FILES.items():
        print(name, pdf_pipeline(data, low, high))
