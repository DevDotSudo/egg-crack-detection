import cv2
import numpy as np

from app.detection.models import EggRegion, PaperBaselineResult


class PaperBaselineDetector:
    def detect(self, image: np.ndarray, egg: EggRegion) -> PaperBaselineResult:
        red = image[:, :, 2]
        green = image[:, :, 1]
        red_blur = cv2.GaussianBlur(red, (11, 11), 0)
        _, binary = cv2.threshold(red_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_and(binary, egg.full_mask)
        masked_green = cv2.bitwise_and(green, green, mask=binary)
        edges = cv2.Canny(masked_green, 20, 60, L2gradient=True)
        cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cross, iterations=1)
        boundary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, cross)
        boundary = cv2.dilate(boundary, cross, iterations=2)
        crack = cv2.bitwise_and(closed, cv2.bitwise_not(boundary))
        crack = cv2.bitwise_and(crack, egg.inner_mask)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(crack, connectivity=8)
        accepted = np.zeros_like(crack)
        components = 0
        score = 0.0
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            width = int(stats[index, cv2.CC_STAT_WIDTH])
            height = int(stats[index, cv2.CC_STAT_HEIGHT])
            span = float(np.hypot(width, height))
            if area < 5 or span < egg.minor_axis * 0.06:
                continue
            component = np.where(labels == index, 255, 0).astype(np.uint8)
            accepted = cv2.bitwise_or(accepted, component)
            components += 1
            score = max(score, min(1.0, span / max(egg.minor_axis * 0.35, 1.0)))
        return PaperBaselineResult(
            mask=accepted,
            crack=components > 0,
            score=float(score),
            components=components,
            steps={
                'paper_red_blur': red_blur,
                'paper_binary_egg': binary,
                'paper_green_roi': masked_green,
                'paper_edges': edges,
                'paper_morphology': closed,
                'paper_crack_mask': accepted,
            },
        )
