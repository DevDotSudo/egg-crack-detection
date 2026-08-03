import cv2
import numpy as np

from app.detection.models import ComponentFeatures, EggRegion


class OverlayRenderer:
    def render(
        self,
        image: np.ndarray,
        egg: EggRegion,
        crack_mask: np.ndarray,
        components: list[ComponentFeatures],
    ) -> np.ndarray:
        overlay = image.copy()
        cv2.drawContours(overlay, [egg.contour], -1, (0, 220, 0), 1, cv2.LINE_8)
        if cv2.countNonZero(crack_mask) > 0:
            contours, _ = cv2.findContours(
                (crack_mask > 0).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1, cv2.LINE_8)
        for index, component in enumerate([value for value in components if value.accepted], start=1):
            x, y, width, height = component.bbox
            cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 255), 1, cv2.LINE_8)
            cv2.putText(
                overlay,
                str(index),
                (x, max(16, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        return overlay
