# Backend 2.3 fixes

- Camera and uploaded images preserve their received orientation by default.
- Optional orientation correction remains available through `EGG_CAMERA_ORIENTATION_FIX`.
- Removed the final skeleton-only mask that covered only the crack centerline.
- The final mask now contains the complete validated crack component area.
- The one-pixel crack polygon follows the outer boundary of the complete detected area.
- Narrow directional extension still completes faint connected crack sections without broad texture flooding.
- Multiple visible cracks remain supported.
- Mamdani egg-size and crack-size classification remain unchanged.
- 36 tests pass.
