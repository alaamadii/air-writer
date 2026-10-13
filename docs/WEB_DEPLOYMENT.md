# Web Application and Vercel Deployment

This guide covers the browser version of Air Writer. The Python/OpenCV desktop application is
still available and is documented in the root `README.md`.

## Why the Web Version Is Separate

Vercel cannot run an interactive OpenCV desktop window or access a visitor's webcam from a
Python server. The browser is the correct runtime for a public camera application:

1. `getUserMedia()` opens the camera after explicit user permission.
2. MediaPipe Tasks Vision runs the Hand Landmarker locally through WebAssembly.
3. The browser tries the GPU delegate first and falls back to CPU when GPU execution is not
   available.
4. Canvas renders the drawing, eraser, hand landmarks, and cursor at the camera resolution.
5. Vercel serves static HTML, CSS, JavaScript, the model, and cache/security headers only.

This design avoids server round trips on every frame. It preserves interaction quality and
latency while keeping the video private.

## Runtime Architecture

```text
Camera (browser permission)
        |
        v
HTMLVideoElement -- MediaPipe Hand Landmarker (VIDEO mode, GPU -> CPU fallback)
        |                         |
        |                         v
        |              21 normalized + world landmarks
        |                         |
        |                         v
        |              3D finger geometry / gestures
        |                         |
        |              One Euro filter / time stabilizer
        |                         |
        +-------------------------+-------------------+
                                  v                   v
                         persistent draw canvas   landmark overlay
```

The web port keeps the desktop safeguards:

- Joint-angle and palm-scale finger classification, independent of screen rotation/handedness.
- World landmarks when available, normalized landmark fallback otherwise.
- 50 ms gesture stabilization and 100 ms release grace.
- 600 ms one-shot open-palm clear hold.
- Timestamp-aware One Euro index-tip smoothing.
- Stroke reset on mode changes, hand loss, toolbar contact, or large position jumps.

## Local Development

Install Node.js 20.19 or newer, then run from the repository root:

```bash
npm install
npm run dev
```

Vite prints a local URL, usually `http://localhost:5173`. `localhost` is allowed to request the
camera without a manually configured TLS certificate.

Production verification:

```bash
npm run test:web
npm run build
npm run preview
```

The optimized site is generated in `dist/`. The MediaPipe model is emitted as a fingerprinted
asset. The build also copies the package's WebAssembly runtime into `dist/mediapipe-wasm/`, so
the deployed application has no runtime dependency on a third-party CDN. Both receive long-lived
cache headers.

## Free Vercel Deployment from GitHub

Vercel's Hobby plan is free for personal, non-commercial projects. Air Writer is fully static,
so it does not consume server-function execution for hand tracking.

1. Push this repository to GitHub.
2. Sign in at [vercel.com](https://vercel.com/) with GitHub.
3. Select **Add New → Project**.
4. Import `alaamadii/air-writer`.
5. Keep **Root Directory** as the repository root (`./`).
6. Vercel reads `vercel.json`; confirm these detected settings:
   - Framework Preset: `Vite`
   - Build Command: `npm run build`
   - Output Directory: `dist`
   - Install Command: `npm install` (or the automatic default)
7. Do not add environment variables; the application has no API keys or backend secrets.
8. Select **Deploy**.

Every later push to `main` creates a new production deployment. Pull requests receive preview
deployments. HTTPS is automatic, which is required by browsers for production camera access.

## Browser Use

1. Open the HTTPS deployment.
2. Press **تشغيل الكاميرا** and allow camera permission.
3. Keep one hand visible with reasonable front lighting.
4. Use these gestures:

| Gesture | Action |
| --- | --- |
| Index finger | Draw |
| Index + middle | Move/select a toolbar item after a 450 ms hold |
| Index + middle + ring | Erase |
| Open palm for about 600 ms | Clear once |

Mouse and touch input can also select toolbar controls. **حفظ صورة** downloads one PNG containing
the mirrored camera frame and current drawing. It is generated and downloaded locally.

## Quality and Performance

- The requested camera constraint is 1280×720 at an ideal 30 FPS (up to 60 when available).
- MediaPipe video mode reuses tracking between detections instead of running palm detection for
  every frame.
- The video and both canvas layers share the camera's native dimensions, avoiding coordinate
  offsets and unnecessary image scaling.
- Rendering uses `requestAnimationFrame`, and inference runs only when the video has a new frame.
- The model and application chunks are fingerprinted and cached by Vercel's CDN.

Performance still depends on the visitor's device, browser, camera, lighting, and GPU driver. A
CPU fallback keeps the application functional on devices where the WebGL/GPU delegate is not
available.

## Privacy and Security

- Camera permission is controlled by the browser and can be revoked at any time.
- No camera frame, landmark, or drawing is sent to this repository or a Vercel function.
- There is no analytics SDK, account system, database, or server upload endpoint.
- `Permissions-Policy: camera=(self)` limits camera permission to the deployed origin.
- The stop button closes all active media tracks; leaving the page does the same.

## Troubleshooting

### Camera permission was denied

Use the lock/camera icon in the browser address bar, allow camera access for the site, then reload.

### Camera only fails on a custom host

Confirm the page uses `https://`. Browsers block production camera access on insecure HTTP pages.

### The model does not load

Check that the Vercel build completed, that `hand_landmarker.task` appears as a generated file
under `dist/assets/`, and that `dist/mediapipe-wasm/` contains the `.js` and `.wasm` runtime files.

### Tracking is slow on an older device

Close camera-heavy tabs and other video applications. Use current Chrome, Edge, or Safari. The
diagnostics control shows inference FPS and whether `GPU` or `CPU` is active.

### Mirroring or coordinates look wrong

The preview is mirrored for natural interaction, and the landmark x-coordinate is mirrored once
in JavaScript. Do not add another CSS transform to either canvas.
