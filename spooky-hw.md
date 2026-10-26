# Spooky HW: Build a Real-Time Monster Mirror in the Browser

**Course:** Computer Vision (undergraduate / early graduate)
**Format:** individual programming assignment, 4 tasks, each building on the last
**Suggested time:** 2 weeks (about 12–18 hours)
**Language/platform:** TypeScript in the browser (runs on a laptop webcam or a phone)

## Overview

In this assignment you will build the core of **ScareMeter**, a game that watches your face through a webcam and scores how scary you look. You will build it in four steps:

1. **Mirror the video.** Show the webcam as a selfie-style mirror on a `<canvas>`.
2. **Detect face landmarks.** Run a neural face-mesh model on every frame and draw its 478 points on screen.
3. **Measure the mouth.** Turn the landmarks into a live "mouth open" meter bar.
4. **Add a face filter.** Draw cartoon demon ears and glowing demon eyes that follow your head.

All of the computer vision runs **on the device**. No video leaves the browser, and once loaded the app works without a network connection. This is the same design used by the production app at <https://scaremeter.app>.

The ScareMeter source code is a **reference implementation**. Each task lists the files that solve the same problem. Read them to learn *how* a production app handles the details, but write your own code: you will be asked to explain every line you submit.

### Learning objectives

By the end of this assignment you should be able to:

- Capture camera frames and reason about **image coordinate systems** (pixels vs. normalized coordinates, mirrored vs. raw).
- Apply **2-D affine transforms** (translate, rotate, scale, reflect) and explain them as matrices.
- Run a pretrained **face landmark model** (MediaPipe Face Landmarker) in a real-time video loop and explain its pipeline (detector → mesh regressor → blendshapes).
- Design a **scale-invariant geometric feature** (mouth aspect ratio) and compare it with a learned feature (the `jawOpen` blendshape).
- Handle real-time signal problems: **jitter, calibration to a neutral baseline, normalization and temporal smoothing**.
- Render **face-anchored graphics** in a local "face coordinate frame" that follows head position, scale and roll.

## Tools and modules

Everything you need is free and open source. Use these references throughout.

| Tool / module | What it is used for | Reference |
|---|---|---|
| **Node.js 22+** | Runs the build tools | <https://nodejs.org/en/download> |
| **pnpm** (or npm) | Installs packages | <https://pnpm.io/installation> |
| **Vite** | Dev server and bundler (serves `localhost`, which browsers treat as secure for camera access) | <https://vite.dev/guide/> |
| **TypeScript** | Typed JavaScript; catches coordinate/units mistakes early | <https://www.typescriptlang.org/docs/handbook/intro.html> |
| **`MediaDevices.getUserMedia()`** | Opens the webcam (and microphone) | <https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia> |
| **Secure contexts** | Why the camera only works on `https://` or `localhost` | <https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts> |
| **`<video>` / `HTMLVideoElement`** | Receives the camera stream | <https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement> |
| **Canvas 2D API** | Draws video, points, bars and filters | <https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D> |
| Canvas transforms (`translate`, `rotate`, `scale`, `save`/`restore`) | Mirroring and face-space drawing | <https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial/Transformations> |
| `createRadialGradient`, `globalCompositeOperation` | Glowing eyes (additive "lighter" blending) | <https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/createRadialGradient>, <https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/globalCompositeOperation> |
| **`requestAnimationFrame`** | The render loop (once per display refresh) | <https://developer.mozilla.org/en-US/docs/Web/API/Window/requestAnimationFrame> |
| **`requestVideoFrameCallback`** | Runs inference once per *new camera frame* (with an rAF fallback) | <https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback> |
| **MediaPipe Tasks Vision** (`@mediapipe/tasks-vision`) | The face landmark model runtime (WebAssembly + WebGL/GPU) | npm: <https://www.npmjs.com/package/@mediapipe/tasks-vision> |
| MediaPipe **Face Landmarker** guide | Model overview, outputs, options | <https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker> |
| Face Landmarker **web guide** | JavaScript setup, `detectForVideo`, running modes | <https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/web_js> |
| Face Landmarker **model file** (`face_landmarker.task`, float16) | The trained model bundle | <https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task> |
| **Face mesh landmark index map** | Which index is which point (lips, eyes, cheeks…) | <https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png> |
| **Blendshape names** (52 ARKit-style coefficients such as `jawOpen`, `eyeWideLeft`) | Learned expression features | <https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation> |
| **Vitest** (optional) | Unit-testing your pure math functions | <https://vitest.dev/guide/> |

### Background reading (the models you are using)

- V. Bazarevsky et al., **"BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs,"** 2019. arXiv:[1907.05047](https://arxiv.org/abs/1907.05047). *The first stage: finds the face box.*
- Y. Kartynnik et al., **"Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs,"** 2019. arXiv:[1907.06724](https://arxiv.org/abs/1907.06724). *The second stage: regresses the 3-D face mesh.*
- I. Grishchenko et al., **"Attention Mesh: High-fidelity Face Mesh Prediction in Real-time,"** 2020. arXiv:[2006.10962](https://arxiv.org/abs/2006.10962). *Refined lips, eyes and irises (why the model has 478 points, not 468).*
- I. Grishchenko et al., **"Blendshapes GHUM: Real-time Monocular Facial Blendshape Prediction,"** 2023. arXiv:[2309.05782](https://arxiv.org/abs/2309.05782). *The third stage: landmarks → 52 expression coefficients.*
- G. Casiez, N. Roussel, D. Vogel, **"1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems,"** CHI 2012. <https://gery.casiez.net/1euro/>. *Optional, for the smoothing extension.*

## Setup (Task 0)

1. Install Node.js 22+ and pnpm.
2. Create a project:
   ```sh
   pnpm create vite spooky-hw --template vanilla-ts
   cd spooky-hw
   pnpm add @mediapipe/tasks-vision
   pnpm dev
   ```
3. **Host the model locally.** Download `face_landmarker.task` (link above) into `public/models/`. Copy the MediaPipe WebAssembly runtime from `node_modules/@mediapipe/tasks-vision/wasm/` into `public/mediapipe/wasm/`. Your app must make **no requests to other websites** at runtime. Check the browser DevTools Network tab to confirm.
   *Reference:* `public/models/README.md` and the `mediapipe-wasm` plugin in `vite.config.ts` show how ScareMeter copies the runtime at build time.
4. Open `http://localhost:5173`. To test on a phone you need HTTPS (for example a tunnel, or a static host such as Cloudflare Pages or GitHub Pages).

Suggested file layout (you may change it):

```text
src/
  main.ts        // wiring: start camera, start loops
  camera.ts      // Task 1
  mirror.ts      // Task 1
  landmarks.ts   // Task 2
  mouth.ts       // Task 3 (pure functions, no DOM: easy to unit-test)
  meter.ts       // Task 3
  filter.ts      // Task 4
```

## Task 1 — Mirrored video (15 points)

**Goal:** show the webcam on a full-window `<canvas>` as a mirror image, like a selfie camera.

**Requirements**

1. Request the **front camera** with `getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } } })`. If those constraints fail (`OverconstrainedError`), retry with `{ video: true }`.
2. Attach the stream to a hidden `<video>` element (`muted`, `playsInline`, `autoplay`). `playsInline` is required on iPhones.
3. Every animation frame, draw the video onto the canvas **mirrored horizontally**. Do this with a canvas transform, **not** CSS `transform: scaleX(-1)`.
4. Size the canvas to the video's real resolution (`video.videoWidth × video.videoHeight`) and scale it to the window with CSS (letterboxing is fine).
5. Show a clear message if the user denies permission (`NotAllowedError`), there is no camera (`NotFoundError`), or the camera is in use (`NotReadableError`).

**Hints**

```ts
function drawMirrored(ctx: CanvasRenderingContext2D, video: HTMLVideoElement, w: number, h: number) {
  ctx.save();
  // TODO: a translate and a scale that map pixel x to (w - x)
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();
}
```

**Written questions (answer in your README)**

- Q1.1 Write the 3×3 homogeneous matrix for the mirror transform you used. Show that applying it twice gives the identity.
- Q1.2 Why does the reference app mirror inside the canvas rather than with CSS? *(Hint: the canvas is also the source for recorded clips. Look at `canvas.captureStream()`.)*
- Q1.3 Text drawn on a mirrored canvas reads backwards. How would you draw a score label so that it is readable while the video stays mirrored?

**Reference implementation:** `src/capture/media.ts` (constraints, error mapping) and `drawVideo()` in `src/render/compositor.ts`.

## Task 2 — Face landmarks on screen (25 points)

**Goal:** run MediaPipe Face Landmarker on each new camera frame and draw all 478 landmarks as small dots on the mirrored video.

**Requirements**

1. Create the landmarker from your **local** files:
   ```ts
   import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

   const fileset = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
   const landmarker = await FaceLandmarker.createFromOptions(fileset, {
     baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate: 'GPU' },
     runningMode: 'VIDEO',
     numFaces: 1,
     outputFaceBlendshapes: true, // needed in Task 3
   });
   ```
   If creating with `delegate: 'GPU'` throws, retry with `'CPU'`.
2. Run inference **once per new video frame** with `requestVideoFrameCallback` (fall back to `requestAnimationFrame` where it is missing). Keep inference separate from drawing: the render loop draws the *most recent* result.
3. Call `landmarker.detectForVideo(video, timestampMs)`. MediaPipe requires **strictly increasing** timestamps. Guarantee this even if two callbacks get the same `performance.now()` value.
4. Draw every landmark as a filled circle of radius about 1–2 px. Landmarks are **normalized** to the raw (un-mirrored) frame: `x, y ∈ [0, 1]`. Convert them to canvas pixels and **apply the same mirror** as the video so the dots sit on your face.
5. Draw all dots in **one path** (`beginPath` → many `arc` → one `fill`). Show the measured tracking FPS and inference time (ms) in a corner.
6. When no face is found, draw nothing and show "No face found".

**Hints**

- A normalized point `(x, y)` maps to pixel `(x·W, y·H)` unmirrored, or `((1 − x)·W, y·H)` mirrored.
- `result.faceLandmarks[0]` is an array of `{x, y, z}`; `z` is relative depth (smaller is closer to the camera, roughly on the same scale as `x`).
- Smooth the FPS reading with an exponential moving average, e.g. `fps = 0.9·fps + 0.1·instant`.

**Written questions**

- Q2.1 Face Landmarker is a pipeline of models. Name the stages and what each one outputs (see the readings).
- Q2.2 Why is `runningMode: 'VIDEO'` faster than running in `'IMAGE'` mode on every frame? What does the model reuse between frames?
- Q2.3 Measure your FPS and inference time with the GPU delegate and with the CPU delegate. Report your device and both numbers. Explain the difference.
- Q2.4 Using the landmark index map, list the indices of the inner-lip ring and the two mouth corners. Mark them in a different color on screen and include a screenshot.

**Reference implementation:** `src/vision/trackers.ts` (local model paths, GPU→CPU fallback), `src/vision/engine.ts` (per-frame loop, monotonic timestamps, load shedding when inference is slow), `drawFaceDots()` in `src/render/overlay.ts`.

## Task 3 — Mouth open/closed meter (30 points)

**Goal:** show a vertical meter bar that fills as you open your mouth and empties when you close it.

You will compute the mouth openness **two ways** and compare them.

### 3a. Geometric feature: mouth aspect ratio (MAR)

Using face-mesh indices **13** (upper inner lip), **14** (lower inner lip), **61** and **291** (mouth corners):

```text
MAR = ‖p13 − p14‖ / ‖p61 − p291‖
```

Compute distances in **pixels** (multiply `x` by `W` and `y` by `H` first), because normalized coordinates are not isotropic when `W ≠ H`.

### 3b. Learned feature: the `jawOpen` blendshape

Read `result.faceBlendshapes[0].categories` and find the entry whose `categoryName` is `'jawOpen'`. Its `score` is in `[0, 1]`.

### 3c. Calibration, normalization and smoothing

Raw values differ from person to person (lip thickness, resting mouth shape). Make the meter **relative to the player's own resting face**:

1. **Calibrate:** when the app starts, ask the user to relax their face for 1.5 s. Record the **median** feature value over that window as the baseline `b`. (Why the median and not the mean? Answer in Q3.2.)
2. **Normalize** to `[0, 1]` with a clamped linear ramp:
   ```text
   value = clamp((feature − b − lo) / (hi − lo), 0, 1)
   ```
   Choose `lo` (a dead zone that ignores talking and jitter) and `hi` (a wide-open scream) from your own measurements. The reference app uses `lo = 0.05`, `hi = 0.6` for `jawOpen`.
3. **Smooth** the displayed value so the bar doesn't flicker. Use an exponential moving average `s ← s + α(value − s)` and pick α.
4. **Draw the bar:** a rounded rectangle that fills from the bottom, with a green → orange → red vertical gradient and a label. Draw the label so it reads correctly on the mirrored canvas (your answer to Q1.3).
5. Keep your feature math in `mouth.ts` as **pure functions** (no DOM, no globals) so you can unit-test them.

**Written questions**

- Q3.1 Why is MAR divided by mouth width? What happens to the unnormalized lip gap when you move toward or away from the camera? Show it with a quick experiment.
- Q3.2 Why use the median for calibration?
- Q3.3 Record a 10-second plot (log values to the console, or draw a scrolling chart) of MAR and `jawOpen` while you: keep your mouth closed, talk, open wide, smile broadly, and turn your head 45°. Which feature is more robust to head rotation? Which one confuses a broad smile with an open mouth? Explain using how each one is computed.
- Q3.4 What is the trade-off when choosing α? Measure the meter's lag for two α values.

**Unit tests (required):** at least 3 tests for your pure functions, for example: a closed mouth gives 0, a wide-open mouth gives 1, and MAR is unchanged when all points are scaled by 2.

**Reference implementation:** `toFaceFrame()` in `src/vision/faceFeatures.ts` (reading blendshapes) and its `FACE_MESH` lip indices; `faceBaseline()` and `mouthFrameValue()` in `src/score/faceScore.ts` (median baseline and ramp); `lerp01()` in `src/score/math.ts`; `drawMeter()` in `src/render/overlay.ts`; unit tests in `src/score/faceScore.test.ts`.

## Task 4 — Face filter: demon ears and demon eyes (30 points)

**Goal:** draw two cartoon **demon ears** and two **glowing demon eyes** that stick to the face as you move, turn and tilt your head.

### 4a. A face coordinate frame

Drawing in raw pixels breaks as soon as the head moves closer or tilts. Instead, build a **face frame** each frame:

- **Scale:** face width `s = ‖p234 − p454‖` (cheek to cheek, in pixels).
- **Up vector:** `u = normalize(p10 − p152)` (chin → forehead).
- **Roll angle:** `θ = atan2(u.x, −u.y)` (0 when the head is upright).

To draw a shape anchored at point `a`, use `translate(a)`, then `rotate(θ)`, then `scale(s, s)`. Inside that transform, **1 unit = one face width**, `+x` is across the face and `+y` is down the face. Design every shape once in these units.

Compute landmark pixels **after mirroring** (`x → (1 − x)·W`) so the filter lines up with the mirrored video.

### 4b. Demon ears

- Anchor each ear at a cheek landmark (234 and 454), moved up the face by about `0.1·s` along `u`.
- Draw a pointed, curved ear (two `quadraticCurveTo` calls work well) with a dark outline and a lighter inner shape.
- Design the ear for one side only. For the other side, reflect with `scale(−1, 1)` inside the face frame. Decide which side you are on with the sign of the dot product between `(anchor − noseTip)` and the face's "right" vector `(−u.y, u.x)`.

### 4c. Demon eyes

- Center each eye on its **iris** landmark (468 and 473; these exist only in the 478-point model).
- Size each glow from the eye's width (`‖outer − inner‖`, indices 33/133 and 263/362).
- Draw a radial gradient: white center → red → transparent edge, with `globalCompositeOperation = 'lighter'` (additive blending, so it looks emissive).
- **Make it react:** grow the glow when the eyes open wide (the `eyeWideLeft`/`eyeWideRight` blendshapes) and add a slight flicker over time.

### 4d. Stability

- Smooth the landmark positions before drawing (an EMA per point, α ≈ 0.5). When the face is lost for more than ~0.4 s, reset the smoother so the filter snaps to the new face instead of sliding across the screen.
- **Score the raw frame, not the filtered one.** If the meter from Task 3 used pixels from the canvas *after* the filter was drawn, a filter could change the score. Make sure your features come only from the model's output on the raw video.

**Written questions**

- Q4.1 Write the full 2-D affine matrix for "translate to anchor, rotate by θ, scale by s" and show where face-space point `(0.3, −0.2)` lands for an anchor of `(400, 300)`, θ = 10° and s = 200 px.
- Q4.2 The roll angle is computed from only two landmarks. What happens to the ears when you turn your head (yaw) 60°? Propose a fix that uses the facial transformation matrix (`outputFacialTransformationMatrixes: true`).
- Q4.3 Ears drawn on top of the video also cover your hair and background, even when they should be *behind* the head as you turn. Describe how an **occluder** (an invisible face mesh drawn into a depth buffer, e.g. with Three.js) solves this.
- Q4.4 Compare the EMA smoother with the 1€ filter. Why does the 1€ filter reduce jitter at rest *and* lag during fast motion?

**Reference implementation:** `faceGeom()` and `inFaceSpace()` in `src/effects/geometry.ts` (the face frame) and its `LM` landmark table; `ears()` (demon branch) and `eyes()` (glow branch) in `src/effects/layers.ts`; `LandmarkSmoother` in `src/effects/smoothing.ts`; the `demon` entry in `src/effects/catalog.ts` shows how a filter is described as data.

## Deliverables

Submit a Git repository (or a zip) containing:

1. Your source code, buildable with `pnpm install && pnpm build`.
2. `README.md` with:
   - how to run it,
   - answers to all written questions (Q1.1–Q4.4),
   - your device/browser and measured FPS for each task,
   - screenshots for Task 2 (landmarks, with lips highlighted), Task 3 (meter at rest and wide open) and Task 4 (filter upright and with the head tilted).
3. A **20–40 second screen recording** showing all four tasks working, including head tilt and moving toward/away from the camera.
4. Your unit tests (`pnpm test`).

## Grading rubric (100 points)

| Task | Points | Full credit requires |
|---|---:|---|
| 1. Mirrored video | 15 | Canvas-transform mirror; correct sizing; friendly permission errors; Q1.1–Q1.3 |
| 2. Landmarks | 25 | Local model + WASM (no third-party requests); per-frame inference with monotonic timestamps; dots aligned on the mirrored image; FPS/inference HUD; Q2.1–Q2.4 |
| 3. Mouth meter | 30 | MAR and `jawOpen` both implemented; median calibration; clamped ramp; smoothing; readable bar; ≥ 3 passing unit tests; comparison plot and Q3.1–Q3.4 |
| 4. Demon filter | 30 | Ears and eyes follow position, scale and roll; mirrored ear design; reactive glow; landmark smoothing with reset; scoring uses the raw frame; Q4.1–Q4.4 |

**Deductions:** up to −10 for loading models or scripts from another website at runtime; up to −10 for code you cannot explain in a short check-in; −5 if the app crashes when the face leaves the frame.

## Extensions (up to +10 bonus)

- **Fang Factor:** crop the inner-lip polygon from the *raw* frame, convert pixels to CIELAB and estimate the fraction of tooth-colored pixels. Discuss fairness across skin tones, lip colors and lighting. *(Reference: `src/vision/mouthSampler.ts`, `src/vision/teeth.ts`.)*
- **Scream meter:** add a microphone loudness bar with Web Audio's `AnalyserNode` or an `AudioWorklet` (<https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API>). *(Reference: `src/audio/`.)*
- **Record a clip:** record your canvas with `canvas.captureStream()` and `MediaRecorder` (<https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder>). *(Reference: `src/record/recorder.ts`.)*
- **Head pose:** decompose the facial transformation matrix into yaw/pitch/roll and use it to fix Q4.2. *(Reference: `eulerFromMatrix()` in `src/vision/faceFeatures.ts`.)*
- **1€ filter:** replace the EMA with a 1€ filter and show jitter/lag measurements before and after.

## Academic integrity and privacy

- You may read the ScareMeter reference code and the MediaPipe documentation. Do not copy reference files wholesale; cite any function you adapt in a code comment.
- Never upload classmates' faces or recordings anywhere. Recordings you submit should show only you, or people who have given consent.
- Keep all processing on-device, as the reference app does: no analytics, no cloud APIs, no remote model hosting.
