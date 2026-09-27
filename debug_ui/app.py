import os
import sys
import base64
import tempfile
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure repository root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from flask import Flask, request, jsonify, render_template, render_template_string
import cv2

try:
    from detector.inputs.video_file import VideoFileSource
    from detector.core.detector import PotholeDetector
    from detector.core.depth_estimator import MetricDepthEstimator
    from detector.inputs.gps_file import GPSFileSource, GPSFixQuality
    from detector.packaging.record_builder import RecordBuilder
except ImportError:
    from inputs.video_file import VideoFileSource
    from core.detector import PotholeDetector
    from core.depth_estimator import MetricDepthEstimator
    from inputs.gps_file import GPSFileSource, GPSFixQuality
    from packaging.record_builder import RecordBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("debug_ui")

app = Flask(__name__)
# Allow uploads up to 500 MB
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

# Lazy-loaded singleton detector and depth estimator models to avoid reloading on each request
_detector: Optional[PotholeDetector] = None
_depth_estimator: Optional[MetricDepthEstimator] = None


def get_detector(conf_thresh: float = 0.40) -> PotholeDetector:
    global _detector
    if _detector is None or _detector.conf_thresh != conf_thresh:
        _detector = PotholeDetector(conf_thresh=conf_thresh)
    return _detector


def get_depth_estimator() -> MetricDepthEstimator:
    global _depth_estimator
    if _depth_estimator is None:
        _depth_estimator = MetricDepthEstimator(lazy_load=False)
    return _depth_estimator


HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OmniSight Detector - Local Debug UI</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --badge-bg: #0f172a;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 24px;
      line-height: 1.5;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
    }
    h1 { font-size: 1.6rem; font-weight: 700; color: #60a5fa; }
    .badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      background: var(--border);
      color: var(--text-muted);
    }
    .panel {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }
    .form-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }
    label {
      display: block;
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 6px;
      color: var(--text-muted);
    }
    input[type="file"], input[type="number"] {
      width: 100%;
      padding: 10px 12px;
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      font-size: 0.9rem;
    }
    input[type="file"]::file-selector-button {
      padding: 6px 12px;
      background: var(--primary);
      color: white;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      margin-right: 12px;
    }
    button.btn-submit {
      background: var(--primary);
      color: white;
      border: none;
      padding: 12px 24px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 1rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: background 0.15s ease;
    }
    button.btn-submit:hover { background: var(--primary-hover); }
    button.btn-submit:disabled { opacity: 0.6; cursor: not-allowed; }
    .stats-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .stat-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
      text-align: center;
    }
    .stat-num { font-size: 2rem; font-weight: 700; color: #38bdf8; }
    .stat-label { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; }
    .records-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 20px;
    }
    .record-card {
      background: #1e293b;
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .thumb-wrap {
      width: 100%;
      height: 220px;
      background: #000;
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .thumb-wrap img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }
    .record-body { padding: 16px; flex: 1; display: flex; flex-direction: column; gap: 8px; font-size: 0.85rem; }
    .tag-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 4px; }
    .tag {
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 600;
    }
    .tag-ok { background: #064e3b; color: #34d399; }
    .tag-outlier { background: #450a0a; color: #f87171; }
    .tag-gps-matched { background: #1e3a8a; color: #93c5fd; }
    .tag-gps-interp { background: #312e81; color: #c7d2fe; }
    .tag-gps-nofix { background: #334155; color: #cbd5e1; }
    .record-meta { color: var(--text-muted); line-height: 1.6; }
    .loading-spinner {
      display: none;
      width: 20px;
      height: 20px;
      border: 3px solid rgba(255,255,255,0.3);
      border-radius: 50%;
      border-top-color: white;
      animation: spin 1s ease-in-out infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>OmniSight Road Condition Detector</h1>
        <p style="color: var(--text-muted); font-size: 0.9rem;">Local In-Process Debug & Inspection UI</p>
      </div>
      <span class="badge">Local Tooling</span>
    </header>

    <div class="panel">
      <form id="uploadForm">
        <div class="form-grid">
          <div>
            <label for="videoFile">Video File (.mp4) *</label>
            <input type="file" id="videoFile" name="video" accept=".mp4" required>
          </div>
          <div>
            <label for="gpsFile">GPS CSV Log (optional)</label>
            <input type="file" id="gpsFile" name="gps" accept=".csv,.txt">
          </div>
          <div>
            <label for="fpsInput">Sampling Rate (FPS)</label>
            <input type="number" id="fpsInput" name="fps" value="2.0" step="0.5" min="0.5" max="30">
          </div>
          <div>
            <label for="confInput">Confidence Threshold</label>
            <input type="number" id="confInput" name="conf_thresh" value="0.40" step="0.05" min="0.1" max="1.0">
          </div>
        </div>
        <button type="submit" class="btn-submit" id="submitBtn">
          <span class="loading-spinner" id="spinner"></span>
          <span id="btnText">Analyze Video</span>
        </button>
      </form>
    </div>

    <div id="statsSection" style="display: none;">
      <div class="stats-row">
        <div class="stat-card">
          <div class="stat-num" id="statFrames">0</div>
          <div class="stat-label">Frames Processed</div>
        </div>
        <div class="stat-card">
          <div class="stat-num" id="statDetections">0</div>
          <div class="stat-label">Potholes Detected</div>
        </div>
        <div class="stat-card">
          <div class="stat-num" id="statValidDepth" style="color: #34d399;">0</div>
          <div class="stat-label">Valid Depth (0-30cm)</div>
        </div>
        <div class="stat-card">
          <div class="stat-num" id="statGPS" style="color: #60a5fa;">0%</div>
          <div class="stat-label">GPS Fix Rate</div>
        </div>
      </div>
      <div class="records-grid" id="recordsGrid"></div>
    </div>
  </div>

  <script>
    const form = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');
    const spinner = document.getElementById('spinner');
    const btnText = document.getElementById('btnText');
    const statsSection = document.getElementById('statsSection');
    const recordsGrid = document.getElementById('recordsGrid');

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const videoInput = document.getElementById('videoFile');
      if (!videoInput.files[0]) {
        alert('Please choose a video file.');
        return;
      }

      const formData = new FormData(form);
      submitBtn.disabled = true;
      spinner.style.display = 'inline-block';
      btnText.innerText = 'Analyzing Frames & Estimating Depth...';

      try {
        const response = await fetch('/analyze', {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          const errData = await response.json();
          throw new Error(errData.error || ('HTTP ' + response.status));
        }

        const data = await response.json();
        renderResults(data);
      } catch (err) {
        alert('Analysis error: ' + err.message);
      } finally {
        submitBtn.disabled = false;
        spinner.style.display = 'none';
        btnText.innerText = 'Analyze Video';
      }
    });

    function renderResults(data) {
      statsSection.style.display = 'block';
      document.getElementById('statFrames').innerText = data.total_frames_processed;
      document.getElementById('statDetections').innerText = data.total_detections;

      const validCount = data.records.filter(r => r.depth_quality_flag === 'ok').length;
      document.getElementById('statValidDepth').innerText = validCount;

      const gpsCount = data.records.filter(r => r.gps_fix_quality !== 'no_fix').length;
      const gpsRate = data.records.length > 0 ? ((gpsCount / data.records.length) * 100).toFixed(0) + '%' : (data.gps_provided ? '100%' : 'N/A');
      document.getElementById('statGPS').innerText = gpsRate;

      recordsGrid.innerHTML = '';
      if (data.records.length === 0) {
        recordsGrid.innerHTML = '<div style="color: var(--text-muted); padding: 20px;">No potholes detected above threshold.</div>';
        return;
      }

      data.records.forEach((rec, idx) => {
        const card = document.createElement('div');
        card.className = 'record-card';

        const depthTag = rec.depth_quality_flag === 'ok' 
          ? `<span class="tag tag-ok">Depth: ${rec.depth_cm} cm</span>` 
          : `<span class="tag tag-outlier">Depth: Outlier (${rec.depth_cm ?? 'null'})</span>`;

        let gpsTag = `<span class="tag tag-gps-nofix">GPS: no_fix</span>`;
        if (rec.gps_fix_quality === 'matched') {
          gpsTag = `<span class="tag tag-gps-matched">GPS: matched</span>`;
        } else if (rec.gps_fix_quality === 'interpolated') {
          gpsTag = `<span class="tag tag-gps-interp">GPS: interpolated</span>`;
        }

        const coordsText = (rec.latitude !== null && rec.longitude !== null)
          ? `${rec.latitude.toFixed(6)}, ${rec.longitude.toFixed(6)}`
          : 'null';

        card.innerHTML = `
          <div class="thumb-wrap">
            ${rec.thumbnail_base64 ? `<img src="${rec.thumbnail_base64}" alt="Pothole #${idx+1}">` : '<span style="color:#64748b;">No Image</span>'}
          </div>
          <div class="record-body">
            <div class="tag-row">
              <span class="tag tag-ok">Conf: ${(rec.confidence * 100).toFixed(0)}%</span>
              ${depthTag}
              ${gpsTag}
            </div>
            <div class="record-meta">
              <strong>ID:</strong> <code style="color:#cbd5e1;">${rec.pothole_id.slice(0, 8)}...</code><br>
              <strong>Frame:</strong> #${rec.metadata?.frame_id ?? idx} @ ${rec.timestamp.split('T')[1] || rec.timestamp}<br>
              <strong>Coordinates:</strong> ${coordsText}<br>
              <strong>Box:</strong> [${rec.bounding_box.x1}, ${rec.bounding_box.y1}, ${rec.bounding_box.x2}, ${rec.bounding_box.y2}]
            </div>
          </div>
        `;
        recordsGrid.appendChild(card);
      });
    }
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    """Serves the interactive local debugging dashboard."""
    try:
        return render_template("index.html")
    except Exception:
        return render_template_string(HTML_DASHBOARD)


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze:
    Multipart form upload containing:
      - 'video' (required): an .mp4 video file
      - 'gps' (optional): a GPS CSV file
      - 'fps' (optional, default 2.0): sample rate
      - 'conf_thresh' (optional, default 0.40): YOLO confidence threshold
    """
    if "video" not in request.files or not request.files["video"].filename:
        return jsonify({"error": "Missing required 'video' file in form upload."}), 400

    video_file = request.files["video"]
    gps_file = request.files.get("gps")

    try:
        sample_fps = float(request.form.get("fps", 2.0))
    except (ValueError, TypeError):
        sample_fps = 2.0

    try:
        conf_thresh = float(request.form.get("conf_thresh", 0.40))
    except (ValueError, TypeError):
        conf_thresh = 0.40

    temp_dir = tempfile.mkdtemp(prefix="omnisight_debug_")

    try:
        # 1. Save uploaded video to temp directory
        video_filename = video_file.filename or "uploaded.mp4"
        video_path = os.path.join(temp_dir, video_filename)
        video_file.save(video_path)

        # 2. Save GPS file if provided
        gps_source: Optional[GPSFileSource] = None
        gps_provided = False
        video_start_time = None

        if gps_file and gps_file.filename:
            gps_path = os.path.join(temp_dir, gps_file.filename)
            gps_file.save(gps_path)
            try:
                gps_source = GPSFileSource(gps_path)
                gps_provided = True
                if gps_source.coordinates:
                    video_start_time = gps_source.coordinates[0].timestamp
                    logger.info(f"Synchronized video start time with GPS track: {video_start_time.isoformat()}")
            except Exception as e:
                logger.warning(f"Could not load uploaded GPS file: {e}")
                gps_source = None

        # 3. Instantiate pipeline components in-process
        video_source = VideoFileSource(
            filepath=video_path,
            sample_fps=sample_fps,
            start_time=video_start_time,
        )
        detector = get_detector(conf_thresh=conf_thresh)
        depth_estimator = get_depth_estimator()
        record_builder = RecordBuilder(video_filename=video_filename)

        output_records: List[Dict[str, Any]] = []
        total_frames_processed = 0

        # 4. Process video frames
        for frame in video_source.get_frames():
            total_frames_processed += 1
            bgr_image = frame.bgr_image
            frame_ts = frame.timestamp
            h, w = bgr_image.shape[:2]

            # Detect potholes
            detections = detector.detect(bgr_image)
            if not detections:
                continue

            # Estimate depth map once for the frame
            try:
                depth_map = depth_estimator.estimate_depth_map(bgr_image)
            except Exception as e:
                logger.error(f"Depth estimation failed for frame {frame.frame_id}: {e}")
                depth_map = None

            # GPS matching if GPS source provided
            if gps_source is not None:
                gps_coord, fix_quality = gps_source.get_coordinate_at_time(frame_ts)
            else:
                gps_coord, fix_quality = None, "no_fix"

            # Process each detection in frame
            for det in detections:
                if depth_map is not None:
                    depth_result = depth_estimator.compute_depth_from_map(depth_map, det)
                else:
                    depth_result = (None, "rejected_outlier")

                depth_cm, depth_flag = depth_result

                # Build standard PotholeRecord
                record = record_builder.build_record(
                    frame_data=frame,
                    detection=det,
                    gps_coord=(gps_coord, fix_quality),
                    depth_cm=depth_result,
                )

                record_dict = record_builder.record_to_dict(record)

                # Generate base64 annotated thumbnail
                annotated_crop = bgr_image.copy()
                box_color = (0, 0, 255) if depth_flag == "rejected_outlier" else (0, 255, 0)
                cv2.rectangle(
                    annotated_crop,
                    (det.x1, det.y1),
                    (det.x2, det.y2),
                    box_color,
                    3,
                )
                label_text = f"Pothole {det.confidence:.2f} | {depth_cm}cm ({depth_flag})"
                cv2.putText(
                    annotated_crop,
                    label_text,
                    (det.x1, max(24, det.y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Crop around bounding box with generous padding for clear thumbnail viewing
                pad_x = max(40, int((det.x2 - det.x1) * 0.3))
                pad_y = max(40, int((det.y2 - det.y1) * 0.3))
                crop_x1 = max(0, det.x1 - pad_x)
                crop_y1 = max(0, det.y1 - pad_y)
                crop_x2 = min(w, det.x2 + pad_x)
                crop_y2 = min(h, det.y2 + pad_y)

                thumb_patch = annotated_crop[crop_y1:crop_y2, crop_x1:crop_x2]
                if thumb_patch.size == 0:
                    thumb_patch = annotated_crop

                # Resize thumbnail to standard display dimensions
                thumb_patch = cv2.resize(thumb_patch, (360, 240))
                ret, buf = cv2.imencode(".jpg", thumb_patch, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    record_dict["thumbnail_base64"] = (
                        "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
                    )
                else:
                    record_dict["thumbnail_base64"] = None

                output_records.append(record_dict)

        return jsonify({
            "success": True,
            "video_filename": video_filename,
            "total_frames_processed": total_frames_processed,
            "total_detections": len(output_records),
            "gps_provided": gps_provided,
            "records": output_records,
        })

    except Exception as e:
        logger.exception(f"Pipeline execution error in /analyze: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temporary upload files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f" OmniSight Local Debug UI running at http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
