import os
import sys
import argparse
import logging
from typing import Optional, List
from datetime import datetime

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from detector.inputs.video_file import VideoFileSource
    from detector.core.detector import PotholeDetector
    from detector.core.depth_estimator import MetricDepthEstimator
    from detector.inputs.gps_file import GPSFileSource, GPSFixQuality
    from detector.packaging.record_builder import RecordBuilder
    from detector.transmission.http_transmitter import HttpTransmitter
    from detector.transmission.local_queue import LocalQueueManager
except ImportError:
    from inputs.video_file import VideoFileSource
    from core.detector import PotholeDetector
    from core.depth_estimator import MetricDepthEstimator
    from inputs.gps_file import GPSFileSource, GPSFixQuality
    from packaging.record_builder import RecordBuilder
    from transmission.http_transmitter import HttpTransmitter
    from transmission.local_queue import LocalQueueManager

# Configure clean console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orchestrator")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OmniSight Edge Detector Pipeline: Ingest video, detect potholes, estimate depth, synchronize GPS, and transmit records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video (.mp4)",
    )
    parser.add_argument(
        "--gps",
        type=str,
        required=True,
        help="Path to the timestamped GPS CSV log",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        required=True,
        help="Backend ingestion endpoint URL (e.g. http://127.0.0.1:8000/api/potholes)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Target frame sampling rate in frames per second",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.40,
        help="YOLOv8 pothole confidence threshold",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Video start time (ISO 8601). If omitted, automatically synchronizes with the first GPS fix.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save an annotated debug video drawing bounding boxes and depth labels on detected frames",
    )
    parser.add_argument(
        "--output-debug-video",
        type=str,
        default="output/annotated.mp4",
        help="Output filepath for annotated debug video (used when --visualize is set)",
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace):
    # Verify input paths exist
    if not os.path.exists(args.video):
        logger.error(f"Input video file not found: {args.video}")
        sys.exit(1)
    if not os.path.exists(args.gps):
        logger.error(f"Input GPS log file not found: {args.gps}")
        sys.exit(1)

    logger.info("=" * 65)
    logger.info("STARTING OMNISIGHT DETECTION PIPELINE")
    logger.info(f"Video File:      {args.video}")
    logger.info(f"GPS Track:       {args.gps}")
    logger.info(f"Backend URL:     {args.server_url}")
    logger.info(f"Sampling Rate:   {args.fps} FPS")
    logger.info(f"Conf Threshold:  {args.conf_thresh}")
    logger.info("=" * 65)

    # 1. Initialize GPS & Video Sources
    logger.info("Initializing pipeline modules...")
    gps_source = GPSFileSource(filepath=args.gps)

    video_start_time = None
    if args.start_time:
        video_start_time = datetime.fromisoformat(args.start_time.replace("Z", "+00:00"))
    elif gps_source.coordinates:
        video_start_time = gps_source.coordinates[0].timestamp
        logger.info(f"Synchronized video start time with GPS track start: {video_start_time.isoformat()}")

    video_source = VideoFileSource(filepath=args.video, sample_fps=args.fps, start_time=video_start_time)
    detector = PotholeDetector(conf_thresh=args.conf_thresh)
    depth_estimator = MetricDepthEstimator()
    record_builder = RecordBuilder(video_filename=os.path.basename(args.video))
    queue_manager = LocalQueueManager("queue.db")
    transmitter = HttpTransmitter(server_url=args.server_url, queue_manager=queue_manager)

    # 2. Setup Debug Video Writer if requested
    video_writer = None
    if args.visualize:
        if cv2 is None:
            logger.warning("OpenCV is not available; disabling debug video visualization.")
        else:
            out_dir = os.path.dirname(args.output_debug_video)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            # Use mp4v fourcc
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # Width and height will be initialized on first frame
            logger.info(f"Debug video visualization enabled -> {args.output_debug_video}")

    # Pipeline Metrics
    total_frames = 0
    gps_fixed_frames = 0
    total_detections = 0
    transmitted_count = 0
    queued_count = 0
    quarantined_count = 0

    try:
        for frame in video_source.get_frames():
            total_frames += 1
            frame_ts = frame.timestamp
            frame_id = frame.frame_id
            bgr_image = frame.bgr_image

            # Step 1: GPS Synchronization
            gps_coord, fix_quality = gps_source.get_coordinate_at_time(frame_ts)
            has_fix = fix_quality in (GPSFixQuality.MATCHED, GPSFixQuality.INTERPOLATED)
            if has_fix:
                gps_fixed_frames += 1

            gps_summary = (
                f"({gps_coord.latitude:.5f}, {gps_coord.longitude:.5f}) [{fix_quality}]"
                if gps_coord is not None
                else f"[no_fix]"
            )

            # Step 2: Object Detection (YOLOv8)
            detections = detector.detect(bgr_image)
            num_dets = len(detections)

            logger.info(
                f"[Frame {frame_id:04d} @ {frame_ts.strftime('%H:%M:%S.%f')[:-3]}] "
                f"GPS: {gps_summary} | Potholes Detected: {num_dets}"
            )

            if num_dets == 0:
                continue

            total_detections += num_dets

            # Step 3: Metric Depth Estimation
            # Compute metric depth map once for the entire frame to reuse across detections
            try:
                depth_map = depth_estimator.estimate_depth_map(bgr_image)
            except Exception as e:
                logger.error(f"Depth map estimation failed for frame {frame_id}: {e}")
                depth_map = None

            # Prepare visualization frame if requested
            annotated_frame = bgr_image.copy() if (args.visualize and cv2 is not None) else None

            # Step 4, 5, 6: Process Each Detection
            for idx, det in enumerate(detections, start=1):
                # Calculate depth
                if depth_map is not None:
                    depth_result = depth_estimator.compute_depth_from_map(depth_map, det)
                else:
                    depth_result = (None, "rejected_outlier")

                depth_cm, depth_flag = depth_result

                # Step 5: Build Record and Route ('no_fix' -> quarantine)
                record, should_transmit = record_builder.build_and_route(
                    frame_data=frame,
                    detection=det,
                    gps_coord=(gps_coord, fix_quality),
                    depth_cm=depth_result,
                )

                # Step 6: Transmit or Queue
                if should_transmit:
                    sent = transmitter.transmit(record)
                    if sent:
                        transmitted_count += 1
                        trans_status = "SENT"
                    else:
                        queued_count += 1
                        trans_status = "QUEUED_OFFLINE"
                else:
                    quarantined_count += 1
                    trans_status = "QUARANTINED"

                logger.info(
                    f"  -> Det #{idx} [{det.x1},{det.y1},{det.x2},{det.y2}] conf={det.confidence:.2f} "
                    f"| Depth: {depth_cm}cm ({depth_flag}) | Result: {trans_status}"
                )

                # Step 7: Annotate debug frame
                if annotated_frame is not None:
                    cv2.rectangle(
                        annotated_frame,
                        (det.x1, det.y1),
                        (det.x2, det.y2),
                        (0, 0, 255) if depth_flag == "rejected_outlier" else (0, 255, 0),
                        2,
                    )
                    label = f"Pothole {det.confidence:.2f} | {depth_cm}cm [{depth_flag}]"
                    cv2.putText(
                        annotated_frame,
                        label,
                        (det.x1, max(20, det.y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            # Write annotated frame to debug video
            if annotated_frame is not None and cv2 is not None:
                if video_writer is None:
                    h, w = annotated_frame.shape[:2]
                    video_writer = cv2.VideoWriter(args.output_debug_video, fourcc, args.fps, (w, h))
                video_writer.write(annotated_frame)

    except KeyboardInterrupt:
        logger.warning("\nPipeline execution interrupted by user.")
    finally:
        # Clean up video writer
        if video_writer is not None:
            video_writer.release()
            logger.info(f"Annotated debug video saved to {args.output_debug_video}")

        # Clean up transmitter
        transmitter.close()

    # Step 8: Queue Maintenance & Purge
    purged_records = queue_manager.purge_old_records()

    # Final Execution Summary
    gps_fix_rate = (gps_fixed_frames / total_frames * 100.0) if total_frames > 0 else 0.0

    print("\n" + "=" * 65)
    print("           PIPELINE EXECUTION SUMMARY")
    print("=" * 65)
    print(f"Total Frames Processed:    {total_frames}")
    print(f"GPS Fix Rate:              {gps_fix_rate:.1f}% ({gps_fixed_frames}/{total_frames} frames)")
    print(f"Total Potholes Detected:   {total_detections}")
    print(f"Transmitted to Server:     {transmitted_count}")
    print(f"Queued for Offline Retry:  {queued_count}")
    print(f"Quarantined (No GPS Fix):  {quarantined_count}")
    print(f"Queue Maintenance Purge:   {purged_records} expired records deleted")
    print("=" * 65 + "\n")


def main():
    args = parse_arguments()
    run_pipeline(args)


if __name__ == "__main__":
    main()
