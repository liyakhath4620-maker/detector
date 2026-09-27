import os
import shutil
import tempfile
import numpy as np
import pytest
from datetime import datetime, timezone, timedelta
from typing import Generator, List

from detector.data_models import FrameData, Detection, GPSCoordinate, PotholeRecord
from detector.inputs.gps_file import GPSFileSource, GPSFixQuality
from detector.core.depth_estimator import MetricDepthEstimator, DepthQualityFlag
from detector.packaging.record_builder import RecordBuilder
from detector.transmission.local_queue import LocalQueueManager, QueueStatus
from detector.transmission.http_transmitter import HttpTransmitter
from detector.tests.mock_server import MockOmniSightServer


class MockSyntheticVideoSource:
    """Generates synthetic test video frames with a road surface and pothole depression."""

    def __init__(self, num_frames: int = 5, start_time: Optional[datetime] = None):
        self.num_frames = num_frames
        self.start_time = start_time or datetime(2026, 9, 27, 12, 0, 0, tzinfo=timezone.utc)

    def get_frames(self) -> Generator[FrameData, None, None]:
        for i in range(self.num_frames):
            frame_ts = self.start_time + timedelta(seconds=i * 0.5)
            # Create a 480x640 synthetic BGR frame
            image = np.full((480, 640, 3), 128, dtype=np.uint8)
            # Draw synthetic pothole region
            image[200:300, 250:350] = (40, 40, 40)
            yield FrameData(
                frame_id=i,
                timestamp=frame_ts,
                bgr_image=image,
            )


def test_end_to_end_pipeline():
    """
    Integration test asserting:
      1. Correct frame count.
      2. Valid bounding boxes.
      3. Reasonable depth values.
      4. Correct GPS tagging (matched and interpolated).
      5. Successful delivery to the mock server.
      6. Offline resilience (server down -> SQLite queuing -> server up -> queue flush).
    """
    temp_dir = tempfile.mkdtemp(prefix="omnisight_test_")
    gps_csv_path = os.path.join(temp_dir, "test_gps.csv")
    queue_db_path = os.path.join(temp_dir, "test_pipeline_queue.db")

    try:
        # Create timestamped GPS track (fixes at t=0s, 1s, 2s)
        t0 = datetime(2026, 9, 27, 12, 0, 0, tzinfo=timezone.utc)
        gps_csv_content = (
            "timestamp,latitude,longitude,speed,altitude,accuracy\n"
            f"{t0.isoformat()},37.774900,-122.419400,10.0,15.0,1.5\n"
            f"{(t0 + timedelta(seconds=1.0)).isoformat()},37.775100,-122.419200,12.0,16.0,1.5\n"
            f"{(t0 + timedelta(seconds=2.0)).isoformat()},37.775300,-122.419000,14.0,17.0,1.5\n"
        )
        with open(gps_csv_path, "w", encoding="utf-8") as f:
            f.write(gps_csv_content)

        # 1. Initialize Mock Server on an ephemeral port
        mock_server = MockOmniSightServer(host="127.0.0.1", port=0)
        mock_server.start()
        server_port = mock_server.port

        # 2. Initialize Pipeline Modules
        video_source = MockSyntheticVideoSource(num_frames=5, start_time=t0)
        gps_source = GPSFileSource(gps_csv_path, match_tolerance_sec=1.0, max_interpolation_gap_sec=3.0)
        depth_estimator = MetricDepthEstimator(lazy_load=True, interior_percentile=95.0)
        record_builder = RecordBuilder(video_filename="test_run.mp4", session_id="test_integ_001", quarantine_dir=temp_dir)
        queue_manager = LocalQueueManager(db_path=queue_db_path)
        transmitter = HttpTransmitter(server_url=mock_server.url, queue_manager=queue_manager, timeout=5.0)

        processed_frames: List[FrameData] = []
        generated_records: List[PotholeRecord] = []

        # Synthetic depth map for depth estimator:
        # Road plane at 2.00m, pothole depression at [200:300, 250:350] at 2.12m (+12cm depth)
        synthetic_depth_map = np.full((480, 640), 2.00, dtype=np.float32)
        synthetic_depth_map[200:300, 250:350] = 2.12

        # 3. Execute End-to-End Pipeline Loop
        for frame in video_source.get_frames():
            processed_frames.append(frame)

            # Simulated detection from detector
            det = Detection(x1=250, y1=200, x2=350, y2=300, confidence=0.92, class_id=0, class_label="pothole")

            # Assertion 2: Valid Bounding Boxes
            assert det.x1 < det.x2, f"Invalid bbox width: {det.x1} >= {det.x2}"
            assert det.y1 < det.y2, f"Invalid bbox height: {det.y1} >= {det.y2}"
            assert det.x1 >= 0 and det.y1 >= 0
            assert det.x2 <= frame.bgr_image.shape[1]
            assert det.y2 <= frame.bgr_image.shape[0]

            # Stage 4: Depth Estimation
            depth_result = depth_estimator.compute_depth_from_map(synthetic_depth_map, det)

            # Assertion 3: Reasonable Depth Values
            depth_cm, depth_flag = depth_result
            assert depth_flag == DepthQualityFlag.OK, f"Expected ok depth flag, got {depth_flag}"
            assert depth_cm is not None
            assert 0.0 <= depth_cm <= 30.0, f"Depth {depth_cm}cm is outside reasonable range [0, 30]"
            assert round(depth_cm, 1) == 12.0

            # Stage 5: GPS Coordinate Matching
            gps_coord, fix_quality = gps_source.get_coordinate_at_time(frame.timestamp)

            # Assertion 4: Correct GPS Tagging
            assert gps_coord is not None
            assert fix_quality in (GPSFixQuality.MATCHED, GPSFixQuality.INTERPOLATED)
            assert 37.77 <= gps_coord.latitude <= 37.78
            assert -122.42 <= gps_coord.longitude <= -122.41

            if frame.frame_id in (0, 2, 4):
                # Even frames correspond to whole seconds (t=0s, 1s, 2s) -> matched
                assert fix_quality == GPSFixQuality.MATCHED
            else:
                # Odd frames at t=0.5s, 1.5s -> interpolated
                assert fix_quality == GPSFixQuality.INTERPOLATED

            # Stage 6: Record Building & Routing
            record, should_transmit = record_builder.build_and_route(
                frame_data=frame,
                detection=det,
                gps_coord=gps_coord,
                depth_cm=depth_result,
            )
            assert should_transmit is True
            generated_records.append(record)

            # Stage 7: HTTP Transmission
            sent = transmitter.transmit(record)
            assert sent is True, f"Failed to transmit record {record.pothole_id} to mock server"

        # Assertion 1: Correct Frame Count
        assert len(processed_frames) == 5, f"Expected 5 frames, got {len(processed_frames)}"
        assert len(generated_records) == 5

        # Assertion 5: Successful Delivery to the Mock Server
        received = mock_server.get_received_records()
        assert len(received) == 5, f"Expected 5 delivered records, got {len(received)}"

        for original, posted in zip(generated_records, received):
            assert posted["pothole_id"] == original.pothole_id
            assert posted["confidence"] == original.confidence
            assert posted["depth_cm"] == original.depth_cm
            assert posted["gps_fix_quality"] == original.gps_fix_quality
            assert posted["bounding_box"] == original.bounding_box

        # ----------------------------------------------------------------------
        # Assertion 6: Offline Resilience
        # ----------------------------------------------------------------------
        # 1. Stop the mock server to simulate network outage
        mock_server.stop()

        # 2. Attempt transmitting a new record during outage
        offline_frame = FrameData(frame_id=99, timestamp=t0 + timedelta(seconds=10), bgr_image=None)
        offline_det = Detection(x1=100, y1=100, x2=200, y2=200, confidence=0.85, class_id=0, class_label="pothole")
        offline_gps = GPSCoordinate(timestamp=offline_frame.timestamp, latitude=37.7760, longitude=-122.4180)
        offline_depth = (15.0, "ok")

        offline_record = record_builder.build_record(offline_frame, offline_det, offline_gps, offline_depth)
        sent_offline = transmitter.transmit(offline_record)

        # Must fail delivery and return False
        assert sent_offline is False

        # Confirm the record was stored in SQLite queue with status PENDING
        queued_row = queue_manager.get_record(offline_record.pothole_id)
        assert queued_row is not None, "Offline record was not enqueued in SQLite"
        assert queued_row["status"] == QueueStatus.PENDING.value
        assert queued_row["retry_count"] >= 1

        # 3. Restart the mock server on the exact same port
        mock_server.port = server_port
        mock_server.start()

        # Reset retry schedule to allow immediate queue processing
        queue_manager.mark_status(offline_record.pothole_id, QueueStatus.PENDING, next_retry_at=None)

        # 4. Flush / process the pending queue
        flushed_count = transmitter.process_pending_queue(batch_size=10)
        assert flushed_count >= 1, f"Expected at least 1 record flushed, got {flushed_count}"

        # 5. Confirm delivery to the restarted mock server
        server_all_records = mock_server.get_received_records()
        assert any(r["pothole_id"] == offline_record.pothole_id for r in server_all_records), (
            "Quoted offline record was not delivered to mock server after restart"
        )

        # 6. Confirm database status in SQLite was updated to SENT
        updated_row = queue_manager.get_record(offline_record.pothole_id)
        assert updated_row["status"] == QueueStatus.SENT.value, (
            f"Expected queue status SENT, got {updated_row['status']}"
        )

        print("\nAll integration pipeline assertions passed successfully!")

    finally:
        if "mock_server" in locals():
            mock_server.stop()
        if "transmitter" in locals():
            transmitter.close()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_end_to_end_pipeline()
