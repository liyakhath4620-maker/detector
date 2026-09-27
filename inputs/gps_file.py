import os
import csv
import bisect
import logging
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Any

try:
    from detector.inputs.base import BaseGPSSource
    from detector.data_models import GPSCoordinate
except ImportError:
    from inputs.base import BaseGPSSource
    from data_models import GPSCoordinate

logger = logging.getLogger(__name__)


class GPSFixQuality:
    MATCHED = "matched"
    INTERPOLATED = "interpolated"
    NO_FIX = "no_fix"


class GPSMatchResult(tuple):
    """
    Result container for GPS coordinate matching.
    
    Unpacks as a 2-tuple: (coordinate, gps_fix_quality).
    Forwards coordinate attributes (latitude, longitude, speed, altitude, accuracy)
    when coordinate is not None for maximum compatibility.
    """

    def __new__(cls, coordinate: Optional[GPSCoordinate], gps_fix_quality: str):
        return super().__new__(cls, (coordinate, gps_fix_quality))

    def __init__(self, coordinate: Optional[GPSCoordinate], gps_fix_quality: str):
        self.coordinate = coordinate
        self.gps_fix_quality = gps_fix_quality
        self.fix_quality = gps_fix_quality

    def __getattr__(self, name: str) -> Any:
        if self.coordinate is not None and hasattr(self.coordinate, name):
            return getattr(self.coordinate, name)
        raise AttributeError(f"'GPSMatchResult' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"GPSMatchResult(coordinate={self.coordinate}, gps_fix_quality='{self.gps_fix_quality}')"


class GPSFileSource(BaseGPSSource):
    """
    GPS data source that reads recorded GPS coordinates from a CSV log
    (timestamp, latitude, longitude, speed, altitude, accuracy) and matches them
    against video frame timestamps using binary search (bisect).
    
    Matching behavior:
      - Linearly interpolates between two fixes if the frame falls between them.
      - Matches to the nearest fix within ±1.0s tolerance if outside or across gaps.
      - Returns (coordinate, gps_fix_quality) with flags: 'matched', 'interpolated', or 'no_fix'.
    """

    def __init__(
        self,
        filepath: str,
        match_tolerance_sec: float = 1.0,
        max_interpolation_gap_sec: float = 5.0,
        exact_threshold_sec: float = 0.05,
    ):
        """
        Initializes the GPS file source and parses the GPS CSV log.

        :param filepath: Path to timestamped GPS CSV log.
        :param match_tolerance_sec: Tolerance window (default ±1.0s) for nearest-neighbor fix matching.
        :param max_interpolation_gap_sec: Maximum time gap (seconds) between fixes to permit interpolation.
        :param exact_threshold_sec: Threshold below which two fixes are considered identical.
        """
        self.filepath = filepath
        self.match_tolerance_sec = match_tolerance_sec
        self.max_interpolation_gap_sec = max_interpolation_gap_sec
        self.exact_threshold_sec = exact_threshold_sec

        self.coordinates: List[GPSCoordinate] = []
        self.timestamps: List[float] = []
        self._last_fix_quality: str = GPSFixQuality.NO_FIX

        self._load_file(filepath)

    @property
    def last_fix_quality(self) -> str:
        """Returns the GPS fix quality flag of the most recent coordinate lookup."""
        return self._last_fix_quality

    def _to_epoch(self, dt: datetime) -> float:
        """Converts datetime to UTC timestamp in seconds."""
        if dt.tzinfo is not None:
            return dt.timestamp()
        return dt.replace(tzinfo=timezone.utc).timestamp()

    def _parse_timestamp(self, val: Any) -> datetime:
        """Parses various timestamp representations into a datetime object."""
        if isinstance(val, datetime):
            return val

        if isinstance(val, (int, float)):
            epoch = float(val)
            if epoch > 1e11:
                epoch /= 1000.0
            return datetime.fromtimestamp(epoch, tz=timezone.utc)

        val_str = str(val).strip()
        try:
            epoch = float(val_str)
            if epoch > 1e11:
                epoch /= 1000.0
            return datetime.fromtimestamp(epoch, tz=timezone.utc)
        except ValueError:
            pass

        cleaned = val_str.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(cleaned)
        except ValueError:
            pass

        for fmt in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                return datetime.strptime(val_str, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        raise ValueError(f"Unable to parse timestamp: '{val}'")

    def _load_file(self, filepath: str):
        """Loads and parses the GPS file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"GPS file not found: {filepath}")

        if filepath.lower().endswith(".json"):
            self._load_json(filepath)
        else:
            self._load_csv(filepath)

        # Sort coordinates chronologically
        self.coordinates.sort(key=lambda c: self._to_epoch(c.timestamp))
        self.timestamps = [self._to_epoch(c.timestamp) for c in self.coordinates]
        logger.info(f"Loaded {len(self.coordinates)} GPS coordinates from {filepath}")

    def _load_csv(self, filepath: str):
        """Parses CSV GPS log (timestamp, latitude, longitude, speed, altitude, accuracy)."""
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            return

        first_line = lines[0]
        delimiter = "\t" if "\t" in first_line else ("," if "," in first_line else ";")

        reader = csv.reader(lines, delimiter=delimiter)
        raw_rows = list(reader)

        if not raw_rows:
            return

        headers = [h.strip().lower() for h in raw_rows[0]]

        def find_col(*candidates):
            for i, h in enumerate(headers):
                for cand in candidates:
                    if cand in h:
                        return i
            return None

        time_idx = find_col("time", "date", "ts", "epoch")
        lat_idx = find_col("lat")
        lon_idx = find_col("lon", "lng", "long")
        speed_idx = find_col("speed", "spd", "vel")
        alt_idx = find_col("alt", "ele")
        acc_idx = find_col("acc", "hdop", "prec")

        start_row = 1
        if lat_idx is None or lon_idx is None:
            time_idx = 0
            lat_idx = 1
            lon_idx = 2
            speed_idx = 3 if len(raw_rows[0]) > 3 else None
            alt_idx = 4 if len(raw_rows[0]) > 4 else None
            acc_idx = 5 if len(raw_rows[0]) > 5 else None
            start_row = 0

        for row_idx, row in enumerate(raw_rows[start_row:], start=start_row):
            if not row or len(row) <= max(time_idx, lat_idx, lon_idx):
                continue
            try:
                ts = self._parse_timestamp(row[time_idx])
                lat = float(row[lat_idx])
                lon = float(row[lon_idx])
                speed = float(row[speed_idx]) if speed_idx is not None and speed_idx < len(row) and row[speed_idx].strip() else None
                alt = float(row[alt_idx]) if alt_idx is not None and alt_idx < len(row) and row[alt_idx].strip() else None
                acc = float(row[acc_idx]) if acc_idx is not None and acc_idx < len(row) and row[acc_idx].strip() else None

                self.coordinates.append(
                    GPSCoordinate(
                        timestamp=ts,
                        latitude=lat,
                        longitude=lon,
                        speed=speed,
                        altitude=alt,
                        accuracy=acc,
                    )
                )
            except Exception as e:
                logger.debug(f"Skipping unparseable GPS row {row_idx}: {row} ({e})")

    def _load_json(self, filepath: str):
        """Parses JSON GPS array."""
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for key in ("coordinates", "locations", "points", "fixes", "data"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array of GPS points in {filepath}")

        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                ts_val = item.get("timestamp") or item.get("time") or item.get("datetime")
                lat = float(item.get("latitude") if item.get("latitude") is not None else item.get("lat"))
                lon = float(item.get("longitude") if item.get("longitude") is not None else item.get("lon", item.get("lng")))
                speed = float(item["speed"]) if item.get("speed") is not None else None
                alt = float(item.get("altitude") or item.get("alt") or item.get("elevation", 0)) if (item.get("altitude") or item.get("alt") or item.get("elevation")) is not None else None
                acc = float(item.get("accuracy") or item.get("acc") or item.get("hdop", 0)) if (item.get("accuracy") or item.get("acc") or item.get("hdop")) is not None else None

                self.coordinates.append(
                    GPSCoordinate(
                        timestamp=self._parse_timestamp(ts_val),
                        latitude=lat,
                        longitude=lon,
                        speed=speed,
                        altitude=alt,
                        accuracy=acc,
                    )
                )
            except Exception as e:
                logger.debug(f"Skipping invalid JSON GPS item: {item} ({e})")

    def get_coordinate_at_time(
        self,
        timestamp: datetime,
    ) -> GPSMatchResult:
        """
        Matches a frame timestamp against GPS data using binary search (bisect).
        
        Returns:
          GPSMatchResult tuple: (coordinate, gps_fix_quality)
          - 'matched': matched to nearest fix within ±1.0s tolerance
          - 'interpolated': linearly interpolated between two bounding fixes
          - 'no_fix': nothing close enough
        """
        if not self.coordinates:
            self._last_fix_quality = GPSFixQuality.NO_FIX
            return GPSMatchResult(None, GPSFixQuality.NO_FIX)

        target_epoch = self._to_epoch(timestamp)
        n = len(self.timestamps)

        # Binary search for position in sorted timestamps
        idx = bisect.bisect_left(self.timestamps, target_epoch)

        # Case 1: Frame falls strictly between two recorded fixes (t1 < target_epoch < t2)
        if 0 < idx < n:
            t1 = self.timestamps[idx - 1]
            t2 = self.timestamps[idx]
            c1 = self.coordinates[idx - 1]
            c2 = self.coordinates[idx]

            # Exact match check with boundary points
            if abs(target_epoch - t1) <= self.exact_threshold_sec:
                matched_coord = self._clone_coord(c1, timestamp, GPSFixQuality.MATCHED)
                self._last_fix_quality = GPSFixQuality.MATCHED
                return GPSMatchResult(matched_coord, GPSFixQuality.MATCHED)

            if abs(t2 - target_epoch) <= self.exact_threshold_sec:
                matched_coord = self._clone_coord(c2, timestamp, GPSFixQuality.MATCHED)
                self._last_fix_quality = GPSFixQuality.MATCHED
                return GPSMatchResult(matched_coord, GPSFixQuality.MATCHED)

            gap = t2 - t1
            # Linearly interpolate between the two fixes if within max interpolation gap
            if 0 < gap <= self.max_interpolation_gap_sec:
                alpha = (target_epoch - t1) / gap
                lat = c1.latitude + alpha * (c2.latitude - c1.latitude)
                lon = c1.longitude + alpha * (c2.longitude - c1.longitude)

                speed = (
                    c1.speed + alpha * (c2.speed - c1.speed)
                    if c1.speed is not None and c2.speed is not None
                    else (c1.speed if alpha < 0.5 else c2.speed)
                )
                altitude = (
                    c1.altitude + alpha * (c2.altitude - c1.altitude)
                    if c1.altitude is not None and c2.altitude is not None
                    else (c1.altitude if alpha < 0.5 else c2.altitude)
                )
                accuracy = (
                    c1.accuracy + alpha * (c2.accuracy - c1.accuracy)
                    if c1.accuracy is not None and c2.accuracy is not None
                    else (c1.accuracy if alpha < 0.5 else c2.accuracy)
                )

                interp_coord = GPSCoordinate(
                    timestamp=timestamp,
                    latitude=round(lat, 7),
                    longitude=round(lon, 7),
                    speed=round(speed, 2) if speed is not None else None,
                    altitude=round(altitude, 2) if altitude is not None else None,
                    accuracy=round(accuracy, 2) if accuracy is not None else None,
                )
                interp_coord.fix_quality = GPSFixQuality.INTERPOLATED
                interp_coord.gps_fix_quality = GPSFixQuality.INTERPOLATED
                self._last_fix_quality = GPSFixQuality.INTERPOLATED
                return GPSMatchResult(interp_coord, GPSFixQuality.INTERPOLATED)

            # Gap exceeds interpolation limit: check nearest fix within ±1.0s tolerance
            diff1 = abs(target_epoch - t1)
            diff2 = abs(t2 - target_epoch)
            if diff1 <= self.match_tolerance_sec or diff2 <= self.match_tolerance_sec:
                best_coord = c1 if diff1 <= diff2 else c2
                matched_coord = self._clone_coord(best_coord, timestamp, GPSFixQuality.MATCHED)
                self._last_fix_quality = GPSFixQuality.MATCHED
                return GPSMatchResult(matched_coord, GPSFixQuality.MATCHED)

        # Case 2: Frame is before the first recorded fix
        elif idx == 0:
            diff = self.timestamps[0] - target_epoch
            if diff <= self.match_tolerance_sec:
                matched_coord = self._clone_coord(self.coordinates[0], timestamp, GPSFixQuality.MATCHED)
                self._last_fix_quality = GPSFixQuality.MATCHED
                return GPSMatchResult(matched_coord, GPSFixQuality.MATCHED)

        # Case 3: Frame is after the last recorded fix
        elif idx == n:
            diff = target_epoch - self.timestamps[-1]
            if diff <= self.match_tolerance_sec:
                matched_coord = self._clone_coord(self.coordinates[-1], timestamp, GPSFixQuality.MATCHED)
                self._last_fix_quality = GPSFixQuality.MATCHED
                return GPSMatchResult(matched_coord, GPSFixQuality.MATCHED)

        # Case 4: No GPS fix close enough
        self._last_fix_quality = GPSFixQuality.NO_FIX
        return GPSMatchResult(None, GPSFixQuality.NO_FIX)

    @staticmethod
    def _clone_coord(src: GPSCoordinate, ts: datetime, fix_quality: str) -> GPSCoordinate:
        """Clones a coordinate with the requested timestamp and fix quality attribute."""
        c = GPSCoordinate(
            timestamp=ts,
            latitude=src.latitude,
            longitude=src.longitude,
            speed=src.speed,
            altitude=src.altitude,
            accuracy=src.accuracy,
        )
        c.fix_quality = fix_quality
        c.gps_fix_quality = fix_quality
        return c
