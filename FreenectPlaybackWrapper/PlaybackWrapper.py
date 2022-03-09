
from pathlib import Path
from typing import Optional

import datetime, time
import cv2
import numpy

class FreenectStatus:
    updated_depth: bool = False
    updated_rgb: bool = False


class FreenectPlaybackWrapper:
    def __init__(self, freenect_video_folder: str, realtime_wait: bool =True):
        self.video_folder = freenect_video_folder
        self.video_folder = self.video_folder.replace("\\", "/")
        self.video_folder = Path(self.video_folder)

        if not self.video_folder.exists():
            raise RuntimeError("Provided folder does not exist: {}".format(freenect_video_folder))

        self.index_path = self.video_folder.joinpath("INDEX.txt")

        if not self.index_path.exists():
            raise RuntimeError("Unable to find INDEX.txt inside folder: ".format(freenect_video_folder))

        self.index_length = len([l for l in open(self.index_path).readlines() if l[0] != 'a'])

        self.realtime = realtime_wait

    def __iter__(self):

        previous_file_timestamp: Optional[numpy.float64] = None
        previous_run_timestamp: Optional[numpy.float64] = None

        cached_rgb = None
        cached_depth = None
        reset_new = True

        # Cycle through every line
        for line in open(str(self.index_path), "r"):
            # Line can contain '\n' on some platforms
            line = line.replace("\n", "")

            # If accelerometer data, ignore
            if line[0] == 'a':
                continue

            full_path = self.video_folder.joinpath(line)

            if reset_new:
                new_rgb = None
                new_depth = None

            reset_new = True

            if line[0] == 'r':
                # Read RGB Image
                new_rgb = cv2.imread(str(full_path))
            elif line[0] == 'd':
                # Read Depth Image

                with open(str(full_path), "rb") as depth_reader:
                    format_settings = depth_reader.readline()
                    format_settings = format_settings.replace(b"\n", b"")
                    width, height, max_size = format_settings.split(b" ")[1:]
                    width, height, max_size = int(width), int(height), int(max_size)

                    raw_data = depth_reader.read()

                    image_data = numpy.fromstring(raw_data, dtype="<u2")
                    image_data = numpy.right_shift(image_data, 3)

                    image_data_8bit = image_data.astype(numpy.uint8)
                    image_data_8bit = numpy.reshape(image_data_8bit, (height, width, 1))

                    new_depth = image_data_8bit

                pass

            # Sleep/Skip if realtime
            if self.realtime:
                new_file_timestamp = _get_timestamp_from_filename(str(line))

                if previous_file_timestamp is not None:
                    diff = (new_file_timestamp - previous_file_timestamp) * 1000

                    now = numpy.float64(datetime.datetime.now().timestamp() * 1000)
                    time_to_wait = diff - (now - previous_run_timestamp)

                    if time_to_wait > 0:
                        time.sleep(time_to_wait / 1000)
                    else:
                        reset_new = False
                        continue

                previous_run_timestamp = numpy.float64(datetime.datetime.now().timestamp() * 1000)
                previous_file_timestamp = new_file_timestamp

            status = FreenectStatus()

            if new_rgb is not None:
                cached_rgb = new_rgb
                status.updated_rgb = True
            if new_depth is not None:
                cached_depth = new_depth
                status.updated_depth = True

            yield status, cached_rgb, cached_depth

    def __len__(self):
        return self.index_length


def _get_timestamp_from_filename(filename: str) -> numpy.float64:
    return numpy.float64(filename[2:19])
