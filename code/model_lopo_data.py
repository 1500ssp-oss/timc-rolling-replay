from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

import model_lopo
from segmented_archived_timebase import derive_segmented_archived_timebase
from timc_paths import data_directory


def load_lopo_frames() -> list[pd.DataFrame]:
    project = Path(__file__).resolve().parent / "engine_core"
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from src.real_process_data import load_real_process_data

    data1_dir = data_directory("Data1", required=True)
    result = []
    for frame in load_real_process_data(str(data1_dir)).frames:
        work = frame.copy()
        timebase = derive_segmented_archived_timebase(work, model_lopo.D0, 0.4, 3.0)
        work["effective_dt_s"] = timebase.dt_control_s
        work["distance_step_m"] = timebase.observed_dlength_m
        work["segment_id"] = timebase.segment_id
        result.append(work)
    return result


model_lopo.load_frames = load_lopo_frames

if __name__ == "__main__":
    model_lopo.main()
