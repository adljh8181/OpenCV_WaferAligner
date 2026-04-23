# -*- coding: utf-8 -*-
"""
================================================================================
AUTOFOCUS SERVICE v1.0
================================================================================
Python equivalent of the C# autofocus logic in MainWindowViewModel.cs.

Handles three TCP commands forwarded by the C# VPServerApp when
ComputerVisionType == ZMQ:

  START_AUTOFOCUS_REQ "<imagePath>" <zHeightStart> <stepNumber>
      - Resets the scan session.
      - Measures the Focus Measure Index (FMI) of the first image.
      - Returns direction UP to tell the machine to keep moving.

  FOCUS_REQ "<imagePath>" <zHeightCurrent>
      - Measures FMI of each subsequent image.
      - After <stepNumber> images are collected, runs peak detection.
      - Returns STOP + best Z height when the sharpest position is found.
      - Returns FOCUS_ERR if no peak is detected after the full scan.

  FM_INDEX_REQ "<imagePath>"
      - Standalone, stateless one-shot FMI measurement.
      - Used by ChapAnalyze::PerformVPAutoFocus (machine manages Z logic).

Algorithm
---------
  FMI method  : Laplacian Variance  (stddev^2 of the Laplacian-filtered image)
                Matches C# EmguVision.GetSharpnessLaplacianVariance()
  Peak finder : Sliding-window local maxima with height-above-average threshold
                Matches C# MainWindowViewModel.PeakDetection()

Reply format (dict -- serialised to JSON by zmq_server.py)
-----------
  START_AUTOFOCUS_REQ OK  : {"status":"ok",    "fmi":<float>, "direction":"UP"}
  FOCUS_REQ still scanning: {"status":"ok",    "fmi":<float>, "direction":"UP"}
  FOCUS_REQ peak found    : {"status":"ok",    "fmi":<float>, "direction":"STOP",
                              "best_height":<float>}
  FOCUS_REQ no peak       : {"status":"error", "message":"FOCUS_ERR"}
  FM_INDEX_REQ OK         : {"status":"ok",    "fmi":<float>}
  any ImageNotFound       : {"status":"error", "message":"ImageNotFound"}
================================================================================
"""

import os
import cv2
import numpy as np


# ==============================================================================
#                           CONFIGURATION
# ==============================================================================

class AutofocusConfig:
    """
    Tunable parameters for the peak-detection algorithm.
    Mirrors C# WaferAlignParam.autoFocusParam fields.
    """

    # Sliding-window half-size used in peak detection.
    # A candidate must be strictly greater than every neighbour within
    # +/- PEAK_FILTER_HALF_SIZE positions.
    # Matches C# PeakFilterHalfSizeInPixels (default 2)
    PEAK_FILTER_HALF_SIZE: int = 2

    # A candidate peak must exceed the neighbourhood average by at least
    # this fraction (e.g. 0.05 = 5 %).
    # Matches C# PeakHeightThresholdPercentage (default 0.05)
    PEAK_HEIGHT_THRESHOLD_PCT: float = 0.05


# ==============================================================================
#                           FMI  (Focus Measure Index)
# ==============================================================================

def compute_fmi(image_path: str) -> float:
    """
    Compute the Focus Measure Index of a grayscale image using
    Laplacian Variance.

    Formula
    -------
        fmi = Var(Laplacian(I))  =  stddev(Laplacian(I))^2

    Higher value means sharper / better-focused image.

    Matches
    -------
        C# EmguVision.GetSharpnessLaplacianVariance()

    Parameters
    ----------
    image_path : str
        Absolute path to the image file (BMP / PNG / TIFF ...).

    Returns
    -------
    float
        Focus Measure Index (>= 0).

    Raises
    ------
    FileNotFoundError
        If the image file cannot be opened by OpenCV.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Apply Laplacian filter (CV_64F keeps negative values)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    # Compute variance = stddev^2
    _, std_dev = cv2.meanStdDev(laplacian)
    return float(std_dev[0][0] ** 2)


# ==============================================================================
#                           PEAK DETECTION
# ==============================================================================

def peak_detection(fmis: list,
                   window_half_size: int = AutofocusConfig.PEAK_FILTER_HALF_SIZE,
                   threshold_perc: float = AutofocusConfig.PEAK_HEIGHT_THRESHOLD_PCT
                   ) -> list:
    """
    Sliding-window local-maxima detector on a list of FMI scores.

    A candidate at index i is accepted as a peak when ALL of the
    following hold:
      1. fmis[i] is strictly greater than every neighbour within
         +/- window_half_size.
      2. fmis[i] exceeds the neighbourhood average by at least
         threshold_perc (5% by default).

    Matches
    -------
        C# MainWindowViewModel.PeakDetection()

    Parameters
    ----------
    fmis              : list[float]  FMI scores in scan order (oldest first).
    window_half_size  : int          Half-width of the comparison window.
    threshold_perc    : float        Minimum fractional rise above average.

    Returns
    -------
    list[int]
        Indices of detected peaks (may be empty).
    """
    peaks = []
    n = len(fmis)

    for i in range(n):
        center  = fmis[i]
        start   = max(0, i - window_half_size)
        end     = min(n - 1, i + window_half_size)
        is_peak = True
        avg_sum = 0.0
        count   = 0

        for j in range(start, end + 1):
            if j == i:
                continue
            if fmis[j] >= center:       # not strictly the local maximum
                is_peak = False
                break
            avg_sum += fmis[j]
            count   += 1

        if not is_peak or count == 0:
            continue

        # Height-above-average threshold check
        average = avg_sum / count
        if average == 0:
            continue
        if (center - average) / average >= threshold_perc:
            peaks.append(i)

    return peaks


# ==============================================================================
#                           AUTOFOCUS SESSION
# ==============================================================================

class AutofocusSession:
    """
    Stateful session that accumulates FMI measurements across multiple
    FOCUS_REQ calls and detects the best focus position.

    Usage
    -----
        session = AutofocusSession()

        # Machine sends START_AUTOFOCUS_REQ:
        reply = session.start(image_path, z_height_start, step_count)

        # Machine sends FOCUS_REQ for each step:
        reply = session.step(image_path, z_height_current)
        # Repeat until reply["direction"] == "STOP" or status == "error"

    One instance is typically held by the ZMQ server and reused across
    sessions (start() always resets).
    """

    def __init__(self, config: AutofocusConfig = None):
        self._cfg        = config or AutofocusConfig()
        self._heights    = []   # list[float] -- Z heights in scan order
        self._fmis       = []   # list[float] -- FMI scores in scan order
        self._step_count = 50   # target number of steps (set by start())

    # ------------------------------------------------------------------
    # START_AUTOFOCUS_REQ
    # ------------------------------------------------------------------
    def start(self, image_path: str,
              z_height_start: float,
              step_count: int) -> dict:
        """
        Handle START_AUTOFOCUS_REQ.

        Resets session state, measures FMI of the very first image, and
        tells the machine to keep moving in the UP direction.

        Parameters
        ----------
        image_path     : str    Path to the first image.
        z_height_start : float  Z stage position when this image was taken.
        step_count     : int    Total images to collect before peak detection.

        Returns
        -------
        dict
            {"status":"ok",    "fmi":<float>, "direction":"UP"}
            {"status":"error", "message":"ImageNotFound"}
        """
        if not os.path.isfile(image_path):
            return {"status": "error", "message": "ImageNotFound"}

        # Reset session
        self._heights    = []
        self._fmis       = []
        self._step_count = step_count

        try:
            fmi = compute_fmi(image_path)
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

        self._heights.append(z_height_start)
        self._fmis.append(fmi)

        _delete_file(image_path)

        return {"status": "ok", "fmi": fmi, "direction": "UP"}

    # ------------------------------------------------------------------
    # FOCUS_REQ
    # ------------------------------------------------------------------
    def step(self, image_path: str, z_height: float) -> dict:
        """
        Handle FOCUS_REQ.

        Appends the FMI of the current image to the session list.
        Once enough images have been collected, runs peak detection and
        returns the best Z height (STOP) or FOCUS_ERR.

        Parameters
        ----------
        image_path : str    Path to the current image.
        z_height   : float  Z stage position when this image was taken.

        Returns
        -------
        dict -- one of:
            {"status":"ok",    "fmi":<float>, "direction":"UP"}
            {"status":"ok",    "fmi":<float>, "direction":"STOP",
             "best_height":<float>}
            {"status":"error", "message":"FOCUS_ERR"}
            {"status":"error", "message":"ImageNotFound"}
        """
        if not os.path.isfile(image_path):
            return {"status": "error", "message": "ImageNotFound"}

        try:
            fmi = compute_fmi(image_path)
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

        self._heights.append(z_height)
        self._fmis.append(fmi)
        collected = len(self._fmis)

        _delete_file(image_path)

        # Still collecting -- not enough steps yet
        if collected < self._step_count:
            return {"status": "ok", "fmi": fmi, "direction": "UP"}

        # Enough steps: run peak detection
        peaks = peak_detection(
            self._fmis,
            window_half_size = self._cfg.PEAK_FILTER_HALF_SIZE,
            threshold_perc   = self._cfg.PEAK_HEIGHT_THRESHOLD_PCT,
        )

        if peaks:
            best_idx    = peaks[0]                      # first (bottom-most) peak
            best_height = self._heights[best_idx]
            return {
                "status":      "ok",
                "fmi":         fmi,
                "direction":   "STOP",
                "best_height": best_height,
            }

        return {"status": "error", "message": "FOCUS_ERR"}

    # ------------------------------------------------------------------
    # Properties (read-only, for logging / UI)
    # ------------------------------------------------------------------
    @property
    def collected(self) -> int:
        """Number of images collected in the current session."""
        return len(self._fmis)

    @property
    def step_count(self) -> int:
        """Target number of steps for this session."""
        return self._step_count

    @property
    def heights(self) -> list:
        """Copy of Z heights collected so far."""
        return list(self._heights)

    @property
    def fmis(self) -> list:
        """Copy of FMI scores collected so far."""
        return list(self._fmis)


# ==============================================================================
#                           FM_INDEX_REQ  (stateless)
# ==============================================================================

def focus_measure_index(image_path: str) -> dict:
    """
    Handle FM_INDEX_REQ -- standalone, stateless one-shot FMI.

    Used by ChapAnalyze::PerformVPAutoFocus where the machine controller
    manages the Z-stage logic itself and only needs a sharpness score.

    Parameters
    ----------
    image_path : str  Path to the image.

    Returns
    -------
    dict
        {"status":"ok",    "fmi":<float>}
        {"status":"error", "message":"ImageNotFound"}
    """
    if not os.path.isfile(image_path):
        return {"status": "error", "message": "ImageNotFound"}

    try:
        fmi = compute_fmi(image_path)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    _delete_file(image_path)

    return {"status": "ok", "fmi": fmi}


# ==============================================================================
#                           UTILITY
# ==============================================================================

def _delete_file(path: str) -> None:
    """
    Silently delete a file after it has been processed.
    Matches the C# try { File.Delete(imagePath); } catch { } pattern.
    """
    try:
        os.remove(path)
    except Exception:
        pass
