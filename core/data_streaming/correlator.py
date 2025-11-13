import time
import numpy as np
from collections import deque
import threading
from pylsl import StreamInfo, StreamOutlet

class ChannelCorrelator:
    """
    Detects per-channel expression changes using three timescales:
    - Derivative (instantaneous): catches onsets and brief expressions
    - Fast baseline (~1.5s): detects stability and recent changes
    - Slow baseline (~8s): measures deviation from neutral
    
    Outputs a 52-channel float vector of comodulation magnitudes.
    """
    def __init__(self,
                 window_size=None,
                 fps=30,
                 delta_threshold=None,
                 similarity_threshold=None,
                 slow_window_multiplier=None,
                 derivative_threshold=None,
                 derivative_smooth=None,
                 config=None):
        
        self.config = config or {}
        correlator_config = self.config.get('data_streaming', {}).get('correlator', {})
        
        # Get values from config with fallbacks
        window_size = window_size if window_size is not None else correlator_config.get('fast_window_size', 50)
        delta_threshold = delta_threshold if delta_threshold is not None else correlator_config.get('delta_threshold', 0.005)
        similarity_threshold = similarity_threshold if similarity_threshold is not None else correlator_config.get('similarity_threshold', 0.6)
        slow_window_multiplier = slow_window_multiplier if slow_window_multiplier is not None else correlator_config.get('slow_window_multiplier', 5)
        derivative_threshold = derivative_threshold if derivative_threshold is not None else correlator_config.get('derivative_threshold', 0.03)
        derivative_smooth = derivative_smooth if derivative_smooth is not None else correlator_config.get('derivative_smooth', 0.3)
        
        # Windows for different timescales
        self.w_fast = window_size  # ~1.7s at 30fps
        self.w_slow = window_size * slow_window_multiplier  # ~8.3s at 30fps
        self.fps = fps
        
        # Thresholds
        self.delta_threshold = delta_threshold
        self.similarity_threshold = similarity_threshold
        self.derivative_threshold = derivative_threshold
        self.derivative_smooth = derivative_smooth
        
        # For sustained state machine (using slow baseline)
        self.sustained_states = np.zeros(52, dtype=bool)
        self.sustained_vals = np.zeros(52, dtype=float)
        
        # For transient expressions (using derivative)
        self.transient_vals = np.zeros(52, dtype=float)
        self.transient_decay = correlator_config.get('transient_decay', 0.8)  # Decay rate per frame
        
        # For brief expressions (using slow baseline without latching)
        self.brief_vals = np.zeros(52, dtype=float)
        self.brief_decay = correlator_config.get('brief_decay', 0.9)
        
        # Buffers for baselines
        self.buff1_fast = deque(maxlen=self.w_fast)
        self.buff2_fast = deque(maxlen=self.w_fast)
        self.buff1_slow = deque(maxlen=self.w_slow)
        self.buff2_slow = deque(maxlen=self.w_slow)
        
        # Running sums for efficiency
        self.sum1_fast = np.zeros(52, dtype=float)
        self.sum2_fast = np.zeros(52, dtype=float)
        self.sum1_slow = np.zeros(52, dtype=float)
        self.sum2_slow = np.zeros(52, dtype=float)
        
        # For derivative calculation
        self.prev_vec1 = None
        self.prev_vec2 = None
        self.prev_derivative1 = np.zeros(52, dtype=float)
        self.prev_derivative2 = np.zeros(52, dtype=float)
        
        # For detecting returns to neutral
        self.frames_since_neutral = np.zeros(52, dtype=int)
        neutral_threshold_multiplier = correlator_config.get('neutral_threshold_multiplier', 0.5)
        self.neutral_threshold = delta_threshold * neutral_threshold_multiplier
        
        # Thread safety and LSL
        self._lock = threading.Lock()
        self.outlet = None

    def update(self, vec1: np.ndarray, vec2: np.ndarray):
        """
        Process new samples using three timescales to detect different types
        of comodulation: transient (derivative), brief, and sustained.
        Returns a length-52 float array of comodulation magnitudes.
        """
        with self._lock:
            # Update fast baseline buffers
            if len(self.buff1_fast) == self.w_fast:
                old1_fast = self.buff1_fast[0]
                old2_fast = self.buff2_fast[0]
                self.sum1_fast -= old1_fast
                self.sum2_fast -= old2_fast

            self.buff1_fast.append(vec1)
            self.buff2_fast.append(vec2)
            self.sum1_fast += vec1
            self.sum2_fast += vec2

            # Update slow baseline buffers
            if len(self.buff1_slow) == self.w_slow:
                old1_slow = self.buff1_slow[0]
                old2_slow = self.buff2_slow[0]
                self.sum1_slow -= old1_slow
                self.sum2_slow -= old2_slow

            self.buff1_slow.append(vec1)
            self.buff2_slow.append(vec2)
            self.sum1_slow += vec1
            self.sum2_slow += vec2

            # Need full slow window before processing
            if len(self.buff1_slow) < self.w_slow:
                # Log buffer state during initial buffering
                import sys
                print(f"[Correlator] [BUFFERING] buff1_slow={len(self.buff1_slow)}/{self.w_slow}, "
                      f"buff2_slow={len(self.buff2_slow)}/{self.w_slow} - returning None")
                sys.stdout.flush()
                return None

            # Compute baselines
            baseline1_fast = self.sum1_fast / len(self.buff1_fast)
            baseline2_fast = self.sum2_fast / len(self.buff2_fast)
            baseline1_slow = self.sum1_slow / self.w_slow
            baseline2_slow = self.sum2_slow / self.w_slow
        
        # THREE CORE SIGNALS
        
        # 1. Slow delta: deviation from neutral baseline
        delta1_slow = vec1 - baseline1_slow
        delta2_slow = vec2 - baseline2_slow
        abs_delta1_slow = np.abs(delta1_slow)
        abs_delta2_slow = np.abs(delta2_slow)
        
        # 2. Fast delta: recent changes
        delta1_fast = vec1 - baseline1_fast
        delta2_fast = vec2 - baseline2_fast
        abs_delta1_fast = np.abs(delta1_fast)
        abs_delta2_fast = np.abs(delta2_fast)
        
        # 3. Derivative: instantaneous rate of change
        if self.prev_vec1 is not None:
            derivative1_raw = vec1 - self.prev_vec1
            derivative2_raw = vec2 - self.prev_vec2
            
            # Smooth derivatives to reduce noise
            derivative1 = (self.derivative_smooth * derivative1_raw + 
                          (1 - self.derivative_smooth) * self.prev_derivative1)
            derivative2 = (self.derivative_smooth * derivative2_raw + 
                          (1 - self.derivative_smooth) * self.prev_derivative2)
            
            self.prev_derivative1 = derivative1.copy()
            self.prev_derivative2 = derivative2.copy()
        else:
            derivative1 = derivative2 = np.zeros(52)
        
        # DETECTION LOGIC FOR DIFFERENT TIMESCALES
        
        # A. TRANSIENT DETECTION (using derivative)
        # Catches synchronized onsets and very brief expressions
        abs_deriv1 = np.abs(derivative1)
        abs_deriv2 = np.abs(derivative2)
        synchronized_movement = (
            (abs_deriv1 > self.derivative_threshold) & 
            (abs_deriv2 > self.derivative_threshold) &
            (np.sign(derivative1) == np.sign(derivative2))
        )
        
        # Similarity of movement speed
        deriv_sum = abs_deriv1 + abs_deriv2 + 1e-6
        movement_similarity = 1.0 - np.abs(abs_deriv1 - abs_deriv2) / deriv_sum
        
        # Transient magnitude based on synchronized speed
        transient_magnitude = (abs_deriv1 + abs_deriv2) * 0.5 * movement_similarity
        
        # Update transient values with decay
        self.transient_vals *= self.transient_decay
        mask_transient = synchronized_movement & (movement_similarity > 0.7)
        self.transient_vals[mask_transient] = np.maximum(
            self.transient_vals[mask_transient],
            transient_magnitude[mask_transient]
        )
        
        # B. BRIEF EXPRESSION DETECTION (using slow baseline, no latching)
        # Traditional comodulation but with fast decay
        change1 = abs_delta1_slow > self.delta_threshold
        change2 = abs_delta2_slow > self.delta_threshold
        similarity = np.abs(delta1_slow - delta2_slow) < self.similarity_threshold
        
        brief_mask = change1 & change2 & similarity
        brief_magnitude = (abs_delta1_slow + abs_delta2_slow) * 0.5
        
        # Update brief values with decay
        self.brief_vals *= self.brief_decay
        self.brief_vals[brief_mask] = np.maximum(
            self.brief_vals[brief_mask],
            brief_magnitude[brief_mask]
        )
        
        # C. SUSTAINED EXPRESSION DETECTION (state machine)
        # Only enter sustained state if expression is stable
        is_stable = (abs_delta1_fast < self.delta_threshold * 0.5) & \
                   (abs_delta2_fast < self.delta_threshold * 0.5)
        
        # Track proximity to neutral
        near_neutral1 = abs_delta1_slow < self.neutral_threshold
        near_neutral2 = abs_delta2_slow < self.neutral_threshold
        near_neutral = near_neutral1 & near_neutral2
        
        self.frames_since_neutral[near_neutral] = 0
        self.frames_since_neutral[~near_neutral] += 1
        
        # Sustained state machine
        sustained_comod = brief_magnitude * brief_mask
        
        # Enter sustained state only if stable
        entering = brief_mask & (sustained_comod > self.delta_threshold) & \
                  ~self.sustained_states & is_stable
        self.sustained_states[entering] = True
        self.sustained_vals[entering] = sustained_comod[entering]
        
        # Update sustained values if still comodulating
        in_state = self.sustained_states & brief_mask
        update_mask = in_state & (sustained_comod > self.sustained_vals)
        self.sustained_vals[update_mask] = sustained_comod[update_mask]
        
        # Exit conditions for sustained state
        recently_neutral = self.frames_since_neutral < self.fps  # Within 1 second
        should_exit = (is_stable & (recently_neutral | near_neutral)) | ~brief_mask
        
        exiting = self.sustained_states & should_exit
        self.sustained_states[exiting] = False
        self.sustained_vals[exiting] = 0.0
        
        # COMBINE ALL THREE SIGNALS
        # Take maximum across all timescales
        combined_output = np.maximum(
            self.transient_vals,
            np.maximum(self.brief_vals, self.sustained_states * self.sustained_vals)
        )
        
        # Store previous vectors for next derivative calculation
        self.prev_vec1 = vec1.copy()
        self.prev_vec2 = vec2.copy()
        
        # Push to LSL
        if self.outlet:
            try:
                self.outlet.push_sample(combined_output.tolist())
            except Exception as e:
                print(f"[Correlator] [FATAL] LSL push failed: {e}")
                import traceback
                traceback.print_exc()

        return combined_output

    def run(self, p1, p2, stop_evt):
        """
        Combined loop: read frames from two Participant instances,
        stream their samples, compute and stream comodulation magnitudes.
        """
        while not stop_evt.is_set():
            sample1 = p1.read_frame()
            sample2 = p2.read_frame()

            # Stream individual samples if outlets exist
            if sample1 is not None and getattr(p1, 'outlet', None):
                p1.outlet.push_sample(sample1)
            if sample2 is not None and getattr(p2, 'outlet', None):
                p2.outlet.push_sample(sample2)

            # Compute and stream comodulation magnitudes
            if sample1 is not None and sample2 is not None:
                vec1 = np.array(sample1[:52], dtype=float)
                vec2 = np.array(sample2[:52], dtype=float)
                self.update(vec1, vec2)

            stop_evt.wait(1.0 / self.fps)
        
    def setup_stream(self, fps=None,
                    name: str = "CoModulation",
                    source_id: str = "comodulator_uid"):
        """Call this once when Start is pressed."""
        if self.outlet is not None:
            print("[Correlator] Stream already active")
            return
            
        info = StreamInfo(
            name=name,
            type="Comodulation",
            channel_count=52,
            nominal_srate=fps or self.fps,
            channel_format="float32",
            source_id=source_id
        )
        self.outlet = StreamOutlet(info)
        print(f"[Correlator] Created LSL stream: {name}")

    def close(self):
        """Clean up the LSL outlet"""
        if hasattr(self, 'outlet') and self.outlet:
            # StreamOutlet has no close() method - just clear the reference
            # The outlet will be cleaned up automatically
            self.outlet = None