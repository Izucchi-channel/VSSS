import gradio as gr
import numpy as np
import cv2
import os
import tempfile
import subprocess
import shutil
from pathlib import Path
import time
from typing import Tuple, List, Optional, Dict
from itertools import product
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualCryptographyVideo:
    def __init__(self):
        # Define white share patterns (2 ones, 2 zeros)
        # Bit order: top-left, top-right, bottom-left, bottom-right
        self.white_patterns = [
            [1, 1, 0, 0],  # 1100
            [1, 0, 1, 0],  # 1010
            [1, 0, 0, 1],  # 1001
            [0, 1, 1, 0],  # 0110
            [0, 1, 0, 1],  # 0101
            [0, 0, 1, 1],  # 0011
        ]

        # Define black share patterns (3 ones, 1 zero)
        self.black_patterns = [
            [1, 1, 1, 0],  # 1110
            [1, 1, 0, 1],  # 1101
            [1, 0, 1, 1],  # 1011
            [0, 1, 1, 1],  # 0111
        ]

        # Target results
        self.target_white_patterns = [
            [1, 1, 1, 0],  # 1110
            [1, 1, 0, 1],  # 1101
            [1, 0, 1, 1],  # 1011
            [0, 1, 1, 1],  # 0111
        ]
        self.target_black = [1, 1, 1, 1]  # 1111

        # Build comprehensive lookup table
        self._build_comprehensive_table()

    def _build_comprehensive_table(self):
        """
        Build a complete lookup table for all possible share combinations
        and their resulting outputs.
        """
        all_patterns = self.white_patterns + self.black_patterns

        # For each target type (white/black) and each pair of ref patterns,
        # find all valid share combinations
        self.valid_combinations = {
            'white': {},  # Key: (ref1_black, ref2_black), Value: list of (s1, s2, result)
            'black': {}
        }

        for s1 in all_patterns:
            s1_type = 'black' if s1 in self.black_patterns else 'white'
            for s2 in all_patterns:
                s2_type = 'black' if s2 in self.black_patterns else 'white'

                # Calculate OR result
                result = [a | b for a, b in zip(s1, s2)]

                # Determine target type of result
                if result == self.target_black:
                    result_type = 'black'
                elif result in self.target_white_patterns:
                    result_type = 'white'
                else:
                    continue

                # Store combination
                ref_key = (s1_type == 'black', s2_type == 'black')
                if ref_key not in self.valid_combinations[result_type]:
                    self.valid_combinations[result_type][ref_key] = []

                self.valid_combinations[result_type][ref_key].append({
                    'share1': s1.copy(),
                    'share2': s2.copy(),
                    'result': result.copy()
                })

        # Verify we have combinations for all cases
        print(f"White combinations: {sum(len(v) for v in self.valid_combinations['white'].values())}")
        print(f"Black combinations: {sum(len(v) for v in self.valid_combinations['black'].values())}")

    def process_frame_to_blocks(self, frame: np.ndarray, target_width: int, target_height: int,
                                invert: bool = False) -> np.ndarray:
        """Convert frame to binary and resize to target block dimensions"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        resized = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

        if invert:
            # Invert: pixels > 128 become 0 (black in inverted mode)
            binary = (resized <= 128).astype(np.uint8)
        else:
            binary = (resized > 128).astype(np.uint8)

        return binary

    def generate_shares_with_distribution(self,
                                         target_frame: np.ndarray,
                                         ref1_frame: np.ndarray,
                                         ref2_frame: np.ndarray,
                                         balance_factor: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate shares where BOTH maintain similarity to their reference images.

        balance_factor: Controls the trade-off between share quality and randomness.
        0.0 = maximize reference similarity (may have artifacts)
        1.0 = maximize randomness (shares become noise)
        0.5 = balanced approach (default)
        """
        height, width = target_frame.shape

        # Initialize share frames (2x expansion for 2x2 blocks)
        share1 = np.zeros((height * 2, width * 2), dtype=np.uint8)
        share2 = np.zeros((height * 2, width * 2), dtype=np.uint8)
        result = np.zeros((height * 2, width * 2), dtype=np.uint8)

        # Process each block
        for i in range(height):
            for j in range(width):
                target_pixel = target_frame[i, j]
                ref1_pixel = ref1_frame[i, j] if ref1_frame is not None else 0
                ref2_pixel = ref2_frame[i, j] if ref2_frame is not None else 0

                # Determine desired pattern types
                ref1_black = (ref1_pixel == 1)
                ref2_black = (ref2_pixel == 1)
                target_black = (target_pixel == 1)

                # Get valid combinations
                combinations = self._get_best_combinations(
                    target_black, ref1_black, ref2_black, balance_factor
                )

                # Select one combination (weighted random selection)
                if combinations:
                    selected = combinations[np.random.randint(len(combinations))]
                else:
                    # Fallback: use any valid combination
                    selected = self._get_fallback_combination(target_black)

                # Place patterns in 2x2 blocks
                positions = [
                    (i*2, j*2),      # top-left
                    (i*2, j*2+1),    # top-right
                    (i*2+1, j*2),    # bottom-left
                    (i*2+1, j*2+1)   # bottom-right
                ]

                for k, (y, x) in enumerate(positions):
                    share1[y, x] = selected['share1'][k] * 255
                    share2[y, x] = selected['share2'][k] * 255
                    result[y, x] = selected['result'][k] * 255

        return share1, share2, result

    def _get_best_combinations(self, target_black: bool,
                              ref1_black: bool, ref2_black: bool,
                              balance_factor: float) -> List[Dict]:
        """Get the best share combinations based on reference similarity"""
        target_type = 'black' if target_black else 'white'
        ref_key = (ref1_black, ref2_black)

        if target_type not in self.valid_combinations:
            return []

        if ref_key not in self.valid_combinations[target_type]:
            # Try opposite reference if exact match not available
            ref_key = (not ref1_black, not ref2_black)
            if ref_key not in self.valid_combinations[target_type]:
                return self._get_all_combinations_for_target(target_type)

        all_combinations = self.valid_combinations[target_type][ref_key]

        if not all_combinations:
            return []

        # Score each combination based on how well it matches references
        scored_combinations = []
        for combo in all_combinations:
            score = self._score_combination(combo, ref1_black, ref2_black, balance_factor)
            scored_combinations.append((score, combo))

        # Sort by score (higher is better)
        scored_combinations.sort(key=lambda x: x[0], reverse=True)

        # Return top 50% of combinations
        top_n = max(1, len(scored_combinations) // 2)
        return [combo for _, combo in scored_combinations[:top_n]]

    def _score_combination(self, combo: Dict, ref1_black: bool, ref2_black: bool,
                          balance_factor: float) -> float:
        """Score a combination based on how well it matches reference images"""
        score = 0.0

        # Score share1 based on ref1
        s1 = combo['share1']
        if ref1_black:
            # For black reference, more 1s is better
            score += sum(s1) / 4.0 * (1 - balance_factor)
        else:
            # For white reference, more 0s is better
            score += (4 - sum(s1)) / 4.0 * (1 - balance_factor)

        # Score share2 based on ref2
        s2 = combo['share2']
        if ref2_black:
            score += sum(s2) / 4.0 * (1 - balance_factor)
        else:
            score += (4 - sum(s2)) / 4.0 * (1 - balance_factor)

        # Add randomness bonus
        score += np.random.random() * balance_factor * 2

        return score

    def _get_all_combinations_for_target(self, target_type: str) -> List[Dict]:
        """Get all valid combinations for a target type"""
        all_combos = []
        for ref_key in self.valid_combinations[target_type]:
            all_combos.extend(self.valid_combinations[target_type][ref_key])
        return all_combos

    def _get_fallback_combination(self, target_black: bool) -> Dict:
        """Get a fallback combination when no optimal combination is found"""
        target_type = 'black' if target_black else 'white'

        # Try all combinations
        all_combos = self._get_all_combinations_for_target(target_type)

        if all_combos:
            return all_combos[np.random.randint(len(all_combos))]

        # Absolute fallback: create basic combination
        if target_black:
            return {
                'share1': [1, 1, 1, 0],
                'share2': [0, 1, 1, 1],
                'result': [1, 1, 1, 1]
            }
        else:
            return {
                'share1': [1, 1, 0, 0],
                'share2': [0, 0, 1, 1],
                'result': [1, 1, 1, 0]
            }

class VideoProcessor:
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.vc = VisualCryptographyVideo()

    def check_ffmpeg(self):
        """Check if FFmpeg is available"""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_video_info(self, video_path: str) -> dict:
        """Get video information using FFmpeg"""
        cmd = [
            'ffmpeg', '-i', video_path,
            '-f', 'null', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        info = {'fps': 30, 'duration': 0, 'width': 0, 'height': 0, 'frame_count': 0}

        for line in result.stderr.split('\n'):
            if 'fps' in line.lower() and 'stream' in line.lower():
                parts = line.split(',')
                for part in parts:
                    if 'fps' in part:
                        try:
                            fps_str = part.strip().split()[0]
                            info['fps'] = float(fps_str)
                        except:
                            pass
                    if 'x' in part and 'y' in part.lower():
                        try:
                            dims = part.strip().split()
                            for d in dims:
                                if 'x' in d:
                                    w, h = d.split('x')
                                    info['width'] = int(w)
                                    info['height'] = int(h)
                        except:
                            pass
            if 'Duration' in line:
                time_str = line.split('Duration: ')[1].split(',')[0]
                h, m, s = time_str.split(':')
                info['duration'] = float(h) * 3600 + float(m) * 60 + float(s)
            if 'frame=' in line:
                try:
                    frame_part = line.split('frame=')[1].split()[0]
                    info['frame_count'] = int(frame_part)
                except:
                    pass

        return info

    def extract_frames(self, video_path: str, output_dir: str,
                       start_time: float = 0, duration: Optional[float] = None,
                       frame_skip: int = 1) -> List[str]:
        """
        Extract frames from video with optional frame skipping

        Args:
            frame_skip: Process every Nth frame (1 = all frames, 2 = every other frame)
        """
        frame_pattern = os.path.join(output_dir, 'frame_%06d.png')

        cmd = ['ffmpeg']
        if start_time > 0:
            cmd.extend(['-ss', str(start_time)])

        cmd.extend(['-i', video_path])

        if duration:
            cmd.extend(['-t', str(duration)])

        # Extract frames at original FPS
        cmd.extend(['-vsync', '0', frame_pattern])

        if self.use_gpu:
            cmd = self._add_gpu_params(cmd)

        subprocess.run(cmd, capture_output=True, check=True)

        all_frames = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')])

        # Apply frame skipping
        if frame_skip > 1:
            skipped_frames = all_frames[::frame_skip]
            # Remove skipped frames
            for f in all_frames:
                if f not in skipped_frames:
                    try:
                        os.remove(f)
                    except:
                        pass
            return skipped_frames

        return all_frames

    def _add_gpu_params(self, cmd: List[str]) -> List[str]:
        """Add GPU acceleration parameters to FFmpeg command"""
        if shutil.which('nvidia-smi'):
            try:
                test_cmd = ['ffmpeg', '-hwaccels']
                result = subprocess.run(test_cmd, capture_output=True, text=True)
                if 'cuda' in result.stdout:
                    cmd.insert(1, '-hwaccel')
                    cmd.insert(2, 'cuda')
                    return cmd
            except:
                pass
        return cmd

    def create_video_from_frames(self, frames_dir: str, output_path: str,
                                 fps: float, pixel_size: int = 4,
                                 compression: int = 23, invert_output: bool = False) -> str:
        """
        Create video from frames with specified parameters

        Args:
            fps: Output FPS (same as original target video FPS)
            invert_output: Invert colors in output video
        """
        if pixel_size != 1:
            resized_dir = os.path.join(os.path.dirname(frames_dir), 'resized')
            os.makedirs(resized_dir, exist_ok=True)

            for frame_file in sorted(os.listdir(frames_dir)):
                if frame_file.endswith('.png'):
                    img = cv2.imread(os.path.join(frames_dir, frame_file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        if invert_output:
                            img = 255 - img
                        new_size = (img.shape[1] * pixel_size, img.shape[0] * pixel_size)
                        resized = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)
                        cv2.imwrite(os.path.join(resized_dir, frame_file), resized)

            frames_dir = resized_dir
        elif invert_output:
            # Apply inversion without resizing
            inverted_dir = os.path.join(os.path.dirname(frames_dir), 'inverted')
            os.makedirs(inverted_dir, exist_ok=True)

            for frame_file in sorted(os.listdir(frames_dir)):
                if frame_file.endswith('.png'):
                    img = cv2.imread(os.path.join(frames_dir, frame_file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = 255 - img
                        cv2.imwrite(os.path.join(inverted_dir, frame_file), img)

            frames_dir = inverted_dir

        frame_pattern = os.path.join(frames_dir, 'frame_%06d.png')

        cmd = [
            'ffmpeg',
            '-framerate', str(fps),
            '-i', frame_pattern,
            '-c:v', 'libx264',
            '-crf', str(compression),
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            '-y',
            output_path
        ]

        if self.use_gpu:
            cmd = self._add_gpu_encoding_params(cmd)

        subprocess.run(cmd, capture_output=True, check=True)
        return output_path

    def _add_gpu_encoding_params(self, cmd: List[str]) -> List[str]:
        """Add GPU encoding parameters"""
        if shutil.which('nvidia-smi'):
            try:
                test_cmd = ['ffmpeg', '-encoders']
                result = subprocess.run(test_cmd, capture_output=True, text=True)
                if 'h264_nvenc' in result.stdout:
                    for i, arg in enumerate(cmd):
                        if arg == 'libx264':
                            cmd[i] = 'h264_nvenc'
                            break
            except:
                pass
        return cmd

    def process_video_segment(self,
                             ref1_path: str,
                             ref2_path: str,
                             target_path: str,
                             output_dir: str,
                             start_time: float = 0,
                             duration: Optional[float] = None,
                             pixel_count: int = 100,
                             pixel_size: int = 4,
                             compression: int = 23,
                             balance_factor: float = 0.5,
                             frame_skip: int = 1,
                             invert_ref1: bool = False,
                             invert_ref2: bool = False,
                             invert_target: bool = False,
                             invert_result: bool = False,
                             output_json: bool = True,
                             progress_callback=None) -> Tuple[str, str, str, Optional[str]]:
        """Process a video segment and generate shares"""

        temp_dir = tempfile.mkdtemp()

        try:
            target_info = self.get_video_info(target_path)
            ref1_info = self.get_video_info(ref1_path)
            ref2_info = self.get_video_info(ref2_path)

            target_fps = target_info.get('fps', 30)
            ref1_fps = ref1_info.get('fps', 30)
            ref2_fps = ref2_info.get('fps', 30)

            original_fps = target_fps

            print(f"Target video FPS: {original_fps}, Frame skip: {frame_skip}")
            print(f"Inversion settings - Ref1: {invert_ref1}, Ref2: {invert_ref2}, Target: {invert_target}, Result: {invert_result}")

            # Extract frames
            ref1_frames_dir = os.path.join(temp_dir, 'ref1')
            ref2_frames_dir = os.path.join(temp_dir, 'ref2')
            target_frames_dir = os.path.join(temp_dir, 'target')

            os.makedirs(ref1_frames_dir)
            os.makedirs(ref2_frames_dir)
            os.makedirs(target_frames_dir)

            print("Extracting frames...")

            # Extract frames with skipping
            ref1_frames = self.extract_frames(ref1_path, ref1_frames_dir, start_time, duration, frame_skip)
            ref2_frames = self.extract_frames(ref2_path, ref2_frames_dir, start_time, duration, frame_skip)
            target_frames = self.extract_frames(target_path, target_frames_dir, start_time, duration, frame_skip)

            print(f"Extracted {len(target_frames)} frames (every {frame_skip}th frame)")

            # Determine target dimensions
            sample_frame = cv2.imread(target_frames[0], cv2.IMREAD_GRAYSCALE)
            orig_h, orig_w = sample_frame.shape

            target_block_width = pixel_count
            target_block_height = int(pixel_count * (orig_h / orig_w))

            # Create output directories
            share1_frames_dir = os.path.join(temp_dir, 'share1')
            share2_frames_dir = os.path.join(temp_dir, 'share2')
            result_frames_dir = os.path.join(temp_dir, 'result')

            os.makedirs(share1_frames_dir)
            os.makedirs(share2_frames_dir)
            os.makedirs(result_frames_dir)

            print("Processing frames with balanced distribution...")

            # Store frame data for JSON output
            frame_data = {
                'share1': [],
                'share2': [],
                'result': []
            }

            total_frames = len(target_frames)
            for idx, target_frame_path in enumerate(target_frames):
                target_frame = cv2.imread(target_frame_path, cv2.IMREAD_GRAYSCALE)

                current_time = (idx * frame_skip) / target_fps

                ref1_idx = min(int(current_time * ref1_fps / frame_skip), len(ref1_frames) - 1)
                ref2_idx = min(int(current_time * ref2_fps / frame_skip), len(ref2_frames) - 1)

                ref1_frame = cv2.imread(ref1_frames[ref1_idx], cv2.IMREAD_GRAYSCALE)
                ref2_frame = cv2.imread(ref2_frames[ref2_idx], cv2.IMREAD_GRAYSCALE)

                # Process frames with optional inversion
                target_binary = self.vc.process_frame_to_blocks(target_frame, target_block_width, target_block_height, invert_target)
                ref1_binary = self.vc.process_frame_to_blocks(ref1_frame, target_block_width, target_block_height, invert_ref1)
                ref2_binary = self.vc.process_frame_to_blocks(ref2_frame, target_block_width, target_block_height, invert_ref2)

                # Generate shares with balanced distribution
                share1, share2, result = self.vc.generate_shares_with_distribution(
                    target_binary, ref1_binary, ref2_binary, balance_factor
                )

                # Save frames
                frame_name = f'frame_{idx:06d}.png'
                cv2.imwrite(os.path.join(share1_frames_dir, frame_name), share1)
                cv2.imwrite(os.path.join(share2_frames_dir, frame_name), share2)
                cv2.imwrite(os.path.join(result_frames_dir, frame_name), result)

                if output_json:
                    share1_flat = (share1.flatten() // 255).astype(np.uint8)
                    share2_flat = (share2.flatten() // 255).astype(np.uint8)
                    result_flat = (result.flatten() // 255).astype(np.uint8)

                    frame_data['share1'].append(
                        self.pack_bits(share1_flat)
                    )

                    frame_data['share2'].append(
                        self.pack_bits(share2_flat)
                    )

                    frame_data['result'].append(
                        self.pack_bits(result_flat)
                    )

                if progress_callback and idx % 10 == 0:
                    progress = (idx + 1) / total_frames
                    progress_callback(progress)

            output_fps = original_fps / frame_skip

            # Create output videos with original FPS
            print(f"Creating output videos with original FPS: {output_fps}")

            share1_path = os.path.join(output_dir, 'share1.mp4')
            share2_path = os.path.join(output_dir, 'share2.mp4')
            result_path = os.path.join(output_dir, 'result.mp4')

            # Use original FPS for all output videos
            self.create_video_from_frames(share1_frames_dir, share1_path, output_fps, pixel_size, compression)
            self.create_video_from_frames(share2_frames_dir, share2_path, output_fps, pixel_size, compression)
            self.create_video_from_frames(result_frames_dir, result_path, output_fps, pixel_size, compression, invert_result)

            # Generate JSON file if requested
            json_path = None
            if output_json:
                json_path = self._generate_json_output(
                    output_dir, frame_data, pixel_count, target_block_height,
                    output_fps, pixel_size, compression, balance_factor, frame_skip,
                    invert_ref1, invert_ref2, invert_target, invert_result,
                    output_fps, len(target_frames)
                )

            return share1_path, share2_path, result_path, json_path

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    def pack_bits(self, bits):
        packed = []

        for i in range(0, len(bits), 8):
            value = 0

            for j in range(8):
                if i + j < len(bits):
                    value |= int(bits[i + j]) << (7 - j)

            packed.append(value)

        return packed

    def _generate_json_output(self, output_dir: str, frame_data: Dict,
                             pixel_count: int, pixel_height: int,
                             fps: float, pixel_size: int, compression: int,
                             balance_factor: float, frame_skip: int,
                             invert_ref1: bool, invert_ref2: bool,
                             invert_target: bool, invert_result: bool,
                             original_fps: float, total_frames: int) -> str:
        """Generate JSON output file with all settings and frame data"""

        json_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_frames': total_frames,
                'original_fps': original_fps,
                'settings': {
                    'pixel_count_width': pixel_count,
                    'pixel_count_height': pixel_height,
                    'fps': fps,
                    'pixel_size': pixel_size,
                    'compression': compression,
                    'balance_factor': balance_factor,
                    'frame_skip': frame_skip,
                    'invert_ref1': invert_ref1,
                    'invert_ref2': invert_ref2,
                    'invert_target': invert_target,
                    'invert_result': invert_result
                }
            },
            'data': {
                'share1': frame_data['share1'],
                'share2': frame_data['share2'],
                'result': frame_data['result']
            }
        }

        json_path = os.path.join(output_dir, 'visual_cryptography_data.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(
                json_data,
                f,
                ensure_ascii=False,
                separators=(',', ':')
            )

        print(f"JSON data saved to: {json_path}")
        return json_path

def create_gradio_interface():
    """Create the Gradio interface"""

    processor = VideoProcessor(use_gpu=False)

    def process_preview(ref1_video, ref2_video, target_video,
                       pixel_count, pixel_size, compression,
                       start_second, duration_seconds, use_gpu, balance_factor,
                       frame_skip, invert_ref1, invert_ref2, invert_target, invert_result,
                       output_json):
        """Process preview segment"""
        if not all([ref1_video, ref2_video, target_video]):
            return None, None, None, "Please upload all three videos", None

        processor.use_gpu = use_gpu

        output_dir = tempfile.mkdtemp()

        try:
            share1, share2, result, json_path = processor.process_video_segment(
                ref1_video, ref2_video, target_video,
                output_dir,
                start_time=start_second,
                duration=duration_seconds,
                pixel_count=int(pixel_count),
                pixel_size=int(pixel_size),
                compression=int(compression),
                balance_factor=float(balance_factor),
                frame_skip=int(frame_skip),
                invert_ref1=invert_ref1,
                invert_ref2=invert_ref2,
                invert_target=invert_target,
                invert_result=invert_result,
                output_json=output_json,
                progress_callback=None
            )

            return share1, share2, result, "Preview generation complete!", json_path

        except Exception as e:
            print(f"Error processing preview: {str(e)}")
            return None, None, None, f"Error: {str(e)}", None

    def process_full(ref1_video, ref2_video, target_video,
                    pixel_count, pixel_size, compression,
                    use_gpu, balance_factor, frame_skip,
                    invert_ref1, invert_ref2, invert_target, invert_result,
                    output_json):
        """Process full videos"""
        if not all([ref1_video, ref2_video, target_video]):
            return None, None, None, "Please upload all three videos", None

        processor.use_gpu = use_gpu

        output_dir = tempfile.mkdtemp()

        try:
            share1, share2, result, json_path = processor.process_video_segment(
                ref1_video, ref2_video, target_video,
                output_dir,
                start_time=0,
                duration=None,
                pixel_count=int(pixel_count),
                pixel_size=int(pixel_size),
                compression=int(compression),
                balance_factor=float(balance_factor),
                frame_skip=int(frame_skip),
                invert_ref1=invert_ref1,
                invert_ref2=invert_ref2,
                invert_target=invert_target,
                invert_result=invert_result,
                output_json=output_json,
                progress_callback=None
            )

            return share1, share2, result, "Full video generation complete!", json_path

        except Exception as e:
            print(f"Error processing full video: {str(e)}")
            return None, None, None, f"Error: {str(e)}", None

    # Create the Gradio interface
    with gr.Blocks(title="Visual Cryptography Video Generator - Balanced") as demo:
        gr.Markdown("""
        # 🎬 Visual Cryptography Video Generator (Balanced Distribution)

        Upload three videos (two references and one target).
        **Both shares will maintain similarity to their reference videos** while revealing the target when overlapped.

        ### Key Features:
        - **Frame Skip**: Control processing speed by skipping frames (output FPS always matches target video)
        - **Color Inversion**: Individually invert black/white recognition for each video
        - **JSON Output**: Export all frame data as binary arrays
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📁 Input Videos")
                ref1_video = gr.Video(label="Reference Video 1 (for Share 1)")
                ref2_video = gr.Video(label="Reference Video 2 (for Share 2)")
                target_video = gr.Video(label="Target Video (revealed when combined)")

            with gr.Column():
                gr.Markdown("### ⚙️ Settings")

                with gr.Group():
                    gr.Markdown("#### Resolution")
                    pixel_count = gr.Slider(
                        minimum=10, maximum=200, value=64, step=1,
                        label="Pixel Count (Width)"
                    )
                    pixel_size = gr.Slider(
                        minimum=1, maximum=16, value=4, step=1,
                        label="Output Pixel Size (px)"
                    )

                with gr.Group():
                    gr.Markdown("#### Video Settings")
                    compression = gr.Slider(
                        minimum=1, maximum=51, value=23, step=1,
                        label="FFmpeg Compression (CRF, lower = better quality)"
                    )
                    frame_skip = gr.Slider(
                        minimum=1, maximum=10, value=1, step=1,
                        label="Frame Skip (1=all frames, 2=every other frame, etc.)"
                    )

                with gr.Group():
                    gr.Markdown("#### Performance")
                    use_gpu = gr.Checkbox(
                        value=False,
                        label="Use GPU Acceleration (if available)"
                    )

                with gr.Group():
                    gr.Markdown("#### Color Inversion")
                    gr.Markdown("Invert black/white recognition for each video")
                    invert_ref1 = gr.Checkbox(value=False, label="Invert Reference 1 (white↔black)")
                    invert_ref2 = gr.Checkbox(value=False, label="Invert Reference 2 (white↔black)")
                    invert_target = gr.Checkbox(value=False, label="Invert Target Video (white↔black)")
                    invert_result = gr.Checkbox(value=False, label="Invert Output Result (white↔black)")

                with gr.Group():
                    gr.Markdown("#### Advanced")
                    balance_factor = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.3, step=0.001,
                        label="Balance Factor (0 = maximize reference similarity, 1 = maximize randomness)"
                    )
                    output_json = gr.Checkbox(
                        value=True,
                        label="Output JSON data file"
                    )

        with gr.Row():
            gr.Markdown("### 🎯 Generation Controls")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Preview (Test Segment)")
                with gr.Row():
                    start_second = gr.Number(value=0, label="Start Second", precision=1)
                    duration_seconds = gr.Number(value=3, label="Duration (seconds)", precision=1)
                preview_btn = gr.Button("🔍 Generate Preview", variant="secondary", size="lg")

            with gr.Column():
                gr.Markdown("#### Full Generation")
                gr.Markdown("")
                full_btn = gr.Button("🎬 Generate Full Video", variant="primary", size="lg")

        with gr.Row():
            gr.Markdown("### 📤 Output Videos")

        with gr.Row():
            with gr.Column():
                share1_output = gr.Video(label="Share 1 (Reference 1 pattern)")
            with gr.Column():
                share2_output = gr.Video(label="Share 2 (Reference 2 pattern)")

        with gr.Row():
            result_output = gr.Video(label="Combined Result (Target video revealed)")

        status_output = gr.Textbox(label="Status", interactive=False)
        json_output = gr.File(label="JSON Data File (if enabled)")

        # Connect buttons
        preview_btn.click(
            fn=process_preview,
            inputs=[ref1_video, ref2_video, target_video,
                   pixel_count, pixel_size, compression,
                   start_second, duration_seconds, use_gpu, balance_factor,
                   frame_skip, invert_ref1, invert_ref2, invert_target, invert_result,
                   output_json],
            outputs=[share1_output, share2_output, result_output, status_output, json_output]
        )

        full_btn.click(
            fn=process_full,
            inputs=[ref1_video, ref2_video, target_video,
                   pixel_count, pixel_size, compression,
                   use_gpu, balance_factor, frame_skip,
                   invert_ref1, invert_ref2, invert_target, invert_result,
                   output_json],
            outputs=[share1_output, share2_output, result_output, status_output, json_output]
        )

        gr.Markdown("""
        ### 💡 Tips:
        - **Output FPS**: Automatically matches your target video's original FPS
        - **Frame Skip**: Use 2 or 3 for faster processing with minimal quality loss (playback speed remains the same)
        - **Color Inversion**: Useful when your reference videos have opposite polarity
        - **Invert Target**: Swap white/black interpretation of the target video
        - **Invert Result**: Final output video colors are inverted
        - **JSON Output**: Contains all frame data as binary arrays for further analysis
        """)

    return demo

if __name__ == "__main__":
    test_processor = VideoProcessor()
    if not test_processor.check_ffmpeg():
        print("Warning: FFmpeg not found. Please install FFmpeg to use this application.")
        print("Download from: https://ffmpeg.org/download.html")

    demo = create_gradio_interface()
    demo.launch(share=True, debug=True)
