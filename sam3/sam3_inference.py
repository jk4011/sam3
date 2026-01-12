import torch
import numpy as np
import os
import tempfile
import shutil
import torch.nn.functional as F
from sam3.model_builder import build_sam3_video_predictor
from jhutil import cache_output, print_time
import builtins


class Sam3VideoState:
    """Singleton to persist model state across module reloads (even with autoreload)."""

    def __new__(cls):
        # Store in builtins to survive module reloads
        if not hasattr(builtins, '_sam3_video_state_instance'):
            instance = super().__new__(cls)
            instance.predictor = None
            instance.current_session_id = None
            instance.last_image_paths = None
            builtins._sam3_video_state_instance = instance
        return builtins._sam3_video_state_instance


_state = Sam3VideoState()


@print_time(func_name="init_sam3_predictor")
def init_predictor(gpus_to_use=None):
    """Initialize the SAM3 video predictor.

    Args:
        gpus_to_use: List of GPU IDs to use. Defaults to current device.
    """
    global _state
    if gpus_to_use is None:
        gpus_to_use = [torch.cuda.current_device()] if torch.cuda.is_available() else [0]
    _state.predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

@cache_output(func_name="_sam3_inference", override=False)
def _sam3_inference(video_path, text, image_paths=None, score_threshold=0.5, max_num_objects=3, gpus_to_use=None):
    """Internal function for SAM3 video inference with session reuse optimization.

    Args:
        video_path: str (MP4 file path or image directory). Ignored if image_paths is provided.
        text: str - text prompt for segmentation
        image_paths: list of image file paths (optional). If provided, creates temp dir with symlinks.
        score_threshold: float - confidence threshold for detection (default: 0.5)
        max_num_objects: int - maximum number of objects to track (default: 5)
        gpus_to_use: list of GPU IDs

    Returns:
        dict mapping frame_idx to outputs dict containing:
            - 'out_obj_ids': numpy array of object IDs
            - 'out_binary_masks': numpy array of binary masks
            - 'out_probs': numpy array of probabilities
            - 'out_boxes_xywh': numpy array of bounding boxes
    """
    global _state

    if _state.predictor is None:
        init_predictor(gpus_to_use)

    # Set score threshold and max objects
    _state.predictor.model.score_threshold_detection = score_threshold
    _state.predictor.model.max_num_objects = max_num_objects

    # Check if we can reuse existing session
    can_reuse_session = (
        _state.current_session_id is not None
        and image_paths is not None
        and image_paths == _state.last_image_paths
    )

    temp_dir = None
    session_id = None

    try:
        if can_reuse_session:
            # Reuse existing session - no need to reload images
            session_id = _state.current_session_id
            # Just reset the session to clear previous prompts
            _state.predictor.handle_request(
                request=dict(
                    type="reset_session",
                    session_id=session_id,
                )
            )
        else:
            # Need to create new session
            # First, close existing session if any
            if _state.current_session_id is not None:
                try:
                    _state.predictor.handle_request(
                        request=dict(
                            type="close_session",
                            session_id=_state.current_session_id,
                        )
                    )
                except Exception:
                    # Ignore errors when closing old session
                    pass
                _state.current_session_id = None
                _state.last_image_paths = None

            # Create temporary directory with symlinks if image_paths is provided
            if image_paths is not None:
                temp_dir = tempfile.mkdtemp(prefix="sam3_inference_")
                try:
                    # Create symlinks with sequential naming (0000.ext, 0001.ext, ...)
                    for idx, img_path in enumerate(image_paths):
                        ext = os.path.splitext(img_path)[1]  # Get extension (.png, .jpg, etc.)
                        link_name = f"{idx:04d}{ext}"
                        link_path = os.path.join(temp_dir, link_name)
                        os.symlink(os.path.abspath(img_path), link_path)

                    # Use temp directory as video_path
                    actual_video_path = temp_dir
                except Exception as e:
                    # Clean up temp dir if symlink creation fails
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    raise e
            else:
                actual_video_path = video_path

            # Start new session
            response = _state.predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=actual_video_path,
                )
            )
            session_id = response["session_id"]

            # Update global state
            _state.current_session_id = session_id
            _state.last_image_paths = image_paths

            # Reset session (required after start_session)
            _state.predictor.handle_request(
                request=dict(
                    type="reset_session",
                    session_id=session_id,
                )
            )

        # Add text prompt on frame 0
        _state.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=text,
            )
        )

        # Propagate through video
        outputs_per_frame = {}
        for response in _state.predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]

        return outputs_per_frame

    finally:
        # Clean up temporary directory if created (only for new sessions)
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def close_current_session():
    """Manually close the current SAM3 session to free GPU memory.

    This is useful when you want to explicitly free resources between different
    datasets or when you're done with all inference.
    """
    global _state

    if _state.current_session_id is not None and _state.predictor is not None:
        try:
            _state.predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=_state.current_session_id,
                )
            )
        except Exception:
            # Ignore errors when closing session
            pass
        finally:
            _state.current_session_id = None
            _state.last_image_paths = None


def sam3_video_inference(video_path=None, text=None, image_paths=None, score_threshold=0.5, max_num_objects=3, merge_threshold=0.5, mask_size=(378, 504)):
    """SAM3 video inference with simplified output.

    Args:
        video_path: str (MP4 file path or image directory). Either this or image_paths must be provided.
        text: str - text prompt for segmentation
        image_paths: list of image file paths (optional). If provided, creates temp dir with symlinks.
                     Takes precedence over video_path.
        score_threshold: float - confidence threshold for detection (default: 0.5).
                        Lower values detect more objects with lower confidence.
        max_num_objects: int - maximum number of objects to track (default: 5).
                        Limits memory usage by capping tracked objects.
        merge_threshold: float - minimum average probability to include in merge (default: 0.5).
                        Objects with lower average probability are excluded from merging.
                        If all objects are below threshold, only the highest scoring one is kept.
        gpus_to_use: list of GPU IDs or int (single GPU ID)

    Returns:
        torch.Tensor of shape [num_frames, H, W] with bool dtype.
        All detected objects are merged using logical OR.
    """
    # Validate inputs
    if video_path is None and image_paths is None:
        raise ValueError("Either video_path or image_paths must be provided")

    if text is None:
        raise ValueError("text prompt must be provided")

    outputs_per_frame = _sam3_inference(
        video_path=video_path,
        text=text,
        image_paths=image_paths,
        score_threshold=score_threshold,
        max_num_objects=max_num_objects,
        gpus_to_use=[0]
    )

    # Step 1: Calculate mean probability for each obj_id across all frames
    obj_id_to_probs = {}  # {obj_id: [prob1, prob2, ...]}

    for frame_idx in sorted(outputs_per_frame.keys()):
        outputs = outputs_per_frame[frame_idx]
        obj_ids = outputs["out_obj_ids"]
        probs = outputs["out_probs"]

        for i, obj_id in enumerate(obj_ids):
            if obj_id not in obj_id_to_probs:
                obj_id_to_probs[obj_id] = []
            obj_id_to_probs[obj_id].append(float(probs[i]))

    # Calculate mean probability for each obj_id
    obj_id_to_mean_prob = {
        obj_id: np.mean(prob_list)
        for obj_id, prob_list in obj_id_to_probs.items()
    }

    # Step 2: Determine valid obj_ids based on merge_threshold
    valid_obj_ids = set()
    for obj_id, mean_prob in obj_id_to_mean_prob.items():
        if mean_prob >= merge_threshold:
            valid_obj_ids.add(obj_id)

    # Step 3: If no objects pass threshold, keep only the highest mean prob one
    if len(valid_obj_ids) == 0 and len(obj_id_to_mean_prob) > 0:
        best_obj_id = max(obj_id_to_mean_prob.items(), key=lambda x: x[1])[0]
        valid_obj_ids.add(best_obj_id)

    # Step 4: Merge masks frame by frame, only for valid obj_ids
    # Determine expected number of frames (from image_paths or actual frames)
    if image_paths is not None:
        num_expected_frames = len(image_paths)
    else:
        # For video_path, use the actual number of frames processed
        num_expected_frames = len(outputs_per_frame)

    # Get reference mask shape from first available frame
    reference_shape = None
    for frame_idx in sorted(outputs_per_frame.keys()):
        outputs = outputs_per_frame[frame_idx]
        if len(outputs["out_binary_masks"]) > 0:
            reference_shape = outputs["out_binary_masks"][0].shape
            break

    video_segments = []
    for frame_idx in range(num_expected_frames):
        # Check if this frame was processed
        if frame_idx not in outputs_per_frame:
            # Frame was skipped, create empty mask
            if reference_shape is not None:
                merged_mask = np.zeros(reference_shape, dtype=bool)
            else:
                merged_mask = None
            video_segments.append(merged_mask)
            continue

        outputs = outputs_per_frame[frame_idx]
        obj_ids = outputs["out_obj_ids"]
        binary_masks = outputs["out_binary_masks"]  # shape: [num_objects, H, W]

        if len(binary_masks) > 0:
            # Update reference shape if needed
            if reference_shape is None:
                reference_shape = binary_masks[0].shape

            # Merge only masks whose obj_id is in valid_obj_ids
            merged_mask = np.zeros_like(binary_masks[0], dtype=bool)
            for i, obj_id in enumerate(obj_ids):
                if obj_id in valid_obj_ids:
                    merged_mask = np.logical_or(merged_mask, binary_masks[i] > 0)
        else:
            # No objects detected, create empty mask
            if reference_shape is not None:
                merged_mask = np.zeros(reference_shape, dtype=bool)
            else:
                merged_mask = None

        video_segments.append(merged_mask)

    # Handle None values - replace with empty masks
    if reference_shape is None:
        # No masks found in any frame, need to determine shape from image
        if image_paths is not None and len(image_paths) > 0:
            # Read first image to get shape
            from PIL import Image
            with Image.open(image_paths[0]) as img:
                reference_shape = (img.height, img.width)
        else:
            # Cannot determine shape, raise error
            raise ValueError("No valid masks found in video and no image_paths provided to determine shape")

    # Replace None with empty masks to maintain frame count
    for i in range(len(video_segments)):
        if video_segments[i] is None:
            video_segments[i] = np.zeros(reference_shape, dtype=bool)

    # Stack into single tensor
    video_segments = torch.tensor(np.stack(video_segments), dtype=torch.bool)

    video_segments_resized = F.interpolate(
        video_segments[:, None, :, :].float(),
        size=mask_size,
        mode="bilinear",
        align_corners=False
    )[:, 0] > 0.5

    return video_segments_resized


if __name__ == "__main__":
    import os
    import glob

    # Test parameters (configurable)
    VIDEO_PATH = "/root/data1/jinhyeok/seg123/dataset/neu3d/coffee_martini/cam00.mp4"

    # TODO for claude: check with this parameter for test
    IMAGE_DIR = "/root/data1/jinhyeok/seg123/dataset/neu3d/coffee_martini/cam00/images"

    TEXT_PROMPT = "cup"
    REFERENCE_PATH = "/root/data1/jinhyeok/sam3/assets/outputs_per_frame.pth"
    IOU_THRESHOLD = 0.95

    # Test with image_paths (subset of images)
    SCORE_THRESHOLD = 0.5  # Adjust this to see more/fewer objects
    MAX_NUM_OBJECTS = 5  # Maximum number of objects to track
    MERGE_THRESHOLD = 0.5  # Minimum average probability to include in merge

    print("Running SAM3 inference test with image_paths...")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Text prompt: '{TEXT_PROMPT}'")
    print(f"Score threshold: {SCORE_THRESHOLD}")
    print(f"Max num objects: {MAX_NUM_OBJECTS}")
    print(f"Merge threshold: {MERGE_THRESHOLD}")
    print()

    # Get all image paths and select a subset for testing
    all_image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    print(f"Found {len(all_image_paths)} images in {IMAGE_DIR}")

    # Use every 10th image for faster testing (or use all for full test)
    # test_image_paths = all_image_paths[::10]  # Every 10th image
    test_image_paths = all_image_paths  # All images

    print(f"Using {len(test_image_paths)} images for inference")
    print(f"First image: {os.path.basename(test_image_paths[0])}")
    print(f"Last image: {os.path.basename(test_image_paths[-1])}")
    print()

    # Run inference with image_paths
    video_segments = sam3_video_inference(
        image_paths=test_image_paths,
        text=TEXT_PROMPT,
        score_threshold=SCORE_THRESHOLD,
        max_num_objects=MAX_NUM_OBJECTS,
        merge_threshold=MERGE_THRESHOLD,
    )

    print(f"Output shape: {video_segments.shape}")
    print(f"Output dtype: {video_segments.dtype}")
    print(f"Number of frames: {video_segments.shape[0]}")
    print()

    # Load reference outputs
    if os.path.exists(REFERENCE_PATH):
        print(f"Loading reference from: {REFERENCE_PATH}")
        reference_outputs = torch.load(REFERENCE_PATH, weights_only=False)

        # Convert reference to same format
        # Note: keys are strings in the saved file
        reference_segments = []
        for frame_idx in sorted(reference_outputs.keys(), key=lambda x: int(x)):
            frame_outputs = reference_outputs[frame_idx]
            # frame_outputs is {obj_id: binary_mask} after prepare_masks_for_visualization

            if len(frame_outputs) > 0:
                merged_mask = np.zeros_like(list(frame_outputs.values())[0], dtype=bool)
                for obj_id, mask in frame_outputs.items():
                    merged_mask = np.logical_or(merged_mask, mask > 0)
            else:
                # No objects in this frame, create empty mask
                if len(reference_segments) > 0:
                    merged_mask = np.zeros_like(reference_segments[0], dtype=bool)
                else:
                    # Skip if we don't have a reference size yet
                    continue

            reference_segments.append(merged_mask)

        reference_segments = torch.tensor(np.stack(reference_segments), dtype=torch.bool)

        print(f"Reference shape: {reference_segments.shape}")
        print()

        # Check if shapes match
        if video_segments.shape != reference_segments.shape:
            print(f"⚠ Warning: Shape mismatch!")
            print(f"  Output shape:    {video_segments.shape}")
            print(f"  Reference shape: {reference_segments.shape}")
            print(f"  Resizing output to match reference for comparison...")
            print()

            # Resize video_segments to match reference
            import torch.nn.functional as F
            video_segments_resized = F.interpolate(
                video_segments.unsqueeze(1).float(),  # [T, 1, H, W]
                size=(reference_segments.shape[1], reference_segments.shape[2]),
                mode='nearest'
            ).squeeze(1).bool()  # [T, H, W]
        else:
            video_segments_resized = video_segments

        # Calculate IoU per frame and average
        ious = []
        num_frames = min(len(video_segments_resized), len(reference_segments))

        for i in range(num_frames):
            intersection = (video_segments_resized[i] & reference_segments[i]).sum().item()
            union = (video_segments_resized[i] | reference_segments[i]).sum().item()

            if union == 0:
                iou = 1.0  # Both empty
            else:
                iou = intersection / union

            ious.append(iou)

        mean_iou = np.mean(ious)
        min_iou = np.min(ious)
        max_iou = np.max(ious)

        print(f"IoU Statistics:")
        print(f"  Mean IoU: {mean_iou:.4f}")
        print(f"  Min IoU:  {min_iou:.4f}")
        print(f"  Max IoU:  {max_iou:.4f}")
        print()

        if mean_iou >= IOU_THRESHOLD:
            print(f"✓ TEST PASSED: Mean IoU ({mean_iou:.4f}) >= threshold ({IOU_THRESHOLD})")
        else:
            print(f"✗ TEST FAILED: Mean IoU ({mean_iou:.4f}) < threshold ({IOU_THRESHOLD})")
    else:
        print(f"Warning: Reference file not found at {REFERENCE_PATH}")
        print("Skipping IoU comparison.")
