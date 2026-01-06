import torch
import numpy as np
import os
import tempfile
import shutil
from sam3.model_builder import build_sam3_video_predictor

predictor = None


def init_predictor(gpus_to_use=None):
    """Initialize the SAM3 video predictor.

    Args:
        gpus_to_use: List of GPU IDs to use. Defaults to current device.
    """
    global predictor
    if gpus_to_use is None:
        gpus_to_use = [torch.cuda.current_device()] if torch.cuda.is_available() else [0]
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)


def _sam3_inference(video_path, text, image_paths=None, gpus_to_use=None):
    """Internal function for SAM3 video inference.

    Args:
        video_path: str (MP4 file path or image directory). Ignored if image_paths is provided.
        text: str - text prompt for segmentation
        image_paths: list of image file paths (optional). If provided, creates temp dir with symlinks.
        gpus_to_use: list of GPU IDs

    Returns:
        dict mapping frame_idx to outputs dict containing:
            - 'out_obj_ids': numpy array of object IDs
            - 'out_binary_masks': numpy array of binary masks
            - 'out_probs': numpy array of probabilities
            - 'out_boxes_xywh': numpy array of bounding boxes
    """
    global predictor

    if predictor is None:
        init_predictor(gpus_to_use)

    # Create temporary directory with symlinks if image_paths is provided
    temp_dir = None
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

    try:
        # Start session
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=actual_video_path,
            )
        )
        session_id = response["session_id"]

        try:
            # Reset session
            predictor.handle_request(
                request=dict(
                    type="reset_session",
                    session_id=session_id,
                )
            )

            # Add text prompt on frame 0
            predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=text,
                )
            )

            # Propagate through video
            outputs_per_frame = {}
            for response in predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                )
            ):
                outputs_per_frame[response["frame_index"]] = response["outputs"]

            return outputs_per_frame

        finally:
            # Always close session to free resources
            predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )
    finally:
        # Clean up temporary directory if created
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def sam3_inference(video_path=None, text=None, image_paths=None, gpus_to_use=None):
    """SAM3 video inference with simplified output.

    Args:
        video_path: str (MP4 file path or image directory). Either this or image_paths must be provided.
        text: str - text prompt for segmentation
        image_paths: list of image file paths (optional). If provided, creates temp dir with symlinks.
                     Takes precedence over video_path.
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

    # Convert single GPU ID to list
    if isinstance(gpus_to_use, int):
        gpus_to_use = [gpus_to_use]

    outputs_per_frame = _sam3_inference(
        video_path=video_path,
        text=text,
        image_paths=image_paths,
        gpus_to_use=gpus_to_use
    )

    # Convert to binary masks and merge all objects
    video_segments = []
    for frame_idx in sorted(outputs_per_frame.keys()):
        outputs = outputs_per_frame[frame_idx]

        # outputs contains: out_obj_ids, out_binary_masks, out_probs, out_boxes_xywh
        binary_masks = outputs["out_binary_masks"]  # shape: [num_objects, H, W]

        if len(binary_masks) > 0:
            # Merge all object masks with logical OR
            merged_mask = np.zeros_like(binary_masks[0], dtype=bool)
            for mask in binary_masks:
                merged_mask = np.logical_or(merged_mask, mask > 0)
        else:
            # No objects detected, create empty mask
            # Get size from first frame if available
            if len(video_segments) > 0:
                merged_mask = np.zeros_like(video_segments[0], dtype=bool)
            else:
                # Default size - will be determined from first valid frame
                merged_mask = None

        video_segments.append(merged_mask)

    # Filter out None values and stack into single tensor
    video_segments = [seg for seg in video_segments if seg is not None]
    if len(video_segments) == 0:
        raise ValueError("No valid masks found in video")

    video_segments = torch.tensor(np.stack(video_segments), dtype=torch.bool)

    return video_segments


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
    print("Running SAM3 inference test with image_paths...")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Text prompt: '{TEXT_PROMPT}'")
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
    video_segments = sam3_inference(
        image_paths=test_image_paths,
        text=TEXT_PROMPT,
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
