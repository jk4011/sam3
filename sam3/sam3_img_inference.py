import torch
from PIL import Image
import torch.nn.functional as F
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from jhutil import cache_output
import builtins


class Sam3State:
    """Singleton to persist model state across module reloads (even with autoreload)."""

    def __new__(cls):
        # Store in builtins to survive module reloads
        if not hasattr(builtins, '_sam3_state_instance'):
            instance = super().__new__(cls)
            instance.processor = None
            instance.recent_image_path = None
            instance.inference_state = None
            builtins._sam3_state_instance = instance
        return builtins._sam3_state_instance


_state = Sam3State()


def init_predictor(confidence_threshold=0.):
    global _state
    model = build_sam3_image_model()
    _state.processor = Sam3Processor(model, confidence_threshold=confidence_threshold)


@cache_output(func_name="sam3_img_inference", override=False)
def sam3_img_inference(image_path, text, confidence_threshold=0.0, merge_threshold=0.8, mask_size=None):
    global _state

    if _state.processor is None or _state.processor.confidence_threshold != confidence_threshold:
        init_predictor(confidence_threshold)

    if _state.recent_image_path != image_path:
        image = Image.open(image_path)
        _state.inference_state = _state.processor.set_image(image)
        _state.recent_image_path = image_path

    _state.inference_state = _state.processor.set_text_prompt(state=_state.inference_state, prompt=text)
    scores = _state.inference_state['scores']
    masks = _state.inference_state['masks']

    # Get best mask
    mask = masks[torch.argmax(scores)]
    if (scores > merge_threshold).sum() > 1:
        mask = torch.sum(masks[scores > merge_threshold], dim=0) > 0
    score = torch.max(scores)

    # Resize to match images shape
    if mask_size is not None:
        mask = F.interpolate(
            mask.float().unsqueeze(0),
            size=mask_size,
            mode="bilinear",
            align_corners=False
        )[0]

    return mask, score