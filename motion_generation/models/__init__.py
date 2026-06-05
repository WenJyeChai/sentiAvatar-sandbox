try:
    from .audio_motion_model import AudioMotionTransformer, AudioMotionConfig
except ModuleNotFoundError:
    AudioMotionTransformer = None
    AudioMotionConfig = None

try:
    from .vllm_infill_model import (
        MotionInfillCausalLM,
        MotionInfillCollator,
        MotionInfillSFTDataset,
        load_tokenizer_for_infill,
        maybe_enable_lora,
    )
except ModuleNotFoundError:
    MotionInfillCausalLM = None
    MotionInfillCollator = None
    MotionInfillSFTDataset = None
    load_tokenizer_for_infill = None
    maybe_enable_lora = None
