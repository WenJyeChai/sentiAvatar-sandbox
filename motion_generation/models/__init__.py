try:
    from .audio_motion_model import AudioMotionTransformer, AudioMotionConfig
except ModuleNotFoundError:
    AudioMotionTransformer = None
    AudioMotionConfig = None
