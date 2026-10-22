"""Safe multilingual email preprocessing interfaces."""

from .pipeline import EmailPreprocessor, preprocess_email
from .schemas import EmailMessage, MetadataFeatures, PreprocessingResult, URLFeatures

__all__ = [
    "EmailMessage",
    "EmailPreprocessor",
    "MetadataFeatures",
    "PreprocessingResult",
    "URLFeatures",
    "preprocess_email",
]
