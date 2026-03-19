class AIPlatformError(Exception):
    """Base exception for the platform."""


class ModelError(AIPlatformError):
    """Base model exception."""


class ModelNotFoundError(ModelError):
    """Requested model is not registered."""


class ModelInferenceError(ModelError):
    """Model failed to generate output."""


class ExecutorError(AIPlatformError):
    """Base executor exception."""


class ExecutorNotFoundError(ExecutorError):
    """Executor not found."""


class PipelineError(AIPlatformError):
    """Base pipeline exception."""


class PipelineNotFoundError(PipelineError):
    """Pipeline not found."""


class PipelineStepError(PipelineError):
    """A pipeline step failed or returned invalid data."""


class KnowledgeError(AIPlatformError):
    """Base knowledge exception."""


class QdrantSearchError(KnowledgeError):
    """Qdrant query failed."""


class RerankerError(KnowledgeError):
    """Reranker failed."""


class ValidationError(AIPlatformError):
    """User input validation error."""
