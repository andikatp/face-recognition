from pydantic import BaseModel, Field


class VerifyResponse(BaseModel):
    """Schema for successful verification assessment."""

    status: str = Field(..., description="Decision status: 'success' or 'rejected'")
    message: str = Field(..., description="User-friendly message explaining the decision")
    antispoof_score: float | None = Field(
        None,
        description="Calculated fakeness percentage / score (0.0 = definitely real, 1.0 = definitely spoof)",
    )


class ErrorResponse(BaseModel):
    """Schema for API error messages."""

    status: str = Field("error", description="Error status identifier")
    message: str = Field(..., description="Details regarding the error")
