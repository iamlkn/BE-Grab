from pydantic import BaseModel, Field
from typing import Optional

class GenerateReportResponse(BaseModel):
    report_format: str = "markdown"
    report_content: Optional[str] = None
    report_file_path: Optional[str] = None