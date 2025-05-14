from pydantic import BaseModel

class CodeSnippetResponse(BaseModel):
    language: str
    snippet: str