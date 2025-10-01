from pydantic import BaseModel, HttpUrl
from datetime import datetime


class Paper(BaseModel):
    id: str
    title: str
    summary: str
    published: datetime
    updated: datetime | None = None
    authors: list[str]
    categories: list[str]
    link_pdf: HttpUrl | None = None