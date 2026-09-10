from datetime import datetime
from typing import Any

from pydantic import BaseModel


class BatchCreateRequest(BaseModel):
    metadata: dict[str, Any] | None = None
    ignore_roles: list[str] | None = None


class BatchItemPayload(BaseModel):
    # Wire shape of zep-cloud 3.25 BatchAddItem (only the fields MiroFish sends;
    # unknown SDK extras are ignored by pydantic).
    type: str = "graph_episode"
    data: str = ""
    data_type: str = "text"
    graph_id: str = ""
    metadata: dict[str, Any] | None = None
    source_description: str = ""
    name: str | None = None
    created_at: datetime | None = None


class BatchAddItemsRequest(BaseModel):
    items: list[BatchItemPayload] = []
