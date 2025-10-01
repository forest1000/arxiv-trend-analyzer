from __future__ import annotations
from collections import defaultdict
from typing import Callable, Dict, List, Type, Any
from pydantic import BaseModel




class Event(BaseModel):
    """Base Event."""




class FetchCompleted(Event):
    query: str
    count: int

class EventBus:
    def __init__(self) -> None:
        self._subs: Dict[Type[Event], List[Callable[[Event], Any]]] = defaultdict(list)


    def subscribe(self, event_type: Type[Event], handler: Callable[[Event], Any]) -> None:
        self._subs[event_type].append(handler)


    def publish(self, event: Event) -> None:
        for h in list(self._subs[type(event)]):
            h(event)

# アプリ内シングルトン（必要に応じて差し替え可能）
bus = EventBus()