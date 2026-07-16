from __future__ import annotations

from chargingpilot.simulator.models import ChargingHistoryRecord


class ChargingHistoryLog:
    def __init__(self) -> None:
        self._records: list[ChargingHistoryRecord] = []

    def append(self, record: ChargingHistoryRecord) -> None:
        self._records.append(record)

    def records(self) -> list[ChargingHistoryRecord]:
        return list(self._records)
