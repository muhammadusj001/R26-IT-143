"""Persistence abstraction placeholder for PostgreSQL and future storage backends."""


class StorageService:
    def save_event(self) -> None:
        raise NotImplementedError("Storage integration will be implemented later.")
