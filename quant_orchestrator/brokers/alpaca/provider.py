from __future__ import annotations

from typing import Any

from quant_orchestrator.platform.contracts import ProviderManifest


class AlpacaBroker:
    name = "alpaca"

    def __init__(self, client: Any | None = None) -> None:
        self.client = client

    def get_account(self, **kwargs: Any) -> Any:
        if self.client is None:
            raise ValueError("AlpacaBroker requires a client")
        return self.client.get_account(**kwargs)

    def get_positions(self, **kwargs: Any) -> Any:
        if self.client is None:
            raise ValueError("AlpacaBroker requires a client")
        return self.client.get_positions(**kwargs)

    def submit_order(self, order: Any, **kwargs: Any) -> Any:
        if self.client is None:
            raise ValueError("AlpacaBroker requires a client")
        return self.client.submit_order(order, **kwargs)


alpaca_provider = ProviderManifest(
    name="alpaca",
    category="broker",
    display_name="Alpaca",
    description="Broker adapter shell for Alpaca trading clients.",
    website="https://alpaca.markets",
    capabilities=("account", "positions", "orders", "paper"),
    adapters={"default": AlpacaBroker},
)
