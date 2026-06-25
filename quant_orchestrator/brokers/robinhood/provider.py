from __future__ import annotations

from typing import Any

from quant_orchestrator.platform.contracts import ProviderManifest


class RobinhoodBroker:
    name = "robinhood"

    def __init__(self, client: Any | None = None) -> None:
        self.client = client

    def get_account(self, **kwargs: Any) -> Any:
        if self.client is None:
            raise ValueError("RobinhoodBroker requires a client")
        return self.client.get_account(**kwargs)

    def get_positions(self, **kwargs: Any) -> Any:
        if self.client is None:
            raise ValueError("RobinhoodBroker requires a client")
        return self.client.get_positions(**kwargs)

    def submit_order(self, order: Any, **kwargs: Any) -> Any:
        if self.client is None:
            raise ValueError("RobinhoodBroker requires a client")
        return self.client.submit_order(order, **kwargs)


robinhood_provider = ProviderManifest(
    name="robinhood",
    category="broker",
    display_name="Robinhood",
    description="Broker adapter shell for Robinhood trading clients.",
    website="https://robinhood.com",
    capabilities=("account", "positions", "orders", "options"),
    adapters={"default": RobinhoodBroker},
)
