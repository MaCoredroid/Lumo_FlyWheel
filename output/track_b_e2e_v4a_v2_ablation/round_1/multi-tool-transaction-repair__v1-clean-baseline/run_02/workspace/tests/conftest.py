import pytest

from services.billing import ledger
from services.notifications import queue
from services.orders import store


@pytest.fixture(autouse=True)
def clean_state():
    store.reset()
    ledger.reset()
    queue.reset()
    yield
    store.reset()
    ledger.reset()
    queue.reset()
