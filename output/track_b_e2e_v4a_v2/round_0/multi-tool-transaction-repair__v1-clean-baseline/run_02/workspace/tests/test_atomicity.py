import pytest

from services.billing import ledger
from services.notifications import queue
from services.orders import store
from services.transactions.workflow import checkout


def test_failed_billing_does_not_mark_order_paid():
    with pytest.raises(ledger.BillingError):
        checkout("ord-2", 25, fail_billing=True)

    assert store.ORDERS["ord-2"]["status"] == "cancelled"
    assert not any(row.get("type") == "charge" for row in ledger.LEDGER)


def test_retry_failure_does_not_emit_duplicate_side_effects():
    with pytest.raises(ledger.BillingError):
        checkout("ord-3", 25, fail_billing=True, retry=True)

    failures = [row for row in queue.QUEUE if row["event"] == "payment_failed"]
    assert len(failures) <= 1
