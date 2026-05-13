from services.billing import ledger
from services.notifications import queue
from services.orders import store
from services.transactions.workflow import checkout


def test_checkout_happy_path_marks_paid_and_emits_once():
    result = checkout("ord-1", 25)

    assert result["status"] == "paid"
    assert store.ORDERS["ord-1"]["amount"] == 25
    assert ledger.LEDGER == [{"order_id": "ord-1", "amount": 25, "type": "charge"}]
    assert queue.QUEUE == [{"order_id": "ord-1", "event": "order_paid"}]
