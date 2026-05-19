from services.billing import ledger
from services.notifications import queue
from services.orders import store


def checkout(order_id, amount, *, fail_billing=False, retry=False):
    store.reserve_order(order_id, amount)
    try:
        ledger.charge(order_id, amount, fail=fail_billing)
    except ledger.BillingError:
        store.cancel_order(order_id)
        ledger.refund(order_id)
        if retry:
            queue.emit(order_id, "payment_failed")
        raise
    store.mark_paid(order_id)
    queue.emit(order_id, "order_paid")
    return store.ORDERS[order_id]
