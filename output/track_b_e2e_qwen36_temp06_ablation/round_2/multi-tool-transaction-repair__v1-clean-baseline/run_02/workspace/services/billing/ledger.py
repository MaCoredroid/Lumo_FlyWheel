LEDGER = []


class BillingError(RuntimeError):
    pass


def charge(order_id, amount, *, fail=False):
    if fail:
        raise BillingError("billing gateway rejected charge")
    LEDGER.append({"order_id": order_id, "amount": amount, "type": "charge"})


def refund(order_id):
    LEDGER.append({"order_id": order_id, "type": "refund"})


def reset():
    LEDGER.clear()
