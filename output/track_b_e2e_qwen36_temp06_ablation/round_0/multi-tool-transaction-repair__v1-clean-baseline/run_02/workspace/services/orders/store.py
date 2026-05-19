ORDERS = {}


def reserve_order(order_id, amount):
    ORDERS[order_id] = {"amount": amount, "status": "reserved"}


def mark_paid(order_id):
    ORDERS[order_id]["status"] = "paid"


def cancel_order(order_id):
    if order_id in ORDERS:
        ORDERS[order_id]["status"] = "cancelled"


def reset():
    ORDERS.clear()
