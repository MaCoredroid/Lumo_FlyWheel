QUEUE = []


def emit(order_id, event):
    QUEUE.append({"order_id": order_id, "event": event})


def reset():
    QUEUE.clear()
