from __future__ import annotations

TITLE = "Incident Triage Report"
REPORT_SLUG = "incident-triage"
ACK_SLA_MINUTES = {
    "sev1": 15,
    "sev2": 30,
    "sev3": 60,
}
RECORDS = [
    {
        "service": "billing-api",
        "severity": "sev1",
        "owner": "Jules",
        "minutes_open": 22,
        "acked": False,
    },
    {
        "service": "search-indexer",
        "severity": "sev2",
        "owner": "Ivy",
        "minutes_open": 18,
        "acked": True,
    },
    {
        "service": "checkout-web",
        "severity": "sev2",
        "owner": "Jules",
        "minutes_open": 41,
        "acked": False,
    },
    {
        "service": "fraud-worker",
        "severity": "sev3",
        "owner": "Ava",
        "minutes_open": 12,
        "acked": False,
    },
]
