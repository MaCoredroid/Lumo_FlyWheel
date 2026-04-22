from __future__ import annotations


def build_report(include_watchlist: bool = False) -> dict:
    report = {
        "generated_at": "2026-04-20T15:30:00Z",
        "summary": {
            "total_symbols": 4,
            "gainers": ["NVDA", "MSFT"],
            "laggards": ["TSLA"],
        },
        "alerts": [
            {"symbol": "NVDA", "action": "buy", "reason": "relative strength breakout"},
            {"symbol": "TSLA", "action": "trim", "reason": "failed gap-up continuation"},
        ],
    }
    if include_watchlist:
        report["follow_up"] = {
            "watchlist": ["AAPL", "AMZN"],
            "note": "Review liquidity names before the close.",
        }
    return report
