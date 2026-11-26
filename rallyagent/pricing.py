"""Dollars from a response's model and usage, for the eval runner and the public API's budget.

$ per million tokens (input, output), first-party API; cache writes cost 1.25x input, reads 0.1x.
Claude Haiku 5.5's rate is for prompts up to 100K tokens, which every agent call here is.
"""
PRICES = {"claude-opus-5-5": (4.00, 20.00), "claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}


def cost_usd(usage_by_model):
    total = 0.0
    for model, u in (usage_by_model or {}).items():
        key = next((k for k in PRICES if model.startswith(k)), None)
        if key is None:
            raise ValueError(f"no price for model {model!r}; add it to rallyagent/pricing.py")
        pin, pout = PRICES[key]
        total += (u.get("input_tokens", 0) * pin + u.get("output_tokens", 0) * pout
                  + u.get("cache_creation_input_tokens", 0) * pin * 1.25
                  + u.get("cache_read_input_tokens", 0) * pin * 0.1) / 1e6
    return round(total, 6)
