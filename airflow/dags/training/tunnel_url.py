"""Resolve the current public URL of a Cloudflare Quick Tunnel."""
import re

TUNNEL_URL_PATTERN = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def get_quick_tunnel_url(log_path: str) -> str:
    """Parse a cloudflared Quick Tunnel logfile and return the current URL.

    Quick Tunnel URLs are random and change on every container restart, so
    this must be called at runtime instead of caching the result.
    """
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
    matches = TUNNEL_URL_PATTERN.findall(content)
    if not matches:
        raise RuntimeError(f"No trycloudflare.com URL found in {log_path}")
    return matches[-1]
