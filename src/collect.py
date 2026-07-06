"""
# collect.py
# ==========
Steam data collection using SteamSpy API (no key required).

SteamSpy (steamspy.com/api.php) is a third-party analytics service that
aggregates Steam data. It provides owner estimates, playtime, tags, pricing,
and review counts — all in one place, without requiring a Steam API key.

Primary endpoint:
  GET https://steamspy.com/api.php?request=all&page=N
  Returns 1000 games per page with: owners, playtime, tags, price, reviews.

Enrichment (optional, best-effort):
  GET https://store.steampowered.com/api/appdetails?appids=ID
  Used to fetch game descriptions for NLP sentiment analysis.
"""

import time
import logging
import requests
import pandas as pd
from pathlib import Path
from config import (
    N_GAMES, REQUEST_DELAY, MAX_RETRIES, RETRY_DELAY,
    RAW_DATA_PATH, DATA_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# Helpers


def _get(url: str, params: dict = None) -> dict | None:
    """GET with retry logic and exponential back-off."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            log.warning("HTTP %s on attempt %d -- %s", resp.status_code, attempt, url)
        except requests.RequestException as exc:
            log.warning("Request error attempt %d: %s", attempt, exc)
        time.sleep(RETRY_DELAY * attempt)
    return None


def _parse_owners(owner_str: str) -> int:
    """
    Convert SteamSpy owner range string to a single integer (midpoint).
    e.g. '10,000,000 .. 20,000,000' -> 15_000_000
    """
    try:
        parts = owner_str.replace(",", "").split("..")
        low  = int(parts[0].strip())
        high = int(parts[1].strip())
        return (low + high) // 2
    except Exception:
        return 0


# SteamSpy API


def get_steamspy_page(page: int = 0) -> dict:
    """Fetch one page (1000 games) from SteamSpy."""
    data = _get("https://steamspy.com/api.php", params={"request": "all", "page": page})
    return data or {}


def get_all_steamspy_games(max_pages: int = 5) -> list[dict]:
    """
    Collect up to max_pages * 1000 games from SteamSpy.
    Returns a flat list of game dicts.
    """
    log.info("Fetching game list from SteamSpy (up to %d pages)...", max_pages)
    all_games = []
    for page in range(max_pages):
        log.info("  Fetching SteamSpy page %d...", page)
        page_data = get_steamspy_page(page)
        if not page_data:
            log.warning("  Empty response on page %d, stopping.", page)
            break
        all_games.extend(page_data.values())
        log.info("  Got %d games so far.", len(all_games))
        time.sleep(1.5)  # SteamSpy rate limit: ~1 req/sec
    log.info("SteamSpy returned %d total games.", len(all_games))
    return all_games


# SteamSpy Details


def get_steamspy_details(app_id: int) -> dict | None:
    """Fetch full details for a single app from SteamSpy."""
    return _get("https://steamspy.com/api.php", params={"request": "appdetails", "appid": app_id})


# Steam Store enrichment (description for NLP)


def get_steam_description(app_id: int) -> str:
    """
    Fetch the short description from Steam Store for NLP.
    Returns empty string on failure (graceful degradation).
    """
    try:
        data = _get(
            "https://store.steampowered.com/api/appdetails",
            params={"appids": app_id, "cc": "us", "l": "en", "filters": "short_description"}
        )
        if data and str(app_id) in data:
            payload = data[str(app_id)]
            if payload.get("success"):
                return payload.get("data", {}).get("short_description", "")
    except Exception:
        pass
    return ""


# Record builder


def _parse_steamspy_record(game: dict, description: str = "") -> dict:
    """
    Convert a SteamSpy game dict into a flat feature record.

    SteamSpy fields:
      appid, name, developer, publisher, score_rank,
      owners (range string), average_forever, average_2weeks,
      median_forever, median_2weeks, price (cents str),
      initialprice, discount, ccu, positive, negative,
      languages, genre, tags (dict of tag->votes)
    """
    owners_mid  = _parse_owners(game.get("owners", "0 .. 0"))
    price_usd   = int(game.get("price", 0) or 0) / 100.0
    positive    = int(game.get("positive", 0) or 0)
    negative    = int(game.get("negative", 0) or 0)
    total_rev   = positive + negative
    pct_pos     = (positive / total_rev * 100) if total_rev > 0 else 0.0

    # Tags: dict like {"Action": 3522, "FPS": 2987}
    tags_dict   = game.get("tags", {}) or {}
    # Join top tags by vote count as a pipe-separated string
    top_tags    = "|".join(
        k for k, _ in sorted(tags_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    )

    genres_raw  = game.get("genre", "") or ""
    # SteamSpy returns genres as comma-separated string
    genres      = "|".join(g.strip() for g in genres_raw.split(",") if g.strip())

    return {
        "app_id":            int(game.get("appid", 0)),
        "name":              game.get("name", ""),
        "developer":         game.get("developer", ""),
        "publisher":         game.get("publisher", ""),
        "price_usd":         price_usd,
        "is_free":           int(price_usd == 0.0),
        "genres":            genres,
        "user_tags":         top_tags,
        "owners_estimate":   owners_mid,
        "avg_playtime_forever": int(game.get("average_forever", 0) or 0),
        "avg_playtime_2w":   int(game.get("average_2weeks", 0) or 0),
        "median_playtime":   int(game.get("median_forever", 0) or 0),
        "ccu":               int(game.get("ccu", 0) or 0),
        "positive_reviews":  positive,
        "negative_reviews":  negative,
        "total_reviews":     total_rev,
        "pct_positive":      round(pct_pos, 2),
        "score_rank":        int(game.get("score_rank", 0) or 0),
        "discount":          int(game.get("discount", 0) or 0),
        "languages":         game.get("languages", ""),
        "short_description": description,
    }


# Main collector


def collect_dataset(n: int = N_GAMES, resume: bool = True, enrich_descriptions: bool = True) -> pd.DataFrame:
    """
    Collect `n` games from SteamSpy, optionally enriching with Steam Store
    descriptions for NLP analysis. Saves checkpoints every 100 games.

    Parameters
    ----------
    n                   : Target number of games.
    resume              : Skip already-collected IDs if raw file exists.
    enrich_descriptions : Fetch descriptions from Steam Store for NLP.
    """
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    raw_path = Path(RAW_DATA_PATH)

    # Resume support
    collected_ids: set[int] = set()
    rows: list[dict] = []
    if resume and raw_path.exists():
        existing      = pd.read_csv(raw_path)
        rows          = existing.to_dict("records")
        collected_ids = set(int(x) for x in existing["app_id"].tolist())
        log.info("Resuming -- %d games already collected.", len(rows))

    # Fetch the full game list from SteamSpy
    # 3 pages = 3000 candidates, more than enough to filter down to n
    needed_pages = max(1, (n // 1000) + 2)
    all_games = get_all_steamspy_games(max_pages=needed_pages)

    # Filter: skip already collected, skip entries with no name
    candidates = [
        g for g in all_games
        if g.get("name", "").strip() and int(g.get("appid", 0)) not in collected_ids
    ]
    log.info("%d new candidates to process.", len(candidates))

    for game in candidates:
        if len(rows) >= n:
            break

        app_id = int(game.get("appid", 0))
        if not app_id:
            continue

        # Fetch full details from SteamSpy (contains genres, tags, languages)
        details = get_steamspy_details(app_id)
        if not details or "name" not in details:
            details = game
        else:
            if "appid" not in details:
                details["appid"] = app_id

        # Optional: fetch description from Steam Store for NLP
        description = ""
        if enrich_descriptions:
            description = get_steam_description(app_id)
            time.sleep(REQUEST_DELAY)
        else:
            time.sleep(0.35)  # Safe delay for SteamSpy rate limit

        record = _parse_steamspy_record(details, description)
        rows.append(record)
        collected_ids.add(app_id)

        log.info("[%d/%d] %s (owners: %s)", len(rows), n,
                 record["name"][:50], f"{record['owners_estimate']:,}")

        # Checkpoint every 100 games
        if len(rows) % 100 == 0:
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            log.info("Checkpoint saved (%d games).", len(rows))

    df = pd.DataFrame(rows)
    df.to_csv(raw_path, index=False)
    log.info("Collection complete. %d games saved to %s", len(df), raw_path)
    return df
