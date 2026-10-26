"""Self-contained asyncio load generator for the API (httpx only).

    uv run python scripts/loadtest.py --scenario mix --concurrency 50 --duration 30

Writes reports/loadtest_<scenario>.json (or --out). A side probe hits /health every 250 ms on its
own client so event-loop / threadpool starvation shows up as probe latency, not as load-worker
latency.
"""

import argparse
import asyncio
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import httpx

FEATURE_KEYS = (
    "origin_port", "destination_port", "cargo_type", "container_count",
    "weight_tons", "planned_departure", "planned_arrival",
)  # fmt: skip


class Data:
    """Real ids/routes/dates fetched once from the API."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.routes: list[tuple[str, str]] = []
        self.origins: list[str] = []
        self.total = 0
        self.date_min = self.date_max = ""


async def load_data(c: httpx.AsyncClient) -> Data:
    d = Data()
    r = await c.get("/shipments", params={"limit": 500, "offset": 0})
    r.raise_for_status()
    d.total = r.json()["total"]
    d.rows = r.json()["items"]
    for off in range(500, d.total, 500):
        d.rows += (await c.get("/shipments", params={"limit": 500, "offset": off})).json()["items"]
    r = await c.get("/routes/stats", params={"limit": 100, "min_completed": 0})
    d.routes = [(x["origin_port"], x["destination_port"]) for x in r.json()]
    d.origins = sorted({o for o, _ in d.routes})
    dates = sorted(x["booking_date"] for x in d.rows)
    d.date_min, d.date_max = dates[0][:10], dates[-1][:10]
    return d


# Each scenario op returns (label, method, url, kwargs). Labels group stats per endpoint shape.
def op_health(d, rnd):
    return "GET /health", "GET", "/health", {}


def op_shipment(d, rnd):
    return "GET /shipments/{id}", "GET", f"/shipments/{rnd.choice(d.rows)['shipment_id']}", {}


def op_ship_origin(d, rnd):
    p = {"origin": rnd.choice(d.origins), "limit": 50}
    return "GET /shipments?origin&limit=50", "GET", "/shipments", {"params": p}


def op_route(d, rnd):
    o, de = rnd.choice(d.routes)
    return "GET /routes/{o}/{d}/stats", "GET", f"/routes/{o}/{de}/stats", {}


def op_ports(d, rnd):
    return "GET /ports", "GET", "/ports", {}


def op_pred_id(d, rnd):
    body = {"shipment_id": rnd.choice(d.rows)["shipment_id"]}
    return "POST /predict-delay (id)", "POST", "/predict-delay", {"json": body}


def op_pred_feat(d, rnd):
    row = rnd.choice(d.rows)
    body = {"features": {k: row[k] for k in FEATURE_KEYS}}
    return "POST /predict-delay (features)", "POST", "/predict-delay", {"json": body}


def op_route_stats(d, rnd):
    p = {"limit": 100, "min_completed": 1}
    return "GET /routes/stats?limit=100", "GET", "/routes/stats", {"params": p}


def op_big_page(d, rnd):
    p = {"limit": 500, "offset": rnd.randrange(0, max(1, d.total - 500))}
    return "GET /shipments?limit=500&offset=N", "GET", "/shipments", {"params": p}


def op_wide_window(d, rnd):
    # window covers most of the data range; limit 500 forces a large sorted scan + serialisation
    start = rnd.randrange(0, 20)
    p = {"booked_from": f"{d.date_min}T00:00:00", "booked_to": f"{d.date_max}T23:59:59",
         "limit": 500, "offset": start * 100}  # fmt: skip
    return "GET /shipments wide window", "GET", "/shipments", {"params": p}


READ = [(1, op_health), (3, op_shipment), (2, op_ship_origin), (2, op_route), (1, op_ports)]
PREDICT = [(1, op_pred_id), (1, op_pred_feat)]
HEAVY = [(2, op_route_stats), (3, op_big_page), (2, op_wide_window)]
MIX = (
    [(w * 6, f) for w, f in READ]
    + [(w * 5, f) for w, f in PREDICT]
    + [(w * 3, f) for w, f in HEAVY]
)  # ~60% read / 25% predict / 15% heavy
SCENARIOS = {"read": READ, "predict": PREDICT, "heavy": HEAVY, "mix": MIX}


class Stats:
    def __init__(self) -> None:
        self.lat: dict[str, list[float]] = defaultdict(list)
        self.http_err: dict[str, int] = defaultdict(int)
        self.exc_err: dict[str, int] = defaultdict(int)
        self.status: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.exc_kinds: dict[str, int] = defaultdict(int)

    def pct(self, label: str, q: float) -> float:
        xs = sorted(self.lat[label])
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


async def worker(c, d, ops, stats, stop_at, rnd):
    weights = [w for w, _ in ops]
    fns = [f for _, f in ops]
    while time.perf_counter() < stop_at:
        label, method, url, kw = rnd.choices(fns, weights)[0](d, rnd)
        t = time.perf_counter()
        try:
            r = await c.request(method, url, **kw)
            ms = (time.perf_counter() - t) * 1000
            stats.lat[label].append(ms)
            stats.status[label][r.status_code] += 1
            if not 200 <= r.status_code < 300:
                stats.http_err[label] += 1
        except Exception as e:  # noqa: BLE001 - every failure is a data point
            stats.lat[label].append((time.perf_counter() - t) * 1000)
            stats.exc_err[label] += 1
            stats.exc_kinds[type(e).__name__] += 1


async def health_probe(base_url, stop_at, out):
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as c:
        while time.perf_counter() < stop_at:
            t = time.perf_counter()
            try:
                r = await c.get("/health")
                out.append(((time.perf_counter() - t) * 1000, r.status_code))
            except Exception:  # noqa: BLE001
                out.append(((time.perf_counter() - t) * 1000, 0))
            await asyncio.sleep(0.25)


def summarize(stats: Stats, wall: float) -> list[dict]:
    rows = []
    for label in sorted(stats.lat):
        n = len(stats.lat[label])
        rows.append({
            "endpoint": label, "requests": n,
            "http_errors": stats.http_err[label], "exceptions": stats.exc_err[label],
            "status_counts": dict(stats.status[label]),
            "p50_ms": round(stats.pct(label, 0.50), 1), "p95_ms": round(stats.pct(label, 0.95), 1),
            "p99_ms": round(stats.pct(label, 0.99), 1), "max_ms": round(max(stats.lat[label]), 1),
            "rps": round(n / wall, 1),
        })  # fmt: skip
    return rows


def probe_summary(probe: list[tuple[float, int]]) -> dict:
    lat = sorted(p[0] for p in probe)
    if not lat:
        return {}
    return {
        "samples": len(lat), "non_200": sum(1 for _, s in probe if s != 200),
        "p50_ms": round(lat[len(lat) // 2], 1), "p95_ms": round(lat[int(len(lat) * 0.95)], 1),
        "max_ms": round(lat[-1], 1),
    }  # fmt: skip


def print_table(rows: list[dict], total: dict, probe: dict) -> None:
    cols = [("endpoint", 38)] + [(c, 8) for c in ("reqs", "http_err", "exc", "p50", "p95")]
    cols += [(c, 8) for c in ("p99", "max", "rps")]
    hdr = f"{'endpoint':38}" + "".join(f"{c:>{w}}" for c, w in cols[1:])
    print(hdr + "\n" + "-" * len(hdr))
    for r in [*rows, total]:
        print(
            f"{r['endpoint']:38}{r['requests']:>8}{r['http_errors']:>9}{r['exceptions']:>5}"
            f"{r['p50_ms']:>8}{r['p95_ms']:>8}{r['p99_ms']:>8}{r['max_ms']:>8}{r['rps']:>8}"
        )
    print(f"health probe (separate client, 4/s): {probe}")


async def run(args) -> dict:
    limits = httpx.Limits(
        max_connections=args.concurrency + 10, max_keepalive_connections=args.concurrency + 10
    )
    async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout, limits=limits) as c:
        d = await load_data(c)
        print(f"data: {len(d.rows)} shipments, {len(d.routes)} routes, {d.date_min}..{d.date_max}")
        stats, probe = Stats(), []
        ops = SCENARIOS[args.scenario]
        t0 = time.perf_counter()
        stop_at = t0 + args.duration
        rnd = random.Random(args.seed)
        probe_task = asyncio.create_task(health_probe(args.base_url, stop_at, probe))
        await asyncio.gather(
            *(
                worker(c, d, ops, stats, stop_at, random.Random(rnd.random()))
                for _ in range(args.concurrency)
            )
        )
        wall = time.perf_counter() - t0
        await probe_task
        rows = summarize(stats, wall)
        allv = sorted(x for v in stats.lat.values() for x in v)
        total = {
            "endpoint": "TOTAL",
            "requests": len(allv),
            "http_errors": sum(stats.http_err.values()),
            "exceptions": sum(stats.exc_err.values()),
            "p50_ms": round(allv[len(allv) // 2], 1),
            "p95_ms": round(allv[int(len(allv) * 0.95)], 1),
            "p99_ms": round(allv[int(len(allv) * 0.99)], 1),
            "max_ms": round(allv[-1], 1),
            "rps": round(len(allv) / wall, 1),
        }
        ps = probe_summary(probe)
        print_table(rows, total, ps)
        if stats.exc_kinds:
            print("exception kinds:", dict(stats.exc_kinds))
        # recovery / post-check
        r = await c.get("/health")
        print(f"post-run GET /health -> {r.status_code}")
        rec = []
        if args.recovery:
            end = time.perf_counter() + args.recovery
            win = 0
            while time.perf_counter() < end:
                s2, _p2 = Stats(), []
                await asyncio.gather(
                    *(
                        worker(c, d, READ, s2, time.perf_counter() + 2, random.Random(i))
                        for i in range(5)
                    )
                )
                allr = sorted(x for v in s2.lat.values() for x in v)
                rec.append(
                    {
                        "t_after_end_s": win * 2,
                        "p50_ms": round(allr[len(allr) // 2], 1),
                        "p95_ms": round(allr[int(len(allr) * 0.95)], 1),
                        "n": len(allr),
                        "errors": sum(s2.http_err.values()) + sum(s2.exc_err.values()),
                    }
                )
                win += 1
            print("recovery (read @5, 2 s windows):", json.dumps(rec))
        return {
            "scenario": args.scenario, "concurrency": args.concurrency, "duration_s": args.duration,
            "seed": args.seed, "wall_s": round(wall, 2), "total": total, "endpoints": rows,
            "health_probe": ps, "exception_kinds": dict(stats.exc_kinds),
            "post_health_status": r.status_code, "recovery": rec,
        }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="mix")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument(
        "--recovery", type=float, default=0, help="seconds of post-run light-load sampling"
    )
    ap.add_argument("--out", help="JSON path (default reports/loadtest_<scenario>.json)")
    args = ap.parse_args()
    result = asyncio.run(run(args))
    out = Path(args.out or f"reports/loadtest_{args.scenario}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
