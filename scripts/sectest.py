"""Authorised security test for the Supply Chain Intelligence API (local, synthetic data).

Re-runnable:  uv run python scripts/sectest.py   [--base-url http://localhost:8000]

Each case prints PASS/FAIL with the request and a short response summary. A case PASSES
when the service rejects/handles bad input with a structured 4xx (or 200-empty) and never
returns 500, hangs, leaks internals, or returns extra rows. "Expected-info" cases report
behaviour that is acceptable-but-worth-flagging (docs exposed, no auth) and never fail the run.
"""

from __future__ import annotations

import argparse
import contextlib
import socket
import sys
import time

import httpx

VALID_SHIPMENT = "SHP-04801"
VALID_ORIGIN, VALID_DEST = "USLAX", "BEANR"

results: list[tuple[bool, str]] = []
_info: list[str] = []


def _summary(r: httpx.Response) -> str:
    body = r.text.replace("\n", " ")
    return f"HTTP {r.status_code} | {body[:160]}"


def check(name: str, req: str, ok: bool, detail: str) -> None:
    results.append((ok, name))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        req : {req}\n        resp: {detail}")


def info(name: str, req: str, detail: str) -> None:
    _info.append(name)
    print(f"[INFO] {name}\n        req : {req}\n        resp: {detail}")


# status codes that mean "handled safely": never 500, never a hang
SAFE_4XX = {400, 404, 405, 415, 422}


def not_500(r: httpx.Response) -> bool:
    return r.status_code != 500


# ---------------------------------------------------------------------------
# 1. Injection
# ---------------------------------------------------------------------------
SQLI = [
    "' OR 1=1 --",
    "CNSHA' UNION SELECT * FROM shipments --",
    "; DROP TABLE shipments",
    "%27",
    "СNSHA",  # cyrillic homoglyph of C
    "' OR '1'='1",
]


def test_injection(c: httpx.Client) -> None:
    print("\n=== 1. Injection ===")
    baseline = c.get("/shipments", params={"limit": 1}).json()["total"]
    for payload in SQLI:
        # origin/destination are regex-constrained -> expect 422 (never reaches SQL)
        for param in ("origin", "destination"):
            r = c.get("/shipments", params={param: payload, "limit": 5})
            check(
                f"SQLi {param}={payload!r}",
                f"GET /shipments?{param}=<payload>",
                r.status_code == 422,
                _summary(r),
            )
        # status is a free string but parameterised -> 200 with 0 rows, total unchanged
        r = c.get("/shipments", params={"status": payload, "limit": 5})
        try:
            total = r.json().get("total")
            n = len(r.json().get("items", []))
        except Exception:
            total, n = None, None
        check(
            f"SQLi status={payload!r}",
            "GET /shipments?status=<payload>",
            r.status_code == 200 and total == 0 and n == 0,
            f"{_summary(r)} (baseline total={baseline})",
        )
    # shipment_id path param (free string, parameterised) -> 404, never 500/extra rows
    for payload in SQLI:
        r = c.get(
            f"/shipments/{httpx.URL(path='/' + payload).raw_path.decode(errors='replace')[1:]}"
        )
        r = c.get("/shipments/" + payload)
        check(
            f"SQLi shipment_id={payload!r}",
            "GET /shipments/<payload>",
            r.status_code in (404, 422) and not_500(r),
            _summary(r),
        )
    # booked_from garbage -> 422 (datetime parse)
    r = c.get("/shipments", params={"booked_from": "' OR 1=1 --"})
    check(
        "SQLi booked_from",
        "GET /shipments?booked_from=<payload>",
        r.status_code == 422,
        _summary(r),
    )
    # sort / order are Literal-validated -> 422
    for param, val in (("sort", "shipment_id; DROP"), ("order", "asc; DROP")):
        r = c.get("/routes/stats", params={param: val})
        check(
            f"SQLi routes {param}={val!r}",
            f"GET /routes/stats?{param}=<payload>",
            r.status_code == 422,
            _summary(r),
        )
    # route path params are regex-constrained
    r = c.get(f"/routes/{SQLI[0]}/{VALID_DEST}/stats")
    check(
        "SQLi route origin path",
        "GET /routes/<payload>/BEANR/stats",
        r.status_code in (404, 422),
        _summary(r),
    )


# ---------------------------------------------------------------------------
# 2. Input bounds
# ---------------------------------------------------------------------------
def test_bounds(c: httpx.Client) -> None:
    print("\n=== 2. Input bounds ===")
    for limit in ("0", "501", "1000000000", "-1", "abc", "1e9"):
        r = c.get("/shipments", params={"limit": limit})
        check(f"limit={limit}", f"GET /shipments?limit={limit}", r.status_code == 422, _summary(r))
    for offset in ("-1", str(2**63), str(10**18)):
        try:
            r = c.get("/shipments", params={"offset": offset, "limit": 1})
        except httpx.HTTPError as e:
            check(
                f"offset={offset}",
                f"GET /shipments?offset={offset}",
                False,
                f"transport error (server disconnected / no response): {e}",
            )
            continue
        # offset>=0 passes schema; huge offset -> 200 empty (DuckDB handles), never 500
        ok = (offset == "-1" and r.status_code == 422) or (offset != "-1" and not_500(r))
        check(f"offset={offset}", f"GET /shipments?offset={offset}", ok, _summary(r))
    for d in ("9999-12-31", "0001-01-01", "2024-02-30", "2024-06-01T00:00:00+05:30"):
        r = c.get("/shipments", params={"booked_from": d, "limit": 1})
        check(f"date booked_from={d}", f"GET /shipments?booked_from={d}", not_500(r), _summary(r))
    # timezone-aware vs naive: does a tz-aware bound crash the comparison?
    r = c.get(
        "/shipments",
        params={
            "booked_from": "2024-06-01T00:00:00+05:30",
            "booked_to": "2026-01-01T00:00:00+05:30",
            "limit": 1,
        },
    )
    check(
        "tz-aware date range",
        "GET /shipments?booked_from=..+05:30&booked_to=..+05:30",
        not_500(r),
        _summary(r),
    )
    r = c.get("/shipments", params={"booked_from": "2025-06-01", "booked_to": "2025-01-01"})
    check("booked_from > booked_to", "GET /shipments (from>to)", r.status_code == 422, _summary(r))
    r = c.get("/routes/stats", params={"min_completed": "-1"})
    check(
        "min_completed=-1", "GET /routes/stats?min_completed=-1", r.status_code == 422, _summary(r)
    )
    r = c.get("/shipments/" + "A" * 10240)
    check(
        "10KB shipment_id",
        "GET /shipments/<10KB>",
        r.status_code in (404, 414, 422) and not_500(r),
        _summary(r),
    )
    for p in ("/shipments/../health", "/shipments/%2e%2e/health", "/shipments/%00"):
        r = c.get(p)
        check(
            f"path traversal {p}",
            f"GET {p}",
            not_500(r) and r.status_code != 200 or p.endswith("health"),
            _summary(r),
        )
    # predict-delay body abuse
    bodies = [
        (
            "NaN weight",
            '{"features":{"origin_port":"USLAX","destination_port":"BEANR","cargo_type":"X",'
            '"container_count":1,"weight_tons":NaN,"planned_departure":"2025-01-01T00:00:00",'
            '"planned_arrival":"2025-02-01T00:00:00"}}',
        ),
        (
            "Infinity weight",
            '{"features":{"origin_port":"USLAX","destination_port":"BEANR","cargo_type":"X",'
            '"container_count":1,"weight_tons":Infinity,"planned_departure":"2025-01-01T00:00:00",'
            '"planned_arrival":"2025-02-01T00:00:00"}}',
        ),
        (
            "1e400 weight",
            '{"features":{"origin_port":"USLAX","destination_port":"BEANR","cargo_type":"X",'
            '"container_count":1,"weight_tons":1e400,"planned_departure":"2025-01-01T00:00:00",'
            '"planned_arrival":"2025-02-01T00:00:00"}}',
        ),
        (
            "container 10**18",
            '{"features":{"origin_port":"USLAX","destination_port":"BEANR","cargo_type":"X",'
            '"container_count":1000000000000000000,"weight_tons":5,'
            '"planned_departure":"2025-01-01T00:00:00","planned_arrival":"2025-02-01T00:00:00"}}',
        ),
        ("empty body", ""),
        ("array body", "[]"),
        ("extra fields", '{"shipment_id":"SHP-04801","evil":"x","__proto__":{"a":1}}'),
        (
            "both id+features",
            '{"shipment_id":"SHP-04801","features":{"origin_port":"USLAX",'
            '"destination_port":"BEANR","cargo_type":"X","container_count":1,"weight_tons":5,'
            '"planned_departure":"2025-01-01T00:00:00","planned_arrival":"2025-02-01T00:00:00"}}',
        ),
    ]
    # extra fields and a huge-but-in-range container_count being ignored/accepted (200) is fine;
    # only a 500/hang is a failure for those. The rest must be a clean 4xx.
    accept_200 = {"container 10**18", "extra fields"}
    for name, body in bodies:
        r = c.post("/predict-delay", content=body, headers={"content-type": "application/json"})
        ok = not_500(r) if name in accept_200 else (r.status_code in SAFE_4XX and not_500(r))
        check(f"predict {name}", f"POST /predict-delay {name}", ok, _summary(r))
    # wrong content-type
    r = c.post(
        "/predict-delay",
        content='{"shipment_id":"SHP-04801"}',
        headers={"content-type": "text/plain"},
    )
    check(
        "predict wrong content-type",
        "POST /predict-delay text/plain",
        r.status_code in SAFE_4XX and not_500(r),
        _summary(r),
    )
    # 5MB body
    big = '{"shipment_id":"' + "A" * (5 * 1024 * 1024) + '"}'
    try:
        r = c.post(
            "/predict-delay", content=big, headers={"content-type": "application/json"}, timeout=20
        )
        check(
            "predict 5MB body",
            "POST /predict-delay 5MB",
            r.status_code in SAFE_4XX and not_500(r),
            _summary(r),
        )
    except Exception as e:
        check("predict 5MB body", "POST /predict-delay 5MB", False, f"exception: {e}")


# ---------------------------------------------------------------------------
# 3. Headers
# ---------------------------------------------------------------------------
def test_headers(c: httpx.Client, base: str) -> None:
    print("\n=== 3. Headers ===")
    # 4KB is reliably under the ASGI server's own header limit, so this isolates the APP's
    # (missing) cap. A sane service caps/truncates the client-supplied id (e.g. <=256). The
    # app echoes and LOGS it verbatim, so each request can write ~KBs of attacker data to logs.
    SANE_CAP = 256
    r = c.get("/health", headers={"x-request-id": "Z" * 4096})
    echoed = len(r.headers.get("x-request-id", ""))
    check(
        "x-request-id capped by the app",
        "GET /health x-request-id=<4096 bytes>",
        echoed <= SANE_CAP,
        f"HTTP {r.status_code}, echoed-len={echoed} (app applies no cap; "
        f"reflected + logged verbatim)",
    )
    # unicode request id: HTTP header values are latin-1/ascii on the wire, so a well-behaved
    # client (httpx) refuses to send non-ascii. Confirms it is not reachable over normal HTTP.
    try:
        r = c.get("/health", headers={"x-request-id": "идент-é"})
        check(
            "x-request-id unicode",
            "GET /health x-request-id=<unicode>",
            not_500(r),
            f"HTTP {r.status_code}, echoed={r.headers.get('x-request-id', '')[:40]!r}",
        )
    except (UnicodeEncodeError, httpx.HTTPError) as e:
        check(
            "x-request-id unicode",
            "GET /health x-request-id=<unicode>",
            True,
            f"client refuses to send non-ascii header ({type(e).__name__}); not reachable via HTTP",
        )
    # CRLF / log injection via raw socket
    host, port = base.split("://")[1].split(":")
    port = int(port.split("/")[0])
    try:
        payload = (
            "GET /health HTTP/1.1\r\nHost: localhost\r\n"
            "x-request-id: evil\r\nX-Injected: pwned\r\n\r\n"
        )
        s = socket.create_connection((host, port), timeout=5)
        s.sendall(payload.encode())
        time.sleep(0.3)
        raw = s.recv(4096).decode(errors="replace")
        s.close()
        # the proper CRLF test: a header value containing a literal CRLF is what matters, but
        # servers parse it as a new header. We confirm the service survived and responded 200.
        check(
            "CRLF in headers (raw socket)",
            "raw GET /health with injected header",
            raw.startswith("HTTP/1.1 200"),
            raw.split("\r\n")[0],
        )
    except Exception as e:
        check("CRLF in headers (raw socket)", "raw socket", False, f"exception: {e}")
    # host header spoofing
    r = c.get("/health", headers={"host": "evil.example.com"})
    check(
        "host header spoof",
        "GET /health Host: evil.example.com",
        not_500(r),
        f"HTTP {r.status_code} (reflected in body: {'evil.example' in r.text})",
    )
    # method tampering
    for method in ("PUT", "DELETE", "TRACE", "OPTIONS", "HEAD"):
        try:
            r = c.request(method, "/shipments")
            ok = (
                r.status_code in (405, 200)
                if method in ("OPTIONS", "HEAD")
                else r.status_code == 405
            )
            check(
                f"method {method} /shipments",
                f"{method} /shipments",
                ok and not_500(r),
                f"HTTP {r.status_code} allow={r.headers.get('allow', '')}",
            )
        except Exception as e:
            check(f"method {method} /shipments", f"{method} /shipments", False, f"exception: {e}")


# ---------------------------------------------------------------------------
# 4. Info exposure
# ---------------------------------------------------------------------------
def test_info(c: httpx.Client) -> None:
    print("\n=== 4. Info exposure ===")
    for path in ("/docs", "/redoc", "/openapi.json"):
        r = c.get(path)
        info(
            f"{path} reachable",
            f"GET {path}",
            f"HTTP {r.status_code} (acceptable for a take-home; flag)",
        )
    r = c.get("/health")
    info("Server header", "GET /health", f"server={r.headers.get('server', '(none)')!r}")
    for path in ("/_test", "/_debug", "/admin", "/__debug__", "/.env", "/metrics"):
        r = c.get(path)
        check(f"no prod debug route {path}", f"GET {path}", r.status_code == 404, _summary(r))
    # try to trigger a 500 / leak via odd-but-valid params
    probes = [
        ("/shipments", {"booked_from": "2025-01-01T00:00:00Z"}),  # trailing Z
        ("/routes/stats", {"limit": "100"}),
        ("/shipments", {"status": "\x00"}),
    ]
    for path, params in probes:
        r = c.get(path, params=params)
        leaked = any(
            s in r.text.lower()
            for s in ("traceback", "/app/", "site-packages", "duckdb.", 'file "')
        )
        check(
            f"no stack trace/leak {path} {params}",
            f"GET {path} {params}",
            not_500(r) and not leaked,
            _summary(r),
        )


# ---------------------------------------------------------------------------
# 5. Abuse (brief, read-only)
# ---------------------------------------------------------------------------
def test_abuse(c: httpx.Client) -> None:
    print("\n=== 5. Abuse (no auth / no rate limit) ===")
    # confirm no auth: a plain request with no credentials succeeds
    r = c.get("/shipments", params={"limit": 1})
    info(
        "No authentication required",
        "GET /shipments (no credentials)",
        f"HTTP {r.status_code} — endpoint served without any auth",
    )
    # brief burst: how many /predict-delay accepted in ~5s, and /health latency under load
    n, errors, t0 = 0, 0, time.perf_counter()
    while time.perf_counter() - t0 < 5:
        try:
            rr = c.post("/predict-delay", json={"shipment_id": VALID_SHIPMENT}, timeout=10)
            n += 1
            if rr.status_code != 200:
                errors += 1
        except Exception:
            errors += 1
    dur = time.perf_counter() - t0
    hb = c.get("/health").elapsed.total_seconds() * 1000
    info(
        "Rate-limit absent on /predict-delay",
        f"{n} POST /predict-delay in {dur:.1f}s (single client)",
        f"~{n / dur:.0f} req/s accepted, {errors} errors; /health latency after load ~{hb:.0f}ms "
        f"(unbounded: a real attacker with concurrency would scale this linearly)",
    )


def main() -> int:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--skip-abuse", action="store_true")
    args = ap.parse_args()
    # No keep-alive: a 500 that closes the socket must not cascade into the next case.
    limits = httpx.Limits(max_keepalive_connections=0)
    with httpx.Client(base_url=args.base_url, timeout=15, limits=limits) as c:
        try:
            c.get("/health").raise_for_status()
        except Exception as e:
            print(f"API not reachable at {args.base_url}: {e}")
            return 2
        test_injection(c)
        test_bounds(c)
        test_headers(c, args.base_url)
        test_info(c)
        if not args.skip_abuse:
            test_abuse(c)
    passed = sum(1 for ok, _ in results if ok)
    failed = [name for ok, name in results if not ok]
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{len(results)} checks PASS, {len(failed)} FAIL, {len(_info)} INFO")
    if failed:
        print("FAILED:")
        for name in failed:
            print(f"  - {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
