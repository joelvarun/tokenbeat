"""Loopback SSE harness: real sockets, real sleeps, real arrival timestamps.

Nothing here is a model of timing. A server thread writes SSE frames into a
real TCP connection on a schedule of absolute deadlines, and the client thread
timestamps frames as its own `recv` calls return. Everything the estimators then
see is measured: the send syscall cost, the scheduler's wakeup error, and the
coalescing that puts several frames into one read and so records a gap of zero.

Absolute deadlines are what makes head-of-line blocking appear. A frame that
misses its deadline does not push the frames behind it, so the gap after a long
delay collapses, exactly as in-order delivery behaves on a real path. That is
the artefact the estimators have to survive.
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np

from .simulate import StackConfig, draw_burst_lengths

SSE_HEADERS = (
    b"HTTP/1.1 200 OK\r\n"
    b"Content-Type: text/event-stream\r\n"
    b"Cache-Control: no-cache\r\n"
    b"Connection: close\r\n\r\n"
)
SPIN_MARGIN_S = 1.2e-3


def _spin_until(deadline: float) -> float:
    """Sleep most of the way, then busy wait, so deadline error stays microseconds."""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return time.perf_counter()
        if remaining > SPIN_MARGIN_S:
            time.sleep(remaining - SPIN_MARGIN_S)


def build_schedule(cfg: StackConfig, n_tokens: int, rng: np.random.Generator) -> dict:
    """Absolute send offsets for every frame, plus the burst structure behind them."""
    lens = draw_burst_lengths(cfg.alpha, cfg.gamma, n_tokens, rng)
    offsets, burst_id, position = [], [], []
    clock = 0.0
    log_step = float(np.log(cfg.step_ms * 1e-3))
    for bi, k in enumerate(lens):
        if rng.random() < cfg.jump_prob:
            log_step += float(rng.normal(0.0, cfg.jump_log_sigma))
        clock += float(np.exp(log_step + rng.normal(0.0, cfg.step_log_sigma)))
        offsets.append(clock)
        burst_id.append(bi)
        position.append(0)
        for pos in range(1, int(k)):
            clock += float(np.exp(cfg.flush_log_mean + rng.normal(0.0, cfg.flush_log_sigma)))
            offsets.append(clock)
            burst_id.append(bi)
            position.append(pos)

    offsets = np.array(offsets[:n_tokens])
    delay = cfg.jitter_scale * np.exp(
        cfg.net_log_mean + rng.normal(0.0, cfg.net_log_sigma, size=offsets.size)
    )
    hit = rng.random(offsets.size) < cfg.contam_prob
    if hit.any():
        contam_median = cfg.contam_median_steps * cfg.step_ms * 1e-3
        delay[hit] = contam_median * np.exp(
            rng.normal(0.0, cfg.contam_log_sigma, size=int(hit.sum()))
        )
    return {
        "targets": offsets + delay,
        "burst_id": np.array(burst_id[:n_tokens]),
        "position": np.array(position[:n_tokens]),
        "contaminated": hit,
    }


def _serve(listener: socket.socket, targets: np.ndarray) -> None:
    conn, _ = listener.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    buf = b""
    while b"\r\n\r\n" not in buf:
        part = conn.recv(4096)
        if not part:
            conn.close()
            return
        buf += part
    conn.sendall(SSE_HEADERS)
    base = time.perf_counter()
    try:
        for i, offset in enumerate(targets):
            _spin_until(base + float(offset))
            conn.sendall(b'data: {"i":%d,"delta":"tok"}\n\n' % i)
    except OSError:
        pass
    finally:
        try:
            conn.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        conn.close()


def run_stream(cfg: StackConfig, n_tokens: int, rng: np.random.Generator) -> dict:
    """Stream n_tokens over a real loopback connection and time their arrival."""
    plan = build_schedule(cfg, n_tokens, rng)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    thread = threading.Thread(target=_serve, args=(listener, plan["targets"]), daemon=True)
    thread.start()

    arrivals: list[float] = []
    client = socket.create_connection(("127.0.0.1", port), timeout=30.0)
    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    client.sendall(
        b"GET /v1/messages HTTP/1.1\r\nHost: 127.0.0.1\r\nAccept: text/event-stream\r\n\r\n"
    )
    buf = b""
    try:
        while b"\r\n\r\n" not in buf:
            part = client.recv(65536)
            if not part:
                break
            buf += part
        stamp = time.perf_counter()
        buf = buf.split(b"\r\n\r\n", 1)[1] if b"\r\n\r\n" in buf else b""
        while b"\n\n" in buf:
            _, buf = buf.split(b"\n\n", 1)
            arrivals.append(stamp)
        while True:
            part = client.recv(65536)
            stamp = time.perf_counter()
            if not part:
                break
            buf += part
            while b"\n\n" in buf:
                _, buf = buf.split(b"\n\n", 1)
                arrivals.append(stamp)
    finally:
        client.close()
        thread.join(timeout=10.0)
        listener.close()

    got = len(arrivals)
    arr = np.array(arrivals[:got])
    return {
        "arrivals": arr - arr[0] if got else arr,
        "burst_id": plan["burst_id"][:got],
        "position": plan["position"][:got],
        "contaminated": plan["contaminated"][:got],
        "truth": {"alpha": cfg.alpha, "gamma": cfg.gamma, "step_ms": cfg.step_ms},
    }
