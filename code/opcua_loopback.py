"""Local-only OPC UA read-compute-write integration exercise.

This intentionally tests protocol behaviour, not plant-side control or PLC
integration.  The server and client share one host and are disposable test
endpoints; results must be reported as a local loopback integration test.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from asyncua import Client, Server, ua


ENDPOINT = "opc.tcp://127.0.0.1:48431/rolling-replay/"
BOUND = 0.55


async def start_server() -> tuple[Server, dict[str, object]]:
    server = Server()
    await server.init()
    server.set_endpoint(ENDPOINT)
    server.set_server_name("Rolling Replay Local Loopback Test Server")
    namespace = await server.register_namespace("urn:rolling-replay:local-loopback")
    objects = server.nodes.objects
    test = await objects.add_object(namespace, "LocalLoopback")
    nodes = {
        "heartbeat": await test.add_variable(namespace, "Heartbeat", 0),
        "measurement": await test.add_variable(namespace, "Measurement", 0.0),
        "model_available": await test.add_variable(namespace, "ModelAvailable", True),
        "u_speed": await test.add_variable(namespace, "CommandSpeed", 0.0),
        "u_gap": await test.add_variable(namespace, "CommandGap", 0.0),
        "u_shape": await test.add_variable(namespace, "CommandShape", 0.0),
        "fallback": await test.add_variable(namespace, "FallbackActive", False),
    }
    for node in nodes.values():
        await node.set_writable()
    await server.start()
    return server, nodes


async def bind_client(client: Client) -> dict[str, object]:
    root = client.nodes.root
    return {
        "heartbeat": await root.get_child(["0:Objects", "2:LocalLoopback", "2:Heartbeat"]),
        "measurement": await root.get_child(["0:Objects", "2:LocalLoopback", "2:Measurement"]),
        "model_available": await root.get_child(["0:Objects", "2:LocalLoopback", "2:ModelAvailable"]),
        "u_speed": await root.get_child(["0:Objects", "2:LocalLoopback", "2:CommandSpeed"]),
        "u_gap": await root.get_child(["0:Objects", "2:LocalLoopback", "2:CommandGap"]),
        "u_shape": await root.get_child(["0:Objects", "2:LocalLoopback", "2:CommandShape"]),
        "fallback": await root.get_child(["0:Objects", "2:LocalLoopback", "2:FallbackActive"]),
    }


async def main_async(cycles: int, out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    server, server_nodes = await start_server()
    client = Client(url=ENDPOINT, timeout=2.0)
    await client.connect()
    client_nodes = await bind_client(client)
    rows: list[dict[str, object]] = []
    fallbacks = {"stale": 0, "bad_status": 0, "model_load_failure": 0}
    reconnects = 0
    last_heartbeat = None
    stale_run = 0
    restart_at = int(cycles * 0.80)
    stale_start, stale_end = int(cycles * 0.20), int(cycles * 0.22)
    bad_start, bad_end = int(cycles * 0.40), int(cycles * 0.42)
    model_start, model_end = int(cycles * 0.60), int(cycles * 0.62)
    try:
        for cycle in range(cycles):
            if cycle == restart_at:
                await client.disconnect()
                await server.stop()
                await asyncio.sleep(0.05)
                await server.start()
                await client.connect()
                client_nodes = await bind_client(client)
                reconnects += 1
                last_heartbeat, stale_run = None, 0

            stale_injected = stale_start <= cycle < stale_end
            bad_injected = bad_start <= cycle < bad_end
            model_failure_injected = model_start <= cycle < model_end
            if not stale_injected:
                await server_nodes["heartbeat"].write_value(cycle + 1)
            if bad_injected:
                bad = ua.DataValue(ua.Variant(float(cycle)), StatusCode=ua.StatusCode(ua.StatusCodes.BadNoCommunication))
                await server_nodes["measurement"].write_value(bad)
            else:
                await server_nodes["measurement"].write_value(float(cycle))
            await server_nodes["model_available"].write_value(not model_failure_injected)

            t0 = time.perf_counter_ns()
            heartbeat = await client_nodes["heartbeat"].read_value()
            measurement = await client_nodes["measurement"].read_data_value(raise_on_bad_status=False)
            model_available = await client_nodes["model_available"].read_value()
            stale_run = stale_run + 1 if heartbeat == last_heartbeat else 0
            last_heartbeat = heartbeat
            stale = stale_run >= 2
            bad_status = not measurement.StatusCode.is_good()
            fallback_reason = ""
            if stale:
                fallback_reason = "stale"
            elif bad_status:
                fallback_reason = "bad_status"
            elif not model_available:
                fallback_reason = "model_load_failure"
            if fallback_reason:
                command = np.zeros(3, dtype=float)
                fallbacks[fallback_reason] += 1
            else:
                command = np.array((0.20 * math.sin(cycle / 29.0), 0.15 * math.cos(cycle / 31.0), 0.10 * math.sin(cycle / 37.0)))
            command = np.clip(command, -BOUND, BOUND)
            await client_nodes["u_speed"].write_value(float(command[0]))
            await client_nodes["u_gap"].write_value(float(command[1]))
            await client_nodes["u_shape"].write_value(float(command[2]))
            await client_nodes["fallback"].write_value(bool(fallback_reason))
            latency_ms = (time.perf_counter_ns() - t0) / 1e6
            rows.append({
                "cycle": cycle, "latency_ms": latency_ms, "heartbeat": heartbeat,
                "stale_injected": int(stale_injected), "bad_status_injected": int(bad_injected),
                "model_failure_injected": int(model_failure_injected), "fallback_reason": fallback_reason,
                "u_speed": command[0], "u_gap": command[1], "u_shape": command[2],
            })
    finally:
        try:
            await client.disconnect()
        finally:
            await server.stop()

    frame = np.asarray([[row["u_speed"], row["u_gap"], row["u_shape"]] for row in rows], dtype=float)
    latency = np.asarray([row["latency_ms"] for row in rows], dtype=float)
    results = {
        "test_name": "local_loopback_opcua_integration",
        "endpoint": ENDPOINT,
        "cycles": cycles,
        "server_restart_events": 1,
        "client_reconnects": reconnects,
        "fallback_counts": fallbacks,
        "command_boundary_violations": int((np.abs(frame) > BOUND + 1e-12).any(axis=1).sum()),
        "command_finite_failures": int((~np.isfinite(frame)).any(axis=1).sum()),
        "latency_ms_p50": float(np.quantile(latency, 0.50)),
        "latency_ms_p95": float(np.quantile(latency, 0.95)),
        "latency_ms_p99": float(np.quantile(latency, 0.99)),
        "scope_limit": "Local server/client loopback only; no plant-side endpoint, PLC, target network, or industrial latency claim.",
    }
    with (out_dir / "cycle_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cycles", type=int, default=10000)
    args = parser.parse_args()
    result = asyncio.run(main_async(args.cycles, Path(args.out_dir)))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
