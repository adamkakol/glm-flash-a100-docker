"""Validate stable loopback ports and fail before loading GPU model weights.

These are availability checks, not lasting reservations. Docker/the frontend's
actual bind remains authoritative if another application races the launch.
"""
from contextlib import ExitStack
import errno
import ipaddress
import json
import socket
import subprocess


HOST = "127.0.0.1"


class PortConflictError(RuntimeError):
    """A configured host TCP port is already allocated."""


def validate_port_layout(config: dict, gateway_port: int | None = None) -> dict:
    """Pure validation; None uses config.gateway_port when present."""
    base = config.get("base_port")
    if type(base) is not int or not 1024 <= base <= 65533:
        raise ValueError("base_port must be an integer from 1024 to 65533, leaving room for three backends")
    gateway = config.get("gateway_port") if gateway_port is None else gateway_port
    if gateway is not None and (type(gateway) is not int or not 1024 <= gateway <= 65535):
        raise ValueError("gateway_port / serve --port must be an integer from 1024 to 65535")
    backends = list(range(base, base + 3))
    if gateway in backends:
        raise ValueError(f"gateway port {HOST}:{gateway} overlaps the backend block "
                         f"{HOST}:{base}-{base + 2}; choose a different gateway_port / serve --port or base_port")
    return {"host": HOST, "backend_ports": backends, "gateway_port": gateway}


def _run(command, *, timeout=15):
    return subprocess.run(command, text=True, capture_output=True, check=False, timeout=timeout)


def _docker_bindings(runner) -> list[dict]:
    """Read only IDs, names and published port metadata, never Docker env."""
    try:
        listing = runner(["docker", "ps", "--quiet", "--no-trunc"], timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot check Docker published ports: {exc}") from exc
    if listing.returncode:
        raise RuntimeError("cannot check Docker published ports; Docker must be running and accessible to this user")
    ids = listing.stdout.split()
    if not ids:
        return []
    # Named fields avoid exposing container environment variables or credentials.
    template = ('{"id":{{json .Id}},"name":{{json .Name}},'
                '"ports":{{json .NetworkSettings.Ports}},'
                '"bindings":{{json .HostConfig.PortBindings}}}')
    try:
        result = runner(["docker", "inspect", "--format", template, *ids], timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot inspect Docker published ports: {exc}") from exc
    if result.returncode:
        raise RuntimeError("cannot inspect Docker port reservations; a container may have changed during preflight; retry")
    bindings = []
    try:
        records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        if len(records) != len(ids):
            raise ValueError("incomplete Docker inspect result")
        for record in records:
            for mapping in (record.get("ports") or {}, record.get("bindings") or {}):
                for container_port, publications in mapping.items():
                    if not container_port.endswith("/tcp"):
                        continue
                    for publication in publications or []:
                        port = int(publication["HostPort"])
                        if port:
                            bindings.append({"host": publication.get("HostIp", ""), "port": port,
                                             "container": record.get("name", "").lstrip("/") or record["id"][:12],
                                             "target": container_port})
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise RuntimeError("cannot parse Docker published-port metadata; refusing an incomplete check") from exc
    return bindings


def _overlaps_loopback(host: str) -> bool:
    if not host:
        return True
    try:
        address = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        # Docker normally supplies numeric addresses; unknown bindings fail closed.
        return True
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    # Conservatively include Docker's IPv6 wildcard publication: its IPv4
    # behavior depends on the daemon's dual-stack/userland-proxy settings.
    return address.is_unspecified or str(address) == HOST


def _listener_detail(port: int, runner) -> str:
    try:
        result = runner(["ss", "-H", "-ltnp", f"sport = :{port}"], timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            return " Listener: " + " ".join(result.stdout.split())[:700]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return " Inspect it with: ss -ltnp 'sport = :" + str(port) + "'."


def check_ports(config: dict, backend_count: int = 3, gateway_port: int | None = None,
                *, runner=None) -> dict:
    """Check exact IPv4 loopback TCP ports, including Docker NAT reservations.

    A busy port is never killed, reassigned, or automatically remapped. Socket
    probes use SO_REUSEADDR for clean repeated lifecycles, but not SO_REUSEPORT.
    """
    layout = validate_port_layout(config, gateway_port)
    if type(backend_count) is not int or not 1 <= backend_count <= 3:
        raise ValueError("backend_count must be 1, 2, or 3")
    runner = runner or _run
    labels = {port: f"backend {i}" for i, port in enumerate(layout["backend_ports"][:backend_count])}
    if layout["gateway_port"] is not None:
        labels[layout["gateway_port"]] = "gateway"
    conflicts = []
    for binding in _docker_bindings(runner):
        port = binding["port"]
        if port in labels and _overlaps_loopback(binding["host"]):
            conflicts.append(f"{labels[port]} {HOST}:{port}: Docker container "
                             f"{binding['container']} publishes {binding['host'] or '*'}:{port} "
                             f"to {binding['target']}")
    if conflicts:
        raise PortConflictError("Configured TCP port unavailable: " + "; ".join(dict.fromkeys(conflicts)) +
                                ". Choose a free base_port or gateway_port / serve --port; no service was stopped.")
    # Keep all probes open until this batch is checked, then release them before
    # Docker's real bind. A socket probe also catches host-network containers.
    with ExitStack() as stack:
        for port, label in labels.items():
            try:
                probe = stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind((HOST, port))
                probe.listen(1)
            except OSError as exc:
                if exc.errno == errno.EADDRINUSE:
                    raise PortConflictError(f"Configured {label} TCP port {HOST}:{port} is already in use."
                                            + _listener_detail(port, runner) +
                                            " Choose a free configured port; no service was stopped.") from exc
                raise RuntimeError(f"cannot check {label} TCP port {HOST}:{port}: {exc}") from exc
    return {**layout, "backend_ports": layout["backend_ports"][:backend_count],
            "checked_ports": list(labels)}
