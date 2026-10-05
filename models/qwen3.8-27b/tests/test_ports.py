"""Port validation and reservations; Docker responses are always mocked."""
import errno
import json
from pathlib import Path
import socket
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_bench.ports import PortConflictError, check_ports, validate_port_layout


class DockerPorts:
    def __init__(self, records=(), error=False, listener=""):
        self.records, self.error, self.listener = list(records), error, listener
        self.calls = []

    def __call__(self, cmd, timeout=15):
        self.calls.append(cmd)
        if cmd[:2] == ["docker", "ps"]:
            output = "\n".join(str(i) * 64 for i in range(1, len(self.records) + 1))
        elif cmd[:2] == ["docker", "inspect"]:
            output = "\n".join(json.dumps(v) for v in self.records)
        else:
            output = self.listener
        return subprocess.CompletedProcess(cmd, 1 if self.error else 0, output, "")


def publication(host, port, protocol="tcp", *, configured_only=False):
    binding = {"8000/" + protocol: [{"HostIp": host, "HostPort": str(port)}]}
    return {"id": "a" * 64, "name": "/existing-inference", "ports": None if configured_only else binding,
            "bindings": binding}


class Probe:
    def __init__(self, sockets, fail_port=None, err=errno.EADDRINUSE):
        sockets.append(self)
        self.fail_port, self.err = fail_port, err
        self.closed, self.listening = False, False
        self.options = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def setsockopt(self, *args):
        self.options.append(args)

    def bind(self, address):
        self.address = address
        if address[1] == self.fail_port:
            raise OSError(self.err, "test bind error")

    def listen(self, backlog):
        self.listening = True


class PortTests(unittest.TestCase):
    def setUp(self):
        self.config = {"base_port": 18100, "gateway_port": 18080}
        self.sockets = []

    def probes(self, fail_port=None, err=errno.EADDRINUSE):
        return patch("qwen_bench.ports.socket.socket", side_effect=lambda *args: Probe(self.sockets, fail_port, err))

    def test_layout_rejects_bool_strings_ranges_and_all_gateway_collisions(self):
        with patch("qwen_bench.ports.subprocess.run") as run:
            for base in (True, "18100", 1023, 65534):
                with self.subTest(base=base), self.assertRaisesRegex(ValueError, "base_port"):
                    validate_port_layout({**self.config, "base_port": base})
            for gateway in (False, "8000", 80, 65536, 18100, 18101, 18102):
                with self.subTest(gateway=gateway), self.assertRaises(ValueError):
                    validate_port_layout({**self.config, "gateway_port": gateway})
            run.assert_not_called()

    def test_serve_override_and_container_internal_port_do_not_confuse_host_ports(self):
        self.assertEqual(8000, validate_port_layout(self.config, 8000)["gateway_port"])
        self.assertEqual([65533, 65534, 65535], validate_port_layout({"base_port": 65533})["backend_ports"])
        with self.assertRaisesRegex(ValueError, "overlaps"):
            validate_port_layout({"base_port": 8000, "gateway_port": 18080}, 8000)

    def test_docker_nat_reservations_caught_without_a_socket_listener(self):
        for host in ("127.0.0.1", "0.0.0.0", "::", "", "::ffff:127.0.0.1"):
            runner = DockerPorts([publication(host, 18100, configured_only=True)])
            with self.subTest(host=host), self.probes(), self.assertRaisesRegex(PortConflictError, "existing-inference") as error:
                check_ports(self.config, runner=runner)
            self.assertIn("127.0.0.1:18100", str(error.exception))
            self.assertEqual([], self.sockets)
            self.assertNotIn(".Config.Env", runner.calls[-1][3])

    def test_gateway_docker_collision_reported_before_any_backend_probe(self):
        with self.probes(), self.assertRaisesRegex(PortConflictError, "gateway 127.0.0.1:8000"):
            check_ports(self.config, gateway_port=8000, runner=DockerPorts([publication("0.0.0.0", 8000)]))
        self.assertEqual([], self.sockets)

    def test_docker_udp_and_distinct_specific_ip_do_not_conflict(self):
        runner = DockerPorts([publication("0.0.0.0", 18100, "udp"), publication("192.0.2.10", 18100)])
        with self.probes():
            result = check_ports(self.config, backend_count=1, runner=runner)
        self.assertEqual([18100, 18080], result["checked_ports"])
        self.assertTrue(all(s.closed and s.listening for s in self.sockets))

    def test_dynamic_docker_publication_uses_actual_host_port(self):
        record = publication("0.0.0.0", 18101)
        record["bindings"]["8000/tcp"] = [{"HostIp": "", "HostPort": "0"}]
        # Make independent mappings, as Docker supplies them.
        record["ports"] = {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "18101"}]}
        with self.probes(), self.assertRaisesRegex(PortConflictError, "18101"):
            check_ports(self.config, runner=DockerPorts([record]))

    def test_host_listener_error_names_port_and_process(self):
        runner = DockerPorts(listener='LISTEN 0 128 0.0.0.0:18080 0.0.0.0:* users:(("python",pid=123,fd=4))')
        with self.probes(fail_port=18080), self.assertRaises(PortConflictError) as error:
            check_ports(self.config, runner=runner)
        self.assertIn("gateway TCP port 127.0.0.1:18080", str(error.exception))
        self.assertIn("pid=123", str(error.exception))
        self.assertTrue(all(s.closed for s in self.sockets))

    def test_permission_failure_is_not_misreported_as_busy_port(self):
        with self.probes(fail_port=18100, err=errno.EACCES), self.assertRaises(RuntimeError) as error:
            check_ports(self.config, runner=DockerPorts())
        self.assertIn("cannot check backend 0 TCP port 127.0.0.1:18100", str(error.exception))
        self.assertNotIn("already in use", str(error.exception))

    def test_docker_access_failure_fails_closed(self):
        with self.probes(), self.assertRaisesRegex(RuntimeError, "Docker must be running"):
            check_ports(self.config, runner=DockerPorts(error=True))
        self.assertEqual([], self.sockets)

    def test_repeated_checks_release_every_probe_and_never_use_reuseport(self):
        with self.probes():
            for _ in range(2):
                check_ports(self.config, runner=DockerPorts())
        self.assertEqual(8, len(self.sockets))
        self.assertTrue(all(s.closed and s.listening for s in self.sockets))
        self.assertTrue(all(s.options == [(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)] for s in self.sockets))

    def test_real_wildcard_listener_conflicts_with_loopback_when_sockets_available(self):
        try:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except PermissionError:
            self.skipTest("sandbox does not permit sockets")
        with listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("0.0.0.0", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            if port > 65533:
                self.skipTest("ephemeral port cannot start the three-port block")
            with self.assertRaises(PortConflictError):
                check_ports({"base_port": port}, backend_count=1, runner=DockerPorts())


if __name__ == "__main__":
    unittest.main()
