"""Deployment configuration: env-var upstreams, validation, address formatting."""

import unittest
from unittest import mock

from adaptible._src.wrap import config
from adaptible._src.wrap.__main__ import parser as wrapper_parser


class UpstreamUrlTest(unittest.TestCase):
    def test_explicit_argument_wins_over_environment_and_default(self):
        with mock.patch.dict(
            "os.environ", {"ADAPTIBLE_OLLAMA_URL": "http://127.0.0.1:9"}
        ):
            self.assertEqual(
                config.upstream_url("ollama", "http://127.0.0.1:11435"),
                "http://127.0.0.1:11435",
            )

    def test_environment_overrides_the_default(self):
        cases = (
            ("ollama", "ADAPTIBLE_OLLAMA_URL", "http://10.0.0.4:11434"),
            ("lm-studio", "ADAPTIBLE_LM_STUDIO_URL", "http://10.0.0.5:1234"),
        )
        for service, variable, value in cases:
            with self.subTest(service=service):
                with mock.patch.dict("os.environ", {variable: value}):
                    self.assertEqual(config.upstream_url(service), value)

    def test_default_is_used_when_nothing_is_supplied(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(config.upstream_url("ollama"), config.OLLAMA_URL)
            self.assertEqual(config.upstream_url("lm-studio"), config.LM_STUDIO_URL)

    def test_trailing_slash_is_removed(self):
        self.assertEqual(
            config.upstream_url("ollama", "http://127.0.0.1:11434/"),
            "http://127.0.0.1:11434",
        )

    def test_managed_services_refuse_an_upstream(self):
        for service in ("llama-cpp", "vllm"):
            with self.subTest(service=service):
                with self.assertRaisesRegex(ValueError, "managed server"):
                    config.upstream_url(service, "http://127.0.0.1:8080")

    def test_malformed_urls_are_rejected(self):
        cases = (
            "127.0.0.1:11434",  # no scheme
            "ftp://127.0.0.1:11434",  # wrong scheme
            "http://",  # no host
            "http://user:pw@127.0.0.1:11434",  # credentials
            "http://127.0.0.1:11434?a=b",  # query
            "http://127.0.0.1:11434#frag",  # fragment
            "http://127.0.0.1:0",  # port 0
            "http://127.0.0.1:11434 ",  # whitespace
        )
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    config.upstream_url("ollama", value)

    def test_a_malformed_environment_value_is_rejected_too(self):
        with mock.patch.dict("os.environ", {"ADAPTIBLE_OLLAMA_URL": "not-a-url"}):
            with self.assertRaises(ValueError):
                config.upstream_url("ollama")


class ValidateServerOptionsTest(unittest.TestCase):
    def options(self, *extra):
        return wrapper_parser().parse_args(["ollama", "existing", *extra])

    def test_parser_defaults_pass_for_every_service(self):
        for service in ("ollama", "llama-cpp", "lm-studio", "vllm"):
            with self.subTest(service=service):
                args = wrapper_parser().parse_args([service, "existing"])
                config.validate_server_options(args)

    def test_parser_defaults_come_from_this_module(self):
        args = self.options()
        self.assertEqual(args.host, config.DEFAULT_HOST)
        self.assertEqual(args.port, config.DEFAULT_PORT)
        self.assertEqual(args.max_tokens, config.DEFAULT_MAX_TOKENS)
        self.assertEqual(args.context_size, config.DEFAULT_CONTEXT_SIZE)
        self.assertEqual(args.idle_seconds, config.DEFAULT_IDLE_SECONDS)

    def test_out_of_range_options_are_rejected(self):
        cases = (
            (["--port", "0"], "port"),
            (["--port", "70000"], "port"),
            (["--host", " "], "host"),
            (["--idle-seconds", "-1"], "idle-seconds"),
            (["--idle-seconds", "nan"], "idle-seconds"),
            (["--max-tokens", "0"], "max-tokens"),
            (["--context-size", "127"], "context-size"),
        )
        for extra, expected in cases:
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(ValueError, expected):
                    config.validate_server_options(self.options(*extra))

    def test_an_upstream_for_a_managed_service_is_rejected(self):
        for service in ("llama-cpp", "vllm"):
            with self.subTest(service=service):
                args = wrapper_parser().parse_args(
                    [service, "existing", "--upstream", "http://127.0.0.1:8080"]
                )
                with self.assertRaisesRegex(ValueError, "managed server"):
                    config.validate_server_options(args)


class ServerUrlTest(unittest.TestCase):
    def test_ipv4_and_hostnames_are_formatted_plainly(self):
        self.assertEqual(config.server_url("127.0.0.1", 8000), "http://127.0.0.1:8000")
        self.assertEqual(config.server_url("localhost", 80), "http://localhost:80")

    def test_ipv6_literals_are_bracketed_once(self):
        self.assertEqual(config.server_url("::1", 8000), "http://[::1]:8000")
        self.assertEqual(config.server_url("[::1]", 8000), "http://[::1]:8000")


if __name__ == "__main__":
    unittest.main()
