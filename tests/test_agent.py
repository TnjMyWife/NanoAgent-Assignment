import importlib.util
import sys
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = REPO_ROOT / "agent.py"


def load_agent_module():
    fake_openai = types.ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=None))

    setattr(fake_openai, "OpenAI", FakeOpenAI)
    previous_openai = sys.modules.get("openai")
    sys.modules["openai"] = fake_openai
    try:
        spec = importlib.util.spec_from_file_location("nanoagent_agent", AGENT_PATH)
        assert spec is not None
        module = importlib.util.module_from_spec(cast(ModuleSpec, spec))
        loader = spec.loader
        assert loader is not None
        loader.exec_module(module)
        return module
    finally:
        if previous_openai is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = previous_openai


def make_response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class AgentRegressionTests(unittest.TestCase):
    def setUp(self):
        self.agent = load_agent_module()

    def test_parse_tool_arguments_reports_invalid_json(self):
        parsed = self.agent.parse_tool_arguments('{"command":')
        self.assertIn("_argument_error", parsed)
        self.assertIn("Invalid JSON arguments", parsed["_argument_error"])

    def test_parse_tool_arguments_ignores_non_object_payloads(self):
        self.assertEqual(self.agent.parse_tool_arguments('["ls"]'), {})

    def test_run_agent_returns_unknown_tool_error_to_model_loop(self):
        captured_messages = []

        def fake_create(*, model, messages, tools):
            captured_messages.append(messages)
            if len(captured_messages) == 1:
                return make_response(
                    SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="tc-1",
                                function=SimpleNamespace(
                                    name="missing_tool", arguments="{}"
                                ),
                            )
                        ],
                    )
                )
            return make_response(SimpleNamespace(content="done", tool_calls=[]))

        setattr(
            self.agent,
            "client",
            SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
            ),
        )

        result = self.agent.run_agent("test unknown tool", max_iterations=2)

        self.assertEqual(result, "done")
        tool_messages = [
            m
            for m in captured_messages[1]
            if isinstance(m, dict) and m.get("role") == "tool"
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("Unknown tool 'missing_tool'", tool_messages[0]["content"])

    def test_run_agent_returns_argument_errors_to_model_loop(self):
        captured_messages = []

        def fake_create(*, model, messages, tools):
            captured_messages.append(messages)
            if len(captured_messages) == 1:
                return make_response(
                    SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="tc-1",
                                function=SimpleNamespace(
                                    name="read_file",
                                    arguments='{"path":',
                                ),
                            )
                        ],
                    )
                )
            return make_response(SimpleNamespace(content="done", tool_calls=[]))

        setattr(
            self.agent,
            "client",
            SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
            ),
        )

        result = self.agent.run_agent("test invalid args", max_iterations=2)

        self.assertEqual(result, "done")
        tool_messages = [
            m
            for m in captured_messages[1]
            if isinstance(m, dict) and m.get("role") == "tool"
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("Invalid JSON arguments", tool_messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
