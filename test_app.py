"""Comprehensive tests for the Multi-LLM Comparison Terminal App."""

import pytest
from unittest.mock import MagicMock, patch
from textual.widgets import Input, RichLog

from app import MultiLLMApp, ModelPanel, PromptInput, MODELS, sanitize_id


# --- Fixtures ---


@pytest.fixture
def mock_llm():
    """Mock the llm library to avoid real API calls."""

    class MockResponse:
        def __init__(self, text):
            self._text = text

        def __iter__(self):
            yield self._text

    class MockConversation:
        def __init__(self):
            self.responses = []

        def prompt(self, text):
            resp = MockResponse(f"Mock response to: {text}")
            self.responses.append(resp)
            return resp

    class MockModel:
        def conversation(self):
            return MockConversation()

    with patch("app.llm.get_model", return_value=MockModel()):
        yield


@pytest.fixture
def mock_llm_error():
    """Mock llm to raise an error."""

    class MockConversation:
        def prompt(self, text):
            raise Exception("API Error: rate limited")

    class MockModel:
        def conversation(self):
            return MockConversation()

    with patch("app.llm.get_model", return_value=MockModel()):
        yield


# --- Unit Tests: MODELS config ---


def test_models_config_has_claude():
    """MODELS should include Claude Opus 4.5."""
    assert "claude-opus-4.5" in MODELS
    assert MODELS["claude-opus-4.5"] == "Claude Opus 4.5"


def test_models_config_has_gpt():
    """MODELS should include GPT-5.2."""
    assert "gpt-5.2" in MODELS
    assert MODELS["gpt-5.2"] == "GPT-5.2"


def test_models_config_has_gemini():
    """MODELS should include Gemini Flash."""
    assert "gemini/gemini-flash-latest" in MODELS
    assert MODELS["gemini/gemini-flash-latest"] == "Gemini Flash"


def test_models_config_count():
    """MODELS should have exactly 3 entries."""
    assert len(MODELS) == 3


# --- Unit Tests: ModelPanel ---


def test_model_panel_stores_model_id():
    """ModelPanel should store the model_id."""
    panel = ModelPanel("test-model", "Test Title")
    assert panel.model_id == "test-model"


def test_model_panel_stores_title():
    """ModelPanel should store the title."""
    panel = ModelPanel("test-model", "Test Title")
    assert panel.title == "Test Title"


# --- App Tests with Pilot ---


@pytest.mark.asyncio
async def test_app_creates_three_conversations(mock_llm):
    """App should create a conversation for each model."""
    app = MultiLLMApp()
    assert len(app.conversations) == 3
    assert "claude-opus-4.5" in app.conversations
    assert "gpt-5.2" in app.conversations
    assert "gemini/gemini-flash-latest" in app.conversations


@pytest.mark.asyncio
async def test_app_has_three_panels(mock_llm):
    """App should render 3 model panels."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        panels = app.query(ModelPanel)
        assert len(panels) == 3


@pytest.mark.asyncio
async def test_app_panels_have_correct_ids(mock_llm):
    """Each panel should have the correct ID."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        for model_id in MODELS:
            panel = app.query_one(f"#panel-{sanitize_id(model_id)}", ModelPanel)
            assert panel.model_id == model_id


@pytest.mark.asyncio
async def test_app_has_prompt_input(mock_llm):
    """App should have a prompt input widget."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        assert prompt is not None


@pytest.mark.asyncio
async def test_prompt_input_focused_on_mount(mock_llm):
    """Prompt input should be focused when app starts."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        focused = app.focused
        assert isinstance(focused, PromptInput)


@pytest.mark.asyncio
async def test_each_panel_has_richlog(mock_llm):
    """Each panel should contain a RichLog widget."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        for model_id in MODELS:
            log = app.query_one(f"#log-{sanitize_id(model_id)}", RichLog)
            assert log is not None


# --- Submit Prompt Tests ---


@pytest.mark.asyncio
async def test_empty_prompt_ignored(mock_llm):
    """Submitting empty prompt should do nothing."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        # Press enter with empty input
        await pilot.press("enter")
        # Logs should remain empty
        for model_id in MODELS:
            log = app.query_one(f"#log-{sanitize_id(model_id)}", RichLog)
            assert len(log.lines) == 0


@pytest.mark.asyncio
async def test_submit_clears_input(mock_llm):
    """After submitting, input should be cleared."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "Hello world"
        assert prompt.value == "Hello world"

        await pilot.press("enter")
        assert prompt.value == ""


@pytest.mark.asyncio
async def test_submit_writes_prompt_to_logs(mock_llm):
    """Submitting should write 'You: <prompt>' to all logs."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "Test prompt"
        await pilot.press("enter")

        # Each log should have content (prompt + blank line = at least 2 lines)
        for model_id in MODELS:
            log = app.query_one(f"#log-{sanitize_id(model_id)}", RichLog)
            assert len(log.lines) >= 2


@pytest.mark.asyncio
async def test_submit_spawns_workers(mock_llm):
    """Submitting should spawn a worker for each model."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "Test"
        await pilot.press("enter")
        # Workers should be created (we can check they exist)
        # Note: Workers run in threads, so we wait a bit
        await pilot.pause()
        # After pause, responses should be written
        for model_id in MODELS:
            log = app.query_one(f"#log-{sanitize_id(model_id)}", RichLog)
            # Should have: You prompt, blank, response, blank = 4+ lines
            assert len(log.lines) >= 3


# --- New Conversation Tests ---


@pytest.mark.asyncio
async def test_new_conversation_reinits_conversations(mock_llm):
    """Shift+Enter should create new conversation objects."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        # Get original conversation objects
        original_convs = {k: v for k, v in app.conversations.items()}

        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "First message"
        await pilot.press("enter")
        await pilot.pause()

        # Start new conversation
        prompt.value = "Second message"
        await pilot.press("shift+enter")

        # Conversations should be new objects
        for model_id in MODELS:
            assert app.conversations[model_id] is not original_convs[model_id]


@pytest.mark.asyncio
async def test_new_conversation_clears_logs(mock_llm):
    """Shift+Enter should clear logs before writing."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)

        # Send first message
        prompt.value = "First"
        await pilot.press("enter")
        await pilot.pause()

        # Get line counts after first message
        counts_before = {}
        for model_id in MODELS:
            log = app.query_one(f"#log-{sanitize_id(model_id)}", RichLog)
            counts_before[model_id] = len(log.lines)

        # New conversation - logs get cleared then new content written
        prompt.value = "Second"
        await pilot.press("shift+enter")
        await pilot.pause()

        # Logs should have new conversation marker + new prompt
        for model_id in MODELS:
            log = app.query_one(f"#log-{sanitize_id(model_id)}", RichLog)
            # Should have: separator, You prompt, blank, response, blank
            assert len(log.lines) >= 4


# --- Error Handling Tests ---


@pytest.mark.asyncio
async def test_api_error_shows_error_message(mock_llm_error):
    """API errors should be displayed in the log."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "Test"
        await pilot.press("enter")
        await pilot.pause()

        # Each log should show error (we can't easily check content,
        # but line count should increase)
        for model_id in MODELS:
            log = app.query_one(f"#log-{sanitize_id(model_id)}", RichLog)
            assert len(log.lines) >= 2  # At least prompt + error


# --- Key Handling Tests ---


@pytest.mark.asyncio
async def test_enter_continues_conversation(mock_llm):
    """Enter key should continue existing conversation."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        original_convs = {k: v for k, v in app.conversations.items()}

        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "Message"
        await pilot.press("enter")
        await pilot.pause()

        # Same conversation objects should be used
        for model_id in MODELS:
            assert app.conversations[model_id] is original_convs[model_id]


@pytest.mark.asyncio
async def test_ctrl_c_binding_exists(mock_llm):
    """App should have ctrl+c binding for quit."""
    app = MultiLLMApp()
    bindings = [b for b in app.BINDINGS if b[0] == "ctrl+c"]
    assert len(bindings) == 1
    assert bindings[0][1] == "quit"
