"""Comprehensive tests for the Multi-LLM Comparison Terminal App."""

import pytest
from unittest.mock import MagicMock, patch
from textual.widgets import Input, RichLog

from app import MultiLLMApp, ModelPanel, PromptInput, MODELS, sanitize_id, SLASH_COMMANDS, SlashCommandAutoComplete


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


# --- Slash Command Registry Tests ---


def test_slash_commands_has_new():
    """SLASH_COMMANDS should include /new command."""
    assert "/new" in SLASH_COMMANDS


def test_slash_command_has_required_fields():
    """Each slash command should have name, description, and handler."""
    for cmd_name, cmd in SLASH_COMMANDS.items():
        assert cmd.name, f"{cmd_name} missing name"
        assert cmd.description, f"{cmd_name} missing description"
        assert cmd.handler, f"{cmd_name} missing handler"


# --- /new Command Functionality Tests ---


@pytest.mark.asyncio
async def test_slash_new_clears_all_logs(mock_llm):
    """/new command should clear all logs."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)

        # Send a message first to populate logs
        prompt.value = "Hello"
        await pilot.press("enter")
        await pilot.pause()

        # Verify logs have content
        for model_id in MODELS:
            log = app.query_one(f"#log-{sanitize_id(model_id)}", RichLog)
            assert len(log.lines) > 0

        # Execute /new command
        prompt.value = "/new"
        await pilot.press("enter")

        # Logs should only have the confirmation message
        for model_id in MODELS:
            log = app.query_one(f"#log-{sanitize_id(model_id)}", RichLog)
            assert len(log.lines) == 1  # Just confirmation


@pytest.mark.asyncio
async def test_slash_new_reinits_conversations(mock_llm):
    """/new command should create new conversation objects."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        # Get original conversation objects
        original_convs = {k: v for k, v in app.conversations.items()}

        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/new"
        await pilot.press("enter")

        # Conversations should be new objects
        for model_id in MODELS:
            assert app.conversations[model_id] is not original_convs[model_id]


@pytest.mark.asyncio
async def test_slash_new_shows_confirmation(mock_llm):
    """/new command should show confirmation message."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/new"
        await pilot.press("enter")

        # Each log should have the confirmation message
        for model_id in MODELS:
            log = app.query_one(f"#log-{sanitize_id(model_id)}", RichLog)
            assert len(log.lines) == 1


@pytest.mark.asyncio
async def test_slash_new_clears_input(mock_llm):
    """/new command should clear the input field."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/new"
        await pilot.press("enter")

        assert prompt.value == ""


# --- Invalid Slash Command Tests ---


@pytest.mark.asyncio
async def test_invalid_slash_command_blocks_submission(mock_llm):
    """Invalid slash commands should not submit and leave input unchanged."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/invalid"
        await pilot.press("enter")

        # Input should remain unchanged (not cleared)
        assert prompt.value == "/invalid"

        # Logs should remain empty
        for model_id in MODELS:
            log = app.query_one(f"#log-{sanitize_id(model_id)}", RichLog)
            assert len(log.lines) == 0


@pytest.mark.asyncio
async def test_partial_slash_command_autocompletes(mock_llm):
    """Partial slash commands like /ne get autocompleted, then execute on second Enter."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/ne"
        await pilot.press("enter")

        # Autocomplete fills in /new but doesn't execute yet
        assert prompt.value == "/new"

        # Press Enter again to execute the completed command
        await pilot.press("enter")
        assert prompt.value == ""


@pytest.mark.asyncio
async def test_slash_only_autocompletes(mock_llm):
    """Just '/' gets autocompleted, then executes on second Enter."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/"
        await pilot.press("enter")

        # Autocomplete fills in /new but doesn't execute yet
        assert prompt.value == "/new"

        # Press Enter again to execute the completed command
        await pilot.press("enter")
        assert prompt.value == ""


@pytest.mark.asyncio
async def test_no_matching_slash_command_blocks(mock_llm):
    """Slash commands that don't match anything should block submission."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/xyz"
        await pilot.press("enter")

        # No autocomplete match, invalid command - input should remain
        assert prompt.value == "/xyz"

        # Logs should remain empty
        for model_id in MODELS:
            log = app.query_one(f"#log-{sanitize_id(model_id)}", RichLog)
            assert len(log.lines) == 0


# --- Normal Prompt Handling Tests ---


@pytest.mark.asyncio
async def test_regular_prompt_still_works(mock_llm):
    """Non-slash input should submit normally."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "Hello world"
        await pilot.press("enter")
        await pilot.pause()

        # Input should be cleared
        assert prompt.value == ""

        # Logs should have content
        for model_id in MODELS:
            log = app.query_one(f"#log-{sanitize_id(model_id)}", RichLog)
            assert len(log.lines) >= 2


# --- Autocomplete Tests ---


@pytest.mark.asyncio
async def test_autocomplete_widget_exists(mock_llm):
    """SlashCommandAutoComplete widget should exist in the app."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        autocomplete = app.query_one(SlashCommandAutoComplete)
        assert autocomplete is not None


# --- Integration Tests ---


@pytest.mark.asyncio
async def test_new_then_prompt_workflow(mock_llm):
    """Can use /new then send normal prompts."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)

        # Start with /new
        prompt.value = "/new"
        await pilot.press("enter")

        # Then send a normal prompt
        prompt.value = "Hello"
        await pilot.press("enter")
        await pilot.pause()

        # Logs should have confirmation + prompt + response
        for model_id in MODELS:
            log = app.query_one(f"#log-{sanitize_id(model_id)}", RichLog)
            assert len(log.lines) >= 3


@pytest.mark.asyncio
async def test_multiple_new_commands(mock_llm):
    """Can use /new multiple times in a session."""
    app = MultiLLMApp()
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)

        # First /new
        prompt.value = "/new"
        await pilot.press("enter")
        original_convs = {k: v for k, v in app.conversations.items()}

        # Second /new
        prompt.value = "/new"
        await pilot.press("enter")

        # Conversations should be different from first /new
        for model_id in MODELS:
            assert app.conversations[model_id] is not original_convs[model_id]


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
