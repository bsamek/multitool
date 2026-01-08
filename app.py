"""Multi-LLM Comparison Terminal App - Compare responses from multiple LLMs side by side."""

from collections.abc import Callable
from dataclasses import dataclass

import llm
from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, RichLog, Static
from textual_autocomplete import AutoComplete, DropdownItem, TargetState

MODELS = {
    "claude-opus-4.5": "Claude Opus 4.5",
    "gpt-5.2": "GPT-5.2",
    "gemini/gemini-flash-latest": "Gemini Flash",
}


@dataclass
class SlashCommand:
    """Definition of a slash command."""

    name: str
    description: str
    handler: str  # method name on MultiLLMApp


SLASH_COMMANDS: dict[str, SlashCommand] = {
    "/new": SlashCommand("new", "Clear logs and start fresh conversations", "cmd_new"),
}


def sanitize_id(model_id: str) -> str:
    """Convert model ID to valid CSS identifier (replace . and / with -)."""
    return model_id.replace(".", "-").replace("/", "-")


class SlashCommandAutoComplete(AutoComplete):
    """AutoComplete that shows slash commands when input starts with '/'."""

    def __init__(self, target: Input, commands: dict[str, SlashCommand]):
        self.commands = commands
        super().__init__(target, candidates=self.get_candidates)

    def get_candidates(self, state: TargetState) -> list[DropdownItem]:
        """Return command candidates when input starts with '/'."""
        text = state.text
        if not text.startswith("/"):
            return []

        candidates = []
        search = text.lower()
        for cmd_name, cmd in sorted(self.commands.items()):
            if cmd_name.lower().startswith(search):
                candidates.append(
                    DropdownItem(
                        main=cmd_name,
                        prefix=cmd.description,
                    )
                )
        return candidates


class ModelPanel(Vertical):
    """A panel displaying output from a single LLM."""

    def __init__(self, model_id: str, title: str, **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.title = title

    def compose(self) -> ComposeResult:
        yield Static(self.title, classes="panel-title")
        yield RichLog(id=f"log-{sanitize_id(self.model_id)}", markup=True, wrap=True, min_width=0)


class PromptInput(Input):
    """Custom input that handles Enter and slash commands."""

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()

            text = self.value.strip()
            if text.startswith("/"):
                # Only submit if it's a valid slash command
                if text.lower() in SLASH_COMMANDS:
                    self.app.handle_slash_command(text)
                # Invalid slash command - do nothing (block submission)
            else:
                self.app.submit_prompt()


class MultiLLMApp(App):
    """A Textual app to compare LLM responses side by side."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #panels {
        height: 1fr;
    }

    ModelPanel {
        width: 1fr;
        border: solid green;
        padding: 0 1;
    }

    .panel-title {
        text-align: center;
        text-style: bold;
        background: $primary;
        color: $text;
        padding: 0 1;
    }

    RichLog {
        height: 1fr;
        scrollbar-gutter: stable;
        text-wrap: wrap;
    }

    #prompt-container {
        height: auto;
        max-height: 5;
        padding: 1;
    }

    #prompt {
        width: 100%;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.conversations: dict[str, llm.Conversation] = {}
        self._init_conversations()

    def _init_conversations(self) -> None:
        """Initialize conversation objects for each model."""
        for model_id in MODELS:
            model = llm.get_model(model_id)
            self.conversations[model_id] = model.conversation()

    def get_model_log(self, model_id: str) -> RichLog:
        """Get the RichLog widget for a specific model.

        Centralizes widget lookup to avoid scattered string-based ID queries.
        """
        return self.query_one(f"#log-{sanitize_id(model_id)}", RichLog)

    def for_each_model_log(self, callback: Callable[[str, RichLog], None]) -> None:
        """Execute a callback for each model's log widget.

        Args:
            callback: Function that receives (model_id, log) for each model.
        """
        for model_id in MODELS:
            log = self.get_model_log(model_id)
            callback(model_id, log)

    def compose(self) -> ComposeResult:
        with Horizontal(id="panels"):
            for model_id, title in MODELS.items():
                yield ModelPanel(model_id, title, id=f"panel-{sanitize_id(model_id)}")
        with Vertical(id="prompt-container"):
            prompt_input = PromptInput(
                placeholder="Enter prompt (Enter=send, /new=new chat)",
                id="prompt",
            )
            yield prompt_input
            yield SlashCommandAutoComplete(prompt_input, SLASH_COMMANDS)

    def on_mount(self) -> None:
        """Focus the prompt input on startup."""
        self.query_one("#prompt", Input).focus()

    def submit_prompt(self) -> None:
        """Submit the current prompt to all models."""
        prompt_input = self.query_one("#prompt", Input)
        prompt_text = prompt_input.value.strip()

        if not prompt_text:
            return

        # Clear input
        prompt_input.value = ""

        # Add prompt to each log
        def write_prompt(_model_id: str, log: RichLog) -> None:
            log.write(f"[bold yellow]You:[/] {prompt_text}")
            log.write("")  # blank line before response

        self.for_each_model_log(write_prompt)

        # Stream responses from all models concurrently
        for model_id in MODELS:
            self.stream_to_model(model_id, prompt_text)

    def handle_slash_command(self, command_text: str) -> None:
        """Parse and execute a slash command."""
        cmd = command_text.strip().lower()

        if cmd in SLASH_COMMANDS:
            handler_name = SLASH_COMMANDS[cmd].handler
            handler = getattr(self, handler_name, None)
            if handler:
                self.query_one("#prompt", Input).value = ""
                handler()

    def cmd_new(self) -> None:
        """Handler for /new command - clears logs and reinitializes conversations."""
        # Clear all logs
        self.for_each_model_log(lambda _, log: log.clear())

        # Reinitialize conversations
        self._init_conversations()

        # Show confirmation
        self.for_each_model_log(
            lambda _, log: log.write("[bold cyan]--- New Conversation Started ---[/]")
        )

    @work(thread=True, group="llm")
    def stream_to_model(self, model_id: str, prompt: str) -> None:
        """Stream a response from a model to its panel."""
        log = self.get_model_log(model_id)
        conversation = self.conversations[model_id]

        try:
            response = conversation.prompt(prompt)
            # Accumulate all chunks then write as single response
            full_response = "".join(chunk for chunk in response)
            self.call_from_thread(log.write, full_response, scroll_end=True)
            self.call_from_thread(log.write, "")  # blank line after response

        except Exception as e:
            self.call_from_thread(log.write, f"[bold red]Error: {e}[/]")


def main():
    """Entry point for the application."""
    app = MultiLLMApp()
    app.run()


if __name__ == "__main__":  # pragma: no cover
    main()
