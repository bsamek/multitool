"""Multi-LLM Comparison Terminal App - Compare responses from multiple LLMs side by side."""

import llm
from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, RichLog, Static

MODELS = {
    "claude-opus-4.5": "Claude Opus 4.5",
    "gpt-5.2": "GPT-5.2",
    "gemini/gemini-flash-latest": "Gemini Flash",
}


def sanitize_id(model_id: str) -> str:
    """Convert model ID to valid CSS identifier (replace . and / with -)."""
    return model_id.replace(".", "-").replace("/", "-")


class ModelPanel(Vertical):
    """A panel displaying output from a single LLM."""

    def __init__(self, model_id: str, title: str, **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.title = title

    def compose(self) -> ComposeResult:
        yield Static(self.title, classes="panel-title")
        yield RichLog(id=f"log-{sanitize_id(self.model_id)}", markup=True, wrap=True)


class PromptInput(Input):
    """Custom input that handles Enter vs Shift+Enter."""

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.app.submit_prompt(new_conversation=False)
        elif event.key == "shift+enter":
            event.prevent_default()
            event.stop()
            self.app.submit_prompt(new_conversation=True)


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

    def compose(self) -> ComposeResult:
        with Horizontal(id="panels"):
            for model_id, title in MODELS.items():
                yield ModelPanel(model_id, title, id=f"panel-{sanitize_id(model_id)}")
        with Vertical(id="prompt-container"):
            yield PromptInput(placeholder="Enter prompt (Enter=continue, Shift+Enter=new conversation)", id="prompt")

    def on_mount(self) -> None:
        """Focus the prompt input on startup."""
        self.query_one("#prompt", Input).focus()

    def submit_prompt(self, new_conversation: bool = False) -> None:
        """Submit the current prompt to all models."""
        prompt_input = self.query_one("#prompt", Input)
        prompt_text = prompt_input.value.strip()

        if not prompt_text:
            return

        # Clear input
        prompt_input.value = ""

        # Reset conversations if new conversation requested
        if new_conversation:
            self._init_conversations()
            # Clear all logs
            for model_id in MODELS:
                log = self.query_one(f"#log-{sanitize_id(model_id)}", RichLog)
                log.clear()

        # Add separator and prompt to each log
        for model_id in MODELS:
            log = self.query_one(f"#log-{sanitize_id(model_id)}", RichLog)
            if new_conversation:
                log.write("[bold cyan]--- New Conversation ---[/]")
            log.write(f"[bold yellow]You:[/] {prompt_text}")
            log.write("")  # blank line before response

        # Stream responses from all models concurrently
        for model_id in MODELS:
            self.stream_to_model(model_id, prompt_text)

    @work(thread=True, group="llm")
    def stream_to_model(self, model_id: str, prompt: str) -> None:
        """Stream a response from a model to its panel."""
        log = self.query_one(f"#log-{sanitize_id(model_id)}", RichLog)
        conversation = self.conversations[model_id]

        try:
            response = conversation.prompt(prompt)
            # Accumulate all chunks then write as single response
            full_response = "".join(chunk for chunk in response)
            self.call_from_thread(log.write, full_response, scroll_end=True)
            self.call_from_thread(log.write, "")  # blank line after response

        except Exception as e:
            self.call_from_thread(log.write, f"[bold red]Error: {e}[/]")


if __name__ == "__main__":
    app = MultiLLMApp()
    app.run()
