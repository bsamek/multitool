# Multitool

A terminal application for comparing responses from multiple LLMs side-by-side in real-time.

## Features

- Send prompts to Claude Opus 4.5, GPT-5.2, and Gemini Flash simultaneously
- View streaming responses in parallel panels
- Slash commands with autocomplete (e.g., `/new` to start fresh)

## Setup

Requires Python 3.13+.

```bash
uv sync
```

## Usage

```bash
python app.py
```

- Type a prompt and press **Enter** to send to all models
- Type `/new` to clear logs and start new conversations
- Press **Ctrl+C** to exit

## Testing

```bash
pytest
```

Runs 42 tests covering UI components, slash commands, and integration scenarios. Tests use mocked LLM responses.
