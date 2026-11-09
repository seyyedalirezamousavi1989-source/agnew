# Local Ollama chat

## Use

Use for ordinary text questions and conversation. The application discovers this core capability as the fallback for every valid request.

## Inputs and outputs

- Input: validated conversation messages and an installed Ollama model identifier.
- Output: streamed assistant text from the selected local model.

## Dependencies and limitations

- Ollama must be running at `127.0.0.1:11434`, and the selected model must be installed.
- The server uses a 4096-token context and model quality varies.
- Prompts are sent to the local Ollama service; web data is added only when web search is selected.
