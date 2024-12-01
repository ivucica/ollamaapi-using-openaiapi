# ollamaapi-using-openaiapi

This is a small proxy allowing use of software that expects the Ollama-style
API, even if the service exposes the OpenAI-style API.

For instance, this allows using Privy extension for VSCode while serving the
API using LM Studio, which may be useful due to an easy way to use MLX models
on M2 hardware.
