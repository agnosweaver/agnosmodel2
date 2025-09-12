# agnosmodel2 - Universal GenAI Provider Abstraction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: PolyForm NC](https://img.shields.io/badge/license-PolyForm%20NC-brightgreen)](https://polyformproject.org/licenses/noncommercial/1.0.0)

> **⚠️ Alpha Release** - This is the debut release of agnosmodel2. While functional, the API may change in future versions.

**Universal Generative AI Provider Interface** - Switch between any GenAI provider (OpenAI, Anthropic, local models) without changing your code. Define providers once, switch seamlessly at runtime.

Part of the Agnosweaver™ project suite.

## 🔥 Why agnosmodel2?

Tired of AI vendor lock-in? Frustrated with rewriting code every time you want to try a different model? **agnosmodel2** solves this by providing a unified interface across all GenAI providers.

```python
# Define once, switch anytime
manager = GenManager()
manager.add_provider('gpt', 'openai', {'model': 'gpt-4', 'api_key': '...'})
manager.add_provider('claude', 'anthropic', {'model': 'claude-3', 'api_key': '...'})

response = manager.generate("Hello world")  # Uses current provider
manager.switch_provider('claude')
response = manager.generate("Hello world")  # Now uses Claude - same code!
```

## ✨ Features

- **🔄 Provider Agnostic**: Switch between OpenAI, Anthropic, local models, or any GenAI provider
- **📊 Datatype Agnostic**: Handle JSON, XML, binary, protobuf, or any data format seamlessly
- **🏗️ Extensible Architecture**: Bring-your-own provider implementations
- **⚡ Sync & Async**: Full support for both synchronous and asynchronous operations
- **🌊 Streaming Support**: Real-time streaming responses when supported by providers
- **📦 Minimal Core**: Lightweight interface without bloat
- **🎯 Standardized Responses**: Consistent response format across all providers
- **🔌 Plugin System**: Registry-based provider discovery and registration

## 📦 Installation

```bash
# PyPI
pip install agnosmodel2

# GitHub
pip install git+https://github.com/agnosweaver/agnosmodel2.git
```

**Requirements**: Python 3.11 or higher

## 🚀 Quick Start

### 1. Basic Setup

```python
from agnosmodel2 import GenManager, ProviderRegistry, BaseGenProvider

# Create manager
manager = GenManager()
```

### 2. Implement a Provider

Since agnosmodel2 provides the core interface, you need to implement providers for your specific needs:

```python
class OpenAIProvider(BaseGenProvider):
    def generate(self, prompt: str, **kwargs):
        # Your OpenAI implementation
        response = self.call_openai_api(prompt)
        return GenResponse(
            content=response.text,
            provider=self.name,
            model=self.model,
            timestamp=datetime.now().isoformat(),
            content_type="text/plain"
        )
    
    async def agenerate(self, prompt: str, **kwargs):
        # Your async OpenAI implementation
        pass
    
    def validate(self):
        return 'api_key' in self.config

# Register your provider
ProviderRegistry.register('openai', OpenAIProvider)
```

### 3. Use It!

```python
# Add providers
manager.add_provider('gpt4', 'openai', {
    'model': 'gpt-4',
    'api_key': 'your-key'
})

manager.add_provider('claude', 'anthropic', {
    'model': 'claude-3-sonnet',
    'api_key': 'your-key'
})

# Generate responses
response = manager.generate("Explain quantum computing")
print(f"Response from {response.provider}: {response.content}")

# Switch providers seamlessly
manager.switch_provider('claude')
response = manager.generate("Explain quantum computing")  # Same call, different provider!
```

## 🎯 Common Use Cases

### Model A/B Testing
```python
# Test different models with the same prompt
results = {}
for provider in ['gpt4', 'claude', 'local-llama']:
    manager.switch_provider(provider)
    response = manager.generate("Write a haiku about AI")
    results[provider] = response.content
```

### Fallback Strategy
```python
def generate_with_fallback(prompt):
    providers = ['premium-model', 'standard-model', 'local-backup']
    
    for provider in providers:
        try:
            manager.switch_provider(provider)
            return manager.generate(prompt)
        except Exception as e:
            print(f"{provider} failed: {e}")
            continue
    
    raise Exception("All providers failed")
```

### Cost Optimization
```python
# Route simple queries to cheaper models
def smart_generate(prompt):
    if len(prompt) < 100:
        manager.switch_provider('gpt-3.5')
    else:
        manager.switch_provider('gpt-4')
    
    return manager.generate(prompt)
```

### Streaming Responses
```python
# Stream from any provider that supports it
for chunk in manager.generate_stream("Tell me a long story"):
    print(chunk.content, end='', flush=True)
    if chunk.is_final:
        break
```

### Multi-Modal Generation
```python
# Stream from any provider that supports it
for chunk in manager.generate_stream("Tell me a long story"):
    print(chunk.content, end='', flush=True)
    if chunk.is_final:
        break

# Handle different data types seamlessly
audio_response = manager.generate("Say hello", output_type="audio")
video_response = manager.generate("Create a short clip", output_type="video")

# Content can be any type: str, bytes, custom objects
if isinstance(audio_response.content, bytes):
    with open("output.wav", "wb") as f:
        f.write(audio_response.content)
```

## 📚 API Reference

### Core Classes

#### `GenManager`
Main interface for managing and switching between providers.

**Methods:**
- `add_provider(name, provider_type, config)` - Add a new provider
- `switch_provider(name)` - Switch to a different provider
- `generate(prompt, **kwargs)` - Generate content synchronously
- `agenerate(prompt, **kwargs)` - Generate content asynchronously
- `generate_stream(prompt, **kwargs)` - Stream content synchronously
- `agenerate_stream(prompt, **kwargs)` - Stream content asynchronously

#### `BaseGenProvider`
Abstract base class for all providers.

**Must Implement:**
- `generate(prompt, **kwargs)` - Sync generation
- `agenerate(prompt, **kwargs)` - Async generation
- `validate()` - Configuration validation

**Optional Override:**
- `generate_stream(prompt, **kwargs)` - Sync streaming
- `agenerate_stream(prompt, **kwargs)` - Async streaming

#### `ProviderRegistry`
Registry for provider discovery and instantiation.

**Methods:**
- `register(provider_type, provider_class)` - Register a provider class
- `create(provider_type, name, config)` - Create provider instance
- `list_types()` - List registered provider types

### Data Structures

#### `GenResponse`
```python
@dataclass
class GenResponse:
    content: Union[str, bytes, Any]
    provider: str
    model: str
    timestamp: str
    content_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```

#### `GenStreamChunk`
```python
@dataclass
class GenStreamChunk:
    content: Union[str, bytes, Any]
    provider: str
    model: str
    timestamp: str
    content_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    is_final: bool = False
```

## 🏗️ Architecture

agnosmodel2 follows a clean, extensible architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   GenManager    │────│ ProviderRegistry │────│ BaseGenProvider │
│   (Router)      │    │   (Discovery)    │    │ (Implementation)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                      │
         │              ┌────────┴────────┐             │
         │              │                 │             │
         ▼              ▼                 ▼             ▼
┌─────────────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────────┐
│  Your App Code  │ │ OpenAI   │ │  Anthropic   │ │ Local Model  │
│                 │ │ Provider │ │   Provider   │ │   Provider   │
└─────────────────┘ └──────────┘ └──────────────┘ └──────────────┘
```

**Key Principles:**
- **Separation of Concerns**: Core routing logic separate from provider implementations
- **Plugin Architecture**: Providers register themselves via the registry
- **Consistent Interface**: All providers expose the same methods
- **Transport Agnostic**: Supports HTTP, gRPC, local calls, anything

## ⚡ Advanced Features

### Custom Transport Layer
```python
class HTTPTransport(BaseModelTransport):
    def send(self, request_data, **kwargs):
        # Custom HTTP implementation
        pass

class gRPCTransport(BaseModelTransport):
    def send(self, request_data, **kwargs):
        # Custom gRPC implementation
        pass
```

### Response Parsing
```python
class JSONResponseParser(BaseResponseParser):
    def parse_response(self, raw_response):
        return json.loads(raw_response)['content']
    
    def get_content_type(self, raw_response):
        return "application/json"
```

## 🤝 Contributing

**Coming Soon**: A dedicated repository for community contributions and ideas is being set up. Stay tuned!

For now, if you have suggestions or find issues, please reach out via email.

## 📄 License

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).

**⚠️ Important**: This license **prohibits commercial use**. If you need to use agnosmodel2 in a commercial project, please contact [agnosweaver@gmail.com](mailto:agnosweaver@gmail.com) for a commercial license.

## 📞 Contact

- **Email**: [agnosweaver@gmail.com](mailto:agnosweaver@gmail.com)
- **Project**: Part of the Agnosweaver™ project suite
- **Repository**: [https://github.com/agnosweaver/agnosmodel2](https://github.com/agnosweaver/agnosmodel2)

---

**Built for developers who hate being boxed in.**

*Pick, switch, and combine GenAI providers freely.*