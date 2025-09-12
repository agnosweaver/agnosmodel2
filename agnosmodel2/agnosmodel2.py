"""
agnosmodel2 - Universal Generative AI Provider Interface

Switch between any GenAI provider (OpenAI, Anthropic, local models) without changing your code.
Define providers once, switch seamlessly at runtime.
Core interface only - Bring-your-own provider implementations.

Example:
    manager = GenManager()
    manager.add_provider('gpt', 'openai', {'model': 'gpt-4', 'api_key': '...'})
    manager.add_provider('claude', 'anthropic', {'model': 'claude-3', 'api_key': '...'})
    
    response = manager.generate("Hello world")  # Uses current provider
    manager.switch_provider('claude')
    response = manager.generate("Hello world")  # Now uses Claude

#### Copyright (c) 2025 gauravkg32
#### Part of the Agnosweaver™ project suite
####
#### Licensed under the PolyForm Noncommercial License 1.0.0
#### See the LICENSE file for details.

Project Homepage: https://github.com/agnosweaver/agnosmodel2
"""


from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, AsyncGenerator, Generator, Type
from dataclasses import dataclass


# ============================================================================
# Core Data Structures and Exceptions
# ============================================================================


@dataclass
class GenResponse:
    """Standardized response format across all providers"""
    content: Union[str, bytes, Any]
    provider: str
    model: str
    timestamp: str
    content_type: Optional[str] = None # MIME type of the content, ("text/plain", "image/png", Default=None) specifying encouraged
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GenStreamChunk:
    """Standardized streaming chunk format"""
    content: Union[str, bytes, Any]     # In case you are wondering why the same field names are used twice
    provider: str                       # It's because python's weird dataclass inheritance makes
    model: str                          # DRY impossible and not worth the cost of complexity, so much that WET is preferable
    timestamp: str
    content_type: Optional[str] = None # MIME type of the content, ("text/plain", "image/png", Default=None) specifying encouraged
    metadata: Optional[Dict[str, Any]] = None
    is_final: bool = False


class AgnosmodelError(Exception):
    """Base exception for all Agnosmodel-related errors."""
    pass


# ============================================================================
# Core Abstractions
# ============================================================================


class BaseGenProvider(ABC):
    """
    Minimal base class for all GenAI providers

    Providers are expected to implement both sync and async variants.
    They are responsible for orchestrating requests, handling provider-specific quirks,
    and returning standardized responses.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.model = config.get('model', 'default')
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> GenResponse:
        """Generate data synchronously"""
        pass
    
    @abstractmethod
    async def agenerate(self, prompt: str, **kwargs) -> GenResponse:
        """Generate data asynchronously"""
        pass
    
    def generate_stream(self, prompt: str, **kwargs) -> Generator[GenStreamChunk, None, None]:
        """Stream generate data synchronously - optional override"""
        raise NotImplementedError("Streaming not supported by this provider")
    
    async def agenerate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[GenStreamChunk, None]:
        """Stream generate data asynchronously - optional override"""
        raise NotImplementedError("Async streaming not supported by this provider")
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate provider configuration"""
        pass


# Transport abstraction layer
class BaseModelTransport(ABC):
    """Abstract base class for transport mechanisms"""
    
    @abstractmethod
    def send(self, request_data: Any, **kwargs) -> Any:
        """Send request synchronously"""
        pass
    
    @abstractmethod
    async def asend(self, request_data: Any, **kwargs) -> Any:
        """Send request asynchronously"""
        pass
    
    def send_stream(self, request_data: Any, **kwargs) -> Generator[Any, None, None]:
        """Send request with streaming response synchronously. Optional: override if supports streaming"""
        pass
    
    async def asend_stream(self, request_data: Any, **kwargs) -> AsyncGenerator[Any, None]:
        """Send request with streaming response asynchronously. Optional: override if supports streaming"""
        pass


# Response parsing abstraction
class BaseResponseParser(ABC):
    """Abstract base class for response parsing"""
    
    @abstractmethod
    def parse_response(self, raw_response: Any) -> Union[str, bytes, Any]:
        """Parse complete response and extract content"""
        pass
    
    @abstractmethod
    def parse_stream_chunk(self, raw_chunk: Any) -> Optional[Union[str, bytes, Any]]:
        """Parse streaming chunk and extract content, return None if chunk should be skipped"""
        pass
    
    def get_content_type(self, raw_response: Any) -> Optional[str]:
        """Detect content type from raw response"""
        return None  # Default implementation returns None


# ============================================================================
# Provider Registry (The Magic Happens Here)
# ============================================================================


class ProviderRegistry:
    """
    Registry for provider discovery and instantiation
    
    This allows external modules to register their provider implementations
    without modifying core code.
    """
    
    _providers: Dict[str, Type[BaseGenProvider]] = {}
    
    @classmethod
    def register(cls, provider_type: str, provider_class: Type[BaseGenProvider]) -> None:
        """Register a provider class"""
        cls._providers[provider_type] = provider_class
    
    @classmethod
    def create(cls, provider_type: str, name: str, config: Dict[str, Any]) -> BaseGenProvider:
        """Create provider instance"""
        if provider_type not in cls._providers:
            available = list(cls._providers.keys())
            raise AgnosmodelError(f"Unknown provider type '{provider_type}'. Available: {available}")
        
        provider_class = cls._providers[provider_type]
        return provider_class(name, config)
    
    @classmethod
    def list_types(cls) -> List[str]:
        """List registered provider types"""
        return list(cls._providers.keys())


# ============================================================================
# The Manager (The Core of It All)
# ============================================================================


class GenManager:
    """
    GenAI Provider Manager - switch providers without changing code

    Register multiple GenAI providers, switch between them seamlessly.
    All providers expose the same generate/agenerate/stream interface.
    Not responsible for provider-specific logic - that is up to the providers themselves.
    """
    
    def __init__(self):
        self.providers: Dict[str, BaseGenProvider] = {}
        self.current_provider: Optional[str] = None
    
    def register_provider(self, name: str, provider: BaseGenProvider) -> None:
        """Register a provider instance"""
        if not provider.validate():
            raise AgnosmodelError(f"Provider '{name}' failed validation")
        
        self.providers[name] = provider
        
        # Set as current if first provider
        if not self.current_provider:
            self.current_provider = name
    
    def add_provider(self, name: str, provider_type: str, config: Dict[str, Any]) -> None:
        """Add provider using registry"""
        provider = ProviderRegistry.create(provider_type, name, config)
        self.register_provider(name, provider)
    
    def switch_provider(self, name: str) -> None:
        """Switch to specified provider"""
        if name not in self.providers:
            available = list(self.providers.keys())
            raise AgnosmodelError(f"Provider '{name}' not found. Available: {available}")
        
        self.current_provider = name
    
    def get_current_provider(self) -> Optional[str]:
        """Get current provider name"""
        return self.current_provider
    
    def list_providers(self) -> List[str]:
        """List registered providers"""
        return list(self.providers.keys())
    
    def get_provider(self, name: Optional[str] = None) -> BaseGenProvider:
        """Get provider instance"""
        target = name or self.current_provider
        
        if not target:
            raise AgnosmodelError("No provider specified and no current provider set")
        
        if target not in self.providers:
            available = list(self.providers.keys())
            raise AgnosmodelError(f"Provider '{target}' not found. Available: {available}")
        
        return self.providers[target]
    
    # Core routing methods - delegate to provider
    def generate(self, prompt: str, provider: Optional[str] = None, **kwargs) -> GenResponse:
        """Generate text using specified or current provider"""
        return self.get_provider(provider).generate(prompt, **kwargs)
    
    async def agenerate(self, prompt: str, provider: Optional[str] = None, **kwargs) -> GenResponse:
        """Async generate text using specified or current provider"""
        return await self.get_provider(provider).agenerate(prompt, **kwargs)
    
    def generate_stream(self, prompt: str, provider: Optional[str] = None, **kwargs) -> Generator[GenStreamChunk, None, None]:
        """Stream generate text using specified or current provider"""
        yield from self.get_provider(provider).generate_stream(prompt, **kwargs)
    
    async def agenerate_stream(self, prompt: str, provider: Optional[str] = None, **kwargs) -> AsyncGenerator[GenStreamChunk, None]:
        """Async stream generate text using specified or current provider"""
        async for chunk in self.get_provider(provider).agenerate_stream(prompt, **kwargs):
            yield chunk


# ============================================================================
# The End
# ============================================================================