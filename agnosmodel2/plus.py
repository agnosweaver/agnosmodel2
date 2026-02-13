"""
agnosmodel2.plus - For Batties-Included behavior and advanced features
Comes with GenericProvider which supports HTTPTransport and WebSocketTransport,
and JSONResponseParser, PlainTextResponseParser, BinaryResponseParser, and CustomResponseParser.
Includes enhanced error handling and a minimal CLI for testing.

#### Copyright (c) 2025 gauravkg32
#### Part of the Agnosweaver™ project suite
####
#### Licensed under the PolyForm Noncommercial License 1.0.0
#### See the LICENSE file for details.

Project Homepage: https://github.com/agnosweaver/agnosmodel2
"""


import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, AsyncGenerator, Generator, Callable
from datetime import datetime, UTC


from .agnosmodel2 import (BaseGenProvider, GenManager, ProviderRegistry,
                          GenResponse, GenStreamChunk, AgnosmodelError)


# ============================================================================
# Plus Exceptions
# ============================================================================


class GenProviderError(AgnosmodelError):
    """Base exception for GenAI provider errors"""
    pass


class ConfigurationError(AgnosmodelError):
    """Raised when provider configuration is invalid"""
    pass


class APIError(GenProviderError):
    """Raised when API requests fail"""
    pass


# ============================================================================
# Plus Abstractions
# ============================================================================


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
# Transport Implementations
# ============================================================================


class HTTPTransport(BaseModelTransport):
    """HTTP transport implementation"""
    
    def __init__(self, endpoint: str, method: str = 'POST', headers: Optional[Dict[str, str]] = None):
        self.endpoint = endpoint
        self.method = method
        self.headers = headers or {}
    
    def send(self, request_data: Any, **kwargs) -> Any:
        """Sync HTTP request"""
        import requests
        
        try:
            response = requests.request(
                method=self.method,
                url=self.endpoint,
                headers=self.headers,
                json=request_data,
                timeout=kwargs.get('timeout', 30)
            )
            response.raise_for_status()
            return response
            
        except requests.RequestException as e:
            raise APIError("HTTP request failed") from e
    
    async def asend(self, request_data: Any, **kwargs) -> Any:
        """Async HTTP request"""
        import aiohttp
        
        try:
            timeout = aiohttp.ClientTimeout(total=kwargs.get('timeout', 30))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    method=self.method,
                    url=self.endpoint,
                    headers=self.headers,
                    json=request_data
                ) as response:
                    response.raise_for_status()
                    return response
                    
        except (aiohttp.ClientError, Exception) as e:
            raise APIError("HTTP request failed") from e
    
    def send_stream(self, request_data: Any, **kwargs) -> Generator[Any, None, None]:
        """Sync streaming HTTP request"""
        import requests
        
        try:
            with requests.request(
                method=self.method,
                url=self.endpoint,
                headers=self.headers,
                json=request_data,
                timeout=kwargs.get('timeout', 30),
                stream=True
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        yield line
                        
        except requests.RequestException as e:
            raise APIError("HTTP streaming request failed") from e
    
    async def asend_stream(self, request_data: Any, **kwargs) -> AsyncGenerator[Any, None]:
        """Async streaming HTTP request"""
        import aiohttp
        
        try:
            timeout = aiohttp.ClientTimeout(total=kwargs.get('timeout', 30))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    method=self.method,
                    url=self.endpoint,
                    headers=self.headers,
                    json=request_data
                ) as response:
                    response.raise_for_status()
                    async for line in response.content:
                        if line:
                            yield line.decode('utf-8')
                            
        except (aiohttp.ClientError, Exception) as e:
            raise APIError("HTTP streaming request failed") from e


class WebSocketTransport(BaseModelTransport):
    """WebSocket transport implementation"""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
    
    def send(self, request_data: Any, **kwargs) -> Any:
        """Sync WebSocket request - not typically used"""
        raise NotImplementedError("Synchronous WebSocket not supported")
    
    async def asend(self, request_data: Any, **kwargs) -> Any:
        """Async WebSocket request"""
        import websockets
        
        try:
            async with websockets.connect(self.endpoint) as websocket:
                await websocket.send(json.dumps(request_data))
                response = await websocket.recv()
                return response
                
        except Exception as e:
            raise APIError("WebSocket request failed") from e

    def send_stream(self, request_data: Any, **kwargs) -> Generator[Any, None, None]:
        """Sync WebSocket streaming - not typically used"""
        raise NotImplementedError("Synchronous WebSocket streaming not supported")
    
    async def asend_stream(self, request_data: Any, **kwargs) -> AsyncGenerator[Any, None]:
        """Async WebSocket streaming"""
        import websockets
        
        try:
            async with websockets.connect(self.endpoint) as websocket:
                await websocket.send(json.dumps(request_data))
                
                while True:
                    try:
                        response = await websocket.recv()
                        yield response
                    except websockets.exceptions.ConnectionClosed:
                        break
                        
        except Exception as e:
            raise APIError("WebSocket streaming request failed") from e


# ============================================================================
# Response Parsers
# ============================================================================


class JSONResponseParser(BaseResponseParser):
    """JSON response parser using path-based extraction"""
    
    def __init__(self, response_path: List[str]):
        self.response_path = response_path
    
    def parse_response(self, raw_response: Any) -> Union[str, bytes, Any]:
        """Parse JSON response using configured path"""
        try:
            # Handle different response types
            if hasattr(raw_response, 'json'):
                response_data = raw_response.json()
            elif hasattr(raw_response, 'text'):
                response_data = json.loads(raw_response.text)
            elif isinstance(raw_response, (str, bytes)):
                response_data = json.loads(raw_response)
            else:
                response_data = raw_response
            
            return self._extract_content(response_data)
            
        except (json.JSONDecodeError, KeyError) as e:
            raise APIError("JSON parsing failed") from e

    def parse_stream_chunk(self, raw_chunk: Any) -> Optional[Union[str, bytes, Any]]:
        """Parse JSON streaming chunk"""
        try:
            # Handle Server-Sent Events format
            chunk_str = raw_chunk.strip() if isinstance(raw_chunk, str) else raw_chunk.decode('utf-8').strip()
            
            # Skip empty lines and comments
            if not chunk_str or chunk_str.startswith(':'):
                return None
            
            # Handle SSE data prefix
            if chunk_str.startswith('data: '):
                chunk_str = chunk_str[6:]
            
            # Handle end-of-stream marker
            if chunk_str == '[DONE]':
                return None
            
            chunk_data = json.loads(chunk_str)
            return self._extract_content(chunk_data)
            
        except (json.JSONDecodeError, KeyError):
            # Skip malformed chunks
            return None
    
    def _extract_content(self, response_data: Dict[str, Any]) -> Union[str, bytes, Any]:
        """Extract content using configured path"""
        current = response_data
        
        for key in self.response_path:
            if isinstance(current, list):
                current = current[int(key)]
            else:
                current = current[key]
        
        # Return as-is instead of converting to string to preserve data types
        return current


class PlainTextResponseParser(BaseResponseParser):
    """Plain text response parser"""
    
    def parse_response(self, raw_response: Any) -> Union[str, bytes, Any]:
        """Parse plain text response"""
        if hasattr(raw_response, 'text'):
            return raw_response.text
        elif isinstance(raw_response, bytes):
            return raw_response.decode('utf-8')
        elif isinstance(raw_response, str):
            return raw_response
        else:
            return str(raw_response)
    
    def parse_stream_chunk(self, raw_chunk: Any) -> Optional[Union[str, bytes, Any]]:
        """Parse plain text streaming chunk"""
        if isinstance(raw_chunk, bytes):
            return raw_chunk.decode('utf-8')
        elif isinstance(raw_chunk, str):
            return raw_chunk
        else:
            return str(raw_chunk) if raw_chunk else None


class BinaryResponseParser(BaseResponseParser):
    """Binary response parser for images, audio, etc."""
    
    def __init__(self, content_type: str = "application/octet-stream"):
        self.content_type = content_type
    
    def parse_response(self, raw_response: Any) -> Union[str, bytes, Any]:
        """Parse binary response"""
        if hasattr(raw_response, 'content'):
            return raw_response.content  # bytes
        elif isinstance(raw_response, bytes):
            return raw_response
        else:
            return raw_response
    
    def parse_stream_chunk(self, raw_chunk: Any) -> Optional[Union[str, bytes, Any]]:
        """Parse binary streaming chunk"""
        if isinstance(raw_chunk, (bytes, str)):
            return raw_chunk
        else:
            return raw_chunk if raw_chunk else None
    
    def get_content_type(self, raw_response: Any) -> str:
        """Return configured content type"""
        return self.content_type


class CustomResponseParser(BaseResponseParser):
    """Custom response parser using user-provided functions"""
    
    def __init__(self, 
                 parse_func: Callable[[Any], Union[str, bytes, Any]],
                 parse_stream_func: Optional[Callable[[Any], Optional[Union[str, bytes, Any]]]] = None,
                 content_type: str = "text/plain"):
        self.parse_func = parse_func
        self.parse_stream_func = parse_stream_func or self._default_stream_parser
        self.content_type = content_type
    
    def parse_response(self, raw_response: Any) -> Union[str, bytes, Any]:
        """Parse response using custom function"""
        return self.parse_func(raw_response)
    
    def parse_stream_chunk(self, raw_chunk: Any) -> Optional[Union[str, bytes, Any]]:
        """Parse streaming chunk using custom function"""
        return self.parse_stream_func(raw_chunk)
    
    def get_content_type(self, raw_response: Any) -> str:
        """Return configured content type"""
        return self.content_type
    
    def _default_stream_parser(self, raw_chunk: Any) -> Optional[Union[str, bytes, Any]]:
        """Default stream parser - return chunk as-is"""
        return raw_chunk if raw_chunk else None
    

# ============================================================================
# GenProvider (The GOAT)
# ============================================================================


# Enhanced generic provider with transport and parser abstraction
class GenericProvider(BaseGenProvider):
    """Generic provider supporting multiple transports and response formats"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        # Required config
        self.request_template = config['request_template']
        
        # Initialize transport
        self._init_transport(config)
        
        # Initialize response parser
        self._init_parser(config)
    
        self._setup()
    
    def _setup(self) -> None:
        """Basic setup - API key resolution from environment"""
        if 'api_key' not in self.config:
            env_key = f"{self.name.upper()}_API_KEY"
            api_key = os.getenv(env_key)
            if api_key:
                self.config['api_key'] = api_key

    def _init_transport(self, config: Dict[str, Any]) -> None:
        """Initialize transport based on configuration"""
        transport_type = config.get('transport_type', 'http')
        
        if transport_type == 'http':
            endpoint = config['endpoint']
            method = config.get('method', 'POST')
            headers = config.get('headers', {})
            
            # Handle API key authentication
            api_key = config.get('api_key')
            api_key_header = config.get('api_key_header')
            api_key_prefix = config.get('api_key_prefix', '')
            
            if api_key and api_key_header:
                headers[api_key_header] = f"{api_key_prefix}{api_key}"
            
            self.transport = HTTPTransport(endpoint, method, headers)
        
        elif transport_type == 'websocket':
            endpoint = config['endpoint']
            self.transport = WebSocketTransport(endpoint)
        
        else:
            raise ConfigurationError(f"Unsupported transport type: {transport_type}")
    
    def _init_parser(self, config: Dict[str, Any]) -> None:
        """Initialize response parser based on configuration"""
        parser_type = config.get('parser_type', 'json')
        
        if parser_type == 'json':
            response_path = config.get('response_path', [])
            self.parser = JSONResponseParser(response_path)
        
        elif parser_type == 'text':
            self.parser = PlainTextResponseParser()
        
        elif parser_type == 'binary':
            content_type = config.get('content_type', 'application/octet-stream')
            self.parser = BinaryResponseParser(content_type)
        
        elif parser_type == 'custom':
            parse_func = config['parse_func']
            parse_stream_func = config.get('parse_stream_func')
            content_type = config.get('content_type', 'text/plain')
            self.parser = CustomResponseParser(parse_func, parse_stream_func, content_type)
        
        else:
            raise ConfigurationError(f"Unsupported parser type: {parser_type}")
    
    def generate(self, prompt: Any, **kwargs) -> GenResponse:
        if not isinstance(prompt, str):
            raise TypeError("GenericProvider only supports string prompts")
        """Sync generation"""
        try:
            request_data = self._prepare_request(prompt, **kwargs)
            raw_response = self.transport.send(request_data, **kwargs)
            content = self.parser.parse_response(raw_response)
            content_type = self.parser.get_content_type(raw_response)
            
            return GenResponse(
                content=content,
                provider=self.name,
                model=self.model,
                timestamp=datetime.now(UTC).isoformat(),
                content_type=content_type,
                metadata=self._extract_metadata(raw_response)
            )
            
        except Exception as e:
            raise APIError("Generation failed") from e
    
    async def agenerate(self, prompt: Any, **kwargs) -> GenResponse:
        """Async generation"""
        if not isinstance(prompt, str):
            raise TypeError("GenericProvider only supports string prompts")
        try:
            request_data = self._prepare_request(prompt, **kwargs)
            raw_response = await self.transport.asend(request_data, **kwargs)
            content = self.parser.parse_response(raw_response)
            content_type = self.parser.get_content_type(raw_response)
            
            return GenResponse(
                content=content,
                provider=self.name,
                model=self.model,
                timestamp=datetime.now(UTC).isoformat(),
                content_type=content_type,
                metadata=self._extract_metadata(raw_response)
            )
            
        except Exception as e:
            raise APIError("Async generation failed") from e
    
    def generate_stream(self, prompt: Any, **kwargs) -> Generator[GenStreamChunk, None, None]:
        """Sync streaming generation"""
        if not isinstance(prompt, str):
            raise TypeError("GenericProvider only supports string prompts")
        try:
            request_data = self._prepare_request(prompt, **kwargs)
            
            for raw_chunk in self.transport.send_stream(request_data, **kwargs):
                content = self.parser.parse_stream_chunk(raw_chunk)
                
                if content is not None:
                    # For streaming, we use a default content type or infer from parser
                    content_type = getattr(self.parser, 'content_type', 'text/plain')
                    
                    yield GenStreamChunk(
                        content=content,
                        provider=self.name,
                        model=self.model,
                        timestamp=datetime.now(UTC).isoformat(),
                        content_type=content_type,
                        metadata=self._extract_chunk_metadata(raw_chunk),
                        is_final=False
                    )
            
            # Send final chunk
            yield GenStreamChunk(
                content=None,
                provider=self.name,
                model=self.model,
                timestamp=datetime.now(UTC).isoformat(),
                content_type=content_type,
                is_final=True
            )
            
        except Exception as e:
            raise APIError("Streaming generation failed") from e
    
    async def agenerate_stream(self, prompt: Any, **kwargs) -> AsyncGenerator[GenStreamChunk, None]:
        """Async streaming generation with true streaming support"""
        if not isinstance(prompt, str):
            raise TypeError("GenericProvider only supports string prompts")
        try:
            request_data = self._prepare_request(prompt, **kwargs)
            
            async for raw_chunk in self.transport.asend_stream(request_data, **kwargs):
                content = self.parser.parse_stream_chunk(raw_chunk)
                
                if content is not None:
                    # For streaming, we use a default content type or infer from parser
                    content_type = getattr(self.parser, 'content_type', 'text/plain')
                    
                    yield GenStreamChunk(
                        content=content,
                        provider=self.name,
                        model=self.model,
                        timestamp=datetime.now(UTC).isoformat(),
                        content_type=content_type,
                        metadata=self._extract_chunk_metadata(raw_chunk),
                        is_final=False
                    )
            
            # Send final chunk
            yield GenStreamChunk(
                content=None,
                provider=self.name,
                model=self.model,
                timestamp=datetime.now(UTC).isoformat(),
                content_type=content_type,
                is_final=True
            )
            
        except Exception as e:
            raise APIError("Async streaming generation failed") from e
    
    def _prepare_request(self, prompt: Any, **kwargs) -> Dict[str, Any]:
        """Prepare request with template substitution"""
        if not isinstance(prompt, str):
            raise TypeError("GenericProvider only supports string prompts")
        request_data = json.loads(json.dumps(self.request_template))
        
        substitutions = {
            'prompt': prompt,
            'model': self.model,
            **{k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))}
        }
        
        return self._substitute_template(request_data, substitutions)
    
    def _substitute_template(self, data: Any, subs: Dict[str, Any]) -> Any:
        """Recursive template substitution"""
        if isinstance(data, dict):
            return {k: self._substitute_template(v, subs) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._substitute_template(item, subs) for item in data]
        elif isinstance(data, str):
            for key, value in subs.items():
                data = data.replace(f'{{{{{key}}}}}', str(value))
            return data
        return data
    
    def _extract_metadata(self, raw_response: Any) -> Dict[str, Any]:
        """Extract metadata from raw response"""
        metadata = {}
        
        if hasattr(raw_response, 'status_code'):
            metadata['status_code'] = raw_response.status_code
        elif hasattr(raw_response, 'status'):
            metadata['status_code'] = raw_response.status
        
        if hasattr(raw_response, 'headers'):
            metadata['headers'] = dict(raw_response.headers)
        
        return metadata
    
    def _extract_chunk_metadata(self, raw_chunk: Any) -> Dict[str, Any]:
        """Extract metadata from raw streaming chunk"""
        return {'chunk_type': type(raw_chunk).__name__}
    
    def validate(self) -> bool:
        """Validate configuration"""
        required = ['request_template']
        return all(field in self.config for field in required)


# Register the enhanced generic provider
ProviderRegistry.register('generic', GenericProvider)


# ============================================================================
# Public API
# ============================================================================


__all__ = ['GenericProvider', 'HTTPTransport', 'WebSocketTransport',
           'JSONResponseParser', 'PlainTextResponseParser', 'BinaryResponseParser',
           'CustomResponseParser', 'GenProviderError', 'ConfigurationError', 'APIError', 'GenManager']


# ============================================================================
# Minimal CLI for testing
# ============================================================================

if __name__ == "__main__":
    # Minimal CLI for testing core functionality
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python agnosmodel.py <command>")
        print("Commands: list-types, test-generic, test-streaming, test-multi-model, test-binary")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "list-types":
        print("Registered provider types:")
        for provider_type in ProviderRegistry.list_types():
            print(f"  {provider_type}")
    
    elif command == "test-generic":
        # Test with a mock HTTP provider
        manager = GenManager()
        
        config = {
            'transport_type': 'http',
            'endpoint': 'https://httpbin.org/post',
            'request_template': {'prompt': '{{prompt}}'},
            'parser_type': 'json',
            'response_path': ['json', 'prompt']
        }
        
        manager.add_provider('test', 'generic', config)
        print(f"Current provider: {manager.get_current_provider()}")
        print(f"Available providers: {manager.list_providers()}")
        
        try:
            response = manager.generate("Hello, world!")
            print(f"Response: {response.content}")
            print(f"Content type: {response.content_type}")
        except Exception as e:
            print(f"Test failed: {e}")
    
    elif command == "test-streaming":
        # Test streaming functionality
        import asyncio
        
        async def test_stream():
            manager = GenManager()
            
            config = {
                'transport_type': 'http',
                'endpoint': 'https://httpbin.org/stream/5',
                'request_template': {'prompt': '{{prompt}}'},
                'parser_type': 'text'
            }
            
            manager.add_provider('stream_test', 'generic', config)
            
            print("Testing async streaming...")
            try:
                async for chunk in manager.agenerate_stream("test prompt"):
                    print(f"Chunk: {chunk.content[:50]}...")
                    print(f"Content type: {chunk.content_type}")
                    if chunk.is_final:
                        print("Stream complete!")
                        break
            except Exception as e:
                print(f"Streaming test failed: {e}")
        
        asyncio.run(test_stream())
    
    elif command == "test-multi-model":
        # Test multi-model support with different content types
        manager = GenManager()
        
        # Text model
        text_config = {
            'transport_type': 'http',
            'endpoint': 'https://httpbin.org/post',
            'request_template': {'model': 'text-model', 'prompt': '{{prompt}}'},
            'parser_type': 'json',
            'response_path': ['json', 'prompt']
        }
        
        # Binary model (simulated)
        binary_config = {
            'transport_type': 'http',
            'endpoint': 'https://httpbin.org/bytes/1024',
            'request_template': {'model': 'image-model', 'prompt': '{{prompt}}'},
            'parser_type': 'binary',
            'content_type': 'image/png'
        }
        
        # Custom parser model
        def custom_parse(response):
            return f"Custom parsed: {getattr(response, 'text', str(response))}"
        
        custom_config = {
            'transport_type': 'http',
            'endpoint': 'https://httpbin.org/json',
            'request_template': {'model': 'custom-model', 'prompt': '{{prompt}}'},
            'parser_type': 'custom',
            'parse_func': custom_parse,
            'content_type': 'application/json'
        }
        
        # Register all models
        manager.add_provider('text_model', 'generic', text_config)
        manager.add_provider('binary_model', 'generic', binary_config)
        manager.add_provider('custom_model', 'generic', custom_config)
        
        print(f"Available providers: {manager.list_providers()}")
        
        # Test each model type
        for provider_name in ['text_model', 'binary_model', 'custom_model']:
            try:
                manager.switch_provider(provider_name)
                print(f"\nTesting {provider_name}...")
                
                response = manager.generate("test prompt")
                print(f"Provider: {response.provider}")
                print(f"Content type: {response.content_type}")
                print(f"Content preview: {str(response.content)[:100]}...")
                
            except Exception as e:
                print(f"Test failed for {provider_name}: {e}")
    
    elif command == "test-binary":
        # Focused test for binary content handling
        manager = GenManager()
        
        config = {
            'transport_type': 'http',
            'endpoint': 'https://httpbin.org/bytes/256',
            'method': 'GET',
            'request_template': {},
            'parser_type': 'binary',
            'content_type': 'application/octet-stream'
        }
        
        manager.add_provider('binary_test', 'generic', config)
        
        try:
            print("Testing binary content...")
            response = manager.generate("")  # Empty prompt for GET request
            
            print(f"Content type: {response.content_type}")
            print(f"Content is bytes: {isinstance(response.content, bytes)}")
            print(f"Content length: {len(response.content) if hasattr(response.content, '__len__') else 'unknown'}")
            print(f"First 16 bytes (hex): {response.content[:16].hex() if isinstance(response.content, bytes) else 'N/A'}")
            
        except Exception as e:
            print(f"Binary test failed: {e}")
    
    else:
        print(f"Unknown command: {command}")
        print("Available commands: list-types, test-generic, test-streaming, test-multi-model, test-binary")
        sys.exit(1)