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


from .agnosmodel2 import GenResponse, GenStreamChunk, AgnosmodelError
from .agnosmodel2 import BaseGenProvider, BaseModelTransport, BaseResponseParser
from .agnosmodel2 import ProviderRegistry, GenManager

__version__ = "0.0.1a1"
__all__ = [ 'GenResponse', 'GenStreamChunk', 'AgnosmodelError',
            'BaseGenProvider', 'BaseModelTransport', 'BaseResponseParser',
            'ProviderRegistry', 'GenManager'
]