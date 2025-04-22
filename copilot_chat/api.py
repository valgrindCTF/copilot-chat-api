"""
Reverse-engineered Github Copilot Chat API client for Python.
"""

import requests
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import List, Any, Optional, TypeVar, Callable, Type, cast, Literal, Generator, AsyncGenerator
import json
import uuid
import aiohttp


T = TypeVar("T")
EnumT = TypeVar("EnumT", bound=Enum)


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def to_enum(c: Type[EnumT], x: Any) -> EnumT:
    assert isinstance(x, c)
    return x.value


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except Exception:
            pass
    assert False


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


class SupportedMediaType(Enum):
    IMAGE_GIF = "image/gif"
    IMAGE_HEIC = "image/heic"
    IMAGE_HEIF = "image/heif"
    IMAGE_JPEG = "image/jpeg"
    IMAGE_PNG = "image/png"
    IMAGE_WEBP = "image/webp"


@dataclass
class Vision:
    max_prompt_image_size: int
    max_prompt_images: int
    supported_media_types: List[SupportedMediaType]

    @staticmethod
    def from_dict(obj: Any) -> "Vision":
        assert isinstance(obj, dict)
        max_prompt_image_size = from_int(obj.get("max_prompt_image_size"))
        max_prompt_images = from_int(obj.get("max_prompt_images"))
        supported_media_types = from_list(
            SupportedMediaType, obj.get("supported_media_types")
        )
        return Vision(max_prompt_image_size, max_prompt_images, supported_media_types)

    def to_dict(self) -> dict:
        result: dict = {}
        result["max_prompt_image_size"] = from_int(self.max_prompt_image_size)
        result["max_prompt_images"] = from_int(self.max_prompt_images)
        result["supported_media_types"] = from_list(
            lambda x: to_enum(SupportedMediaType, x), self.supported_media_types
        )
        return result


@dataclass
class Limits:
    max_context_window_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_prompt_tokens: Optional[int] = None
    vision: Optional[Vision] = None
    max_inputs: Optional[int] = None

    @staticmethod
    def from_dict(obj: Any) -> "Limits":
        assert isinstance(obj, dict)
        max_context_window_tokens = from_union(
            [from_int, from_none], obj.get("max_context_window_tokens")
        )
        max_output_tokens = from_union(
            [from_int, from_none], obj.get("max_output_tokens")
        )
        max_prompt_tokens = from_union(
            [from_int, from_none], obj.get("max_prompt_tokens")
        )
        vision = from_union([Vision.from_dict, from_none], obj.get("vision"))
        max_inputs = from_union([from_int, from_none], obj.get("max_inputs"))
        return Limits(
            max_context_window_tokens,
            max_output_tokens,
            max_prompt_tokens,
            vision,
            max_inputs,
        )

    def to_dict(self) -> dict:
        result: dict = {}
        if self.max_context_window_tokens is not None:
            result["max_context_window_tokens"] = from_union(
                [from_int, from_none], self.max_context_window_tokens
            )
        if self.max_output_tokens is not None:
            result["max_output_tokens"] = from_union(
                [from_int, from_none], self.max_output_tokens
            )
        if self.max_prompt_tokens is not None:
            result["max_prompt_tokens"] = from_union(
                [from_int, from_none], self.max_prompt_tokens
            )
        if self.vision is not None:
            result["vision"] = from_union(
                [lambda x: to_class(Vision, x), from_none], self.vision
            )
        if self.max_inputs is not None:
            result["max_inputs"] = from_union([from_int, from_none], self.max_inputs)
        return result


class CapabilitiesObject(Enum):
    MODEL_CAPABILITIES = "model_capabilities"


@dataclass
class Supports:
    streaming: Optional[bool] = None
    tool_calls: Optional[bool] = None
    parallel_tool_calls: Optional[bool] = None
    vision: Optional[bool] = None
    structured_outputs: Optional[bool] = None
    dimensions: Optional[bool] = None

    @staticmethod
    def from_dict(obj: Any) -> "Supports":
        assert isinstance(obj, dict)
        streaming = from_union([from_bool, from_none], obj.get("streaming"))
        tool_calls = from_union([from_bool, from_none], obj.get("tool_calls"))
        parallel_tool_calls = from_union(
            [from_bool, from_none], obj.get("parallel_tool_calls")
        )
        vision = from_union([from_bool, from_none], obj.get("vision"))
        structured_outputs = from_union(
            [from_bool, from_none], obj.get("structured_outputs")
        )
        dimensions = from_union([from_bool, from_none], obj.get("dimensions"))
        return Supports(
            streaming,
            tool_calls,
            parallel_tool_calls,
            vision,
            structured_outputs,
            dimensions,
        )

    def to_dict(self) -> dict:
        result: dict = {}
        if self.streaming is not None:
            result["streaming"] = from_union([from_bool, from_none], self.streaming)
        if self.tool_calls is not None:
            result["tool_calls"] = from_union([from_bool, from_none], self.tool_calls)
        if self.parallel_tool_calls is not None:
            result["parallel_tool_calls"] = from_union(
                [from_bool, from_none], self.parallel_tool_calls
            )
        if self.vision is not None:
            result["vision"] = from_union([from_bool, from_none], self.vision)
        if self.structured_outputs is not None:
            result["structured_outputs"] = from_union(
                [from_bool, from_none], self.structured_outputs
            )
        if self.dimensions is not None:
            result["dimensions"] = from_union([from_bool, from_none], self.dimensions)
        return result


class Tokenizer(Enum):
    CL100_K_BASE = "cl100k_base"
    O200_K_BASE = "o200k_base"


class TypeEnum(Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDINGS = "embeddings"


@dataclass
class Capabilities:
    family: str
    object: CapabilitiesObject
    supports: Supports
    tokenizer: Tokenizer
    type: TypeEnum
    limits: Optional[Limits] = None

    @staticmethod
    def from_dict(obj: Any) -> "Capabilities":
        assert isinstance(obj, dict)
        family = from_str(obj.get("family"))
        object = CapabilitiesObject(obj.get("object"))
        supports = Supports.from_dict(obj.get("supports"))
        tokenizer = Tokenizer(obj.get("tokenizer"))
        type = TypeEnum(obj.get("type"))
        limits = from_union([Limits.from_dict, from_none], obj.get("limits"))
        return Capabilities(family, object, supports, tokenizer, type, limits)

    def to_dict(self) -> dict:
        result: dict = {}
        result["family"] = from_str(self.family)
        result["object"] = to_enum(CapabilitiesObject, self.object)
        result["supports"] = to_class(Supports, self.supports)
        result["tokenizer"] = to_enum(Tokenizer, self.tokenizer)
        result["type"] = to_enum(TypeEnum, self.type)
        if self.limits is not None:
            result["limits"] = from_union(
                [lambda x: to_class(Limits, x), from_none], self.limits
            )
        return result


class State(Enum):
    ENABLED = "enabled"
    UNCONFIGURED = "unconfigured"


@dataclass
class Policy:
    state: State
    terms: str

    @staticmethod
    def from_dict(obj: Any) -> "Policy":
        assert isinstance(obj, dict)
        state = State(obj.get("state"))
        terms = from_str(obj.get("terms"))
        return Policy(state, terms)

    def to_dict(self) -> dict:
        result: dict = {}
        result["state"] = to_enum(State, self.state)
        result["terms"] = from_str(self.terms)
        return result


class Vendor(Enum):
    ANTHROPIC = "Anthropic"
    AZURE_OPEN_AI = "Azure OpenAI"
    GOOGLE = "Google"
    OPEN_AI = "OpenAI"


@dataclass
class Model:
    capabilities: Capabilities
    id: str
    model_picker_enabled: bool
    name: str
    object: Literal["model"]
    preview: bool
    vendor: Vendor
    version: str
    policy: Optional[Policy] = None

    @staticmethod
    def from_dict(obj: Any) -> "Model":
        assert isinstance(obj, dict)
        capabilities = Capabilities.from_dict(obj.get("capabilities"))
        id = from_str(obj.get("id"))
        model_picker_enabled = from_bool(obj.get("model_picker_enabled"))
        name = from_str(obj.get("name"))
        object = obj.get("object")
        preview = from_bool(obj.get("preview"))
        vendor = Vendor(obj.get("vendor"))
        version = from_str(obj.get("version"))
        policy = from_union([Policy.from_dict, from_none], obj.get("policy"))
        return Model(
            capabilities,
            id,
            model_picker_enabled,
            name,
            object,
            preview,
            vendor,
            version,
            policy,
        )

    @staticmethod
    def from_list(obj: Any) -> List["Model"]:
        assert isinstance(obj, list)
        return [Model.from_dict(x) for x in obj]

    def to_dict(self) -> dict:
        result: dict = {}
        result["capabilities"] = to_class(Capabilities, self.capabilities)
        result["id"] = from_str(self.id)
        result["model_picker_enabled"] = from_bool(self.model_picker_enabled)
        result["name"] = from_str(self.name)
        result["object"] = from_str(self.object)
        result["preview"] = from_bool(self.preview)
        result["vendor"] = to_enum(Vendor, self.vendor)
        result["version"] = from_str(self.version)
        if self.policy is not None:
            result["policy"] = from_union(
                [lambda x: to_class(Policy, x), from_none], self.policy
            )
        return result


class CopilotAPI:
    """
    A class to interact with the Copilot Chat API.
    """

    def __init__(self, gh_token: str, vscode_machine: str, vscode_session: str) -> None:
        """
        gh_token should be a valid GitHub oauth token (`gho_***`).
        """
        self.gh_token = gh_token
        self.vscode_machine = vscode_machine
        self.vscode_session = vscode_session
        self.authorization = None
        self.token_expires_at = None
        self._fetch_auth()

    def _fetch_auth(self) -> None:
        """
        Fetches the authorization token from the GitHub API.
        """
        url = "https://api.github.com/copilot_internal/v2/token"
        payload = {}
        headers = {
            "host": "api.github.com",
            "connection": "keep-alive",
            "authorization": f"token {self.gh_token}",
            "editor-plugin-version": "copilot-chat/0.26.5",
            "editor-version": "vscode/1.99.3",
            "user-agent": "GitHubCopilotChat/0.26.5",
            "x-github-api-version": "2025-04-01",
            "x-vscode-user-agent-library-version": "electron-fetch",
            "sec-fetch-site": "none",
            "sec-fetch-mode": "no-cors",
            "sec-fetch-dest": "empty",
            "accept-encoding": "gzip, deflate, br, zstd",
        }
        r = requests.request("GET", url, headers=headers, data=payload)
        r.raise_for_status()
        self.authorization = r.json().get("token")
        if not self.authorization:
            raise ValueError("Failed to fetch authorization token.")
        self.token_expires_at = datetime.fromtimestamp(r.json().get("expires_at"))
        print(f"Token expires at: {self.token_expires_at}")

    def _ensure_valid_token(self) -> None:
        """
        Ensures the token is valid and refreshes it if expired.
        """
        if not self.token_expires_at or datetime.now() >= self.token_expires_at:
            print("Token expired or not set. Fetching a new token...")
            self._fetch_auth()

    def models(self) -> List[Model]:
        """
        Fetches the models available in the Copilot Chat API.
        """
        self._ensure_valid_token()
        url = "https://api.individual.githubcopilot.com/models"

        payload = {}
        headers = {
            "host": "api.individual.githubcopilot.com",
            "connection": "keep-alive",
            "authorization": f"Bearer {self.authorization}",
            "copilot-integration-id": "vscode-chat",
            "editor-plugin-version": "copilot-chat/0.26.5",
            "editor-version": "vscode/1.99.3",
            "openai-intent": "model-access",
            "user-agent": "GitHubCopilotChat/0.26.5",
            "vscode-machineid": self.vscode_machine,
            "vscode-sessionid": self.vscode_session,
            "x-github-api-version": "2025-04-01",
            "x-interaction-type": "model-access",
            "x-vscode-user-agent-library-version": "electron-fetch",
            "x-request-id": str(uuid.uuid4()),
            "sec-fetch-site": "none",
            "sec-fetch-mode": "no-cors",
            "sec-fetch-dest": "empty",
            "accept-encoding": "gzip, deflate, br, zstd",
        }

        r = requests.request("GET", url, headers=headers, data=payload)
        r.raise_for_status()
        return Model.from_list(r.json()["data"])

    def chat(self, body: dict) -> Generator[str, None, None]:
        """
        Send a chat request and stream the response.
        
        Returns a generator that yields chunks of the response as they arrive.
        Each chunk contains a delta with the incremental content.
        """
        self._ensure_valid_token()
        url = "https://api.individual.githubcopilot.com/chat/completions"
        body["stream"] = True 
        payload = json.dumps(body)
        headers = {
            "host": "api.individual.githubcopilot.com",
            "connection": "keep-alive",
            "authorization": f"Bearer {self.authorization}",
            "content-type": "application/json",
            "copilot-integration-id": "vscode-chat",
            "editor-plugin-version": "copilot-chat/0.26.5",
            "editor-version": "vscode/1.99.3",
            "openai-intent": "conversation-panel",
            "user-agent": "GitHubCopilotChat/0.26.5",
            "vscode-machineid": self.vscode_machine,
            "vscode-sessionid": self.vscode_session,
            "x-github-api-version": "2025-04-01",
            "x-initiator": "agent",
            "x-interaction-id": str(uuid.uuid4()),
            "x-interaction-type": "conversation-panel",
            "x-request-id": str(uuid.uuid4()),
            "x-vscode-user-agent-library-version": "electron-fetch",
            "sec-fetch-site": "none",
            "sec-fetch-mode": "no-cors",
            "sec-fetch-dest": "empty",
            "accept-encoding": "gzip, deflate, br, zstd",
        }
    
        with requests.request("POST", url, headers=headers, data=payload, stream=True) as r:
            r.raise_for_status()
            
            for line in r.iter_lines():
                if not line:
                    continue
                    
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    line = line[6:]
                    
                if line == '[DONE]':
                    break
                    
                try:
                    chunk = json.loads(line)
                    if len(chunk['choices']) == 0:
                        continue
                    yield chunk['choices'][0].get("delta", {}).get("content", "")
                except json.JSONDecodeError:
                    print(f"Error parsing JSON: {line}")
                    continue

    def generate(self, body: dict) -> str:
        res = ""
        for chunk in self.chat(body):
            if not chunk:
                continue
            res += chunk
        return res

    async def async_chat(self, body: dict) -> AsyncGenerator[str, None]:
        """
        Send a chat request and stream the response asynchronously.
        
        Returns an async generator that yields chunks of the response as they arrive.
        Each chunk contains a delta with the incremental content.
        """
        self._ensure_valid_token()
        url = "https://api.individual.githubcopilot.com/chat/completions"
        body["stream"] = True 
        payload = json.dumps(body)
        headers = {
            "host": "api.individual.githubcopilot.com",
            "connection": "keep-alive",
            "authorization": f"Bearer {self.authorization}",
            "content-type": "application/json",
            "copilot-integration-id": "vscode-chat",
            "editor-plugin-version": "copilot-chat/0.26.5",
            "editor-version": "vscode/1.99.3",
            "openai-intent": "conversation-panel",
            "user-agent": "GitHubCopilotChat/0.26.5",
            "vscode-machineid": self.vscode_machine,
            "vscode-sessionid": self.vscode_session,
            "x-github-api-version": "2025-04-01",
            "x-initiator": "agent",
            "x-interaction-id": str(uuid.uuid4()),
            "x-interaction-type": "conversation-panel",
            "x-request-id": str(uuid.uuid4()),
            "x-vscode-user-agent-library-version": "electron-fetch",
            "sec-fetch-site": "none",
            "sec-fetch-mode": "no-cors",
            "sec-fetch-dest": "empty",
            "accept-encoding": "gzip, deflate, br, zstd",
        }
    
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=payload) as response:
                response.raise_for_status()
                
                buffer = ""
                async for chunk in response.content.iter_any():
                    buffer += chunk.decode('utf-8')
                    lines = buffer.split('\n')
                    
                    if len(lines) > 1:
                        for line in lines[:-1]:
                            if not line.strip():
                                continue
                                
                            if line.startswith('data: '):
                                line = line[6:]
                                
                            if line == '[DONE]':
                                return
                                
                            try:
                                data = json.loads(line)
                                if len(data['choices']) == 0:
                                    continue
                                content = data['choices'][0].get("delta", {}).get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                print(f"Error parsing JSON: {line}")
                                continue
                        
                        buffer = lines[-1]

    async def async_generate(self, body: dict) -> str:
        """
        Asynchronously generate a complete response from the Copilot Chat API.
        
        Args:
            body: The request body to send to the API
            
        Returns:
            The complete text response from the API
        """
        res = ""
        async for chunk in self.async_chat(body):
            res += chunk
        return res

