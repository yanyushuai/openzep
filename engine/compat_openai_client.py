import copy
import json
import logging
import os
import re
import time
from typing import get_origin
from typing import Any

import openai
from pydantic import BaseModel

from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, ModelSize
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.llm_client.openai_generic_client import (
    DEFAULT_MODEL,
    OpenAIGenericClient,
)
from graphiti_core.prompts.models import Message

logger = logging.getLogger(__name__)

# Qwen3-family reasoning models burn thousands of hidden chain-of-thought
# tokens per structured call (~5x latency). Gateways backed by vLLM/SGLang
# accept chat_template_kwargs to disable thinking; measured on this stack:
# 13.3s -> 2.5s per extraction call with identical JSON output.
_DISABLE_THINKING = os.getenv("LLM_DISABLE_THINKING", "").strip().lower() in {
    "1",
    "true",
    "yes",
}


_SCHEMA_ERROR_STRONG_TOKENS = (
    "json_schema",
    "response_format",
    "additional_properties",
    "additionalproperties",
    "structured output",
    "structured_output",
)
# "strict" alone is too generic (appears in many unrelated errors), so it only
# counts as a signal when a structured-output-specific token is also present.
_SCHEMA_ERROR_WEAK_TOKENS = ("strict",)
_SCHEMA_ERROR_CONTEXT_TOKENS = ("schema", "response", "json", "structured")


def _looks_like_schema_error(exc: Exception) -> bool:
    """True when an LLM API error indicates the structured-output schema was rejected.

    Requires a specific token (json_schema / response_format / additional_properties).
    Bare "strict" is not enough on its own — too many unrelated errors mention it.
    """
    message = str(exc).lower()
    if any(token in message for token in _SCHEMA_ERROR_STRONG_TOKENS):
        return True
    return any(token in message for token in _SCHEMA_ERROR_WEAK_TOKENS) and any(
        token in message for token in ("schema", "response", "json")
    )


class CompatOpenAIGenericClient(OpenAIGenericClient):
    """OpenAI-compatible client with tolerant JSON extraction for loose proxies."""

    @staticmethod
    def _normalize_strict_schema(schema: dict) -> dict:
        """Recursively normalize a JSON schema to be OpenAI strict-compatible.

        For every object node: sets additionalProperties=False and populates
        required with all property keys. Does not mutate the input.
        """
        schema = copy.deepcopy(schema)

        def _walk(node: Any) -> Any:
            if not isinstance(node, dict):
                return node

            # Recurse into $defs / definitions first so nested types are normalized
            for defs_key in ("$defs", "definitions"):
                if defs_key in node:
                    node[defs_key] = {k: _walk(v) for k, v in node[defs_key].items()}

            # Recurse into allOf / anyOf / oneOf entries
            for combiner in ("allOf", "anyOf", "oneOf"):
                if combiner in node:
                    node[combiner] = [_walk(entry) for entry in node[combiner]]

            # Recurse into array items
            if "items" in node:
                node["items"] = _walk(node["items"])

            # Recurse into properties values and enforce object constraints
            if "properties" in node or node.get("type") == "object":
                if "properties" in node:
                    node["properties"] = {k: _walk(v) for k, v in node["properties"].items()}
                    props = list(node["properties"].keys())
                    # Preserve order, dedupe
                    seen: set[str] = set()
                    unique_props = []
                    for p in props:
                        if p not in seen:
                            seen.add(p)
                            unique_props.append(p)
                    node["required"] = unique_props

                # `additionalProperties` as a schema object signals a map/dict
                # field (dict[str, X]). Such nodes must keep their value-type
                # schema — clobbering it to False would change the type's meaning
                # from "any key -> X" to "no extra keys allowed". Only force the
                # bool closure for fixed-property objects.
                if not isinstance(node.get("additionalProperties"), dict):
                    node["additionalProperties"] = False

            return node

        return _walk(schema)

    @staticmethod
    def _is_list_field(response_model: type[BaseModel], field_name: str) -> bool:
        field = response_model.model_fields[field_name]
        return get_origin(field.annotation) is list

    @staticmethod
    def _extract_json_text(raw: str) -> str:
        text = (raw or "").strip()
        if not text:
            return text

        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            if text.lower().startswith("json"):
                text = text[4:].lstrip()

        decoder = json.JSONDecoder()
        for idx, ch in enumerate(text):
            if ch not in "{[":
                continue
            try:
                obj, end = decoder.raw_decode(text[idx:])
                return json.dumps(obj, ensure_ascii=False)
            except json.JSONDecodeError:
                continue

        return text

    @staticmethod
    def _normalize_payload(
        payload: Any,
        response_model: type[BaseModel] | None,
        messages: list[Message],
    ) -> Any:
        if response_model is None:
            return payload

        field_names = list(response_model.model_fields.keys())
        if len(field_names) == 1:
            field_name = field_names[0]
            expects_list = CompatOpenAIGenericClient._is_list_field(response_model, field_name)

            if isinstance(payload, list):
                payload = {field_name: payload}
            elif expects_list and isinstance(payload, dict):
                if field_name not in payload:
                    payload = {field_name: [payload]}
                elif isinstance(payload[field_name], dict):
                    payload = {**payload, field_name: [payload[field_name]]}

        if (
            response_model.__name__ == "ExtractedEntities"
            and isinstance(payload, dict)
            and isinstance(payload.get("extracted_entities"), list)
        ):
            type_map = CompatOpenAIGenericClient._extract_entity_type_map(messages)
            normalized_entities = []
            for item in payload["extracted_entities"]:
                if not isinstance(item, dict):
                    normalized_entities.append(item)
                    continue

                normalized = dict(item)
                if "name" not in normalized and "entity_name" in normalized:
                    normalized["name"] = normalized.pop("entity_name")

                if "entity_type_id" not in normalized and "entity_type_name" in normalized:
                    normalized["entity_type_id"] = type_map.get(str(normalized["entity_type_name"]), 0)

                normalized_entities.append(normalized)

            payload["extracted_entities"] = normalized_entities

        return payload

    @staticmethod
    def _extract_entity_type_map(messages: list[Message]) -> dict[str, int]:
        pattern = re.compile(r"<ENTITY TYPES>\s*(.*?)\s*</ENTITY TYPES>", re.DOTALL)
        for message in messages:
            match = pattern.search(message.content)
            if not match:
                continue
            block = match.group(1).strip()
            try:
                data = json.loads(block)
            except json.JSONDecodeError:
                continue

            mapping = {}
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("entity_type_name")
                    entity_type_id = item.get("entity_type_id")
                    if isinstance(name, str) and isinstance(entity_type_id, int):
                        mapping[name] = entity_type_id
            return mapping

        return {}

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        openai_messages = []
        for message in messages:
            message.content = self._clean_input(message.content)
            if message.role in {"user", "system"}:
                openai_messages.append({"role": message.role, "content": message.content})

        strict_schema: dict[str, Any] | None = None
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        thinking_kwargs = (
            {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
            if _DISABLE_THINKING
            else {}
        )
        started = time.monotonic()
        try:
            response_format: dict[str, Any] = {"type": "json_object"}
            if response_model is not None:
                strict_schema = self._normalize_strict_schema(response_model.model_json_schema())
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": getattr(response_model, "__name__", "structured_response"),
                        "schema": strict_schema,
                        "strict": True,
                    },
                }

            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=effective_max_tokens,
                response_format=response_format,  # type: ignore[arg-type]
                **thinking_kwargs,
            )
        except openai.RateLimitError as exc:
            raise RateLimitError from exc
        except openai.BadRequestError as exc:
            if not _looks_like_schema_error(exc) or response_model is None:
                raise
            logger.warning(
                "Provider rejected strict json_schema, retrying with json_object: %s", exc
            )
            fallback_messages = list(openai_messages)
            if fallback_messages and fallback_messages[-1].get("role") == "user":
                hint = (
                    "\n\nYou MUST respond with a valid JSON object matching this schema exactly: "
                    f"{json.dumps(strict_schema, ensure_ascii=False)}"
                )
                fallback_messages[-1] = dict(fallback_messages[-1])
                fallback_messages[-1]["content"] = str(fallback_messages[-1]["content"]) + hint
            else:
                fallback_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Respond with a valid JSON object matching this schema exactly: "
                            f"{json.dumps(strict_schema, ensure_ascii=False)}"
                        ),
                    }
                )
            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=fallback_messages,
                temperature=self.temperature,
                max_tokens=effective_max_tokens,
                response_format={"type": "json_object"},  # type: ignore[arg-type]
                **thinking_kwargs,
            )

        usage = getattr(response, "usage", None)
        logger.info(
            "llm call: model=%s size=%s %.0fms prompt_tokens=%s completion_tokens=%s",
            self.model or DEFAULT_MODEL,
            getattr(model_size, "value", model_size),
            (time.monotonic() - started) * 1000,
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )

        raw_content = response.choices[0].message.content or ""
        normalized = self._extract_json_text(raw_content)
        if not normalized:
            raise json.JSONDecodeError("Empty content", raw_content, 0)

        parsed = json.loads(normalized)
        return self._normalize_payload(parsed, response_model, messages)
