import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


DOCUMENTATION_DIR = Path(__file__).resolve().parent / "capability_docs"
_REGISTRY = {}


@dataclass(frozen=True)
class Capability:
    id: str
    name: str
    description: str
    purpose: str
    activation_conditions: tuple
    inputs: tuple
    outputs: tuple
    dependencies: tuple
    limitations: tuple
    documentation: str
    availability: object

    def summary(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": "available" if self.availability() else "unavailable",
        }

    def metadata(self):
        return {
            **self.summary(),
            "purpose": self.purpose,
            "activation_conditions": list(self.activation_conditions),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "dependencies": list(self.dependencies),
            "limitations": list(self.limitations),
            "documentation": self.documentation,
        }


def register_capability(
    *,
    id,
    name,
    description,
    purpose,
    activation_conditions,
    inputs,
    outputs,
    dependencies,
    limitations,
    documentation,
    availability=lambda: True,
):
    capability = Capability(
        id=id,
        name=name,
        description=description,
        purpose=purpose,
        activation_conditions=tuple(activation_conditions),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        dependencies=tuple(dependencies),
        limitations=tuple(limitations),
        documentation=documentation,
        availability=availability,
    )
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", id):
        raise ValueError(f"Invalid capability id: {id}")
    if id in _REGISTRY:
        raise ValueError(f"Capability is already registered: {id}")
    documentation_path = (DOCUMENTATION_DIR / documentation).resolve()
    if (
        documentation_path.parent != DOCUMENTATION_DIR.resolve()
        or not documentation_path.is_file()
    ):
        raise ValueError(f"Missing or invalid capability documentation: {documentation}")
    if not callable(availability):
        raise TypeError("Capability availability must be callable")

    def decorate(implementation):
        _REGISTRY[id] = capability
        implementation.capability_id = id
        return implementation

    return decorate


def list_capabilities():
    return [capability.summary() for capability in _REGISTRY.values()]


def discover_capabilities(query):
    normalized_query = unicodedata.normalize("NFKC", query).casefold()
    matches = []
    for capability in _REGISTRY.values():
        if not capability.availability():
            continue
        matching_conditions = [
            condition
            for condition in capability.activation_conditions
            if unicodedata.normalize("NFKC", condition).casefold()
            in normalized_query
        ]
        if capability.id == "local_chat":
            matching_conditions.append("")
        if matching_conditions:
            matches.append((len(matching_conditions), capability))
    matches.sort(key=lambda result: result[0], reverse=True)
    return [
        capability.summary()
        for _score, capability in matches[:5]
    ]


def get_capability_documentation(capability_id):
    capability = _REGISTRY.get(capability_id)
    if not capability:
        return None
    if not capability.availability():
        return None
    documentation_path = (DOCUMENTATION_DIR / capability.documentation).resolve()
    if documentation_path.parent != DOCUMENTATION_DIR.resolve():
        raise ValueError(f"Invalid documentation path for {capability_id}")
    return {
        "capability": capability.metadata(),
        "documentation": documentation_path.read_text(encoding="utf-8"),
    }
