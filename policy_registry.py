"""Data-driven registration of curated policies; adding one does not change routing code."""

from dataclasses import dataclass
import json
from pathlib import Path
import re


REGISTRY_PATH = Path(__file__).parent / "policy_data/registry.json"


@dataclass(frozen=True)
class RegisteredPolicy:
    id: str
    label: str
    scope: str
    path: Path
    include: str
    exclude: str
    route: str

    def matches(self, query):
        return bool(re.search(self.include, query, re.I)) and not bool(
            self.exclude and re.search(self.exclude, query, re.I)
        )


def registered_policies(path=REGISTRY_PATH):
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    policies = []
    for entry in raw["policies"]:
        policy = RegisteredPolicy(**{**entry, "path": path.parent / entry["path"]})
        re.compile(policy.include)
        re.compile(policy.exclude)
        if any(p.id == policy.id or p.label == policy.label for p in policies):
            raise ValueError("Duplicate registered policy")
        policies.append(policy)
    return policies


def select_policy(query, mode="auto", policies=None):
    policies = registered_policies() if policies is None else policies
    if mode == "hybrid":
        return None
    if mode == "auto":
        return next((p for p in policies if p.matches(query)), None)
    return next((p for p in policies if p.id == mode), None)


def mode_for_label(label):
    if label == "Automatic":
        return "auto"
    if label == "Standard search":
        return "hybrid"
    for policy in registered_policies():
        if policy.label == label:
            return policy.id
    raise ValueError("Unknown policy selection")
