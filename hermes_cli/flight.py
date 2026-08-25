"""Durable, reversible local-model continuity for disconnected operation."""

from __future__ import annotations

import copy
import json
import os
import shutil
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from hermes_constants import get_hermes_home
from utils import atomic_replace

LOCAL_PROVIDER = "flight-local"
# Hermes requires a >=64K context window; stock Ollama tags load at their
# baked num_ctx (often 4096-32768), which stalls or refuses the agent.
# ``ensure_local_model`` creates this derivative with num_ctx 65536.
DEFAULT_LOCAL_MODEL = "hermes-flight"
DEFAULT_LOCAL_BASE = "qwen3-coder:30b"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_RECOVERY_THRESHOLD = 1
DEFAULT_COOLDOWN_SECONDS = 15
NETWORK_MARKERS = (
    "http://",
    "https://",
    "github",
    "gitlab",
    "deploy",
    "production",
    "aws",
    "vercel",
    "email",
    "gmail",
    "slack",
    "telegram",
    "whatsapp",
    "browser",
    "web search",
    "internet",
    "download",
    "upload",
)


def _flight_dir(home: Path) -> Path:
    return home / "flight-mode"


def _state_path(home: Path) -> Path:
    return _flight_dir(home) / "state.json"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    atomic_replace(tmp, path)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    _atomic_write_bytes(path, (json.dumps(data, indent=2, sort_keys=True) + "\n").encode())


def new_state(
    *,
    now: float,
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
    recovery_threshold: int = DEFAULT_RECOVERY_THRESHOLD,
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
) -> dict[str, Any]:
    return {
        "version": 1,
        "mode": "online",
        "normal_failures": 0,
        "normal_successes": 0,
        "failure_threshold": max(1, int(failure_threshold)),
        "recovery_threshold": max(1, int(recovery_threshold)),
        "cooldown_seconds": max(0, int(cooldown_seconds)),
        "created_at": now,
        "last_probe_at": None,
        "last_transition_at": None,
        "last_transition": None,
        "last_evidence": {},
        "saved_restore_target": None,
        "task_overrides": {},
    }


def observe_connectivity(
    state: dict[str, Any],
    *,
    normal_ok: bool,
    local_ok: bool,
    now: float,
) -> Optional[str]:
    """Update hysteresis counters and return ``enter`` / ``exit`` when due."""
    state["last_probe_at"] = now
    mode = state.get("mode", "online")
    if normal_ok:
        state["normal_failures"] = 0
        state["normal_successes"] = int(state.get("normal_successes", 0)) + 1
    else:
        state["normal_successes"] = 0
        state["normal_failures"] = int(state.get("normal_failures", 0)) + 1

    last_transition = state.get("last_transition_at")
    cooled = last_transition is None or now - float(last_transition) >= int(
        state.get("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS)
    )
    if mode == "online":
        if (
            not normal_ok
            and local_ok
            and int(state["normal_failures"]) >= int(state.get("failure_threshold", 3))
            and cooled
        ):
            return "enter"
    elif (
        normal_ok
        and int(state["normal_successes"]) >= int(state.get("recovery_threshold", 2))
        and cooled
    ):
        return "exit"
    return None


def build_local_config(
    config: dict[str, Any],
    *,
    model: str,
    base_url: str,
    max_in_progress: int,
) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result.setdefault("providers", {})[LOCAL_PROVIDER] = {
        "name": "Flight mode local Ollama",
        "base_url": base_url.rstrip("/"),
        "api_key": "ollama",
        "api_mode": "openai_chat_completions",
        "default_model": model,
        "discover_models": True,
    }
    result["model"] = {"provider": LOCAL_PROVIDER, "default": model}
    result["fallback_providers"] = []
    result.pop("fallback_model", None)
    kanban = result.setdefault("kanban", {})
    kanban["max_in_progress"] = max(1, int(max_in_progress))
    kanban["max_in_progress_per_profile"] = 1
    kanban["max_spawn"] = min(max(1, int(max_in_progress)), int(kanban.get("max_spawn") or max_in_progress))
    return result


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _is_local_config(config: dict[str, Any]) -> bool:
    return (config.get("model") or {}).get("provider") == LOCAL_PROVIDER


def _is_degraded_restore(config: dict[str, Any], live: dict[str, Any]) -> bool:
    if _is_local_config(config):
        return True
    return bool(live.get("fallback_providers")) and not config.get("fallback_providers")


def _resolve_api_key(provider: dict[str, Any]) -> str:
    key_env = str(provider.get("key_env") or provider.get("api_key_env") or "").strip()
    if key_env:
        try:
            from hermes_cli.config import get_env_value

            return os.getenv(key_env) or get_env_value(key_env) or ""
        except Exception:
            return os.getenv(key_env, "")
    return str(provider.get("api_key") or "")


def _probe_openai(
    base_url: str,
    model: str,
    api_key: str,
    *,
    timeout: float = 20.0,
) -> tuple[bool, str]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly OK"}],
            "max_tokens": 64,
            "temperature": 0,
        }
    ).encode()
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode())
        choices = payload.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        text = str(message.get("content") or "").strip()
        # Reasoning models can spend the whole budget thinking; generated
        # reasoning tokens still prove live inference.
        reasoning = str(message.get("reasoning") or message.get("reasoning_content") or "").strip()
        if text:
            return True, f"inference ok ({text[:80]})"
        if reasoning:
            return True, "inference ok (reasoning-only response)"
        return False, "empty inference response"
    except urllib.error.HTTPError as exc:
        detail = exc.read(512).decode(errors="replace")
        return False, f"HTTP {exc.code}: {detail}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _normal_target(config: dict[str, Any]) -> tuple[str, str, str]:
    model_cfg = config.get("model") or {}
    provider_name = str(model_cfg.get("provider") or "").strip()
    model = str(model_cfg.get("default") or model_cfg.get("model") or "").strip()
    provider = (config.get("providers") or {}).get(provider_name) or {}
    base_url = str(provider.get("base_url") or model_cfg.get("base_url") or "").strip()
    return base_url, model, _resolve_api_key(provider)


def _task_is_local_safe(task: Any) -> bool:
    text = f"{getattr(task, 'title', '')}\n{getattr(task, 'body', '')}".lower()
    return not any(marker in text for marker in NETWORK_MARKERS)


def ensure_local_model(
    model: str = DEFAULT_LOCAL_MODEL,
    base: str = DEFAULT_LOCAL_BASE,
    base_url: str = DEFAULT_LOCAL_BASE_URL,
) -> tuple[bool, str]:
    """Create the >=64K-context flight model in Ollama if it is missing.

    Purely local: ``ollama create`` from an already-pulled base layer, so it
    works with no internet. Returns (ok, detail).
    """
    try:
        url = base_url.rstrip("/").removesuffix("/v1") + "/api/tags"
        with urllib.request.urlopen(url, timeout=5) as response:
            names = {m.get("name") for m in json.loads(response.read()).get("models", [])}
        if model in names or f"{model}:latest" in names:
            return True, f"{model} present"
        if base not in names:
            return False, f"base model {base} not pulled locally"
    except Exception as exc:
        return False, f"ollama unreachable: {exc}"
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".modelfile", delete=False) as handle:
        handle.write(f"FROM {base}\nPARAMETER num_ctx 65536\n")
        path = handle.name
    try:
        proc = subprocess.run(
            ["ollama", "create", model, "-f", path],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            return False, f"ollama create failed: {proc.stderr[-300:]}"
        return True, f"created {model} from {base} (num_ctx 65536)"
    except Exception as exc:
        return False, f"ollama create error: {exc}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def port_queued_tasks(conn: sqlite3.Connection, *, local_model: str, local_provider: str) -> dict[str, dict[str, Any]]:
    from hermes_cli import kanban_db as kb

    saved: dict[str, dict[str, Any]] = {}
    for status in ("ready", "todo", "triage", "blocked"):
        for task in kb.list_tasks(conn, status=status):
            if not task.assignee or not _task_is_local_safe(task):
                continue
            saved[task.id] = {"model": task.model_override, "provider": task.provider_override}
            kb.set_model_override(conn, task.id, local_model, local_provider)
    return saved


def restore_task_overrides(conn: sqlite3.Connection, saved: dict[str, dict[str, Any]]) -> None:
    from hermes_cli import kanban_db as kb

    for task_id, override in saved.items():
        if kb.get_task(conn, task_id) is None:
            continue
        kb.set_model_override(conn, task_id, override.get("model"), override.get("provider"))


def _session_store(home: Path, store: Any = None) -> Any:
    if store is not None:
        return store
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore

    sessions_dir = home / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return SessionStore(sessions_dir=sessions_dir, config=GatewayConfig())


def snapshot_and_pin_sessions(
    home: Path,
    *,
    local_model: str,
    local_provider: str,
    local_base_url: str,
    store: Any = None,
) -> dict[str, Any]:
    """Snapshot each session's /model pin, then point it at local inference."""
    session_store = _session_store(home, store)
    saved: dict[str, Any] = {}
    local_override = {
        "model": local_model,
        "provider": local_provider,
        "base_url": local_base_url.rstrip("/"),
    }
    for entry in session_store.list_sessions():
        saved[entry.session_key] = (
            dict(entry.model_override) if entry.model_override else None
        )
        session_store.set_model_override(entry.session_key, local_override)
    return saved


def restore_session_overrides(
    home: Path,
    saved: dict[str, Any],
    *,
    store: Any = None,
    live_overrides: Optional[dict[str, Any]] = None,
    apply_live: Optional[Callable[[str, Optional[dict[str, Any]]], None]] = None,
) -> int:
    """Restore pre-flight /model pins so sessions resume on the normal cascade."""
    session_store = _session_store(home, store)
    restored = 0
    for session_key, override in (saved or {}).items():
        session_store.set_model_override(session_key, override)
        if live_overrides is not None:
            if override:
                live_overrides[session_key] = dict(override)
            else:
                live_overrides.pop(session_key, None)
        if apply_live is not None:
            apply_live(session_key, override)
        restored += 1
    return restored


class FlightManager:
    def __init__(
        self,
        *,
        home: Optional[Path] = None,
        local_model: str = DEFAULT_LOCAL_MODEL,
        local_base_url: str = DEFAULT_LOCAL_BASE_URL,
        max_in_progress: int = 2,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        recovery_threshold: int = DEFAULT_RECOVERY_THRESHOLD,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
        probe: Optional[Callable[..., tuple[bool, str]]] = None,
    ) -> None:
        self.home = Path(home or get_hermes_home())
        self.local_model = local_model
        self.local_base_url = local_base_url
        self.max_in_progress = max_in_progress
        self.failure_threshold = failure_threshold
        self.recovery_threshold = recovery_threshold
        self.cooldown_seconds = cooldown_seconds
        self.probe = probe or self._probe

    def _load_state(self, now: Optional[float] = None) -> dict[str, Any]:
        path = _state_path(self.home)
        if path.exists():
            try:
                state = json.loads(path.read_text())
                if isinstance(state, dict):
                    return state
            except Exception:
                pass
        return new_state(
            now=time.time() if now is None else now,
            failure_threshold=self.failure_threshold,
            recovery_threshold=self.recovery_threshold,
            cooldown_seconds=self.cooldown_seconds,
        )

    def _save_state(self, state: dict[str, Any]) -> None:
        _atomic_write_json(_state_path(self.home), state)

    def _probe(self, kind: str, config: Optional[dict[str, Any]] = None) -> tuple[bool, str]:
        if kind == "local":
            return _probe_openai(self.local_base_url, self.local_model, "ollama")
        base_url, model, api_key = _normal_target(config or {})
        if not base_url or not model:
            return False, "normal route missing base_url or model"
        return _probe_openai(base_url, model, api_key)

    def _backup_profile(self, profile_home: Path) -> Optional[dict[str, Any]]:
        config_path = profile_home / "config.yaml"
        raw = config_path.read_bytes() if config_path.exists() else b""
        if raw and _is_local_config(_read_config(config_path)):
            return None
        backup_name = f"config-{len(list(_flight_dir(self.home).glob('config-*.yaml'))):03d}.yaml"
        backup_path = _flight_dir(self.home) / backup_name
        _atomic_write_bytes(backup_path, raw)
        return {"home": str(profile_home), "config": str(config_path), "backup": str(backup_path), "existed": config_path.exists()}

    def _profile_homes(self) -> list[Path]:
        homes = [self.home]
        profiles = self.home / "profiles"
        if profiles.is_dir():
            homes.extend(sorted(path for path in profiles.iterdir() if path.is_dir() and path.name != "offline"))
        return homes

    def _clean_backup_for(
        self,
        item: dict[str, Any],
        generation_size: int,
    ) -> Optional[Path]:
        config_path = Path(item["config"])
        live = _read_config(config_path)
        recorded = Path(item["backup"])
        candidates = [recorded]
        try:
            recorded_index = int(recorded.stem.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            recorded_index = -1
        if recorded_index >= 0 and generation_size:
            generation_position = recorded_index % generation_size
            for index in range(
                recorded_index - generation_size,
                generation_position - 1,
                -generation_size,
            ):
                candidates.append(recorded.with_name(f"config-{index:03d}.yaml"))
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                config = yaml.safe_load(candidate.read_text()) or {}
            except Exception:
                continue
            if isinstance(config, dict) and not _is_degraded_restore(config, live):
                return candidate
        return None

    def _recover_restore_target(self) -> dict[str, Any]:
        homes = [home for home in self._profile_homes() if (home / "config.yaml").exists()]
        snapshots = sorted(_flight_dir(self.home).glob("config-*.yaml"))
        if not homes or not snapshots or len(snapshots) % len(homes):
            return {"profiles": [], "configs": []}
        generation_size = len(homes)
        configs = []
        for position, profile_home in enumerate(homes):
            matching = [
                path
                for path in snapshots
                if int(path.stem.rsplit("-", 1)[1]) % generation_size == position
            ]
            if not matching:
                continue
            probe = {
                "home": str(profile_home),
                "config": str(profile_home / "config.yaml"),
                "backup": str(matching[-1]),
                "existed": True,
            }
            clean = self._clean_backup_for(probe, generation_size)
            if clean is not None:
                probe["backup"] = str(clean)
                configs.append(probe)
        return {"profiles": [item["home"] for item in configs], "configs": configs}

    def enter(self, *, now: Optional[float] = None, port_tasks: bool = True) -> dict[str, Any]:
        now = time.time() if now is None else now
        state = self._load_state(now)
        if state.get("mode") == "local":
            return state
        configs_on_disk = [
            _read_config(home / "config.yaml")
            for home in self._profile_homes()
            if (home / "config.yaml").exists()
        ]
        if any(_is_local_config(config) for config in configs_on_disk):
            if not (state.get("saved_restore_target") or {}).get("configs"):
                state["saved_restore_target"] = self._recover_restore_target()
            state["mode"] = "local"
            state.setdefault("warnings", []).append(
                "configs already use flight-local; refused to snapshot them as restore targets"
            )
            self._save_state(state)
            return state
        if self.local_model == DEFAULT_LOCAL_MODEL:
            ok, detail = ensure_local_model(self.local_model, base_url=self.local_base_url)
            if not ok:
                state.setdefault("warnings", []).append(f"local model setup: {detail}")
        targets = []
        for profile_home in self._profile_homes():
            config_path = profile_home / "config.yaml"
            if not config_path.exists():
                continue
            target = self._backup_profile(profile_home)
            if target is None:
                continue
            local = build_local_config(
                _read_config(config_path),
                model=self.local_model,
                base_url=self.local_base_url,
                max_in_progress=self.max_in_progress,
            )
            _atomic_write_bytes(config_path, yaml.safe_dump(local, sort_keys=False).encode())
            targets.append(target)
        task_overrides: dict[str, dict[str, Any]] = {}
        if port_tasks:
            try:
                from hermes_cli import kanban_db as kb

                with kb.connect_closing() as conn:
                    task_overrides = port_queued_tasks(
                        conn, local_model=self.local_model, local_provider=LOCAL_PROVIDER
                    )
            except Exception as exc:
                state.setdefault("warnings", []).append(f"kanban port skipped: {exc}")
        session_overrides: dict[str, Any] = {}
        try:
            session_overrides = snapshot_and_pin_sessions(
                self.home,
                local_model=self.local_model,
                local_provider=LOCAL_PROVIDER,
                local_base_url=self.local_base_url,
            )
        except Exception as exc:
            state.setdefault("warnings", []).append(f"session pin skipped: {exc}")
        state.update(
            {
                "mode": "local",
                "normal_failures": 0,
                "normal_successes": 0,
                "last_transition_at": now,
                "last_transition": {"from": "online", "to": "local", "at": now},
                "saved_restore_target": {"profiles": [target["home"] for target in targets], "configs": targets},
                "task_overrides": task_overrides,
                "session_overrides": session_overrides,
            }
        )
        self._save_state(state)
        return state

    def exit(self, *, now: Optional[float] = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        state = self._load_state(now)
        if state.get("mode") != "local":
            return state
        target = state.get("saved_restore_target") or {}
        configs = target.get("configs") or []
        generation_size = len(configs)
        for item in configs:
            config_path = Path(item["config"])
            backup_path = self._clean_backup_for(item, generation_size)
            if item.get("existed"):
                if backup_path is None:
                    state.setdefault("warnings", []).append(
                        f"config restore skipped for {config_path}: no clean snapshot"
                    )
                    continue
                _atomic_write_bytes(config_path, backup_path.read_bytes())
            elif config_path.exists():
                config_path.unlink()
        try:
            from hermes_cli import kanban_db as kb

            with kb.connect_closing() as conn:
                restore_task_overrides(conn, state.get("task_overrides") or {})
        except Exception as exc:
            state.setdefault("warnings", []).append(f"kanban restore skipped: {exc}")
        try:
            restore_session_overrides(self.home, state.get("session_overrides") or {})
        except Exception as exc:
            state.setdefault("warnings", []).append(f"session restore skipped: {exc}")
        state.update(
            {
                "mode": "online",
                "normal_failures": 0,
                "normal_successes": 0,
                "last_transition_at": now,
                "last_transition": {"from": "local", "to": "online", "at": now},
            }
        )
        self._save_state(state)
        return state

    def status(self, *, probe: bool = True) -> dict[str, Any]:
        state = self._load_state()
        if probe:
            normal_config: dict[str, Any] = {}
            target = state.get("saved_restore_target") or {}
            configs = target.get("configs") or []
            if state.get("mode") == "local" and configs:
                raw = Path(configs[0]["backup"]).read_text()
                normal_config = yaml.safe_load(raw) or {}
            else:
                normal_config = _read_config(self.home / "config.yaml")
            normal_ok, normal_detail = self.probe("normal", normal_config)
            local_ok, local_detail = self.probe("local", None)
            state["live_evidence"] = {
                "normal": {"ok": normal_ok, "detail": normal_detail},
                "local": {"ok": local_ok, "detail": local_detail},
            }
        return state

    def tick(self, *, now: Optional[float] = None, port_tasks: bool = True) -> dict[str, Any]:
        now = time.time() if now is None else now
        state = self._load_state(now)
        normal_config: dict[str, Any]
        configs = ((state.get("saved_restore_target") or {}).get("configs") or [])
        if state.get("mode") == "local" and configs:
            normal_config = yaml.safe_load(Path(configs[0]["backup"]).read_text()) or {}
        else:
            normal_config = _read_config(self.home / "config.yaml")
        normal_ok, normal_detail = self.probe("normal", normal_config)
        local_ok, local_detail = self.probe("local", None)
        state["last_evidence"] = {
            "normal": {"ok": normal_ok, "detail": normal_detail},
            "local": {"ok": local_ok, "detail": local_detail},
            "at": now,
        }
        action = observe_connectivity(state, normal_ok=normal_ok, local_ok=local_ok, now=now)
        self._save_state(state)
        if action == "enter":
            result = self.enter(now=now, port_tasks=port_tasks)
            result["tick_action"] = "enter"
            return result
        if action == "exit":
            result = self.exit(now=now)
            result["tick_action"] = "exit"
            return result
        state["tick_action"] = None
        return state


def cmd_flight(args: Any) -> int:
    manager = FlightManager(
        local_model=getattr(args, "model", None) or DEFAULT_LOCAL_MODEL,
        local_base_url=getattr(args, "base_url", None) or DEFAULT_LOCAL_BASE_URL,
        max_in_progress=getattr(args, "max_in_progress", None) or 2,
        failure_threshold=getattr(args, "failure_threshold", None) or DEFAULT_FAILURE_THRESHOLD,
        recovery_threshold=getattr(args, "recovery_threshold", None) or DEFAULT_RECOVERY_THRESHOLD,
        cooldown_seconds=int(getattr(args, "cooldown", None) or DEFAULT_COOLDOWN_SECONDS),
    )
    command = getattr(args, "flight_command", None) or "status"
    if command == "enter":
        result = manager.enter(port_tasks=not getattr(args, "no_port_tasks", False))
    elif command == "exit":
        result = manager.exit()
    elif command == "tick":
        result = manager.tick(port_tasks=not getattr(args, "no_port_tasks", False))
    elif command == "supervise":
        interval = max(5, int(getattr(args, "interval", 30)))
        while True:
            result = manager.tick(port_tasks=not getattr(args, "no_port_tasks", False))
            print(json.dumps(result, sort_keys=True), flush=True)
            time.sleep(interval)
    else:
        result = manager.status(probe=not getattr(args, "no_probe", False))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser(subparsers: Any) -> Any:
    parser = subparsers.add_parser(
        "flight",
        help="Automatically switch between the normal model cascade and local Ollama",
    )
    commands = parser.add_subparsers(dest="flight_command")
    for name in ("enter", "exit", "status", "tick", "supervise"):
        child = commands.add_parser(name)
        child.add_argument("--model", default=DEFAULT_LOCAL_MODEL)
        child.add_argument("--base-url", default=DEFAULT_LOCAL_BASE_URL)
        child.add_argument("--max-in-progress", type=int, default=2)
        child.add_argument("--failure-threshold", type=int, default=DEFAULT_FAILURE_THRESHOLD)
        child.add_argument("--recovery-threshold", type=int, default=DEFAULT_RECOVERY_THRESHOLD)
        child.add_argument("--cooldown", type=int, default=DEFAULT_COOLDOWN_SECONDS)
        if name in {"enter", "tick", "supervise"}:
            child.add_argument("--no-port-tasks", action="store_true")
        if name == "status":
            child.add_argument("--no-probe", action="store_true")
        if name == "supervise":
            child.add_argument("--interval", type=int, default=30)
        child.set_defaults(func=cmd_flight)
    parser.set_defaults(func=cmd_flight)
    return parser
