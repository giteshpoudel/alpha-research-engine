"""Containment for agent-generated strategy code.

Two gates before any generated strategy is trusted:
1. **AST allowlist** — only ``numpy``/``pandas``/``math`` imports; no ``while``,
   no dunder attribute access, no ``eval``/``exec``/``open``/``__import__`` etc.
2. **Subprocess smoke test** — the code runs in a *separate process* with CPU and
   address-space rlimits and a wall-clock timeout, so an infinite loop or
   memory bomb is killed rather than hanging the optimizer.
"""

from __future__ import annotations

import ast
import multiprocessing as mp

ALLOWED_IMPORTS = {"numpy", "pandas", "math"}
FORBIDDEN_NAMES = {"eval", "exec", "compile", "open", "__import__", "input",
                   "globals", "locals", "vars", "getattr", "setattr", "delattr",
                   "memoryview", "breakpoint"}

CPU_SECONDS = 5
ADDRESS_SPACE_BYTES = 4_000_000_000  # ~4 GiB address space (pandas is hungry)


def validate_source(code: str) -> list[str]:
    """Return a list of policy violations (empty means acceptable)."""
    errors: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    errors.append(f"import not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                errors.append(f"import not allowed: {node.module}")
        elif isinstance(node, ast.While):
            errors.append("while loops are not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            errors.append(f"forbidden name: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append(f"forbidden attribute: {node.attr}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            errors.append("global/nonlocal not allowed")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "signals" for n in tree.body):
        errors.append("module must define signals(close, **params)")
    return errors


def _child(code: str, sample: list[float], queue) -> None:
    import resource

    try:
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
        resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE_BYTES, ADDRESS_SPACE_BYTES))
    except Exception:
        pass
    try:
        import pandas as pd

        namespace: dict = {}
        exec(compile(code, "<generated>", "exec"), namespace)  # noqa: S102 (sandboxed)
        signals = namespace.get("signals")
        if not callable(signals):
            queue.put(("error", "no signals function"))
            return
        close = pd.Series(sample, dtype="float64")
        entries, exits = signals(close)
        if entries.dtype != bool or exits.dtype != bool:
            queue.put(("error", "signals must return boolean Series"))
            return
        queue.put(("ok", int(entries.sum())))
    except BaseException as exc:  # noqa: BLE001 (report anything back)
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def smoke_test(code: str, sample: list[float] | None = None, timeout: float = 10.0) -> dict:
    """Run generated code on a sample series in a limited subprocess."""
    sample = sample or [100.0 + (i % 10) for i in range(200)]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_child, args=(code, sample, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(1)
        return {"ok": False, "error": f"timeout after {timeout}s"}
    try:
        status, payload = queue.get_nowait()
    except Exception:
        return {"ok": False, "error": "no result from sandbox"}
    return ({"ok": True, "entries": payload, "error": None} if status == "ok"
            else {"ok": False, "error": payload})
