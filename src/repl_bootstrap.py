"""
Standalone REPL bootstrap — included via include_str! in repl_executor.rs.
No relative imports. socket_path is passed explicitly so this works inside
the daemon process without relying on environment discovery.
"""

import ast
import json
import socket as _socket
import sys
import traceback
from dataclasses import dataclass
from io import StringIO


# ── Daemon call ───────────────────────────────────────────────────────────────

def _daemon_call(tool: str, args: dict, socket_path: str) -> dict:
    sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    sock.settimeout(30)
    try:
        sock.connect(socket_path)
        request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": tool, "arguments": args}}
        sock.sendall((json.dumps(request) + "\n").encode())
        response = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            response += chunk
            if b"\n" in chunk:
                break
        result = json.loads(response.decode("utf-8", errors="replace"))
        if "error" in result:
            raise RuntimeError(result["error"].get("message", str(result["error"])))
        res = result.get("result", {})
        structured = res.get("structured", {})
        if structured:
            return structured
        content = res.get("content", [{}])
        return {"text": content[0].get("text", "") if content else ""}
    finally:
        sock.close()


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class Memory:
    id: int
    content: str
    score: float = 0.0
    tags: list = None
    created_at: str = ""

    def __post_init__(self):
        self.tags = self.tags or []

    def __repr__(self):
        preview = self.content[:60] + "..." if len(self.content) > 60 else self.content
        return f"Memory({self.id}, score={self.score:.2f}, '{preview}')"


@dataclass
class Triplet:
    subject: str
    predicate: str
    object: str

    def __repr__(self):
        return f"({self.subject}) --[{self.predicate}]--> ({self.object})"


# ── SoulAPI ───────────────────────────────────────────────────────────────────

class SoulAPI:
    def __init__(self, socket_path: str):
        self._sp = socket_path
        self._trajectory = []

    def _call(self, tool: str, args: dict) -> dict:
        return _daemon_call(tool, args, self._sp)

    def _track(self, method: str, args: dict, result):
        self._trajectory.append({
            "method": method,
            "args": args,
            "result_preview": str(result)[:200],
        })

    def _mem_id(self, m: dict) -> int:
        v = m.get("id", 0)
        if isinstance(v, str) and v.startswith("00000000"):
            try:
                return int(v.split("-")[-1], 16)
            except Exception:
                return 0
        return v if isinstance(v, int) else 0

    def search(self, query: str, limit: int = 20, threshold: float = 0.0) -> list:
        r = self._call("recall", {"query": query, "limit": limit})
        out = []
        for m in r.get("memories", r.get("results", [])):
            score = m.get("score", m.get("relevance", m.get("similarity", 0)))
            if score >= threshold:
                out.append(Memory(id=self._mem_id(m),
                                  content=m.get("content", m.get("text", m.get("summary", ""))),
                                  score=score, tags=m.get("tags", []),
                                  created_at=m.get("created_at", "")))
        self._track("search", {"query": query, "limit": limit}, out)
        return out

    def recall(self, query: str, limit: int = 10) -> list:
        r = self._call("recall", {"query": query, "limit": limit})
        out = [Memory(id=self._mem_id(m),
                      content=m.get("content", m.get("text", "")),
                      score=m.get("combined_score", m.get("score", m.get("relevance", 0))),
                      tags=m.get("tags", []), created_at=m.get("created_at", ""))
               for m in r.get("memories", r.get("results", []))]
        self._track("recall", {"query": query, "limit": limit}, out)
        return out

    def expand(self, memory_id: int, depth: int = 3) -> dict:
        r = self._call("expand_memory", {"id": memory_id, "depth": depth})
        self._track("expand", {"memory_id": memory_id, "depth": depth}, r)
        return r

    def triplets(self, subject=None, predicate=None, object=None, limit: int = 50) -> list:
        args = {"limit": limit}
        if subject:   args["subject"]   = subject
        if predicate: args["predicate"] = predicate
        if object:    args["object"]    = object
        r = self._call("query_graph", args)
        out = [Triplet(t["subject"], t["predicate"], t["object"])
               for t in r.get("triplets", [])]
        self._track("triplets", args, out)
        return out

    def recent(self, hours: int = 24, limit: int = 50) -> list:
        r = self._call("explore_recall", {"query": "*", "hours": hours, "limit": limit})
        out = [Memory(id=self._mem_id(m),
                      content=m.get("content", m.get("text", "")),
                      score=m.get("recency_score", m.get("score", 1.0)),
                      tags=m.get("tags", []), created_at=m.get("created_at", ""))
               for m in r.get("memories", r.get("results", []))]
        self._track("recent", {"hours": hours, "limit": limit}, out)
        return out

    def remember(self, content: str, tags=None, importance: float = 0.5) -> int:
        args = {"content": content, "importance": importance}
        if tags:
            args["tags"] = tags
        r = self._call("remember", args)
        mid = r.get("id", r.get("memory_id", 0))
        self._track("remember", {"content": content[:50]}, mid)
        return mid

    def symbols(self, pattern=None, kind=None, limit: int = 50) -> list:
        args = {"limit": limit}
        if pattern: args["pattern"] = pattern
        if kind:    args["kind"]    = kind
        r = self._call("search_symbols", args)
        out = r.get("symbols", [])
        self._track("symbols", args, out)
        return out

    def read_symbol(self, name: str, file_path=None) -> dict:
        args = {"name": name}
        if file_path:
            args["file_path"] = file_path
        r = self._call("read_symbol", args)
        self._track("read_symbol", args, r)
        return r

    def stats(self) -> dict:
        r = self._call("health_check", {})
        self._track("stats", {}, r)
        return r

    def trajectory(self) -> list:
        return list(self._trajectory)

    def clear_trajectory(self):
        self._trajectory.clear()

    def __repr__(self):
        return f"SoulAPI(calls={len(self._trajectory)})"


# ── Sandbox ───────────────────────────────────────────────────────────────────

SAFE_BUILTINS = {
    "True": True, "False": False, "None": None,
    "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "type": type, "object": object, "bytes": bytes,
    "len": len, "range": range, "enumerate": enumerate, "zip": zip,
    "map": map, "filter": filter, "sorted": sorted, "reversed": reversed,
    "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
    "all": all, "any": any, "isinstance": isinstance, "issubclass": issubclass,
    "hasattr": hasattr, "getattr": getattr, "setattr": setattr,
    "repr": repr, "format": format, "chr": chr, "ord": ord,
    "iter": iter, "next": next, "print": print,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError, "AttributeError": AttributeError,
    "RuntimeError": RuntimeError, "StopIteration": StopIteration,
}

_FORBIDDEN = {ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal}


def _validate(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    for node in ast.walk(tree):
        if type(node) in _FORBIDDEN:
            return False, f"Forbidden: {type(node).__name__}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return False, f"Private attribute: {node.attr}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("exec", "eval", "compile", "__import__",
                                "open", "input", "breakpoint"):
                return False, f"Forbidden: {node.func.id}"
    return True, ""


_SERIAL = (str, int, float, bool, type(None), list, dict, tuple)


def _serialize_ns(ns: dict) -> dict:
    out = {}
    skip = {"soul", "Memory", "Triplet", "__builtins__"}
    for k, v in ns.items():
        if k.startswith("_") or k in skip or not isinstance(v, _SERIAL):
            continue
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            pass
    return out


# ── Entry point ───────────────────────────────────────────────────────────────

def repl_execute_main(code: str, initial_ns_json: str, socket_path: str,
                      max_output: int) -> str:
    soul = SoulAPI(socket_path)
    ns = {
        "__builtins__": SAFE_BUILTINS,
        "soul": soul,
        "Memory": Memory,
        "Triplet": Triplet,
    }
    if initial_ns_json:
        try:
            for k, v in json.loads(initial_ns_json).items():
                if not k.startswith("_") and k not in ("soul", "Memory", "Triplet"):
                    ns[k] = v
        except Exception:
            pass

    ok, err = _validate(code)
    if not ok:
        return json.dumps({"success": False, "output": "", "error": f"Validation: {err}",
                           "namespace_json": "{}", "trajectory": []})

    old_stdout = sys.stdout
    captured = StringIO()
    sys.stdout = captured
    error = ""
    try:
        tree = ast.parse(code)
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body.pop()
        if tree.body:
            exec(compile(tree, "<repl>", "exec"), ns)
        if last_expr:
            val = eval(compile(ast.Expression(last_expr.value), "<repl>", "eval"), ns)
            if val is not None:
                print(repr(val))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()
    if len(output) > max_output:
        output = output[:max_output] + f"\n... (truncated, {len(output)} total)"

    return json.dumps({
        "success": not error,
        "output": output,
        "error": error,
        "namespace_json": json.dumps(_serialize_ns(ns)),
        "trajectory": soul.trajectory(),
    })
