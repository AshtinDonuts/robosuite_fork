#!/usr/bin/env python3
"""
Inspect a pickle (.pk/.pkl) file by printing a structured summary.

Examples:
  python myscripts/view_pk.py /path/to/file.pk
  python myscripts/view_pk.py /path/to/file.pk --depth 4 --max-items 30 --show-values
  python myscripts/view_pk.py /path/to/file.pk --path "obs/state" --show-values
"""

from __future__ import annotations

import argparse
import os
import pickle
import pprint
import sys
from dataclasses import dataclass
from typing import Any, Iterable


def _optional_import_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception:
        return None


def _optional_import_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def _is_namedtuple(obj: Any) -> bool:
    t = type(obj)
    return isinstance(obj, tuple) and hasattr(obj, "_fields") and hasattr(t, "_asdict")


def _safe_repr(obj: Any, max_len: int = 200) -> str:
    try:
        s = repr(obj)
    except Exception as e:
        s = f"<repr failed: {type(e).__name__}: {e}>"
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _shape_dtype_str(obj: Any) -> str | None:
    np = _optional_import_numpy()
    torch = _optional_import_torch()

    if np is not None and isinstance(obj, getattr(np, "ndarray")):
        return f"shape={tuple(obj.shape)} dtype={obj.dtype}"
    if torch is not None and isinstance(obj, getattr(torch, "Tensor")):
        try:
            device = str(obj.device)
        except Exception:
            device = "?"
        try:
            dtype = str(obj.dtype)
        except Exception:
            dtype = "?"
        try:
            shape = tuple(obj.shape)
        except Exception:
            shape = "?"
        return f"shape={shape} dtype={dtype} device={device}"
    return None


def _iter_preview(it: Iterable[Any], max_items: int) -> list[Any]:
    out: list[Any] = []
    for i, x in enumerate(it):
        if i >= max_items:
            break
        out.append(x)
    return out


@dataclass(frozen=True)
class InspectOptions:
    depth: int
    max_items: int
    max_str: int
    show_values: bool
    sort_keys: bool


def _inspect(obj: Any, *, opts: InspectOptions, indent: int = 0, depth: int = 0) -> None:
    pad = " " * indent
    tname = type(obj).__name__
    extra = _shape_dtype_str(obj)
    if extra:
        print(f"{pad}{tname} ({extra})")
    else:
        print(f"{pad}{tname}")

    if depth >= opts.depth:
        if opts.show_values:
            print(f"{pad}  value={_safe_repr(obj, max_len=opts.max_str)}")
        return

    # Dict-like
    if isinstance(obj, dict):
        keys = list(obj.keys())
        if opts.sort_keys:
            try:
                keys = sorted(keys)
            except Exception:
                pass
        n = len(keys)
        print(f"{pad}  len={n}")
        for k in keys[: opts.max_items]:
            print(f"{pad}  [{_safe_repr(k, max_len=opts.max_str)}] -> ", end="")
            _inspect(obj[k], opts=opts, indent=indent + 4, depth=depth + 1)
        if n > opts.max_items:
            print(f"{pad}  ... ({n - opts.max_items} more keys)")
        return

    # Namedtuple
    if _is_namedtuple(obj):
        try:
            d = obj._asdict()
        except Exception:
            d = None
        if isinstance(d, dict):
            print(f"{pad}  fields={list(d.keys())}")
            for k in list(d.keys())[: opts.max_items]:
                print(f"{pad}  .{k} -> ", end="")
                _inspect(d[k], opts=opts, indent=indent + 4, depth=depth + 1)
            return

    # List / tuple
    if isinstance(obj, (list, tuple)):
        n = len(obj)
        print(f"{pad}  len={n}")
        for i, x in enumerate(obj[: opts.max_items]):
            print(f"{pad}  [{i}] -> ", end="")
            _inspect(x, opts=opts, indent=indent + 4, depth=depth + 1)
        if n > opts.max_items:
            print(f"{pad}  ... ({n - opts.max_items} more items)")
        return

    # Set / frozenset
    if isinstance(obj, (set, frozenset)):
        n = len(obj)
        print(f"{pad}  len={n}")
        for i, x in enumerate(_iter_preview(obj, opts.max_items)):
            print(f"{pad}  [{i}] -> ", end="")
            _inspect(x, opts=opts, indent=indent + 4, depth=depth + 1)
        if n > opts.max_items:
            print(f"{pad}  ... ({n - opts.max_items} more items)")
        return

    # Fallback value printing
    if opts.show_values:
        print(f"{pad}  value={_safe_repr(obj, max_len=opts.max_str)}")


def _get_by_path(obj: Any, path: str) -> Any:
    """
    Path syntax:
      - slash-separated components: "a/b/0/c"
      - dict keys: component as-is (string); if key not found, also try int/float/bool parsing
      - list/tuple indices: integer component
    """
    if not path:
        return obj
    cur = obj
    for raw in path.split("/"):
        if raw == "":
            continue
        if isinstance(cur, (list, tuple)):
            idx = int(raw)
            cur = cur[idx]
            continue
        if isinstance(cur, dict):
            if raw in cur:
                cur = cur[raw]
                continue
            # Try to interpret as non-string keys
            candidates: list[Any] = []
            try:
                candidates.append(int(raw))
            except Exception:
                pass
            try:
                candidates.append(float(raw))
            except Exception:
                pass
            if raw.lower() in ("true", "false"):
                candidates.append(raw.lower() == "true")
            found = False
            for k in candidates:
                if k in cur:
                    cur = cur[k]
                    found = True
                    break
            if found:
                continue
            raise KeyError(f"Path component {raw!r} not found in dict keys")
        # attribute access as last resort
        if hasattr(cur, raw):
            cur = getattr(cur, raw)
            continue
        raise TypeError(f"Cannot descend into {type(cur).__name__} with component {raw!r}")
    return cur


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Inspect a pickle (.pk/.pkl) file")
    p.add_argument("file", help="Path to .pk/.pkl file")
    p.add_argument("--path", default="", help="Optional subpath inside object (e.g. 'obs/state/0')")
    p.add_argument("--depth", type=int, default=3, help="Max recursion depth for inspection")
    p.add_argument("--max-items", type=int, default=20, help="Max keys/items to show per container")
    p.add_argument("--max-str", type=int, default=200, help="Max repr length for values/keys")
    p.add_argument("--show-values", action="store_true", help="Print repr(value) for leaf nodes")
    p.add_argument("--sort-keys", action="store_true", help="Sort dict keys when possible")
    p.add_argument("--pp", action="store_true", help="Pretty-print the selected object (can be huge)")
    args = p.parse_args(argv)

    file_path = os.path.expanduser(args.file)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}", file=sys.stderr)
        return 2

    try:
        obj = _load_pickle(file_path)
    except Exception as e:
        print(f"Failed to load pickle: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    try:
        sel = _get_by_path(obj, args.path)
    except Exception as e:
        print(f"Failed to select --path {args.path!r}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"file={file_path}")
    if args.path:
        print(f"selected_path={args.path}")

    if args.pp:
        pprint.pprint(sel, width=120, compact=True, sort_dicts=args.sort_keys)
        return 0

    opts = InspectOptions(
        depth=max(0, args.depth),
        max_items=max(1, args.max_items),
        max_str=max(20, args.max_str),
        show_values=bool(args.show_values),
        sort_keys=bool(args.sort_keys),
    )
    _inspect(sel, opts=opts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
