#!/usr/bin/env python3
"""Scan vLLM source for its `ray` API usage and check beam's shim covers it.

Static AST scan: collects every `ray.*` attribute access and every
`from ray... import ...`, resolves each against the beam shim, and reports what
is covered vs missing. Run it whenever vLLM is bumped so the stub surface stays
in sync.

    uv run --with cloudpickle python scripts/scan_vllm_ray.py --src /path/to/vllm

(cloudpickle must be importable, since the shim re-exports it as ray.cloudpickle.)

Exit code is non-zero if any used symbol is missing from the shim, so it works
as a CI gate.

Ceiling: a static scan cannot see methods called on values *returned* by ray
(e.g. ray.get_runtime_context().get_node_id()) or fully dynamic getattr. It
tracks the top-level surface; returned-object methods are covered by the e2e
demo instead.
"""

import argparse
import ast
import importlib
import os
import sys


class RayUsageVisitor(ast.NodeVisitor):
    def __init__(self):
        self.aliases = {"ray"}  # local names bound to the ray package
        self.attrs = set()  # dotted paths like "ray.util.placement_group"
        self.imports = set()  # ("ray.util", "placement_group")

    def visit_Import(self, node):
        for a in node.names:
            if a.name == "ray" or a.name.startswith("ray."):
                self.aliases.add(a.asname or a.name.split(".")[0])
                if a.name != "ray":
                    self.imports.add((a.name, None))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and (node.module == "ray" or node.module.startswith("ray.")):
            for a in node.names:
                if a.name != "*":
                    self.imports.add((node.module, a.name))
        self.generic_visit(node)

    def visit_Attribute(self, node):
        dotted = self._dotted(node)
        if dotted:
            self.attrs.add(dotted)
        self.generic_visit(node)

    def _dotted(self, node):
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name) and cur.id in self.aliases:
            parts.append("ray")  # normalize alias back to the real root
            return ".".join(reversed(parts))
        return None


def scan_tree(src):
    v = RayUsageVisitor()
    for root, _, files in os.walk(src):
        if "/test" in root or "/.git" in root:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            try:
                with open(path, encoding="utf-8") as fh:
                    tree = ast.parse(fh.read(), path)
            except (SyntaxError, UnicodeDecodeError):
                continue
            v.visit(tree)
    return v


# Ray features beam intentionally does not implement: separate opt-in
# integrations and the compiled-DAG data plane, none on the NCCL multi-node
# inference path the vllm-openai image uses. Usage here is reported, not failed.
OUT_OF_SCOPE = (
    "ray.data",  # Ray Data batch inference
    "ray.serve",  # Ray Serve deployments
    "ray.experimental",  # compiled-DAG channels / accelerator context
    "ray._private.accelerators",  # TPU acceleration manager
)


def out_of_scope(sym) -> bool:
    # match on module boundaries: "ray.data" matches "ray.data" and "ray.data.llm"
    # but not a hypothetical "ray.database".
    return any(sym == p or sym.startswith(p + ".") for p in OUT_OF_SCOPE)


def resolve(dotted) -> bool:
    """True if `dotted` resolves as a module or attribute in the loaded shim."""
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            mod = importlib.import_module(".".join(parts[:i]))
        except Exception:
            continue
        obj = mod
        for attr in parts[i:]:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                return False
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.environ.get("VLLM_SRC"), help="path to vLLM source tree")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if not args.src or not os.path.isdir(args.src):
        sys.exit("need --src /path/to/vllm (or set VLLM_SRC); clone vllm-project/vllm")

    # make the shim importable as `ray`
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(here, "python"))

    usage = scan_tree(args.src)

    # required surface: dotted attrs + from-imports turned into dotted paths
    required = set(usage.attrs)
    for mod, name in usage.imports:
        required.add(mod if name is None else f"{mod}.{name}")
    # drop bare "ray" and pure submodule roots we don't need to attribute-check
    required = {r for r in sorted(required) if r != "ray"}

    covered, missing, skipped = [], [], []
    for sym in sorted(required):
        if out_of_scope(sym):
            skipped.append(sym)
        elif resolve(sym):
            covered.append(sym)
        else:
            missing.append(sym)

    if not args.quiet:
        print(
            f"# vLLM ray surface: {len(required)} symbols "
            f"({len(covered)} covered, {len(skipped)} out-of-scope, "
            f"{len(missing)} missing)\n"
        )
        for sym in covered:
            print(f"  ok        {sym}")
        for sym in skipped:
            print(f"  skip      {sym}")
    if missing:
        print(f"\n# MISSING from beam shim ({len(missing)}) -- core path:", file=sys.stderr)
        for sym in missing:
            print(f"  MISSING   {sym}", file=sys.stderr)
        sys.exit(1)
    print("\nall in-scope ray symbols are covered by the beam shim")


if __name__ == "__main__":
    main()
