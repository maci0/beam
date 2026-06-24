"""Unit + fuzz tests for the vLLM-surface scanner: the AST visitor that collects
ray usage, the symbol resolver, and the out-of-scope module-boundary match."""

import ast
import os
import sys

from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import scan_vllm_ray as sc  # noqa: E402


def visit(src):
    v = sc.RayUsageVisitor()
    v.visit(ast.parse(src))
    return v


def test_visitor_plain_attr():
    assert "ray.get" in visit("ray.get(x)").attrs


def test_visitor_nested_attr():
    assert "ray.util.placement_group" in visit("ray.util.placement_group([])").attrs


def test_visitor_import_and_from():
    v = visit("import ray.util\nfrom ray.exceptions import RayActorError")
    assert ("ray.util", None) in v.imports
    assert ("ray.exceptions", "RayActorError") in v.imports


def test_visitor_alias_normalized():
    # `import ray as r; r.get(x)` must normalize back to ray.get
    assert "ray.get" in visit("import ray as r\nr.get(x)").attrs


def test_visitor_ignores_non_ray():
    v = visit("import os\nos.getcwd()\nfoo.bar()")
    assert v.attrs == set() and v.imports == set()


# ---- out_of_scope: module-boundary match ----
def test_out_of_scope_boundaries():
    assert sc.out_of_scope("ray.data")
    assert sc.out_of_scope("ray.data.llm")
    assert sc.out_of_scope("ray.serve.deployment")
    assert not sc.out_of_scope("ray.database")  # not a boundary of ray.data
    assert not sc.out_of_scope("ray.get")


@given(st.text())
def test_out_of_scope_never_crashes(s):
    sc.out_of_scope(s)


# ---- resolve against the live shim ----
def test_resolve_known_symbols():
    assert sc.resolve("ray.get")
    assert sc.resolve("ray.util.placement_group")
    assert sc.resolve("ray.exceptions.RayActorError")


def test_resolve_missing():
    assert not sc.resolve("ray.this_does_not_exist_xyz")


@given(st.lists(st.from_regex(r"[a-z_]{1,8}", fullmatch=True), min_size=1, max_size=5))
def test_resolve_never_crashes(parts):
    sc.resolve("ray." + ".".join(parts))  # arbitrary dotted path, no exception
