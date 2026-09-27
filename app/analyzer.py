"""场景状态可达性静态诊断（纯内存：不写库、不影响请求匹配与场景发布）。

复用 ``parse_scenario`` 的解析结果，把每条规则的 ``next_state`` 与状态超时
迁移组成有向图，从 ``initial_state`` 遍历，回答三个问题：

1. 哪些声明状态从初始状态**不可达**（``unreachable_states``）；
2. 哪些规则**无法触发**（``untriggerable_rules``）——可能源状态均不可达，
   或 state 条件在所有声明状态下都不成立；
3. 每个可达状态的**最短迁移路径**（``shortest_paths``，按迁移步数最少）。

保守原则（宁可少报、不误报不可达）：

- 非 state 条件（body/query/header/path）一律视为"可能满足"，不约束源状态；
- state 条件的比较右值含 Jinja 模板（``{{``/``{%``）或静态求值抛异常时，
  该条件视为无法静态确定，**不约束**源状态（保守保留边）；
- 只有能静态确定为 False 的 (状态, 条件) 组合才从源状态集合中剔除——
  静态求值复用运行时同一个 ``evaluate_condition``，结果与真实匹配一致。
"""
from __future__ import annotations

from collections import deque
from typing import Any

from .matching import evaluate_condition
from .models import Condition, Rule, Scenario


def _is_template(value: Any) -> bool:
    """值（或嵌套元素）是否含 Jinja 模板——静态无法求值。"""
    if isinstance(value, str):
        return "{{" in value or "{%" in value
    if isinstance(value, list):
        return any(_is_template(item) for item in value)
    if isinstance(value, dict):
        return any(_is_template(item) for item in value.values())
    return False


def _condition_possible_states(cond: Condition, states: list[str]) -> set[str] | None:
    """state 条件可能满足的状态集合。

    右值含模板或求值异常时返回 None，表示"无法静态确定"，调用方按
    不约束处理（保守保留所有源状态）。
    """
    if _is_template(cond.value):
        return None
    possible: set[str] = set()
    for state in states:
        try:
            if evaluate_condition(cond, state, True):
                possible.add(state)
        except Exception:  # 如 regex 右值非法：保守保留，不约束
            return None
    return possible


def _rule_source_states(rule: Rule, states: list[str]) -> tuple[set[str], bool]:
    """(规则的可能源状态集合, 是否所有 state 条件都可静态确定)。"""
    possible = set(states)
    fully_static = True
    for cond in rule.conditions:
        if cond.source != "state":
            continue  # 非 state 条件只视为可能满足
        cond_states = _condition_possible_states(cond, states)
        if cond_states is None:
            fully_static = False  # 不能静态确定：保守保留，不约束
            continue
        possible &= cond_states
    return possible, fully_static


def analyze_scenario(scenario: Scenario) -> dict[str, Any]:
    """对已解析场景做状态可达性诊断，返回可 JSON 序列化的诊断报告。"""
    states = list(scenario.states)

    # 逐规则诊断 + 规则迁移边（源状态 -> next_state）
    rules_report: list[dict[str, Any]] = []
    adjacency: dict[str, list[tuple[str, dict[str, Any]]]] = {s: [] for s in states}
    for route in scenario.routes:
        for rule in route.rules:
            possible, fully_static = _rule_source_states(rule, states)
            ordered = [s for s in states if s in possible]
            rules_report.append({
                "route": route.id,
                "rule": rule.name,
                "priority": rule.priority,
                "next_state": rule.next_state,
                "possible_source_states": ordered,
                "state_conditions_static": fully_static,
            })
            if rule.next_state is not None:
                for src in ordered:
                    edge = {"kind": "rule", "route": route.id, "rule": rule.name,
                            "from": src, "to": rule.next_state}
                    adjacency[src].append((rule.next_state, edge))

    # 状态超时迁移边（进入状态后无条件到期触发）
    for state, timeout in scenario.timeouts.items():
        edge = {"kind": "timeout", "from": state, "to": timeout.next_state,
                "after_ms": timeout.after_ms}
        adjacency[state].append((timeout.next_state, edge))

    # BFS：可达集合 + 最短路径（步数最少；同层按路由/状态定义顺序，结果确定）
    dist = {scenario.initial_state: 0}
    prev: dict[str, dict[str, Any]] = {}
    queue = deque([scenario.initial_state])
    while queue:
        current = queue.popleft()
        for nxt, edge in adjacency[current]:
            if nxt not in dist:
                dist[nxt] = dist[current] + 1
                prev[nxt] = edge
                queue.append(nxt)

    reachable = [s for s in states if s in dist]
    unreachable = [s for s in states if s not in dist]

    shortest_paths: dict[str, list[dict[str, Any]]] = {}
    for state in reachable:
        if state == scenario.initial_state:
            continue
        path: list[dict[str, Any]] = []
        cursor = state
        while cursor != scenario.initial_state:
            edge = prev[cursor]
            path.append(edge)
            cursor = edge["from"]
        path.reverse()
        shortest_paths[state] = path

    reachable_set = set(reachable)
    untriggerable: list[dict[str, Any]] = []
    for info in rules_report:
        possible = info["possible_source_states"]
        info["triggerable"] = any(s in reachable_set for s in possible)
        if info["triggerable"]:
            continue
        reason = ("state 条件在所有声明状态下均不成立" if not possible
                  else "规则的可能源状态均不可达")
        untriggerable.append({
            "route": info["route"],
            "rule": info["rule"],
            "possible_source_states": possible,
            "reason": reason,
        })

    return {
        "valid": True,
        "scenario": scenario.name,
        "initial_state": scenario.initial_state,
        "states": states,
        "reachable_states": reachable,
        "unreachable_states": unreachable,
        "rules": rules_report,
        "untriggerable_rules": untriggerable,
        "shortest_paths": shortest_paths,
        "notes": [
            "非 state 条件一律视为可能满足；含模板或无法静态求值的 state 条件"
            "保守保留（不约束源状态），不会误报不可达",
            "诊断基于静态图遍历：不创建会话、不写数据库，不影响请求匹配与场景发布",
        ],
    }
