"""场景状态可达性静态诊断（纯函数，不触碰数据库、不影响运行时匹配）。

把每条规则的 ``next_state`` 与状态超时迁移（``after_ms``/``next_state``）
组成有向图，从 ``initial_state`` 广度优先遍历，回答三个问题：

1. 哪些声明状态**不可达**（没有任何规则迁移或超时迁移链能到达）；
2. 哪些规则**无法触发**（其 state 条件筛选出的可能源状态全部不可达，
   或条件本身在任何声明状态下都不成立）；
3. 每个可达状态从 ``initial_state`` 出发的**最短迁移路径**。

诊断是保守的静态近似：

- 规则的 ``when`` 中只有 ``source=state`` 的条件参与源状态筛选；
  非 state 条件（body/query/header/path）一律视为**可能满足**；
- state 条件的右值含模板语法、或静态求值抛异常（如非法正则）时，
  该条件**不收紧**候选源状态（保守保留边），避免误报不可达；
- 因此图是真实可达关系的超集：这里判定不可达的状态/规则在运行时
  一定也不可达；反过来，判定可达不保证运行时每条边都能走通
  （还取决于请求侧条件是否真能满足）。
"""
from __future__ import annotations

from collections import deque
from typing import Any

from .matching import evaluate_condition
from .models import Rule, Scenario


def _is_static_value(value: Any) -> bool:
    """条件右值是否可静态确定：字符串中含模板语法的视为动态。"""
    return not (isinstance(value, str) and ("{{" in value or "{%" in value))


def _possible_source_states(rule: Rule, states: list[str]) -> tuple[list[str], bool]:
    """按规则的 state 条件收紧可能的源状态集合。

    返回 ``(可能源状态（按 states 声明顺序）, 是否有条件无法静态确定)``。
    非 state 条件不参与收紧（视为可能满足）；单条 state 条件无法静态
    求值时该条件整体不收紧（保守保留候选，避免误报不可达）。
    """
    candidates = set(states)
    uncertain = False
    for cond in rule.conditions:
        if cond.source != "state":
            continue
        if not _is_static_value(cond.value):
            uncertain = True
            continue
        narrowed: set[str] = set()
        for state in candidates:
            try:
                passed = evaluate_condition(cond, state, True)
            except Exception:  # 非法正则等：无法静态确定 -> 保守保留
                narrowed.add(state)
                uncertain = True
                continue
            if passed:
                narrowed.add(state)
        candidates = narrowed
    return [s for s in states if s in candidates], uncertain


def analyze_scenario(scenario: Scenario) -> dict[str, Any]:
    """对已解析校验的场景做状态可达性诊断，返回结构化报告。"""
    states = list(scenario.states)
    edges: list[dict[str, Any]] = []
    rule_reports: list[dict[str, Any]] = []

    # 1) 规则迁移边：可能源状态 -> next_state（next_state 省略为状态不变，
    #    即自环，不影响可达性，不产生边）
    for route in scenario.routes:
        for rule in route.rules:
            sources, uncertain = _possible_source_states(rule, states)
            rule_reports.append({
                "route": route.id,
                "rule": rule.name,
                "priority": rule.priority,
                "next_state": rule.next_state,
                "possible_source_states": sources,
                "state_condition_uncertain": uncertain,
            })
            if rule.next_state is None:
                continue
            for source in sources:
                edges.append({
                    "kind": "rule",
                    "route": route.id,
                    "rule": rule.name,
                    "from": source,
                    "to": rule.next_state,
                })

    # 2) 状态超时迁移边（虚拟时钟到期自动迁移）
    for state, timeout in scenario.timeouts.items():
        edges.append({
            "kind": "timeout",
            "from": state,
            "to": timeout.next_state,
            "after_ms": timeout.after_ms,
        })

    # 3) BFS：从 initial_state 出发求每个状态的最短迁移距离
    adjacency: dict[str, list[int]] = {}
    for index, edge in enumerate(edges):
        adjacency.setdefault(edge["from"], []).append(index)

    initial = scenario.initial_state
    distance: dict[str, int] = {initial: 0}
    previous: dict[str, tuple[str, int]] = {}  # 状态 -> (前驱状态, 边下标)
    queue: deque[str] = deque([initial])
    while queue:
        current = queue.popleft()
        for edge_index in adjacency.get(current, []):
            target = edges[edge_index]["to"]
            if target in distance:
                continue
            distance[target] = distance[current] + 1
            previous[target] = (current, edge_index)
            queue.append(target)

    reachable = [s for s in states if s in distance]
    unreachable = [s for s in states if s not in distance]

    # 4) 最短迁移路径重建（initial 自身距离 0、路径为空）
    shortest_paths: dict[str, Any] = {}
    for state in reachable:
        steps: list[dict[str, Any]] = []
        cursor = state
        while cursor != initial:
            prev_state, edge_index = previous[cursor]
            steps.append(edges[edge_index])
            cursor = prev_state
        steps.reverse()
        shortest_paths[state] = {"distance": distance[state], "steps": steps}

    # 5) 规则可触发性：可能源状态为空（state 条件矛盾）或全部不可达
    untriggerable: list[dict[str, Any]] = []
    for report in rule_reports:
        sources = report["possible_source_states"]
        if not sources:
            reason = "no_possible_source_state"
            detail = "state 条件在任何声明状态下都不成立（条件自相矛盾或取值不在 states 内）"
        elif not any(source in distance for source in sources):
            reason = "source_states_unreachable"
            detail = "所有可能源状态从 initial_state 均不可达"
        else:
            report["triggerable"] = True
            continue
        report["triggerable"] = False
        untriggerable.append({
            "route": report["route"],
            "rule": report["rule"],
            "priority": report["priority"],
            "next_state": report["next_state"],
            "possible_source_states": sources,
            "reason": reason,
            "detail": detail,
        })

    return {
        "analyze": {"persisted": False},
        "valid": True,
        "scenario": scenario.name,
        "initial_state": initial,
        "states": states,
        "reachable_states": reachable,
        "unreachable_states": unreachable,
        "all_states_reachable": not unreachable,
        "rules": rule_reports,
        "untriggerable_rules": untriggerable,
        "graph": {"edges": edges},
        "shortest_paths": shortest_paths,
        "notes": [
            "静态保守分析：非 state 条件视为可能满足；无法静态确定的 state 条件不收紧源状态",
            "判定为不可达的状态/规则在运行时一定不可达；可达不保证运行时每条边都能走通"
            "（还取决于请求侧条件是否真能满足）",
            "本诊断不写入数据库，不改变实际请求匹配与场景发布",
        ],
    }
