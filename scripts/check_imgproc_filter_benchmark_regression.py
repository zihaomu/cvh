#!/usr/bin/env python3
"""
Compare two imgproc filter benchmark CSV files and fail on significant slowdown.

CSV columns expected:
op,kernel,depth,channels,shape,layout,border,dispatch_path,ms_per_iter
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


BenchKey = Tuple[str, str, str, str, str, str, str, str]


@dataclass
class BenchRow:
    key: BenchKey
    ms_per_iter: float


def load_csv(path: pathlib.Path) -> Dict[BenchKey, BenchRow]:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    rows: Dict[BenchKey, BenchRow] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["op"].strip(),
                row["kernel"].strip(),
                row["depth"].strip(),
                row["channels"].strip(),
                row["shape"].strip(),
                row["layout"].strip(),
                row["border"].strip(),
                row["dispatch_path"].strip(),
            )
            rows[key] = BenchRow(key=key, ms_per_iter=float(row["ms_per_iter"]))
    return rows


def format_key(key: BenchKey) -> str:
    op, kernel, depth, channels, shape, layout, border, dispatch = key
    return (
        f"{op:<14} {kernel:<4} {depth:<7} c={channels:<2} shape={shape:<10} "
        f"layout={layout:<17} border={border:<18} dispatch={dispatch}"
    )


@dataclass(frozen=True)
class SlowdownRule:
    selector: str
    max_slowdown: float


def normalize_token(token: str) -> str:
    return token.strip().upper()


def parse_slowdown_rule(text: str) -> SlowdownRule:
    """
    Parse one threshold rule:
      - OP:KERNEL=0.15  (exact match)
      - OP=0.12         (all kernels for op)
      - KERNEL=0.10     (all ops for kernel, e.g. 3x3)
      - *:KERNEL=0.10   (explicit kernel wildcard)
      - OP:*=0.08       (explicit op wildcard)
    """
    if "=" not in text:
        raise ValueError(f"invalid rule '{text}', expected SELECTOR=RATIO")
    selector_raw, ratio_raw = text.split("=", 1)
    selector_raw = selector_raw.strip()
    ratio_raw = ratio_raw.strip()
    if not selector_raw:
        raise ValueError(f"invalid rule '{text}', empty selector")

    try:
        ratio = float(ratio_raw)
    except ValueError as exc:
        raise ValueError(f"invalid ratio in rule '{text}'") from exc
    if ratio < 0.0:
        raise ValueError(f"ratio must be >= 0, got {ratio} in rule '{text}'")

    selector = normalize_token(selector_raw)
    if ":" not in selector:
        if re.match(r"^[0-9]+X[0-9]+$", selector):
            selector = f"*:{selector}"
        else:
            selector = f"{selector}:*"

    left, right = selector.split(":", 1)
    left = left.strip() or "*"
    right = right.strip() or "*"

    if left != "*" and not re.match(r"^[A-Z0-9_]+$", left):
        raise ValueError(f"invalid op selector '{left}' in rule '{text}'")
    if right != "*" and not re.match(r"^[0-9]+X[0-9]+$", right):
        raise ValueError(f"invalid kernel selector '{right}' in rule '{text}'")

    return SlowdownRule(selector=f"{left}:{right}", max_slowdown=ratio)


def resolve_threshold(
    op: str,
    kernel: str,
    default_ratio: float,
    exact_rules: Dict[Tuple[str, str], float],
    op_rules: Dict[str, float],
    kernel_rules: Dict[str, float],
) -> float:
    op_key = normalize_token(op)
    kernel_key = normalize_token(kernel)
    if (op_key, kernel_key) in exact_rules:
        return exact_rules[(op_key, kernel_key)]
    if op_key in op_rules:
        return op_rules[op_key]
    if kernel_key in kernel_rules:
        return kernel_rules[kernel_key]
    return default_ratio


def main() -> int:
    parser = argparse.ArgumentParser(description="Check imgproc filter benchmark regression against baseline CSV")
    parser.add_argument("--baseline", required=True, type=pathlib.Path, help="baseline CSV file")
    parser.add_argument("--current", required=True, type=pathlib.Path, help="current CSV file")
    parser.add_argument(
        "--max-slowdown",
        type=float,
        default=0.08,
        help="maximum allowed slowdown ratio (default: 0.08 == 8%%)",
    )
    parser.add_argument(
        "--max-slowdown-by-op-kernel",
        action="append",
        default=[],
        metavar="RULE",
        help=(
            "override slowdown threshold for specific slices. "
            "Format: OP:KERNEL=RATIO, OP=RATIO, or KERNEL=RATIO. Repeatable."
        ),
    )
    parser.add_argument(
        "--allow-missing-current",
        action="store_true",
        help="do not fail when baseline key is missing in current CSV",
    )
    args = parser.parse_args()

    baseline = load_csv(args.baseline)
    current = load_csv(args.current)

    rules: List[SlowdownRule] = []
    for text in args.max_slowdown_by_op_kernel:
        try:
            rules.append(parse_slowdown_rule(text))
        except ValueError as exc:
            print(f"[filter-benchmark-regression] invalid --max-slowdown-by-op-kernel: {exc}")
            return 2

    exact_rules: Dict[Tuple[str, str], float] = {}
    op_rules: Dict[str, float] = {}
    kernel_rules: Dict[str, float] = {}
    for rule in rules:
        op_selector, kernel_selector = rule.selector.split(":", 1)
        if op_selector != "*" and kernel_selector != "*":
            exact_rules[(op_selector, kernel_selector)] = rule.max_slowdown
        elif op_selector != "*" and kernel_selector == "*":
            op_rules[op_selector] = rule.max_slowdown
        elif op_selector == "*" and kernel_selector != "*":
            kernel_rules[kernel_selector] = rule.max_slowdown

    failures = []
    missing = []
    improved = 0
    compared = 0

    for key, base_row in baseline.items():
        cur_row = current.get(key)
        if cur_row is None:
            missing.append(key)
            continue

        compared += 1
        base = base_row.ms_per_iter
        cur = cur_row.ms_per_iter
        if base <= 0.0:
            continue

        op, kernel, _depth, _channels, _shape, _layout, _border, _dispatch = key
        max_allowed = resolve_threshold(
            op=op,
            kernel=kernel,
            default_ratio=args.max_slowdown,
            exact_rules=exact_rules,
            op_rules=op_rules,
            kernel_rules=kernel_rules,
        )

        ratio = cur / base - 1.0
        if ratio > max_allowed:
            failures.append((key, base, cur, ratio, max_allowed))
        elif ratio < 0.0:
            improved += 1

    if missing and not args.allow_missing_current:
        print("[filter-benchmark-regression] missing cases in current CSV:")
        for key in missing[:20]:
            print("  -", format_key(key))
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        print("Set --allow-missing-current to ignore missing cases.")
        return 2

    print(
        f"[filter-benchmark-regression] compared={compared}, improved_or_equal={compared - len(failures)}, "
        f"improved={improved}, default_threshold={args.max_slowdown * 100:.2f}%, "
        f"override_rules={len(rules)}"
    )

    if failures:
        failures.sort(key=lambda item: item[3], reverse=True)
        print("[filter-benchmark-regression] slowdown violations:")
        for key, base, cur, ratio, max_allowed in failures[:30]:
            print(
                f"  - {format_key(key)} | baseline={base:.6f} ms | current={cur:.6f} ms "
                f"| slowdown={ratio * 100:.2f}% | allowed={max_allowed * 100:.2f}%"
            )
        if len(failures) > 30:
            print(f"  ... and {len(failures) - 30} more")
        return 1

    print("[filter-benchmark-regression] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
