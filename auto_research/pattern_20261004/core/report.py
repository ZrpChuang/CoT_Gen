"""Read completed local metrics; incomplete benchmark cells stay visibly empty."""

import json

from .settings import BENCHMARKS, ROOT

NAMES = {
    "answer_only": "空CoT",
    "paired_plain": "普通段落",
    "paired_steps": "分步标签",
    "student_format": "答案NLL选择",
    "matched_random_format": "答案选择的匹配随机",
    "body_format": "正文NLL选择",
    "body_matched_random": "正文选择的匹配随机",
    "neu_lr1e5": "NEU低LR",
    "sup_lr1e5": "SUP低LR",
    "aug_sup_lr1e5": "AUG-SUP低LR",
    "ssr_lr1e5": "SSR低LR",
}


def primary(path, benchmark, judge="gpt-6-astra"):
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    if result.get("status") != "complete":
        return None
    if benchmark == "ifeval":
        value = result["metrics"]["prompt_strict"]["score"]
    elif benchmark == "multi_challenge":
        value = result["judges"][judge]["overall"]["score"]
    elif benchmark == "eq_bench3":
        value = result["judges"][judge]["rubric_0_100"]["score"]
    elif benchmark == "arena_hard_v2":
        value = result["judges"][judge]["weighted_win_rate"]["hard_prompt"]["score"]
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return float(value)


def render():
    lines = [
        "# CoT格式研究成绩",
        "",
        "## 开发集 screen100",
        "",
        "相对同一9B基座的配对胜负分，50为中性。",
        "",
        "| 方法 | 分数 | 题级95%区间 |",
        "|---|---:|---|",
    ]
    for group in ("anchors", "variants"):
        for path in sorted(
            (ROOT / "results/development" / group).glob("*/metrics.json")
        ):
            obj = json.loads(path.read_text())
            if obj.get("status") != "complete":
                continue
            value = obj["primary"]
            ci = value["ci95"]
            name = path.parent.name
            lines.append(
                f"| {NAMES.get(name, name)} | {value['score']:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] |"
            )
    lines += ["", "## 确认集 confirm200", ""]
    for path in sorted((ROOT / "results/confirmation").glob("*/metrics.json")):
        obj = json.loads(path.read_text())
        value = obj["primary"]
        ci = value["ci95"]
        lines.append(
            f"{NAMES.get(path.parent.name, path.parent.name)}：{value['score']:.2f}，95%区间[{ci[0]:.2f}, {ci[1]:.2f}]。"
        )
    lines += [
        "",
        "## 四benchmark终评",
        "",
        "| 方法 | IFEval | MultiChallenge | EQ-Bench3 | Arena-Hard-v2 |",
        "|---|---:|---:|---:|---:|",
    ]
    completed = 0
    for name in ("body_format", "sup_lr1e5", "aug_sup_lr1e5", "ssr_lr1e5"):
        cells = []
        for bench in BENCHMARKS:
            value = primary(
                ROOT / "results/benchmarks" / name / bench / "metrics.json", bench
            )
            completed += value is not None
            cells.append("—" if value is None else f"{value:.2f}")
        lines.append("| " + " | ".join([NAMES[name], *cells]) + " |")
    lines += [
        "",
        f"完成评分 {completed}/16 项。缺失或未完成项显示“—”，不能据此给出完整终评排名。",
        "",
    ]
    return "\n".join(lines)
