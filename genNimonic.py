#! /usr/bin/env python3

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


TOKEN_RE = re.compile(
    r'\s*(?:(?P<op>&&|\|\||==|!=|>=|<=|[()<>])|'
    r'(?P<not>not(?=\())|'
    r'(?P<string>"(?:[^"\\]|\\.)*")|'
    r'(?P<word>[\w]+(?:\([^()]*\))?))',
    re.UNICODE,
)
JAPANESE_RE = re.compile(r"[ぁ-んァ-ヶ一-龯]")
STEP_END_RE = re.compile(r"\bStep(\d+)_End\b")
STEP_START_RE = re.compile(r"\bStep(\d+)_Start\b")


class ConversionError(ValueError):
    pass


@dataclass
class Station:
    number: str
    name: str
    header: str
    memories: list[str] = field(default_factory=list)
    cycles: list["Cycle"] = field(default_factory=list)


@dataclass
class Action:
    condition: str
    operator: str
    operation: str
    line_number: int
    source_text: str


@dataclass
class Cycle:
    station: Station
    number: str
    name: str
    header: str
    start: str | None = None
    end: str | None = None
    step_ends: dict[int, str] = field(default_factory=dict)
    actions: list[Action] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)
    line_number: int = 0

    @property
    def tag(self) -> str:
        return f"{self.station.number}_{self.number}"


@dataclass(frozen=True)
class Expr:
    kind: str
    value: str = ""
    left: "Expr | None" = None
    right: "Expr | None" = None


class ExpressionParser:
    def __init__(self, text: str, line_number: int):
        self.text = text
        self.line_number = line_number
        self.tokens: list[str] = []
        position = 0
        while position < len(text):
            match = TOKEN_RE.match(text, position)
            if not match:
                raise ConversionError(
                    f"line {line_number}: unsupported expression near {text[position:]!r}"
                )
            self.tokens.append(next(value for value in match.groupdict().values() if value))
            position = match.end()
        self.index = 0

    def parse(self) -> Expr:
        if not self.tokens:
            raise ConversionError(f"line {self.line_number}: empty expression")
        result = self._parse_or()
        if self.index != len(self.tokens):
            raise ConversionError(
                f"line {self.line_number}: unexpected token {self.tokens[self.index]!r}"
            )
        return result

    def _parse_or(self) -> Expr:
        result = self._parse_and()
        while self._peek("||"):
            self.index += 1
            result = Expr("or", left=result, right=self._parse_and())
        return result

    def _parse_and(self) -> Expr:
        result = self._parse_comparison()
        while self._peek("&&"):
            self.index += 1
            result = Expr("and", left=result, right=self._parse_comparison())
        return result

    def _parse_comparison(self) -> Expr:
        result = self._parse_unary()
        if self.index < len(self.tokens) and self.tokens[self.index] in {
            "==", "!=", ">=", "<=", ">", "<"
        }:
            operator = self.tokens[self.index]
            self.index += 1
            result = Expr("compare", operator, result, self._parse_unary())
        return result

    def _parse_unary(self) -> Expr:
        if self._peek("not"):
            self.index += 1
            return Expr("not", left=self._parse_unary())
        if self._peek("("):
            self.index += 1
            result = self._parse_or()
            self._expect(")")
            return result
        if self.index >= len(self.tokens):
            raise ConversionError(f"line {self.line_number}: expected an operand")
        token = self.tokens[self.index]
        if token in {"&&", "||", ")", "==", "!=", ">=", "<=", ">", "<"}:
            raise ConversionError(f"line {self.line_number}: unexpected token {token!r}")
        self.index += 1
        return Expr("atom", token)

    def _peek(self, token: str) -> bool:
        return self.index < len(self.tokens) and self.tokens[self.index] == token

    def _expect(self, token: str) -> None:
        if not self._peek(token):
            raise ConversionError(f"line {self.line_number}: expected {token!r}")
        self.index += 1


def _negated_comparison(operator: str) -> str:
    return {"==": "!=", "!=": "==", ">": "<=", ">=": "<", "<": ">=", "<=": ">"}[
        operator
    ]


def to_dnf(expr: Expr, negate: bool = False) -> list[list[tuple[Expr, bool]]]:
    if expr.kind == "not":
        return to_dnf(expr.left, not negate)
    if expr.kind in {"and", "or"}:
        operator = expr.kind
        if negate:
            operator = "or" if operator == "and" else "and"
        left = to_dnf(expr.left, negate)
        right = to_dnf(expr.right, negate)
        if operator == "or":
            return left + right
        return [left_clause + right_clause for left_clause in left for right_clause in right]
    if expr.kind == "compare" and negate:
        return [[(Expr("compare", _negated_comparison(expr.value), expr.left, expr.right), False)]]
    return [[(expr, negate)]]


def load_symbols(path: Path) -> dict[str, str]:
    symbols: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ConversionError(f"cannot read symbol map {path}: {exc}") from exc

    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConversionError(f"{path}:{line_number}: expected JapaneseName=PLCTag")
        japanese, plc_name = (part.strip() for part in line.split("=", 1))
        if not japanese or not plc_name:
            raise ConversionError(f"{path}:{line_number}: symbol names cannot be empty")
        if JAPANESE_RE.search(plc_name):
            raise ConversionError(
                f"{path}:{line_number}: PLC tag must not contain Japanese characters"
            )
        if japanese in symbols:
            raise ConversionError(f"{path}:{line_number}: duplicate Japanese symbol {japanese!r}")
        symbols[japanese] = plc_name
    return symbols


class SymbolResolver:
    def __init__(self, symbols: dict[str, str]):
        self.symbols = symbols
        self.unknown: dict[str, str] = {}
        self.used_names = set(symbols.values())
        self.next_unknown = 1

    def resolve(self, token: str) -> tuple[str, str | None]:
        if not JAPANESE_RE.search(token):
            self.used_names.add(token)
            return token, None
        if token not in self.unknown and token not in self.symbols:
            while f"XXX{self.next_unknown}" in self.used_names:
                self.next_unknown += 1
            self.unknown[token] = f"XXX{self.next_unknown}"
            self.used_names.add(self.unknown[token])
            self.next_unknown += 1
        return self.symbols.get(token, self.unknown.get(token)), token

    def prime_in_source_order(self, text: str) -> None:
        action_re = re.compile(r"^-\s*(.*?)\s*(-SET->|-RESET->|-TIMER->|->)\s*(.*?)\s*$")
        step_end_re = re.compile(r"^- Step\d+_End:\s*(.*)$")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("### Start:") or line.startswith("### End:"):
                body = line.partition(":")[2]
            elif step_end_re.match(line):
                body = step_end_re.match(line).group(1)
            elif line.startswith('- 記憶"'):
                match = re.fullmatch(r'- 記憶"([^"]+)"', line)
                body = match.group(1) if match else ""
            else:
                action = action_re.match(line)
                if not action:
                    continue
                body = f"{action.group(1)} {action.group(3)}"
                mov_match = re.search(r'MOV\s+("(?:[^"\\]|\\.)*"|[\w]+)\s+([\w]+)', body)
                if mov_match:
                    body = f"{action.group(1)} {mov_match.group(2)}"
            for match in TOKEN_RE.finditer(body):
                token = next(value for value in match.groupdict().values() if value)
                if token not in {"&&", "||", "==", "!=", ">=", "<=", ">", "<", "(", ")"}:
                    if token.startswith('"') and token.endswith('"'):
                        continue
                    self.resolve(token)


def instruction(opcode: str, operands: list[str], resolver: SymbolResolver) -> str:
    converted: list[str] = []
    comments: list[str] = []
    for operand in operands:
        if operand.startswith('"') and operand.endswith('"'):
            value = operand[1:-1]
            converted.append(f"K{value}" if re.fullmatch(r"\d+", value) else operand)
            continue
        if re.fullmatch(r"\d+", operand) and opcode in {"LD=", "AND=", "OR=", "LD<>", "AND<>", "OR<>"}:
            converted.append(f"K{operand}")
            continue
        value, original = resolver.resolve(operand)
        converted.append(value)
        if original and original not in comments:
            comments.append(original)
    suffix = f" ; {'、'.join(comments)}" if comments else ""
    return f"{opcode} {' '.join(converted)}{suffix}".rstrip()


def emit_expression(expr: Expr, resolver: SymbolResolver, line_number: int) -> list[str]:
    clauses = to_dnf(expr)
    if len(clauses) > 256:
        raise ConversionError(f"line {line_number}: expression expands to too many ladder branches")
    output: list[str] = []
    for clause_index, clause in enumerate(clauses):
        for literal_index, (literal, negated) in enumerate(clause):
            if literal.kind == "compare":
                operator = literal.value
                if negated:
                    operator = _negated_comparison(operator)
                mnemonic = {"==": "=", "!=": "<>", ">=": ">=", "<=": "<=", ">": ">", "<": "<"}[
                    operator
                ]
                prefix = "LD" if literal_index == 0 else "AND"
                opcode = f"{prefix}{mnemonic}"
                operands = [literal.left.value, literal.right.value]
            else:
                atom = literal.value
                edge = atom.startswith("fall(") and atom.endswith(")")
                if edge:
                    atom = atom[5:-1]
                if edge and negated:
                    raise ConversionError(
                        f"line {line_number}: negated fall() conditions are not supported"
                    )
                if literal_index == 0:
                    opcode = "LDF" if edge else ("LDI" if negated else "LD")
                else:
                    prefix = "AND"
                    opcode = f"{prefix}{'F' if edge else ('I' if negated else '')}"
                operands = [atom]
            output.append(instruction(opcode, operands, resolver))
        if clause_index:
            output.append("ORB")
    return output


def parse_document(text: str) -> list[Cycle]:
    stations: list[Station] = []
    station: Station | None = None
    cycle: Cycle | None = None
    cycle_header_re = re.compile(r"^## (C\d+):\s*(.*)$")
    station_header_re = re.compile(r"^# (ST\d+):\s*(.*)$")
    step_end_re = re.compile(r"^- Step(\d+)_End:\s*(.*)$")
    memory_re = re.compile(r'^- 記憶"([^"]+)"$')
    action_re = re.compile(r"^-\s*(.*?)\s*(-SET->|-RESET->|-TIMER->|->)\s*(.*?)\s*$")

    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            if cycle:
                cycle.raw_lines.append("")
            continue
        station_match = station_header_re.match(line)
        if station_match:
            station = Station(
                station_match.group(1), station_match.group(2), raw_line
            )
            stations.append(station)
            cycle = None
            continue
        cycle_match = cycle_header_re.match(line)
        if cycle_match:
            if station is None:
                raise ConversionError(f"line {line_number}: cycle appears before a station")
            cycle = Cycle(
                station,
                cycle_match.group(1),
                cycle_match.group(2),
                raw_line,
                raw_lines=[raw_line],
                line_number=line_number,
            )
            station.cycles.append(cycle)
            continue
        memory_match = memory_re.match(line)
        if memory_match:
            if station is None or cycle is not None:
                raise ConversionError(f"line {line_number}: memory directive must be at station level")
            station.memories.append(raw_line)
            continue
        if cycle is None:
            raise ConversionError(f"line {line_number}: expected a station or cycle heading")
        cycle.raw_lines.append(raw_line)
        if line.startswith("### Start:"):
            if cycle.start is not None:
                raise ConversionError(f"line {line_number}: duplicate Start condition")
            cycle.start = line.partition(":")[2].strip()
        elif line.startswith("### End:"):
            if cycle.end is not None:
                raise ConversionError(f"line {line_number}: duplicate End condition")
            cycle.end = line.partition(":")[2].strip()
        else:
            end_match = step_end_re.match(line)
            if end_match:
                step_number = int(end_match.group(1))
                if step_number in cycle.step_ends:
                    raise ConversionError(f"line {line_number}: duplicate Step{step_number}_End")
                cycle.step_ends[step_number] = end_match.group(2).strip()
                continue
            action_match = action_re.match(line)
            if action_match:
                condition = action_match.group(1).strip()
                if re.fullmatch(r"Step\d+", condition):
                    condition += "_Start"
                cycle.actions.append(
                    Action(
                        condition,
                        action_match.group(2),
                        action_match.group(3).strip(),
                        line_number,
                        raw_line,
                    )
                )
                continue
            raise ConversionError(f"line {line_number}: unsupported rule {line!r}")

    cycles = [cycle for item in stations for cycle in item.cycles]
    if not cycles:
        raise ConversionError("input contains no station cycles")
    for item in stations:
        if not item.cycles:
            raise ConversionError(f"{item.number} has no cycles")
    return cycles


def read_spec(path: Path) -> None:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ConversionError(f"cannot read rule reference {path}: {exc}") from exc
    markers = re.findall(r"^;\*\*\* (Input|Output)(\d+) \*\*\*$", "\n".join(lines), re.M)
    inputs = {number for kind, number in markers if kind == "Input"}
    outputs = {number for kind, number in markers if kind == "Output"}
    if not inputs or inputs != outputs:
        raise ConversionError(
            f"{path}: expected matching ;*** InputN *** and ;*** OutputN *** examples"
        )
    if not any(line.strip() and not line.startswith(";***") for line in lines):
        raise ConversionError(f"{path}: rule reference contains no example content")


def validate_cycle(cycle: Cycle) -> list[int]:
    if cycle.start is None:
        raise ConversionError(f"{cycle.station.number} {cycle.number}: missing Start condition")
    if not cycle.step_ends:
        raise ConversionError(f"{cycle.station.number} {cycle.number}: no StepN_End rules")

    referenced_ends: set[int] = set()
    referenced_starts: set[int] = set()
    expressions = [cycle.start, cycle.end or ""]
    expressions.extend(cycle.step_ends.values())
    expressions.extend(action.condition for action in cycle.actions)
    for expression in expressions:
        referenced_ends.update(int(value) for value in STEP_END_RE.findall(expression))
        referenced_starts.update(int(value) for value in STEP_START_RE.findall(expression))
    missing_ends = sorted(referenced_ends - cycle.step_ends.keys())
    if missing_ends:
        missing = ", ".join(f"Step{number}_End" for number in missing_ends)
        raise ConversionError(
            f"{cycle.station.number} {cycle.number}: referenced but undefined {missing}"
        )
    steps = sorted(set(cycle.step_ends) | referenced_starts)
    missing_predecessors = [
        number for number in steps[1:] if not any(previous < number for previous in cycle.step_ends)
    ]
    if missing_predecessors:
        raise ConversionError(
            f"{cycle.station.number} {cycle.number}: no preceding StepN_End for "
            + ", ".join(f"Step{number}_Start" for number in missing_predecessors)
        )
    return steps


def parse_expr(text: str, line_number: int) -> Expr:
    return ExpressionParser(text, line_number).parse()


def qualify_step_references(text: str, cycle: Cycle) -> str:
    return re.sub(
        r"\bStep(\d+)_(Start|End)\b",
        lambda match: f"{cycle.tag}_Step{match.group(1)}_{match.group(2)}",
        text,
    )


def append_gated_coil(
    output: list[str],
    condition_lines: list[str],
    latch_tag: str,
    gate_tag: str,
    resolver: SymbolResolver,
    coil_opcode: str = "OUT",
) -> None:
    output.extend(condition_lines)
    output.append(instruction("LD", [latch_tag], resolver))
    output.append("ORB")
    output.append(instruction("AND", [gate_tag], resolver))
    output.append(instruction(coil_opcode, [latch_tag], resolver))


def generate_cycle(cycle: Cycle, resolver: SymbolResolver) -> list[str]:
    steps = validate_cycle(cycle)
    tag = cycle.tag
    cycle_start = f"{tag}_Start"
    cycle_end = f"{tag}_End"
    output = [f"; {cycle.station.number}: {cycle.station.name}; {cycle.number}: {cycle.name}"]

    start_lines = emit_expression(parse_expr(cycle.start, cycle.line_number), resolver, cycle.line_number)
    start_lines.append(instruction("LD", [cycle_start], resolver))
    start_lines.append("ORB")
    start_lines.append(instruction("AND", ["Ready"], resolver))
    start_lines.append(instruction("ANI", [cycle_end], resolver))
    start_lines.append(instruction("OUT", [cycle_start], resolver))
    output.extend(start_lines)
    output.append("")

    for index, step_number in enumerate(steps):
        step_start = f"{tag}_Step{step_number}_Start"
        if index == 0:
            predecessor = cycle_start
        else:
            previous_steps = [number for number in cycle.step_ends if number < step_number]
            predecessor = f"{tag}_Step{max(previous_steps)}_End"
        output.extend(
            [
                instruction("LD", [predecessor], resolver),
                instruction("AND", ["AlwaysOn"], resolver),
                instruction("LD", [step_start], resolver),
                "ORB",
                instruction("AND", [cycle_start], resolver),
                instruction("OUT", [step_start], resolver),
                "",
            ]
        )

    for step_number in sorted(cycle.step_ends):
        step_start = f"{tag}_Step{step_number}_Start"
        step_end = f"{tag}_Step{step_number}_End"
        condition = cycle.step_ends[step_number] or "AlwaysOn"
        expression = Expr(
            "and",
            left=Expr("atom", step_start),
            right=parse_expr(condition, cycle.line_number),
        )
        condition_lines = emit_expression(expression, resolver, cycle.line_number)
        append_gated_coil(
            output, condition_lines, step_end, cycle_start, resolver
        )
        output.append("")

    last_step_end = f"{tag}_Step{max(cycle.step_ends)}_End"
    end_condition = cycle.end.strip() if cycle.end and cycle.end.strip() else "AlwaysOn"
    end_expression = Expr(
        "and",
        left=Expr("atom", last_step_end),
        right=parse_expr(end_condition, cycle.line_number),
    )
    append_gated_coil(
        output,
        emit_expression(end_expression, resolver, cycle.line_number),
        cycle_end,
        cycle_start,
        resolver,
    )
    output.append("")

    for action in cycle.actions:
        if action.operator in {"-SET->", "-RESET->"}:
            continue
        condition = parse_expr(
            qualify_step_references(action.condition, cycle), action.line_number
        )
        output.extend(emit_expression(condition, resolver, action.line_number))
        operation = action.operation
        if operation.upper().startswith("MOV "):
            mov_match = re.fullmatch(r'MOV\s+("(?:[^"\\]|\\.)*"|[\w]+)\s+([\w]+)', operation)
            if not mov_match:
                raise ConversionError(
                    f"line {action.line_number}: expected MOV <value> <destination>"
                )
            output.append(instruction("MOV", [mov_match.group(1), mov_match.group(2)], resolver))
        elif action.operator == "-TIMER->":
            output.append(instruction("TON", [operation], resolver))
        else:
            output.append(instruction("OUT", [operation], resolver))
        output.append("")
    return output


def merge_out_coils(output_blocks: list[list[str]]) -> list[list[str]]:
    block_items: list[list[tuple[str, object]]] = [[] for _ in output_blocks]
    coil_groups: dict[str, list[list[str]]] = {}
    coil_lines: dict[str, str] = {}

    for block_index, lines in enumerate(output_blocks):
        chunk: list[str] = []

        def save_chunk() -> None:
            if not chunk:
                return
            output_match = re.fullmatch(
                r"OUT\s+(\S+)(?:\s*;.*)?", chunk[-1].strip()
            )
            if output_match:
                target = output_match.group(1)
                branch = chunk[:-1]
                if not branch:
                    raise ConversionError(f"OUT {target} has no condition")
                if target not in coil_groups:
                    coil_groups[target] = []
                    coil_lines[target] = chunk[-1]
                    block_items[block_index].append(("coil", target))
                coil_groups[target].append(branch)
            else:
                block_items[block_index].append(("raw", chunk.copy()))
            chunk.clear()

        for line in lines:
            if not line.strip():
                save_chunk()
            else:
                chunk.append(line)
        save_chunk()

    merged_blocks: list[list[str]] = []
    for items in block_items:
        rendered: list[str] = []
        for kind, value in items:
            if kind == "raw":
                rendered.extend(value)
                rendered.append("")
                continue
            target = value
            branches = coil_groups[target]
            for branch in branches:
                rendered.extend(branch)
            rendered.extend(["ORB"] * (len(branches) - 1))
            rendered.append(coil_lines[target])
            rendered.append("")
        merged_blocks.append(rendered)
    return merged_blocks


def convert(input_text: str, symbols: dict[str, str]) -> str:
    cycles = parse_document(input_text)
    resolver = SymbolResolver(symbols)
    resolver.prime_in_source_order(input_text)
    input_blocks: list[list[str]] = []
    output_blocks: list[list[str]] = []
    for index, cycle in enumerate(cycles, 1):
        input_lines = [cycle.station.header, *cycle.station.memories, *cycle.raw_lines]
        input_blocks.append([";" + line for line in input_lines])
        output_blocks.append(generate_cycle(cycle, resolver))

    memory_names: list[str] = []
    for cycle in cycles:
        for memory_line in cycle.station.memories:
            match = re.fullmatch(r'\s*-\s*記憶"([^"]+)"\s*', memory_line)
            if match and match.group(1) not in memory_names:
                memory_names.append(match.group(1))

    special_actions: dict[str, list[tuple[Cycle, Action]]] = {}
    for cycle in cycles:
        for action in cycle.actions:
            if action.operator in {"-SET->", "-RESET->"}:
                special_actions.setdefault(action.operation, []).append((cycle, action))

    undeclared = sorted(set(special_actions) - set(memory_names))
    if undeclared:
        raise ConversionError(
            "SET/RESET targets must be declared with 記憶: " + ", ".join(undeclared)
        )

    for memory_name in memory_names:
        actions = special_actions.get(memory_name, [])
        set_actions = [(cycle, action) for cycle, action in actions if action.operator == "-SET->"]
        reset_actions = [(cycle, action) for cycle, action in actions if action.operator == "-RESET->"]
        if not set_actions or not reset_actions:
            raise ConversionError(
                f'記憶"{memory_name}" must have at least one -SET-> and one -RESET-> rule'
            )
        input_blocks.append(
            [
                f';- 記憶"{memory_name}"',
                *(";" + action.source_text for _, action in actions),
            ]
        )
        memory_output: list[str] = []

        set_expression = parse_expr(set_actions[0][1].condition, set_actions[0][1].line_number)
        for _, action in set_actions[1:]:
            set_expression = Expr(
                "or",
                left=set_expression,
                right=parse_expr(action.condition, action.line_number),
            )
        memory_output.extend(
            emit_expression(set_expression, resolver, set_actions[0][1].line_number)
        )
        memory_output.append(instruction("LD", [memory_name], resolver))
        memory_output.append("ORB")
        for _, action in reset_actions:
            match = re.fullmatch(r"fall\(([\w]+)\)", action.condition.strip())
            if not match:
                raise ConversionError(
                    f"line {action.line_number}: reset rule for {memory_name} "
                    "must use fall(signal)"
                )
            memory_output.append(instruction("ANFI", [match.group(1)], resolver))
        memory_output.append(f'OUT {memory_name} ; 記憶"{memory_name}"')
        output_blocks.append(memory_output)

    output_blocks = merge_out_coils(output_blocks)
    output: list[str] = []
    for index, (input_lines, output_lines) in enumerate(
        zip(input_blocks, output_blocks), 1
    ):
        output.append(f";*** Input{index} ***")
        output.extend(input_lines)
        output.extend(["", f";*** Output{index} ***"])
        output.extend(output_lines)
        output.extend(["", ""])
    return "\n".join(output).rstrip() + "\n"


def optargs() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert Lad.txt rules into PLC mnemonic output using Lad.mnm as the rule reference."
    )
    parser.add_argument("--input", type=Path, default=base_dir / "lad.txt")
    parser.add_argument("--spec", type=Path, default=base_dir / "lad.mnm")
    parser.add_argument("--symbols", type=Path, default=base_dir / "Symbols.txt")
    parser.add_argument("--output", type=Path, default=base_dir / "lad_out.txt")
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    try:
        read_spec(args.spec)
        symbols = load_symbols(args.symbols)
        input_text = args.input.read_text(encoding="utf-8-sig")
        result = convert(input_text, symbols)
        if args.output.resolve() == args.input.resolve():
            raise ConversionError("output path must differ from input path")
        args.output.write_text(result, encoding="utf-8", newline="\n")
    except (OSError, ConversionError) as exc:
        print(f"GenNimonic: {exc}", file=sys.stderr)
        return 1
    print(f"Converted {args.input} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(optargs()))
