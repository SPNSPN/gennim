#! /usr/bin/env python3

import argparse
import html
import hashlib
import json
import os
import re
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


MARKER_RE = re.compile(r"^;\*\*\* (Input|Output)(\d+) \*\*\*$")
MNEMONIC_RE = re.compile(r"^([A-Z][A-Z0-9]*)\s*(.*)$")
OPERAND_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\S+')
INPUT_STATION_RE = re.compile(r"^# (ST\d+):\s*(.*)$")
INPUT_CYCLE_RE = re.compile(r"^## (C\d+):\s*(.*)$")
INPUT_MEMORY_RE = re.compile(r'^- 記憶"([^"]+)"$')
LOAD_OPS = {"LD", "LDI", "LDF"}
AND_OPS = {"AND", "ANDI", "ANI", "ANFI"}
CONTACT_OPS = LOAD_OPS | AND_OPS
OUTPUT_OPS = {"OUT", "SET", "RST", "MOV", "TON", "TOF"}


class LadderFormatError(ValueError):
    pass


@dataclass
class Section:
    number: str
    input_lines: list[str]
    output_lines: list[str]


def parse_sections(source: str) -> list[Section]:
    input_blocks: dict[str, list[str]] = {}
    output_blocks: dict[str, list[str]] = {}
    current_kind: str | None = None
    current_number: str | None = None
    current_lines: list[str] = []

    def finish_block() -> None:
        if current_kind is None or current_number is None:
            return
        target = input_blocks if current_kind == "Input" else output_blocks
        if current_number in target:
            raise LadderFormatError(f"duplicate {current_kind}{current_number} section")
        target[current_number] = current_lines.copy()

    for line_number, line in enumerate(source.splitlines(), 1):
        marker = MARKER_RE.fullmatch(line.strip())
        if marker:
            finish_block()
            current_kind, current_number = marker.groups()
            current_lines = []
        elif current_kind is not None:
            current_lines.append(line)
        elif line.strip():
            raise LadderFormatError(
                f"line {line_number}: text appears before the first Input/Output marker"
            )
    finish_block()

    if not input_blocks or not output_blocks:
        raise LadderFormatError("expected at least one Input/Output section")
    missing_outputs = sorted(input_blocks.keys() - output_blocks.keys(), key=int)
    missing_inputs = sorted(output_blocks.keys() - input_blocks.keys(), key=int)
    if missing_outputs or missing_inputs:
        details = []
        if missing_outputs:
            details.append("missing Output" + ", Output".join(missing_outputs))
        if missing_inputs:
            details.append("missing Input" + ", Input".join(missing_inputs))
        raise LadderFormatError("; ".join(details))

    return [
        Section(number, input_blocks[number], output_blocks[number])
        for number in sorted(input_blocks, key=int)
    ]


def parse_instruction(line: str) -> tuple[str, list[str], str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith(";"):
        return None
    body, separator, comment = stripped.partition(";")
    match = MNEMONIC_RE.fullmatch(body.strip())
    if not match:
        raise LadderFormatError(f"unsupported instruction line: {line.strip()}")
    mnemonic, operand_text = match.groups()
    operands = OPERAND_RE.findall(operand_text)
    return mnemonic, operands, comment.strip() if separator else ""


def make_contact(mnemonic: str, operands: list[str], comment: str) -> dict:
    return {
        "type": "contact",
        "mnemonic": mnemonic,
        "operand": operands[0] if operands else "(no operand)",
        "annotation": comment,
    }


def combine(kind: str, left: dict, right: dict) -> dict:
    if left.get("type") == kind:
        left["children"].append(right)
        return left
    return {"type": kind, "children": [left, right]}


def finish_rung(
    stack: list[dict], mnemonic: str, operands: list[str], comment: str, warnings: list[str], section_number: str
) -> dict:
    if not stack:
        expression = {"type": "contact", "mnemonic": "?", "operand": "No input", "annotation": ""}
        warnings.append(f"Output{section_number}: {mnemonic} has no preceding contact")
    else:
        expression = stack.pop(0)
        while stack:
            expression = combine("or", expression, stack.pop(0))
            warnings.append(f"Output{section_number}: unjoined contact branches were rendered as OR")
    return {
        "expression": expression,
        "output": {
            "mnemonic": mnemonic,
            "operands": operands,
            "annotation": comment,
        },
    }


def parse_rungs(lines: list[str], section_number: str) -> tuple[list[dict], list[str]]:
    rungs: list[dict] = []
    warnings: list[str] = []
    stack: list[dict] = []
    for line_number, line in enumerate(lines, 1):
        parsed = parse_instruction(line)
        if parsed is None:
            continue
        mnemonic, operands, comment = parsed
        if mnemonic in LOAD_OPS:
            if not operands:
                warnings.append(
                    f"Output{section_number}, line {line_number}: {mnemonic} has no operand"
                )
            stack.append(make_contact(mnemonic, operands, comment))
        elif mnemonic in {"ORB", "OR", "ORI"}:
            if len(stack) < 2:
                warnings.append(
                    f"Output{section_number}, line {line_number}: {mnemonic} has fewer than two branches"
                )
                stack.append(
                    {"type": "contact", "mnemonic": "?", "operand": "Invalid branch", "annotation": ""}
                )
            else:
                right = stack.pop()
                left = stack.pop()
                stack.append(combine("or", left, right))
        elif mnemonic in {"ORL", "ORLD"}:
            warnings.append(
                f"Output{section_number}, line {line_number}: {mnemonic} is not visually supported"
            )
            stack.append(
                {"type": "contact", "mnemonic": mnemonic, "operand": "unsupported", "annotation": ""}
            )
        elif mnemonic in AND_OPS:
            if not operands:
                warnings.append(
                    f"Output{section_number}, line {line_number}: {mnemonic} has no operand"
                )
                continue
            contact = make_contact(mnemonic, operands, comment)
            if stack:
                stack[-1] = combine("and", stack[-1], contact)
            else:
                stack.append(contact)
        elif mnemonic in OUTPUT_OPS or mnemonic not in CONTACT_OPS | {
            "ORB", "OR", "ORI", "ORL", "ORLD"
        }:
            rung = finish_rung(stack, mnemonic, operands, comment, warnings, section_number)
            if mnemonic not in OUTPUT_OPS:
                rung["output"]["unsupported"] = True
                warnings.append(
                    f"Output{section_number}, line {line_number}: unknown mnemonic {mnemonic}"
                )
            rungs.append(rung)
            stack = []
        else:
            warnings.append(
                f"Output{section_number}, line {line_number}: ignored {mnemonic}"
            )
    if stack:
        warnings.append(f"Output{section_number}: contacts remain without an output instruction")
    return rungs, warnings


def describe_section(section: Section) -> dict:
    station: str | None = None
    station_name = ""
    cycle: str | None = None
    cycle_name = ""
    memory = ""
    for raw_line in section.input_lines:
        line = raw_line.strip()
        if line.startswith(";"):
            line = line[1:].strip()
        station_match = INPUT_STATION_RE.fullmatch(line)
        cycle_match = INPUT_CYCLE_RE.fullmatch(line)
        memory_match = INPUT_MEMORY_RE.fullmatch(line)
        if station_match:
            station, station_name = station_match.groups()
        elif cycle_match:
            cycle, cycle_name = cycle_match.groups()
        elif memory_match:
            memory = memory_match.group(1)

    if station and cycle:
        title = f"{station}: {station_name} / {cycle}: {cycle_name}"
        group = f"{station}: {station_name}"
    elif memory:
        title = f"記憶フラグ: {memory}"
        group = "記憶フラグ"
    else:
        title = f"Input{section.number}"
        group = "その他"
    rungs, warnings = parse_rungs(section.output_lines, section.number)
    return {
        "number": section.number,
        "title": title,
        "group": group,
        "station": station or "",
        "cycle": cycle or "",
        "memory": memory,
        "rungs": rungs,
        "warnings": warnings,
    }


def count_unqualified_step_refs(expression: dict) -> int:
    if expression["type"] == "contact":
        return int(bool(re.fullmatch(r"Step\d+_(?:Start|End)", expression["operand"])))
    return sum(count_unqualified_step_refs(child) for child in expression["children"])


def build_model(source: str) -> dict:
    sections = parse_sections(source)
    described_sections = [describe_section(section) for section in sections]
    unqualified_by_section = {
        section["number"]: sum(
            count_unqualified_step_refs(rung["expression"])
            for rung in section["rungs"]
        )
        for section in described_sections
    }
    unqualified_by_section = {
        number: count for number, count in unqualified_by_section.items() if count
    }
    global_warnings = []
    unqualified_count = sum(unqualified_by_section.values())
    if unqualified_count:
        global_warnings.append(
            f"{unqualified_count}件の StepN_Start/StepN_End がステーション・サイクル名なしで"
            "記述されています。表示では入力どおりに扱い、接頭辞を推測していません。"
        )
    return {
        "sections": described_sections,
        "section_count": len(sections),
        "global_warnings": global_warnings,
        "unqualified_steps": unqualified_by_section,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def render_html(template: str, model: dict, signature: str, source_name: str) -> str:
    embedded_model = json.dumps(model, ensure_ascii=False, separators=(",", ":"))
    embedded_model = (
        embedded_model.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    html = template.replace("__MODEL_JSON__", embedded_model)
    html = html.replace("__SOURCE_SIGNATURE__", signature)
    html = html.replace("__SOURCE_NAME_JSON__", html_escape(source_name))
    if "__MODEL_JSON__" in html or "__SOURCE_SIGNATURE__" in html or "__SOURCE_NAME_JSON__" in html:
        raise LadderFormatError("HTML template contains an unresolved placeholder")
    return html


def html_escape(value: str) -> str:
    return html.escape(value, quote=True)


class LadderApplication:
    def __init__(self, source_path: Path, output_path: Path, template_path: Path):
        self.source_path = source_path.resolve()
        self.output_path = output_path.resolve()
        self.template_path = template_path.resolve()
        self.lock = threading.RLock()
        self.signature = ""
        self.error = ""
        self.last_updated = ""
        self.model: dict = {}
        self._source_bytes()
        self._refresh()

    def _source_bytes(self) -> bytes:
        if not self.source_path.is_file():
            raise FileNotFoundError(f"source file not found: {self.source_path}")
        return self.source_path.read_bytes()

    def _refresh(self) -> bool:
        source_bytes = self._source_bytes()
        signature = hashlib.sha256(source_bytes).hexdigest()
        with self.lock:
            if signature == self.signature:
                return False
        try:
            source_text = source_bytes.decode("utf-8-sig")
            model = build_model(source_text)
            template = self.template_path.read_text(encoding="utf-8")
            html = render_html(template, model, signature, self.source_path.name)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.output_path.with_name(self.output_path.name + ".tmp")
            try:
                temporary.write_text(html, encoding="utf-8", newline="\n")
                os.replace(temporary, self.output_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        except (OSError, UnicodeDecodeError, LadderFormatError) as exc:
            with self.lock:
                self.error = str(exc)
                self.signature = signature
            return False
        with self.lock:
            self.model = model
            self.signature = signature
            self.error = ""
            self.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")
        return True

    def watch(self, poll_interval: float) -> None:
        while True:
            time.sleep(poll_interval)
            try:
                self._refresh()
            except OSError as exc:
                with self.lock:
                    self.error = str(exc)
                    self.signature = ""

    def status(self) -> dict:
        with self.lock:
            return {
                "signature": self.signature,
                "error": self.error,
                "last_updated": self.last_updated,
                "source": self.source_path.name,
            }


def make_handler(application: LadderApplication):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = urlparse(self.path).path
            if route == "/api/status":
                body = json.dumps(application.status(), ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if route in {"/", f"/{application.output_path.name}"}:
                try:
                    body = application.output_path.read_bytes()
                except OSError as exc:
                    self.send_error(500, f"Unable to read generated HTML: {exc}")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404, "Not found")

        def log_message(self, format_string: str, *args: object) -> None:
            sys.stdout.write(
                f"{self.log_date_time_string()} {format_string % args}\n"
            )

    return Handler


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Render a read-only HTML ladder viewer and refresh it when the source changes."
    )
    parser.add_argument("--input", type=Path, default=base / "lad_out.txt")
    parser.add_argument("--output", type=Path, default=base / "ladder.html")
    parser.add_argument(
        "--template", type=Path, default=base / "ladder_viewer_template.html"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.poll_interval <= 0:
        print("LadderViewer: --poll-interval must be greater than zero", file=sys.stderr)
        return 2
    if not 0 <= args.port <= 65535:
        print("LadderViewer: --port must be between 0 and 65535", file=sys.stderr)
        return 2
    if args.input.resolve() == args.output.resolve():
        print("LadderViewer: input and output paths must differ", file=sys.stderr)
        return 2
    if args.template.resolve() == args.output.resolve():
        print("LadderViewer: template and output paths must differ", file=sys.stderr)
        return 2
    try:
        application = LadderApplication(args.input, args.output, args.template)
        server = ThreadingHTTPServer(
            (args.host, args.port), make_handler(application)
        )
    except (OSError, LadderFormatError) as exc:
        print(f"LadderViewer: {exc}", file=sys.stderr)
        return 1

    watcher = threading.Thread(
        target=application.watch,
        args=(args.poll_interval,),
        name="ladder-source-watcher",
        daemon=True,
    )
    watcher.start()
    host, port = server.server_address[:2]
    page_url = f"http://{host}:{port}/"
    print(f"HTML: {args.output.resolve()}")
    print(f"Viewer: {page_url}")
    print(f"Watching: {args.input.resolve()}")
    if not args.no_open:
        webbrowser.open(page_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nLadderViewer: stopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
