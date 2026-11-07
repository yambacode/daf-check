"""`daf-check FILE [FILE...] [--json]`: the command-line checker.

Exit status: 0 if a mark in any file says AI-generated, 1 if none does, 2 on a usage error or a
file that could not be read.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from . import __version__
from .check import Result, check

EXIT_AI, EXIT_NONE, EXIT_ERROR = 0, 1, 2


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # usage errors exit 2 (argparse's default too)
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, f"{self.prog}: error: {message}\n")


def _parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="daf-check",
        description=(
            "Checks files for Dream Art Fields' AI marks: the invisible pixel signature and the "
            "signed C2PA manifest. Runs locally; nothing is uploaded."
        ),
        epilog="Exit status: 0 = a mark says AI-generated, 1 = no mark found, 2 = error.",
    )
    p.add_argument("files", nargs="+", metavar="FILE", help="picture, video or audio file")
    p.add_argument("--json", action="store_true", help="print the results as JSON")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _watermark_line(r: Result) -> str:
    w = r.watermark
    if w is not None:
        flag = "AI-generated" if w.ai else "no AI flag"
        return (
            f"found: version {w.version}, {flag}, asset #{w.asset_hash}, "
            f"confidence {w.confidence:.2f}"
        )
    if not r.watermark_checked:
        return f"not checked ({r.watermark_note})" if r.watermark_note else "not checked"
    return "not found" + (f" ({r.watermark_note})" if r.watermark_note else "")


def _c2pa_lines(r: Result) -> list[str]:
    c = r.c2pa
    if not c.present:
        return ["not found"]
    if not c.verified:
        claim = "says AI-generated" if c.ai_generated else "no AI claim found"
        return [f"present, {claim}; signature not checked (install daf-check[c2pa])"]
    if not c.intact:
        lines = [f"present but does not check out ({c.state or 'Invalid'})"]
        lines += [f"  {e}" for e in c.errors[:3]]
        return lines
    trust = "signer on the C2PA trust list" if c.trusted else "signer not on the C2PA trust list"
    lines = [f"signature valid ({c.state}), {trust}"]
    claim = f"yes ({c.source_type.rsplit('/', 1)[-1]})" if c.ai_generated else "no"
    lines.append(f"AI claim: {claim}")
    if c.generator:
        lines.append(f"generator: {c.generator}")
    if c.system or c.provider:
        lines.append(f"system: {c.system or '?'} by {c.provider or '?'}")
    if c.signer or c.issuer:
        org = f" ({c.issuer})" if c.issuer and c.issuer != c.signer else ""
        lines.append(f"signed by: {c.signer}{org}")
    if c.signed_at:
        when = "time-stamped" if c.time_stamped else "signer's clock, not time-stamped"
        lines.append(f"time: {c.signed_at} ({when})")
    return lines


def _text(r: Result) -> str:
    verdict = "no Dream Art Fields mark found"
    if r.ai_generated:
        verdict = "AI-generated (a mark says so)"
    out = [r.source, f"  verdict         : {verdict}", f"  summary         : {r.summary()}"]
    out.append(f"  pixel signature : {_watermark_line(r)}")
    c2pa = _c2pa_lines(r)
    out.append(f"  C2PA manifest   : {c2pa[0]}")
    out += [f"                    {line}" for line in c2pa[1:]]
    if r.binding_matches:
        out.append("  link            : the manifest names the same asset as the pixel signature")
    out += [f"  note            : {n}" for n in r.notes]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    results: list[dict[str, Any]] = []
    any_ai = error = False
    for i, name in enumerate(args.files):
        try:
            r = check(name)
        except OSError as e:
            error = True
            reason = e.strerror or str(e)
            if args.json:
                results.append({"source": name, "error": reason})
            else:
                msg = f"{name}\n  error           : could not read the file ({reason})"
                print(msg, file=sys.stderr)
            continue
        any_ai = any_ai or r.ai_generated
        if args.json:
            results.append(r.as_dict())
        else:
            if i:
                print()
            print(_text(r))
    if args.json:
        print(json.dumps({"daf_check": __version__, "results": results}, indent=2))
    if error:
        return EXIT_ERROR
    return EXIT_AI if any_ai else EXIT_NONE


if __name__ == "__main__":
    raise SystemExit(main())
