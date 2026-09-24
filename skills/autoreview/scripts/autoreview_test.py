#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import io
import json
import os
import re
import runpy
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest import mock

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


SCRIPT_PATH = Path(__file__).with_name("autoreview")
LOADER = SourceFileLoader("autoreview_module", str(SCRIPT_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
AUTOREVIEW = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(AUTOREVIEW)


FINAL_REPORT = {
    "findings": [],
    "overall_correctness": "patch is correct",
    "overall_explanation": "clean",
    "overall_confidence": 0.9,
}

DRAFT_REPORT = {
    "findings": [
        {
            "title": "Draft finding",
            "body": "draft",
            "priority": "P3",
            "confidence": 0.2,
            "category": "maintainability",
            "code_location": {"file_path": "draft.js", "line": 1},
        }
    ],
    "overall_correctness": "patch is incorrect",
    "overall_explanation": "draft",
    "overall_confidence": 0.2,
}


class AutoreviewCursorTests(unittest.TestCase):
    def test_parser_resource_errors_are_invalid_reports(self) -> None:
        args = argparse.Namespace(engine="codex", max_priority="P2")
        for raw in ("[" * 2000 + "]" * 2000, '{"findings":[],"number":' + "9" * 10000 + "}"):
            with self.subTest(length=len(raw)), mock.patch.object(
                AUTOREVIEW, "run_engine", return_value=raw,
            ):
                with self.assertRaises(AUTOREVIEW.ReviewerUnavailable) as caught:
                    AUTOREVIEW.run_reviewer(args, Path.cwd(), "synthetic", set(), [])
                self.assertEqual(caught.exception.reason, "invalid_report")

    def test_container_valued_report_enums_are_invalid_reports(self) -> None:
        args = argparse.Namespace(engine="codex", max_priority="P2")
        finding = copy.deepcopy(DRAFT_REPORT["findings"][0])
        finding["source_attribution"] = {
            "target": "index", "record_id": "record", "source_id": "source",
            "side": "present", "column": 1, "excerpt": "text",
        }
        for field in ("overall_correctness", "priority", "category", "target", "side"):
            for value in ([], {}, None, 42, False):
                report = copy.deepcopy(FINAL_REPORT)
                report["findings"] = [copy.deepcopy(finding)]
                owner = report if field == "overall_correctness" else report["findings"][0]
                if field in {"target", "side"}:
                    owner = owner["source_attribution"]
                owner[field] = value
                with self.subTest(field=field, value=value), mock.patch.object(
                    AUTOREVIEW, "run_engine", return_value=json.dumps({**report, "review_completion": "complete"}),
                ):
                    with self.assertRaises(AUTOREVIEW.ReviewerUnavailable) as caught:
                        AUTOREVIEW.run_reviewer(args, Path.cwd(), "synthetic", {"draft.js"}, [])
                    self.assertEqual(caught.exception.reason, "invalid_report")

    def test_private_completion_is_required_validated_and_stripped(self) -> None:
        args = argparse.Namespace(engine="codex", max_priority="P2")
        for completion in ("complete", "incomplete"):
            provider = {**FINAL_REPORT, "review_completion": completion}
            with self.subTest(completion=completion), mock.patch.object(
                AUTOREVIEW, "run_engine", return_value=json.dumps(provider),
            ):
                result = AUTOREVIEW.run_reviewer(args, Path.cwd(), "synthetic", set(), [])
            self.assertEqual(result.complete, completion == "complete")
            self.assertEqual(result.report["provider_report"], FINAL_REPORT)
            self.assertNotIn("review_completion", result.report)
            self.assertEqual(
                AUTOREVIEW.review_status(result.report, complete=result.complete),
                "scoped-clean" if result.complete else "incomplete",
            )
        for provider in (
            FINAL_REPORT,
            *({**FINAL_REPORT, "review_completion": value}
              for value in ("", "deferred", [], {}, None, 42, False)),
        ):
            with self.subTest(provider=provider), mock.patch.object(
                AUTOREVIEW, "run_engine", return_value=json.dumps(provider),
            ):
                with self.assertRaises(AUTOREVIEW.ReviewerUnavailable) as caught:
                    AUTOREVIEW.run_reviewer(args, Path.cwd(), "synthetic", set(), [])
            self.assertEqual(caught.exception.reason, "invalid_report")
            self.assertIn("missing or invalid review_completion", str(caught.exception))

    def test_provider_schema_keeps_completion_out_of_public_schema(self) -> None:
        self.assertEqual(
            AUTOREVIEW.PROVIDER_SCHEMA["required"],
            [*AUTOREVIEW.SCHEMA["required"], "review_completion"],
        )
        self.assertEqual(
            AUTOREVIEW.PROVIDER_SCHEMA["properties"]["review_completion"],
            {"type": "string", "enum": ["complete", "incomplete"]},
        )
        self.assertFalse(AUTOREVIEW.PROVIDER_SCHEMA["additionalProperties"])
        self.assertNotIn("review_completion", AUTOREVIEW.SCHEMA["properties"])
        prompt = AUTOREVIEW.render_review_prompt(
            "task", "local", None, AUTOREVIEW.ReviewChunk("change"), "", "", (1, 2),
        )
        self.assertIn(json.dumps(AUTOREVIEW.PROVIDER_SCHEMA), prompt)
        self.assertIn("independent, complete assignment", prompt)
        self.assertIn("no shared conversation or future evidence batch", prompt)

    def test_extract_json_prefers_terminal_result_event(self) -> None:
        stream = "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {"role": "assistant", "content": [{"type": "text", "text": json.dumps(DRAFT_REPORT)}]},
                    }
                ),
                json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "result": json.dumps(FINAL_REPORT),
                        "session_id": "session-id",
                        "request_id": "request-id",
                    }
                ),
            ]
        )
        self.assertEqual(AUTOREVIEW.extract_json(stream), FINAL_REPORT)

    def test_extract_json_can_fallback_to_assistant_message(self) -> None:
        stream = json.dumps(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": json.dumps(FINAL_REPORT)}]},
            }
        )
        self.assertEqual(AUTOREVIEW.extract_json(stream), FINAL_REPORT)

    def test_extract_json_does_not_fallback_past_bad_terminal_result(self) -> None:
        stream = "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {"role": "assistant", "content": [{"type": "text", "text": json.dumps(FINAL_REPORT)}]},
                    }
                ),
                json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "result": "not json",
                    }
                ),
            ]
        )
        with self.assertRaises(SystemExit) as exc_info:
            AUTOREVIEW.extract_json(stream)
        self.assertIn("review engine result was not structured JSON", str(exc_info.exception))


class AutoreviewPriorityTests(unittest.TestCase):
    def test_default_priority_is_p0(self) -> None:
        with mock.patch.object(sys, "argv", ["autoreview"]):
            args = AUTOREVIEW.parse_args()
        self.assertEqual(args.max_priority, "P0")

    def test_priority_filter_preserves_lower_findings_and_provider_verdict(self) -> None:
        report = copy.deepcopy(DRAFT_REPORT)
        AUTOREVIEW.filter_findings_by_priority(report, "P0")
        self.assertEqual(report["findings"], [])
        self.assertEqual(report["priority_filtered_findings"], DRAFT_REPORT["findings"])
        for key in ("overall_correctness", "overall_explanation", "overall_confidence"):
            self.assertEqual(report[key], DRAFT_REPORT[key])

    def test_unfinished_assessment_keeps_filtered_observations_incomplete(self) -> None:
        args = argparse.Namespace(engine="codex", max_priority="P0")
        with mock.patch.object(
            AUTOREVIEW, "run_engine",
            return_value=json.dumps({**DRAFT_REPORT, "review_completion": "incomplete"}),
        ):
            result = AUTOREVIEW.run_reviewer(args, Path.cwd(), "synthetic", {"draft.js"}, [])
        self.assertFalse(result.complete)
        self.assertEqual(result.report["provider_report"], DRAFT_REPORT)
        self.assertEqual(result.report["findings"], [])
        self.assertEqual(result.report["priority_filtered_findings"], DRAFT_REPORT["findings"])
        self.assertEqual(AUTOREVIEW.review_status(result.report, complete=result.complete), "incomplete")


class AutoreviewResultScopeTests(unittest.TestCase):
    def test_scope_rejection_preserves_provider_conclusion_and_audit(self) -> None:
        report = copy.deepcopy(DRAFT_REPORT)
        with contextlib.redirect_stderr(io.StringIO()):
            AUTOREVIEW.validate_report(report, Path.cwd(), {"changed.js"}, [])
        self.assertEqual(report["findings"], [])
        for key in ("overall_correctness", "overall_explanation", "overall_confidence"):
            self.assertEqual(report[key], DRAFT_REPORT[key])
        self.assertEqual(report["scope_rejected_findings"], DRAFT_REPORT["findings"])
        report["review_status"] = AUTOREVIEW.review_status(report, complete=True)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            AUTOREVIEW.print_report(report)
        self.assertIn("incomplete", output.getvalue())
        self.assertIn("Draft finding", output.getvalue())
        self.assertIn("draft.js:1", output.getvalue())
        self.assertNotIn("clean:", output.getvalue())

    def test_chunk_merge_keeps_rejections_explanations_and_conservative_confidence(self) -> None:
        rejected = copy.deepcopy(DRAFT_REPORT)
        with contextlib.redirect_stderr(io.StringIO()):
            AUTOREVIEW.validate_report(rejected, Path.cwd(), {"changed.js"}, [])
        reports = [("chunk 1/2", copy.deepcopy(FINAL_REPORT)), ("chunk 2/2", rejected)]
        merged = AUTOREVIEW.merge_chunk_reports(reports)
        self.assertEqual(merged["overall_correctness"], "patch is incorrect")
        self.assertEqual(merged["overall_confidence"], 0.2)
        self.assertEqual(len(merged["scope_rejected_findings"]), 1)
        self.assertEqual(merged["pass_reports"][1]["report"], rejected)
        merged["review_status"] = AUTOREVIEW.review_status(merged, complete=True)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            AUTOREVIEW.print_report(merged)
        self.assertIn("draft", output.getvalue())
        self.assertIn("incomplete", output.getvalue())
        self.assertNotIn("clean:", output.getvalue())

    def test_required_finding_must_survive_priority_filter_for_every_pass_count(self) -> None:
        args = argparse.Namespace(engine="codex", max_priority="P0", require_finding=["Draft finding"])
        for count in (1, 2):
            with self.subTest(count=count), mock.patch.object(
                AUTOREVIEW, "run_engine", return_value=json.dumps({**DRAFT_REPORT, "review_completion": "complete"}),
            ):
                results = AUTOREVIEW.run_review_passes(
                    args, [args], Path.cwd(), ["pack"] * count, {"draft.js"}
                )
            self.assertTrue(all(result.complete for _, result in results))
            reports = [(label, result.report) for label, result in results]
            report = reports[0][1] if count == 1 else AUTOREVIEW.merge_chunk_reports(reports)
            self.assertEqual(
                AUTOREVIEW.missing_required_findings(report, args.require_finding), ["Draft finding"]
            )
            self.assertEqual(report["overall_correctness"], "patch is incorrect")
            self.assertTrue(report["priority_filtered_findings"])

    def test_provider_cannot_supply_local_audit_metadata(self) -> None:
        for key in ("scope_rejected_findings", "priority_filtered_findings", "pass_reports", "review_status", "review_completion"):
            report = copy.deepcopy(FINAL_REPORT)
            report[key] = []
            with self.subTest(key=key), self.assertRaisesRegex(SystemExit, "unexpected top-level"):
                AUTOREVIEW.validate_report(report, Path.cwd(), set(), [])

    def test_required_finding_survives_merge_deduplication_and_body_prefix(self) -> None:
        first = copy.deepcopy(DRAFT_REPORT)
        second = copy.deepcopy(DRAFT_REPORT)
        second["findings"][0]["body"] = "x" * 1980 + " required tail"
        merged = AUTOREVIEW.merge_chunk_reports([("chunk 1/2", first), ("chunk 2/2", second)])
        self.assertEqual(len(merged["findings"]), 1)
        self.assertEqual(AUTOREVIEW.missing_required_findings(merged, ["required tail"]), [])


class AutoreviewTargetResultTests(unittest.TestCase):
    def setUp(self):
        source = AUTOREVIEW.SourceVersion
        self.record = AUTOREVIEW.MixedPath(
            "src/migrate.py", "synthetic-record",
            source("base-source", "100644", "original()\nkeep()\n"),
            source("index-source", "100644", "obsolete()\nkeep()\n"),
            source("working-source", "100644", "corrected()\nkeep()\nbroken()\n"),
            "synthetic staged delta", "synthetic unstaged delta",
            ((1, "original()"),), ((1, "obsolete()"),), (),
        )

    def finding(self, target="index", side="present", line=1, excerpt=None, **changes):
        source = ((self.record.base if target == "index" else self.record.index)
                  if side == "removed" else getattr(self.record, target))
        if excerpt is None:
            excerpt = source.content.splitlines()[line - 1]
        finding = {
            "title": "Synthetic defect", "body": "A concrete synthetic claim.",
            "priority": "P0", "confidence": 0.8, "category": "bug",
            "code_location": {"file_path": self.record.path, "line": line},
            "source_attribution": {"target": target, "record_id": self.record.identity,
                                   "source_id": source.identity, "side": side,
                                   "column": 1, "excerpt": excerpt},
        }
        finding.update(changes)
        return finding

    def validate(self, findings, available=True):
        report = copy.deepcopy(FINAL_REPORT)
        report["findings"] = copy.deepcopy(findings)
        AUTOREVIEW.validate_report(report, Path.cwd(), {self.record.path}, [], (self.record,),
                                   {self.record.identity} if available else set())
        return report

    def test_explicit_targets_anchors_and_pass_availability(self):
        accepted = [self.finding(), self.finding("working_tree", line=3),
                    self.finding("working_tree", line=2), self.finding(side="removed"),
                    self.finding("working_tree", side="removed")]
        self.assertEqual(self.validate(accepted)["findings"], accepted)
        cases = []
        missing = self.finding()
        missing.pop("source_attribution")
        cases.append((missing, "requires explicit"))
        null = self.finding(source_attribution=None)
        cases.append((null, "requires explicit"))
        for field, value, reason in (
            ("record_id", "wrong", "record identity"),
            ("source_id", "wrong", "source identity"),
            ("column", 1000, "excerpt"),
            ("excerpt", "invented()", "excerpt"),
        ):
            finding = self.finding()
            finding["source_attribution"][field] = value
            cases.append((finding, reason))
        cases.extend([
            (self.finding("working_tree", excerpt="obsolete()"), "excerpt"),
            (self.finding("working_tree", line=900, excerpt="broken()"), "out of range"),
            (self.finding("working_tree", side="removed", line=2), "genuinely removed"),
        ])
        for finding, reason in cases:
            with self.subTest(reason=reason, finding=finding):
                report = self.validate([finding])
                self.assertEqual(report["findings"], [])
                self.assertIn(reason, report["attribution_rejected_findings"][0]["attribution_rejection_reason"])
                self.assertEqual(AUTOREVIEW.review_status(report, complete=True), "incomplete")
                self.assertEqual(report["overall_correctness"], "patch is correct")
        report = self.validate([self.finding()], available=False)
        self.assertIn("not available", report["attribution_rejected_findings"][0]["attribution_rejection_reason"])
        original = self.record
        for path in (" src/migrate.py", "src/migrate.py ", " "):
            with self.subTest(path=path):
                self.record = original._replace(path=path)
                finding = self.finding()
                self.assertEqual(self.validate([finding])["findings"], [finding])
            self.record = original

    def test_absence_readd_and_removed_side_are_distinct(self):
        absent = AUTOREVIEW.SourceVersion("absent", None, None)
        for target in ("index", "working_tree"):
            with self.subTest(target=target):
                original = self.record
                if target == "index":
                    self.record = original._replace(index=absent, working_tree_removed=())
                else:
                    self.record = original._replace(working_tree=absent)
                present = self.finding(target, excerpt="obsolete()")
                removed = self.finding(target, side="removed")
                report = self.validate([present, removed])
                self.assertEqual(report["findings"], [removed])
                self.assertIn("absent", report["attribution_rejected_findings"][0]["attribution_rejection_reason"])
                self.record = original

        original = self.record
        for target, side, content, line in (
            ("index", "present", "", 1), ("working_tree", "present", "", 1),
            ("index", "present", "before()\n\n", 2), ("working_tree", "present", "before()\n\n", 2),
            ("index", "removed", "\n", 1), ("working_tree", "removed", "\n", 1),
        ):
            with self.subTest(target=target, side=side, content=content):
                owner = ("base" if target == "index" else "index") if side == "removed" else target
                self.record = original._replace(**{owner: getattr(original, owner)._replace(content=content)})
                if side == "removed":
                    self.record = self.record._replace(**{target + "_removed": ((line, ""),)})
                valid = self.finding(target, side=side, line=line, excerpt="")
                self.assertEqual(self.validate([valid])["findings"], [valid])
                invalid = []
                for key, value in (("record_id", "wrong"), ("source_id", "wrong"),
                                   ("column", 2), ("excerpt", "invented")):
                    bad = copy.deepcopy(valid)
                    bad["source_attribution"][key] = value
                    invalid.append(bad)
                bad = copy.deepcopy(valid)
                bad["code_location"]["line"] = 2 if not content else 900
                invalid.append(bad)
                if side == "present" and content:
                    bad = copy.deepcopy(valid)
                    bad["code_location"]["line"] = 1
                    invalid.append(bad)
                for bad in invalid:
                    report = self.validate([bad])
                    self.assertEqual(report["findings"], [])
                    self.assertEqual(AUTOREVIEW.review_status(report, complete=True), "incomplete")
                self.record = original
        for target in ("index", "working_tree"):
            for side in ("present", "removed"):
                report = self.validate([self.finding(target, side=side, excerpt="")])
                self.assertEqual(report["findings"], [])
            self.record = original._replace(**{target: absent})
            report = self.validate([self.finding(target, excerpt="")])
            self.assertIn("absent", report["attribution_rejected_findings"][0]["attribution_rejection_reason"])
            self.record = original

    def test_title_independent_groups_keep_variants_targets_and_observations(self):
        reports = []
        for index in range(8):
            finding = self.finding(title=f"Index title {index}")
            findings = [finding]
            if index == 7:
                findings += [self.finding(body="Distinct consequence requiring a different fix."),
                             self.finding("working_tree")]
            reports.append((f"pass {index}", self.validate(findings)))
        for selected in (reports, [("single", self.validate([
            finding for _, report in reports for finding in report["findings"]
        ]))]):
            with self.subTest(passes=len(selected)):
                result = AUTOREVIEW.merge_chunk_reports(selected)
                self.assertEqual(len(result["findings"]), 2)
                grouped = result["findings"][0]
                self.assertEqual(len(grouped["claim_variants"]), 2)
                self.assertEqual(len(grouped["claim_variants"][0]["observations"]), 8)
                self.assertEqual(AUTOREVIEW.missing_required_findings(result, ["Index title 7", "different fix"]), [])
                self.assertEqual(len(result["pass_reports"]), len(selected))
                result["review_status"] = AUTOREVIEW.review_status(result, complete=True)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    AUTOREVIEW.print_report(result)
                for text in ("INDEX-only", "WORKING_TREE", "Index title 7", "different fix"):
                    self.assertIn(text, output.getvalue())

    def test_all_engines_keep_raw_reports_before_normalization_and_filters(self):
        captured = AUTOREVIEW.CapturedBundle("delta", {self.record.path}, (self.record,), ())
        prompt = AUTOREVIEW.ReviewPass("synthetic pack", AUTOREVIEW.ReviewChunk("delta", sources=(self.record,)))
        valid = self.finding()
        valid["code_location"]["file_path"] = r".\src\migrate.py"
        stale = self.finding("working_tree", excerpt="obsolete()")
        outside = self.finding(code_location={"file_path": "outside.py", "line": 1})
        provider = {**FINAL_REPORT, "findings": [valid, stale, outside],
                    "overall_correctness": "patch is incorrect", "overall_confidence": 0.43}
        for engine in AUTOREVIEW.ENGINES:
            with self.subTest(engine=engine), mock.patch.object(
                    AUTOREVIEW, "run_engine", return_value=json.dumps({**provider, "review_completion": "complete"})), \
                    mock.patch.object(AUTOREVIEW, "verify_mixed_sources"), contextlib.redirect_stderr(io.StringIO()):
                result = AUTOREVIEW.run_reviewer(argparse.Namespace(engine=engine, max_priority="P0"),
                                                 Path.cwd(), prompt, captured, [])
            self.assertTrue(result.complete)
            report = result.report
            self.assertEqual(report["provider_report"], provider)
            self.assertEqual(report["overall_confidence"], 0.43)
            self.assertEqual(len(report["findings"]), 1)
            self.assertEqual(len(report["scope_rejected_findings"]), 1)
            self.assertEqual(len(report["attribution_rejected_findings"]), 1)
            self.assertEqual(AUTOREVIEW.review_status(report, complete=result.complete), "incomplete")
            self.assertEqual(report["available_source_records"], [self.record.identity])
        low = self.validate([self.finding(priority="P2")])
        AUTOREVIEW.filter_findings_by_priority(low, "P0")
        self.assertEqual(AUTOREVIEW.missing_required_findings(low, ["Synthetic defect"]), ["Synthetic defect"])
        self.assertEqual(AUTOREVIEW.review_status(low, complete=True), "filtered")
        for bad in ({}, {**self.finding()["source_attribution"], "column": True}):
            with self.assertRaisesRegex(SystemExit, "source_attribution"):
                self.validate([self.finding(source_attribution=bad)])

def amp_test_stream(
    cwd: Path,
    *,
    tools: list[object] | None = None,
    mcp_servers: list[object] | None = None,
    trigger: str = "Run the isolated autoreview adapter.",
    tool_name: str = "autoreview_generate",
    tool_input: object = None,
    tool_result_id: str = "amp-tool-use",
    tool_error: bool = False,
    tool_result_content: str | None = None,
) -> str:
    if tool_input is None:
        tool_input = {}
    if tool_result_content is None:
        tool_result_content = (
            "Autoreview generation failed." if tool_error else "Adapter completed."
        )
    return "\n".join(
        [
            json.dumps(
                {
                    "type": "system",
                    "subtype": "init",
                    "cwd": str(cwd),
                    "session_id": "amp-test-session",
                    "tools": ["autoreview_generate"] if tools is None else tools,
                    "mcp_servers": [] if mcp_servers is None else mcp_servers,
                    "agent_mode": "medium",
                }
            ),
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": trigger}],
                    },
                    "parent_tool_use_id": None,
                    "session_id": "amp-test-session",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "amp-tool-use",
                                "name": tool_name,
                                "input": tool_input,
                            }
                        ],
                    },
                    "parent_tool_use_id": None,
                    "session_id": "amp-test-session",
                }
            ),
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_result_id,
                                "content": tool_result_content,
                                "is_error": tool_error,
                            }
                        ],
                    },
                    "parent_tool_use_id": None,
                    "session_id": "amp-test-session",
                }
            ),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Completed."}],
                    },
                    "parent_tool_use_id": None,
                    "session_id": "amp-test-session",
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": "Completed.",
                    "session_id": "amp-test-session",
                }
            ),
        ]
    ) + "\n"


def amp_test_plugin_list(plugin_path: Path) -> str:
    return "\n".join(
        [
            f"✓ {plugin_path} active",
            "  tool: autoreview_generate",
            "  agent: autoreview-adapter",
            "  agent mode: autoreview",
        ]
    ) + "\n"


def amp_test_mcp_denial_result(
    command: list[str],
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    skills_root = Path(env["HOME"]) / ".config" / "agents" / "skills"
    probe_roots = list(skills_root.glob("autoreview-mcp-deny-*"))
    if len(probe_roots) != 1:
        raise AssertionError(f"expected one MCP denial probe, found {probe_roots}")
    mcp_config = json.loads((probe_roots[0] / "mcp.json").read_text(encoding="utf-8"))
    probe_name = next(iter(mcp_config))
    return subprocess.CompletedProcess(
        command,
        0,
        "12 tools available\n",
        f"error connecting to {probe_name}: MCP server is not allowed by MCP permissions\n",
    )


class AutoreviewAmpTests(unittest.TestCase):
    def test_amp_bin_cli_option_and_defaults(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["autoreview", "--engine", "amp", "--amp-bin", "/tmp/trusted-amp"],
        ):
            args = AUTOREVIEW.parse_args()
        reviewer = AUTOREVIEW.reviewer_args(args)[0]
        self.assertEqual(reviewer.amp_bin, "/tmp/trusted-amp")
        self.assertEqual(reviewer.model, "openai/gpt-5.6-sol")
        self.assertEqual(reviewer.thinking, "high")
        self.assertFalse(reviewer.tools)

    @unittest.skipIf(os.name == "nt", "Amp runtime is unsupported on native Windows")
    def test_amp_isolation_probe_requires_api_key_and_flags(self) -> None:
        args = argparse.Namespace(amp_bin="amp")
        required_flags = " ".join(
            [
                "--execute",
                "--stream-json",
                "--stream-json-input",
                "--plugin-ready-timeout",
                "--settings-file",
                "--no-ide",
            ]
        )
        with tempfile.TemporaryDirectory(prefix="autoreview-amp-probe-test.") as tmpdir, mock.patch.dict(
            os.environ,
            {"AMP_API_KEY": "test-key"},
            clear=False,
        ), mock.patch.object(
            AUTOREVIEW,
            "resolve_command",
            return_value="/usr/bin/amp",
        ), mock.patch.object(
            AUTOREVIEW,
            "safe_engine_env",
            return_value={},
        ), mock.patch.object(
            AUTOREVIEW,
            "safe_temp_root",
            return_value=Path(tmpdir),
        ), mock.patch.object(
            AUTOREVIEW,
            "run",
            return_value=subprocess.CompletedProcess(["amp", "--help"], 0, required_flags, ""),
        ):
            self.assertEqual(
                AUTOREVIEW.ensure_amp_isolation_supported(args, Path(tmpdir)),
                "/usr/bin/amp",
            )

        with mock.patch.dict(os.environ, {"AMP_API_KEY": ""}, clear=False), mock.patch.object(
            AUTOREVIEW,
            "resolve_command",
            return_value="/usr/bin/amp",
        ):
            with self.assertRaisesRegex(SystemExit, "requires AMP_API_KEY"):
                AUTOREVIEW.ensure_amp_isolation_supported(args, Path("/tmp/repo"))

    def test_amp_isolation_probe_rejects_native_windows(self) -> None:
        args = argparse.Namespace(amp_bin="amp")
        repo = Path("/tmp/repo")
        context = (
            contextlib.nullcontext()
            if os.name == "nt"
            else mock.patch.object(AUTOREVIEW.os, "name", "nt")
        )
        with context:
            with self.assertRaisesRegex(SystemExit, "native Windows"):
                AUTOREVIEW.ensure_amp_isolation_supported(args, repo)

    @unittest.skipIf(os.name == "nt", "Amp runtime is unsupported on native Windows")
    def test_amp_run_keeps_review_prompt_out_of_outer_agent(self) -> None:
        args = argparse.Namespace(
            amp_bin="amp",
            max_output_chars=2_000_000,
            model="openai/gpt-5.6-sol",
            stream_engine_output=False,
            thinking="high",
        )
        secret_prompt = "review diff PRIVATE_REVIEW_MARKER_8f3c"
        observed: dict[str, object] = {}

        def fake_preflight(
            command: list[str],
            cwd: Path,
            **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            env = kwargs["env"]
            assert isinstance(env, dict)
            runtime_root = Path(str(env["XDG_CONFIG_HOME"])).parent
            plugin_root = Path(str(env["XDG_CONFIG_HOME"])) / "amp" / "plugins"
            plugin_path = next(plugin_root.glob("autoreview-*.ts"))
            if command[-2:] == ["tools", "list"]:
                observed["mcp_preflight_prompt_exists"] = (
                    runtime_root / "review-prompt.txt"
                ).exists()
                return amp_test_mcp_denial_result(command, env)
            observed["preflight_command"] = command
            observed["preflight_prompt_exists"] = (
                runtime_root / "review-prompt.txt"
            ).exists()
            return subprocess.CompletedProcess(
                command,
                0,
                amp_test_plugin_list(plugin_path),
                "",
            )

        def fake_execute(
            command: list[str],
            cwd: Path,
            **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            observed["command"] = command
            observed["cwd"] = cwd
            observed["input"] = kwargs["input_text"]
            observed["env"] = kwargs["env"]
            env = kwargs["env"]
            assert isinstance(env, dict)
            runtime_root = Path(str(env["XDG_CONFIG_HOME"])).parent
            prompt_path = runtime_root / "review-prompt.txt"
            result_path = runtime_root / "review-result.json"
            settings_path = runtime_root / "settings.json"
            plugin_root = Path(str(env["XDG_CONFIG_HOME"])) / "amp" / "plugins"
            plugin_path = next(plugin_root.glob("autoreview-*.ts"))
            observed["prompt"] = prompt_path.read_text(encoding="utf-8")
            observed["settings"] = json.loads(settings_path.read_text(encoding="utf-8"))
            observed["plugin"] = plugin_path.read_text(encoding="utf-8")
            observed["plugin_path"] = plugin_path
            observed["workspace"] = list(cwd.iterdir())
            result_path.write_text(json.dumps(FINAL_REPORT), encoding="utf-8")
            result_path.chmod(0o600)
            return subprocess.CompletedProcess(command, 0, amp_test_stream(cwd), "")

        inherited = {
            "AMP_API_KEY": "test-key",
            "AMP_URL": "https://attacker.invalid",
            "NODE_OPTIONS": "--require=/tmp/attack.js",
            "PYTHONPATH": "/tmp/attack",
            "PLUGINS": "inherited-plugins",
        }
        with tempfile.TemporaryDirectory(prefix="autoreview-amp-run-test.") as tmpdir:
            repo = Path(tmpdir) / "repo"
            repo.mkdir()
            with mock.patch.dict(os.environ, inherited, clear=False), mock.patch.object(
                AUTOREVIEW,
                "ensure_amp_isolation_supported",
                return_value="/usr/bin/amp",
            ), mock.patch.object(
                AUTOREVIEW,
                "run",
                side_effect=fake_preflight,
            ), mock.patch.object(
                AUTOREVIEW,
                "run_with_heartbeat",
                side_effect=fake_execute,
            ):
                output = AUTOREVIEW.run_amp(args, repo, secret_prompt)

        self.assertEqual(json.loads(output), FINAL_REPORT)
        self.assertFalse(observed["mcp_preflight_prompt_exists"])
        self.assertFalse(observed["preflight_prompt_exists"])
        preflight_command = observed["preflight_command"]
        self.assertIsInstance(preflight_command, list)
        assert isinstance(preflight_command, list)
        self.assertEqual(preflight_command[-2:], ["plugins", "list"])
        command = observed["command"]
        self.assertIsInstance(command, list)
        assert isinstance(command, list)
        self.assertIn("--execute", command)
        self.assertIn("--stream-json-input", command)
        self.assertIn("--settings-file", command)
        self.assertNotIn("--orb-execute", command)
        self.assertEqual(command[command.index("--mode") + 1], "autoreview")
        self.assertNotIn(secret_prompt, " ".join(command))
        self.assertNotIn(secret_prompt, str(observed["input"]))
        self.assertEqual(observed["prompt"], secret_prompt)
        self.assertEqual(observed["workspace"], [])
        settings = observed["settings"]
        self.assertIsInstance(settings, dict)
        assert isinstance(settings, dict)
        self.assertNotIn("amp.tools.disable", settings)
        self.assertNotIn("amp.tools.enable", settings)
        self.assertEqual(settings["amp.updates.mode"], "disabled")
        self.assertEqual(
            settings["amp.mcpPermissions"],
            [
                {"matches": {"command": "*"}, "action": "reject"},
                {"matches": {"url": "*"}, "action": "reject"},
            ],
        )
        plugin = observed["plugin"]
        self.assertIsInstance(plugin, str)
        assert isinstance(plugin, str)
        self.assertIn("amp.ai.generate", plugin)
        self.assertIn("amp.registerTool", plugin)
        self.assertIn("amp.createAgent", plugin)
        schema, _ = json.JSONDecoder().raw_decode(plugin.split("schema: ", 1)[1])
        self.assertEqual(schema, {
            "name": "autoreview_report",
            "description": "A security-focused code-review report for the supplied patch.",
            "fields": AUTOREVIEW.PROVIDER_SCHEMA["properties"],
        })
        self.assertIn('tools: ["autoreview_generate"]', plugin)
        self.assertIn("readFileSync", plugin)
        self.assertNotIn(secret_prompt, plugin)
        env = observed["env"]
        self.assertIsInstance(env, dict)
        assert isinstance(env, dict)
        self.assertEqual(env["AMP_API_KEY"], "test-key")
        self.assertNotIn("AMP_URL", env)
        self.assertNotIn("NODE_OPTIONS", env)
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["PLUGINS"], "all")
        plugin_path = observed["plugin_path"]
        self.assertIsInstance(plugin_path, Path)
        assert isinstance(plugin_path, Path)
        self.assertRegex(plugin_path.stem, r"^autoreview-[0-9a-f]{32}$")
        cwd = observed["cwd"]
        self.assertIsInstance(cwd, Path)
        assert isinstance(cwd, Path)
        self.assertNotEqual(cwd.resolve(), repo.resolve())
        self.assertEqual(Path(env["HOME"]).parent, cwd.parent)

    def test_amp_stream_attestation_rejects_bad_events(self) -> None:
        cwd = Path("/tmp/amp-review-empty")
        misplaced_events = [
            json.loads(line) for line in amp_test_stream(cwd).splitlines()
        ]
        tool_use = misplaced_events[2]["message"]["content"].pop()
        misplaced_events[4]["message"]["content"].append(tool_use)
        misplaced = "\n".join(json.dumps(event) for event in misplaced_events) + "\n"
        extra_result_events = [
            json.loads(line) for line in amp_test_stream(cwd).splitlines()
        ]
        extra_result_events[3]["message"]["content"].append(
            {"type": "text", "text": "unexpected"}
        )
        extra_result = (
            "\n".join(json.dumps(event) for event in extra_result_events) + "\n"
        )
        cases = {
            "malformed": "not-json\n",
            "tools": amp_test_stream(cwd, tools=["autoreview_generate", "shell_command"]),
            "mcp": amp_test_stream(cwd, mcp_servers=[{"name": "server"}]),
            "trigger": amp_test_stream(cwd, trigger="untrusted diff"),
            "wrong tool": amp_test_stream(cwd, tool_name="shell_command"),
            "tool input": amp_test_stream(cwd, tool_input={"command": "id"}),
            "tool result": amp_test_stream(cwd, tool_result_id="wrong-id"),
            "unsanitized error": amp_test_stream(
                cwd,
                tool_error=True,
                tool_result_content="provider echoed PRIVATE_REVIEW_MARKER_8f3c",
            ),
            "multiple init": amp_test_stream(cwd).splitlines()[0] + "\n" + amp_test_stream(cwd),
            "multiple result": amp_test_stream(cwd) + amp_test_stream(cwd).splitlines()[-1] + "\n",
            "misplaced tool use": misplaced,
            "extra tool result content": extra_result,
        }
        for label, stream in cases.items():
            with self.subTest(label=label), self.assertRaisesRegex(
                SystemExit,
                "amp isolation attestation failed",
            ):
                AUTOREVIEW.attest_amp_stream(stream, cwd)

        self.assertTrue(AUTOREVIEW.attest_amp_stream(amp_test_stream(cwd), cwd))
        self.assertFalse(
            AUTOREVIEW.attest_amp_stream(amp_test_stream(cwd, tool_error=True), cwd)
        )

    @unittest.skipIf(os.name == "nt", "Amp runtime is unsupported on native Windows")
    def test_amp_run_reports_timeout_before_stream_attestation(self) -> None:
        args = argparse.Namespace(
            amp_bin="amp",
            engine_timeout_seconds=0.01,
            max_output_chars=2_000_000,
            model="openai/gpt-5.6-sol",
            stream_engine_output=False,
            thinking="high",
        )

        def fake_preflight(
            command: list[str],
            cwd: Path,
            **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            env = kwargs["env"]
            assert isinstance(env, dict)
            if command[-2:] == ["tools", "list"]:
                return amp_test_mcp_denial_result(command, env)
            plugin_root = Path(str(env["XDG_CONFIG_HOME"])) / "amp" / "plugins"
            plugin_path = next(plugin_root.glob("autoreview-*.ts"))
            return subprocess.CompletedProcess(
                command,
                0,
                amp_test_plugin_list(plugin_path),
                "",
            )

        def fake_execute(
            command: list[str],
            cwd: Path,
            **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            self.assertEqual(kwargs["max_runtime_seconds"], 0.01)
            return subprocess.CompletedProcess(
                command,
                124,
                '{"type":"system","subtype":"init"}\n',
                "amp engine timed out after 0.01s",
            )

        with tempfile.TemporaryDirectory(prefix="autoreview-amp-timeout-test.") as tmpdir:
            repo = Path(tmpdir) / "repo"
            repo.mkdir()
            with mock.patch.object(
                AUTOREVIEW,
                "ensure_amp_isolation_supported",
                return_value="/usr/bin/amp",
            ), mock.patch.object(
                AUTOREVIEW,
                "run",
                side_effect=fake_preflight,
            ), mock.patch.object(
                AUTOREVIEW,
                "run_with_heartbeat",
                side_effect=fake_execute,
            ), mock.patch.object(
                AUTOREVIEW,
                "attest_amp_stream",
                side_effect=AssertionError("timeout stream must not be attested"),
            ) as attest:
                with self.assertRaises(SystemExit) as exc_info:
                    AUTOREVIEW.run_amp(args, repo, "review")

        message = str(exc_info.exception)
        self.assertIn("amp engine failed (124)", message)
        self.assertIn("amp engine timed out after 0.01s", message)
        attest.assert_not_called()

    def test_amp_failed_process_and_invalid_artifact_keep_runtime_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cases = (
                (subprocess.CompletedProcess([], 7, "", "provider failed"), "expected exactly one leading"),
                (subprocess.CompletedProcess([], 7, '{"type":"system","subtype":"init"}\n', ""), "unexpected adapter event sequence"),
                (subprocess.CompletedProcess([], 0, amp_test_stream(root, tools=["shell_command"]), ""), "exposed tools"),
                (subprocess.CompletedProcess([], 0, amp_test_stream(root), ""), "produced no result file"),
                (subprocess.CompletedProcess([], 0, '{"type":[]}\n', ""), "unexpected stream event type"),
            )
            for result, diagnostic in cases:
                with self.subTest(diagnostic=diagnostic), self.assertRaises(AUTOREVIEW.ReviewerUnavailable) as caught:
                    AUTOREVIEW.amp_review_result(result, root, root / "error", root / "result")
                self.assertEqual(caught.exception.reason, "runtime_validation_failed")
                self.assertIn(diagnostic, str(caught.exception))
                self.assertEqual(caught.exception.returncode, result.returncode)
            for raw in ("[" * 2000 + "]" * 2000, '{"number":' + "9" * 10000 + "}"):
                with self.subTest(length=len(raw)), self.assertRaises(AUTOREVIEW.ReviewerUnavailable) as caught:
                    AUTOREVIEW.amp_review_result(subprocess.CompletedProcess([], 0, raw, ""), root, root / "error", root / "result")
                self.assertEqual(caught.exception.reason, "runtime_validation_failed")

    def test_amp_plugin_inventory_attestation_fails_closed(self) -> None:
        cwd = Path("/tmp/amp-review-empty")
        plugin_path = cwd.parent / "config" / "amp" / "plugins" / "autoreview-token.ts"
        valid = amp_test_plugin_list(plugin_path)
        AUTOREVIEW.attest_amp_plugin_inventory(valid, plugin_path, cwd)

        cases = {
            "missing": "",
            "inactive": valid.replace("✓", "✗", 1).replace(" active", " error", 1),
            "other plugin": valid
            + amp_test_plugin_list(plugin_path.with_name("unexpected.ts")),
            "event handler": valid + "  events: agent.start\n",
            "other tool": valid.replace(
                "  agent: autoreview-adapter",
                "  tool: shell_command\n  agent: autoreview-adapter",
            ),
        }
        for label, output in cases.items():
            with self.subTest(label=label), self.assertRaisesRegex(
                SystemExit,
                "amp plugin isolation preflight failed",
            ):
                AUTOREVIEW.attest_amp_plugin_inventory(output, plugin_path, cwd)

    @unittest.skipIf(os.name == "nt", "Amp runtime is unsupported on native Windows")
    def test_amp_run_surfaces_direct_generation_failure(self) -> None:
        args = argparse.Namespace(
            amp_bin="amp",
            max_output_chars=2_000_000,
            model="openai/gpt-5.6-sol",
            stream_engine_output=False,
            thinking="high",
        )

        def fake_preflight(
            command: list[str],
            cwd: Path,
            **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            env = kwargs["env"]
            assert isinstance(env, dict)
            if command[-2:] == ["tools", "list"]:
                return amp_test_mcp_denial_result(command, env)
            plugin_root = Path(str(env["XDG_CONFIG_HOME"])) / "amp" / "plugins"
            plugin_path = next(plugin_root.glob("autoreview-*.ts"))
            return subprocess.CompletedProcess(
                command,
                0,
                amp_test_plugin_list(plugin_path),
                "",
            )

        def fake_execute(
            command: list[str],
            cwd: Path,
            **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            env = kwargs["env"]
            assert isinstance(env, dict)
            error_path = Path(str(env["XDG_CONFIG_HOME"])).parent / "review-error.txt"
            error_path.write_text("provider rejected model", encoding="utf-8")
            error_path.chmod(0o600)
            return subprocess.CompletedProcess(
                command,
                0,
                amp_test_stream(cwd, tool_error=True),
                "",
            )

        with tempfile.TemporaryDirectory(prefix="autoreview-amp-error-test.") as tmpdir:
            repo = Path(tmpdir) / "repo"
            repo.mkdir()
            with mock.patch.object(
                AUTOREVIEW,
                "ensure_amp_isolation_supported",
                return_value="/usr/bin/amp",
            ), mock.patch.object(
                AUTOREVIEW,
                "run",
                side_effect=fake_preflight,
            ), mock.patch.object(
                AUTOREVIEW,
                "run_with_heartbeat",
                side_effect=fake_execute,
            ):
                with self.assertRaisesRegex(SystemExit, "provider rejected model"):
                    AUTOREVIEW.run_amp(args, repo, "review")


class AutoreviewInputTests(unittest.TestCase):


    def test_every_provider_reviews_each_pack_without_a_scanner(self) -> None:
        for engine in ("codex", "claude", "amp", "pi", "kimi", "grok"):
            with self.subTest(engine=engine), tempfile.TemporaryDirectory() as tempdir:
                args = argparse.Namespace(engine=engine, max_priority="P0")
                prompts = [f"complete pack {index}: unicode π\r\n-context\n+change\n" for index in range(2)]
                with mock.patch.object(AUTOREVIEW, "find_command", side_effect=AssertionError("unexpected scanner lookup")), \
                        mock.patch.object(AUTOREVIEW, "run", side_effect=AssertionError("unexpected scanner process")), \
                        mock.patch.object(
                            AUTOREVIEW, f"run_{engine}",
                            return_value=json.dumps({**FINAL_REPORT, "review_completion": "complete"}),
                        ) as provider:
                    for prompt in prompts:
                        result = AUTOREVIEW.run_reviewer(args, Path(tempdir), prompt, set(), [])
                        self.assertTrue(result.complete)
                        self.assertEqual(result.report["findings"], [])
                self.assertEqual([call.args[2] for call in provider.call_args_list], prompts)

    def test_binary_stdin_preserves_utf8_and_crlf_bytes(self) -> None:
        payload = "unicode \u03c0\r\nnext\n".encode("utf-8")
        with tempfile.TemporaryDirectory() as tempdir, tempfile.TemporaryFile() as source:
            source.write(payload)
            source.seek(0)
            result = AUTOREVIEW.run(
                [sys.executable, "-c", "import sys; print(sys.stdin.buffer.read().hex())"],
                Path(tempdir), stdin=source,
            )
        self.assertEqual(result.stdout.strip(), payload.hex())


class AutoreviewCompatibilityTests(unittest.TestCase):
    def test_default_reviewer_uses_sol_high_with_luna_access_retry(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(sys, "argv", ["autoreview"]):
            reviewer = AUTOREVIEW.reviewer_args(AUTOREVIEW.parse_args())[0]
        self.assertEqual(reviewer.engine, "codex")
        self.assertEqual(reviewer.model, "gpt-6-sol")
        self.assertEqual(reviewer.thinking, "high")
        self.assertEqual(reviewer.fallback_model, "gpt-6-luna")

    def test_astra_rejects_unsupported_effort_from_cli_and_environment(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autoreview-invalid-effort.") as tempdir:
            for effort in ("none", "minimal", "ultra"):
                sources = ("cli", "keyed-cli", "environment", "global-environment")
                for source in sources:
                    with self.subTest(effort=effort, source=source):
                        argv = [sys.executable, str(SCRIPT_PATH), "--engine", "codex",
                                "--codex-bin", str(Path(tempdir) / "missing-codex")]
                        env = {key: value for key, value in os.environ.items()
                               if not key.startswith("AUTOREVIEW_")}
                        if source in {"cli", "keyed-cli"}:
                            prefix = "codex=" if source == "keyed-cli" else ""
                            argv += ["--model", prefix + "gpt-6-astra", "--thinking", prefix + effort]
                        else:
                            prefix = "AUTOREVIEW_CODEX_" if source == "environment" else "AUTOREVIEW_"
                            env.update({prefix + "MODEL": "gpt-6-astra", prefix + "THINKING": effort})
                        # No Git repository or engine exists: rejection must precede preparation.
                        result = subprocess.run(argv, cwd=tempdir, env=env, text=True,
                                                capture_output=True, timeout=30)
                        self.assertEqual(result.returncode, 1, result.stderr)
                        self.assertEqual(result.stdout, "")
                        self.assertEqual(result.stderr.strip(),
                                         f"invalid thinking level for codex model gpt-6-astra: {effort} "
                                         "(valid: high, low, max, medium, xhigh)")

    def test_model_validation_uses_effective_cli_overrides(self) -> None:
        cases = (
            ({}, ["--thinking", "minimal", "--thinking", "codex=high"], "gpt-6-sol", "high"),
            ({"AUTOREVIEW_THINKING": "none", "AUTOREVIEW_CODEX_THINKING": "high"},
             [], "gpt-6-sol", "high"),
            ({"AUTOREVIEW_CODEX_THINKING": "minimal"},
             ["--thinking", "high"], "gpt-6-sol", "high"),
            ({"AUTOREVIEW_CODEX_MODEL": "gpt-6-astra", "AUTOREVIEW_CODEX_THINKING": "none"},
             ["--thinking", "high"], "gpt-6-astra", "high"),
            ({"AUTOREVIEW_CODEX_MODEL": "gpt-6-astra", "AUTOREVIEW_CODEX_THINKING": "minimal"},
             ["--model", "gpt-5.6-sol"], "gpt-5.6-sol", "minimal"),
        )
        for env, overrides, model, effort in cases:
            with self.subTest(overrides=overrides), mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(sys, "argv", ["autoreview", "--engine", "codex", *overrides]):
                reviewer = AUTOREVIEW.reviewer_args(AUTOREVIEW.parse_args())[0]
                self.assertEqual(reviewer.model, model)
                self.assertEqual(reviewer.thinking, effort)

    def test_effort_only_cli_and_environment_preserve_supported_models(self) -> None:
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            selections = (
                (["--thinking", effort], {}),
                (["--thinking", "codex=" + effort], {}),
                ([], {"AUTOREVIEW_THINKING": effort}),
                ([], {"AUTOREVIEW_CODEX_THINKING": effort}),
            )
            for thinking_args, env in selections:
                with self.subTest(effort=effort, thinking_args=thinking_args, env=env):
                    with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
                        sys, "argv", ["autoreview", *thinking_args],
                    ):
                        if effort == "minimal":
                            with self.assertRaisesRegex(SystemExit, "invalid thinking level for codex model gpt-6-sol"):
                                AUTOREVIEW.reviewer_args(AUTOREVIEW.parse_args())
                            continue
                        reviewer = AUTOREVIEW.reviewer_args(AUTOREVIEW.parse_args())[0]
                    self.assertEqual(reviewer.model, "gpt-6-sol")
                    self.assertEqual(reviewer.thinking, effort)
                    self.assertEqual(reviewer.fallback_model, "gpt-6-luna")

    def test_sol_and_luna_validate_effort_and_explicit_model_selections(self) -> None:
        for model, fallback in (("gpt-6-sol", "gpt-6-luna"), ("gpt-6-luna", None)):
            selections = (
                (["--model", model], {}),
                (["--model", "codex=" + model], {}),
                ([], {"AUTOREVIEW_MODEL": model}),
                ([], {"AUTOREVIEW_CODEX_MODEL": model}),
            )
            for model_args, env in selections:
                for effort in (None, "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"):
                    with self.subTest(model=model, model_args=model_args, env=env, effort=effort):
                        argv = ["autoreview", *model_args]
                        if effort:
                            argv += ["--thinking", effort]
                        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(sys, "argv", argv):
                            if effort in {"minimal", "ultra"}:
                                with self.assertRaisesRegex(SystemExit, f"invalid thinking level for codex model {model}"):
                                    AUTOREVIEW.reviewer_args(AUTOREVIEW.parse_args())
                                continue
                            reviewer = AUTOREVIEW.reviewer_args(AUTOREVIEW.parse_args())[0]
                        self.assertEqual(reviewer.model, model)
                        self.assertEqual(reviewer.thinking, effort or "high")
                        self.assertEqual(reviewer.fallback_model, fallback)

    def test_astra_preserves_supported_effort_and_explicit_model(self) -> None:
        selections = (
            (["--model", "gpt-6-astra"], {}),
            (["--model", "codex=gpt-6-astra"], {}),
            ([], {"AUTOREVIEW_MODEL": "gpt-6-astra"}),
            ([], {"AUTOREVIEW_CODEX_MODEL": "gpt-6-astra"}),
        )
        for model_args, env in selections:
            for effort in (None, "low", "medium", "high", "xhigh", "max"):
                with self.subTest(model_args=model_args, env=env, effort=effort):
                    argv = ["autoreview", "--engine", "codex", *model_args]
                    if effort:
                        argv += ["--thinking", effort]
                    with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(sys, "argv", argv):
                        reviewer = AUTOREVIEW.reviewer_args(AUTOREVIEW.parse_args())[0]
                    self.assertEqual(reviewer.model, "gpt-6-astra")
                    self.assertEqual(reviewer.thinking, effort or "high")
                    self.assertIsNone(reviewer.fallback_model)

    def test_astra_effort_restrictions_do_not_change_other_codex_models(self) -> None:
        for effort in ("none", "minimal"):
            with self.subTest(effort=effort), mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                sys, "argv", ["autoreview", "--engine", "codex", "--model", "gpt-5.6-sol", "--thinking", effort],
            ):
                reviewer = AUTOREVIEW.reviewer_args(AUTOREVIEW.parse_args())[0]
                self.assertEqual(reviewer.model, "gpt-5.6-sol")
                self.assertEqual(reviewer.thinking, effort)
                self.assertEqual(reviewer.fallback_model, "gpt-5.6-terra")

    @classmethod
    def setUpClass(cls) -> None:
        cls.home_dir = tempfile.TemporaryDirectory(prefix="autoreview-test-home.")
        cls.home_patch = mock.patch.object(Path, "home", return_value=Path(cls.home_dir.name))
        cls.home_patch.start()
        cls.home_keys = ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH")
        cls.old_home_env = {key: os.environ.get(key) for key in cls.home_keys}
        os.environ["HOME"] = cls.home_dir.name
        os.environ["USERPROFILE"] = cls.home_dir.name
        os.environ.pop("HOMEDRIVE", None)
        os.environ.pop("HOMEPATH", None)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.home_patch.stop()
        for key, value in cls.old_home_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        cls.home_dir.cleanup()

    def test_kimi_bin_cli_option(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["autoreview", "--kimi-bin", "/tmp/trusted-kimi"],
        ):
            args = AUTOREVIEW.parse_args()
        self.assertEqual(args.kimi_bin, "/tmp/trusted-kimi")

    def test_kimi_reviewer_disables_tools(self) -> None:
        args = argparse.Namespace(
            engine="kimi",
            model=None,
            thinking=["on"],
            fallback_model=None,
            codex_config=None,
            codex_speed=None,
            tools=True,
        )

        reviewer = AUTOREVIEW.reviewer_args(args)[0]

        self.assertEqual(reviewer.engine, "kimi")
        self.assertEqual(reviewer.thinking, "on")
        self.assertFalse(reviewer.tools)

    def test_kimi_isolation_requires_current_cli_contract(self) -> None:
        args = argparse.Namespace(kimi_bin="kimi")
        required_flags = " ".join(
            [
                "--agent-file",
                "--skills-dir",
                "--prompt",
                "--output-format",
                "--model",
            ]
        )

        def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            if "--version" in command:
                return subprocess.CompletedProcess(command, 0, "0.31.1", "")
            return subprocess.CompletedProcess(command, 0, required_flags, "")

        with tempfile.TemporaryDirectory(prefix="autoreview-kimi-probe-test.") as tmpdir, mock.patch.object(
            AUTOREVIEW,
            "resolve_command",
            return_value="/usr/bin/kimi",
        ), mock.patch.object(
            AUTOREVIEW,
            "safe_engine_env",
            return_value={},
        ), mock.patch.object(
            AUTOREVIEW,
            "safe_temp_root",
            return_value=Path(tmpdir),
        ), mock.patch.object(
            AUTOREVIEW,
            "run",
            side_effect=fake_run,
        ):
            self.assertEqual(
                AUTOREVIEW.ensure_kimi_isolation_supported(args, Path(tmpdir)),
                "/usr/bin/kimi",
            )

    def test_kimi_invalid_streams_are_unavailable_after_launch(self) -> None:
        args = argparse.Namespace(engine="kimi", kimi_bin="kimi", model="kimi-model",
                                  stream_engine_output=False, thinking="on", max_priority="P2")
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "repo"
            repo.mkdir()
            for stream in ("malformed JSON", '{"role":"meta"}\n', '{"role":"assistant","content":"{}"}'):
                with self.subTest(stream=stream), mock.patch.object(
                    AUTOREVIEW, "ensure_kimi_isolation_supported", return_value="/usr/bin/kimi",
                ), mock.patch.object(
                    AUTOREVIEW, "load_kimi_review_config", return_value=({"telemetry": False}, None),
                ), mock.patch.object(
                    AUTOREVIEW, "run_with_heartbeat", return_value=subprocess.CompletedProcess([], 0, stream, ""),
                ):
                    with self.assertRaises(AUTOREVIEW.ReviewerUnavailable) as caught:
                        AUTOREVIEW.run_reviewer(args, repo, "synthetic pack", set(), [])
                    self.assertEqual(caught.exception.reason, "invalid_report")

    def test_kimi_runs_with_empty_tools_skills_and_mcp(self) -> None:
        args = argparse.Namespace(
            kimi_bin="kimi",
            model="kimi-model",
            stream_engine_output=False,
            thinking="on",
        )
        observed: dict[str, object] = {}

        def fake_run(
            command: list[str],
            cwd: Path,
            **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            observed["command"] = command
            observed["cwd"] = cwd
            observed["env"] = kwargs["env"]
            env = kwargs["env"]
            assert isinstance(env, dict)
            home = Path(str(env["KIMI_CODE_HOME"]))
            observed["agent"] = (home / "reviewer.md").read_text(encoding="utf-8")
            observed["config"] = (home / "config.toml").read_text(encoding="utf-8")
            observed["skills"] = list((home / "skills").iterdir())
            observed["workspace"] = list(cwd.iterdir())
            stream = (
                json.dumps({"role": "meta", "type": "system.version", "version": "0.31.1"})
                + "\n"
                + json.dumps({"role": "assistant", "content": json.dumps(FINAL_REPORT)})
                + "\n"
            )
            return subprocess.CompletedProcess(command, 0, stream, "")

        with tempfile.TemporaryDirectory(prefix="autoreview-kimi-run-test.") as tmpdir:
            repo = Path(tmpdir) / "repo"
            repo.mkdir()
            with mock.patch.object(
                AUTOREVIEW,
                "ensure_kimi_isolation_supported",
                return_value="/usr/bin/kimi",
            ), mock.patch.object(
                AUTOREVIEW,
                "load_kimi_review_config",
                return_value=({"telemetry": False}, None),
            ), mock.patch.object(
                AUTOREVIEW,
                "run_with_heartbeat",
                side_effect=fake_run,
            ):
                output = AUTOREVIEW.run_kimi(args, repo, "review prompt")

        self.assertEqual(json.loads(output), FINAL_REPORT)
        command = observed["command"]
        self.assertIsInstance(command, list)
        assert isinstance(command, list)
        self.assertEqual(command[command.index("--prompt") + 1], "review prompt")
        self.assertEqual(command[command.index("--output-format") + 1], "stream-json")
        self.assertEqual(command[command.index("--model") + 1], "kimi-model")
        self.assertNotIn("--thinking", command)
        agent = observed["agent"]
        self.assertIsInstance(agent, str)
        assert isinstance(agent, str)
        self.assertIn("tools: []", agent)
        self.assertIn("subagents: []", agent)
        config = observed["config"]
        self.assertIsInstance(config, str)
        assert isinstance(config, str)
        self.assertIn("[thinking]", config)
        self.assertIn("enabled = true", config)
        self.assertEqual(observed["skills"], [])
        self.assertEqual(observed["workspace"], [])
        env = observed["env"]
        self.assertIsInstance(env, dict)
        assert isinstance(env, dict)
        self.assertEqual(env["KIMI_DISABLE_TELEMETRY"], "1")
        self.assertEqual(env["KIMI_CODE_NO_AUTO_UPDATE"], "1")
        self.assertNotEqual(Path(str(env["KIMI_CODE_HOME"])), repo)

    def test_grok_bin_cli_option(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["autoreview", "--engine", "grok", "--grok-bin", "/tmp/trusted-grok"],
        ):
            args = AUTOREVIEW.parse_args()
        self.assertEqual(args.grok_bin, "/tmp/trusted-grok")

    def test_grok_reviewer_disables_tools(self) -> None:
        args = argparse.Namespace(
            engine="grok",
            model=None,
            thinking=None,
            fallback_model=None,
            codex_config=None,
            codex_speed=None,
            tools=True,
            web_search=True,
        )

        reviewer = AUTOREVIEW.reviewer_args(args)[0]

        self.assertEqual(reviewer.model, "grok-4.7")
        self.assertEqual(reviewer.thinking, "low")
        self.assertFalse(reviewer.tools)
        self.assertFalse(reviewer.web_search)

    def test_grok_clear_directory_is_snapshot_based_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autoreview-grok-clear-test.") as tmpdir:
            root = Path(tmpdir)
            (root / "keep.txt").write_text("keep", encoding="utf-8")
            (root / "file.txt").write_text("remove", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "child.txt").write_text("remove", encoding="utf-8")
            (root / "link").symlink_to(root / "missing-target")

            AUTOREVIEW.clear_directory(root, keep={"keep.txt"})

            self.assertEqual({path.name for path in root.iterdir()}, {"keep.txt"})

            leftover = root / "left-behind.txt"
            leftover.write_text("remove", encoding="utf-8")
            real_unlink = Path.unlink

            def skip_one_unlink(path: Path, *args: object, **kwargs: object) -> None:
                if path.name != leftover.name:
                    real_unlink(path, *args, **kwargs)

            with mock.patch.object(Path, "unlink", new=skip_one_unlink), self.assertRaisesRegex(
                SystemExit,
                r"isolated Grok runtime could not be cleared: left-behind\.txt",
            ):
                AUTOREVIEW.clear_directory(root, keep={"keep.txt"})

    def test_grok_refuses_unverified_version(self) -> None:
        args = argparse.Namespace(grok_bin="grok")
        required_flags = " ".join(AUTOREVIEW.GROK_REQUIRED_FLAGS)

        def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            if "--version" in command:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    os.environ["AUTOREVIEW_FAKE_GROK_VERSION"],
                    "",
                )
            return subprocess.CompletedProcess(command, 0, required_flags, "")

        with tempfile.TemporaryDirectory(prefix="autoreview-grok-probe-test.") as tmpdir, mock.patch.object(
            AUTOREVIEW, "resolve_command", return_value="/usr/bin/grok",
        ), mock.patch.object(
            AUTOREVIEW, "safe_engine_env", return_value={},
        ), mock.patch.object(
            AUTOREVIEW, "safe_temp_root", return_value=Path(tmpdir),
        ), mock.patch.object(
            AUTOREVIEW, "run", side_effect=fake_run,
        ):
            for version in (
                "grok 1.0.40 (test)",
                "grok 1.0.42 (test)",
                "grok 1.0.41-rc.1 (test)",
                "grok 1.0.41-beta (test)",
                "grok 1.0.41.7 (test)",
            ):
                with self.subTest(version=version), mock.patch.dict(
                    os.environ, {"AUTOREVIEW_FAKE_GROK_VERSION": version}, clear=False,
                ):
                    with self.assertRaisesRegex(
                        SystemExit,
                        rf"unverified Grok Build version {re.escape(version.split()[1])}",
                    ):
                        AUTOREVIEW.ensure_grok_isolation_supported(args, Path(tmpdir))
            with mock.patch.dict(
                os.environ, {"AUTOREVIEW_FAKE_GROK_VERSION": "grok 1.0.41 (test)"}, clear=False,
            ):
                self.assertEqual(
                    AUTOREVIEW.ensure_grok_isolation_supported(args, Path(tmpdir)),
                    "/usr/bin/grok",
                )

    def test_grok_isolation_requires_cli_flags(self) -> None:
        args = argparse.Namespace(grok_bin="grok")
        for missing_flag in ("--verbatim", "--model"):
            def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
                if "--version" in command:
                    return subprocess.CompletedProcess(command, 0, "grok 1.0.41 (test)", "")
                flags = [flag for flag in AUTOREVIEW.GROK_REQUIRED_FLAGS if flag != missing_flag]
                if missing_flag == "--model":
                    flags.append("-m")
                return subprocess.CompletedProcess(command, 0, " ".join(flags), "")

            with self.subTest(missing_flag=missing_flag), tempfile.TemporaryDirectory(
                prefix="autoreview-grok-probe-test.",
            ) as tmpdir, mock.patch.object(
                AUTOREVIEW, "resolve_command", return_value="/usr/bin/grok",
            ), mock.patch.object(
                AUTOREVIEW, "safe_engine_env", return_value={},
            ), mock.patch.object(
                AUTOREVIEW, "safe_temp_root", return_value=Path(tmpdir),
            ), mock.patch.object(
                AUTOREVIEW, "run", side_effect=fake_run,
            ):
                with self.assertRaisesRegex(SystemExit, re.escape(missing_flag)):
                    AUTOREVIEW.ensure_grok_isolation_supported(args, Path(tmpdir))

    def test_grok_version_and_help_probe_timeouts_are_bounded(self) -> None:
        args = argparse.Namespace(grok_bin="grok")
        for timed_out_probe in ("--version", "--help"):
            observed_timeouts: list[float | None] = []

            def fake_run(
                command: list[str], *_args: object, **kwargs: object,
            ) -> subprocess.CompletedProcess[str]:
                timeout = kwargs.get("timeout")
                observed_timeouts.append(timeout if isinstance(timeout, (int, float)) else None)
                if timed_out_probe in command:
                    raise subprocess.TimeoutExpired(command, timeout)
                return subprocess.CompletedProcess(command, 0, "grok 1.0.41 (test)", "")

            with self.subTest(probe=timed_out_probe), tempfile.TemporaryDirectory(
                prefix="autoreview-grok-probe-timeout-test.",
            ) as tmpdir, mock.patch.object(
                AUTOREVIEW, "GROK_PROBE_TIMEOUT_SECONDS", 0.01,
            ), mock.patch.object(
                AUTOREVIEW, "resolve_command", return_value="/usr/bin/grok",
            ), mock.patch.object(
                AUTOREVIEW, "safe_engine_env", return_value={},
            ), mock.patch.object(
                AUTOREVIEW, "safe_temp_root", return_value=Path(tmpdir),
            ), mock.patch.object(
                AUTOREVIEW, "run", side_effect=fake_run,
            ):
                with self.assertRaisesRegex(
                    SystemExit,
                    rf"Grok Build {re.escape(timed_out_probe)} probe timed out after 0.01s",
                ):
                    AUTOREVIEW.ensure_grok_isolation_supported(args, Path(tmpdir))

            self.assertTrue(observed_timeouts)
            self.assertEqual(set(observed_timeouts), {0.01})

    def test_grok_refuses_managed_layers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autoreview-grok-managed-test.") as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            source.mkdir()
            system = root / "etc-grok"
            system.mkdir()
            managed_preferences = (
                root / "managed-preferences" / "ai.x.grok.plist",
                root / "managed-preferences" / "test-user" / "ai.x.grok.plist",
            )
            for managed_path in (
                source / "managed_config.toml",
                source / "requirements.toml",
                system / "managed_config.toml",
                system / "requirements.toml",
            ):
                managed_path.write_text("managed = true\n", encoding="utf-8")
                with self.subTest(path=managed_path), mock.patch.object(
                    AUTOREVIEW, "GROK_SYSTEM_CONFIG_ROOT", system,
                ), mock.patch.object(
                    AUTOREVIEW, "GROK_MACOS_MANAGED_PREFERENCE_PATHS", (),
                ):
                    with self.assertRaisesRegex(SystemExit, "managed Grok Build configuration detected"):
                        AUTOREVIEW.validate_grok_managed_layers(source)
                managed_path.unlink()

            for managed_path in managed_preferences:
                managed_path.parent.mkdir(parents=True, exist_ok=True)
                managed_path.write_bytes(b"opaque managed preference fixture")
                with self.subTest(path=managed_path), mock.patch.object(
                    AUTOREVIEW, "GROK_SYSTEM_CONFIG_ROOT", system,
                ), mock.patch.object(
                    AUTOREVIEW,
                    "GROK_MACOS_MANAGED_PREFERENCE_PATHS",
                    managed_preferences,
                ), mock.patch.object(
                    AUTOREVIEW.sys, "platform", "darwin",
                ):
                    with self.assertRaisesRegex(
                        SystemExit, "managed Grok Build configuration detected",
                    ):
                        AUTOREVIEW.validate_grok_managed_layers(source)
                managed_path.unlink()

            fake_defaults = mock.Mock(
                return_value=subprocess.CompletedProcess(
                    ["defaults", "read", "ai.x.grok"],
                    0,
                    '{ "ordinary-user-setting" = 1; }',
                    "",
                ),
            )
            with mock.patch.object(
                AUTOREVIEW, "GROK_SYSTEM_CONFIG_ROOT", system,
            ), mock.patch.object(
                AUTOREVIEW,
                "GROK_MACOS_MANAGED_PREFERENCE_PATHS",
                managed_preferences,
            ), mock.patch.object(
                AUTOREVIEW.sys, "platform", "darwin",
            ), mock.patch.object(
                AUTOREVIEW.subprocess, "run", fake_defaults,
            ):
                # Ordinary user defaults are not part of the MDM/admin layer.
                self.assertIsNone(
                    AUTOREVIEW.validate_grok_managed_layers(source)
                )
            fake_defaults.assert_not_called()

            managed_root = root / "effective-managed-preferences"
            effective_path = managed_root / "effective-user" / "ai.x.grok.plist"
            login_path = managed_root / "login-user" / "ai.x.grok.plist"
            fake_pwd = mock.Mock()
            fake_pwd.getpwuid.return_value.pw_name = "effective-user"
            for managed_path in (effective_path, login_path):
                managed_path.parent.mkdir(parents=True, exist_ok=True)
                managed_path.write_bytes(b"opaque managed preference fixture")
                with self.subTest(identity_path=managed_path), mock.patch.object(
                    AUTOREVIEW, "GROK_SYSTEM_CONFIG_ROOT", system,
                ), mock.patch.object(
                    AUTOREVIEW,
                    "GROK_MACOS_MANAGED_PREFERENCE_PATHS",
                    (managed_root / "ai.x.grok.plist",),
                ), mock.patch.object(
                    AUTOREVIEW,
                    "GROK_MACOS_MANAGED_PREFERENCE_ROOT",
                    managed_root,
                ), mock.patch.object(
                    AUTOREVIEW, "pwd_module", fake_pwd,
                ), mock.patch.object(
                    AUTOREVIEW.os, "geteuid", return_value=4242,
                ), mock.patch.object(
                    AUTOREVIEW.sys, "platform", "darwin",
                ), mock.patch.dict(
                    os.environ, {"LOGNAME": "login-user", "USER": "login-user"}, clear=False,
                ):
                    with self.assertRaisesRegex(
                        SystemExit, "managed Grok Build configuration detected",
                    ):
                        AUTOREVIEW.validate_grok_managed_layers(source)
                managed_path.unlink()

            fake_pwd.getpwuid.assert_called_with(4242)

    def test_grok_requires_subscription_login(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autoreview-grok-auth-test.") as tmpdir:
            root = Path(tmpdir)
            repo = root / "repo"
            workspace = root / "runtime"
            source = root / "grok-home"
            repo.mkdir()
            workspace.mkdir()
            source.mkdir()
            with mock.patch.dict(os.environ, {"GROK_HOME": str(source)}, clear=False):
                with self.assertRaisesRegex(SystemExit, "grok login"):
                    AUTOREVIEW.validate_grok_runtime_auth_source(repo, workspace)
                inside = repo / "auth.json"
                inside.write_bytes(b"fixture-auth")
                (source / "auth.json").symlink_to(inside)
                with self.assertRaisesRegex(SystemExit, "outside the reviewed repository"):
                    AUTOREVIEW.validate_grok_runtime_auth_source(repo, workspace)
                (source / "auth.json").unlink()
                (repo / "auth.json").write_bytes(b"fixture-auth")
                with mock.patch.dict(os.environ, {"GROK_HOME": str(repo)}, clear=False):
                    with self.assertRaisesRegex(SystemExit, "outside the reviewed repository"):
                        AUTOREVIEW.validate_grok_runtime_auth_source(repo, workspace)

            status_path = root / "status.json"
            reviewer = argparse.Namespace(
                engine="grok", grok_bin="grok", model="grok-4.7", thinking="high",
                fallback_model=None, tools=False,
            )
            dry_args = argparse.Namespace(
                max_priority="P0", commit="HEAD", status_output=str(status_path),
            )
            output = io.StringIO()
            with mock.patch.dict(os.environ, {"GROK_HOME": str(source)}, clear=False), mock.patch.object(
                AUTOREVIEW, "capture_evidence_inputs", return_value=AUTOREVIEW.EvidenceInputs("", [], []),
            ), mock.patch.object(
                AUTOREVIEW, "build_bundle", return_value=AUTOREVIEW.CapturedBundle("", set()),
            ), mock.patch.object(
                AUTOREVIEW, "prepare_review_prompts", return_value=["prompt"],
            ), mock.patch.object(
                AUTOREVIEW, "verify_evidence", return_value=None,
            ), mock.patch.object(
                AUTOREVIEW, "verify_mixed_sources", return_value=None,
            ), mock.patch.object(
                AUTOREVIEW, "find_command", return_value="/usr/bin/grok",
            ), mock.patch.dict(
                AUTOREVIEW.ENGINE_ISOLATION_PROBES,
                {"grok": lambda _reviewer, _repo: "/usr/bin/grok"},
            ), mock.patch.object(
                AUTOREVIEW, "validate_grok_managed_layers", return_value=None,
            ), contextlib.redirect_stdout(output):
                self.assertEqual(
                    AUTOREVIEW.dry_run_preflight(dry_args, [reviewer], repo, "local", None),
                    1,
                )
            self.assertIn("UNAVAILABLE", output.getvalue())
            self.assertIn("grok login", output.getvalue())
            self.assertFalse(status_path.exists())

    def test_grok_runtime_env_and_home(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autoreview-grok-runtime-test.") as tmpdir:
            root = Path(tmpdir)
            repo = root / "repo"
            source = root / "source-home"
            runtime = root / "runtime"
            workspace = root / "workspace"
            repo.mkdir()
            source.mkdir()
            runtime.mkdir()
            workspace.mkdir()
            (source / "auth.json").write_bytes(b"fixture-auth")
            runtime_home = runtime / "GROK_HOME"
            runtime_home.mkdir()
            AUTOREVIEW.prepare_grok_runtime_auth(repo, workspace, runtime_home, source)

            inspect = subprocess.CompletedProcess(
                ["grok", "inspect", "--json"],
                0,
                json.dumps({"skills": [{"name": "alpha"}, {"name": "beta"}]}),
                "",
            )
            with mock.patch.object(AUTOREVIEW, "run", return_value=inspect):
                AUTOREVIEW.write_grok_review_files(
                    "/usr/bin/grok", repo, workspace, runtime_home, runtime / "home",
                )

            self.assertEqual({path.name for path in runtime_home.iterdir()}, {"auth.json", "config.toml"})
            self.assertEqual(stat.S_IMODE((runtime_home / "auth.json").stat().st_mode), 0o600)
            config = tomllib.loads((runtime_home / "config.toml").read_text(encoding="utf-8"))
            self.assertEqual(config["skills"]["disabled"], ["alpha", "beta"])
            self.assertEqual(config["skills"]["ignore"], [
                str(runtime_home / "bundled"),
                str(runtime_home / "skills"),
                str(runtime / "home"),
                str(workspace),
            ])
            self.assertFalse(config["cli"]["auto_update"])
            self.assertFalse(config["features"]["telemetry"])
            self.assertFalse(config["features"]["feedback"])
            self.assertFalse(config["features"]["backend_tools"])
            self.assertFalse(config["features"]["write_file"])
            self.assertFalse(config["features"]["ask_user_question"])
            self.assertFalse(config["features"]["image_gen"])
            self.assertFalse(config["features"]["video_gen"])
            self.assertFalse(config["features"]["campaigns"])
            self.assertFalse(config["features"]["managed_config"])
            self.assertFalse(config["features"]["non_git_warning"])
            self.assertFalse(config["features"]["codebase_indexing"])
            for section in ("goal", "workflows", "subagents", "memory", "managed_mcps"):
                self.assertFalse(config[section]["enabled"])
            for vendor in ("claude", "cursor"):
                self.assertEqual(config["compat"][vendor], {
                    "mcps": False, "rules": False, "skills": False,
                    "agents": False, "hooks": False, "sessions": False,
                })
            self.assertEqual(config["compat"]["codex"], {
                "hooks": False, "skills": False, "sessions": False,
            })
            hostile = {
                "XAI_API_KEY": "xai",
                "GROK_DEPLOYMENT_KEY": "deployment",
                "GROK_CONFIG": "/tmp/config",
                "GROK_CONFIG_PATH": "/tmp/config-path",
                "GROK_AUTH_PROVIDER_COMMAND": "credential-helper",
                "AWS_CONFIG_FILE": "/tmp/aws",
                "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/google",
                "SSL_CERT_FILE": "/tmp/cert",
            }
            extra = AUTOREVIEW.grok_runtime_env(runtime_home, runtime / "home")
            with mock.patch.dict(os.environ, hostile, clear=False), mock.patch.object(
                AUTOREVIEW, "normalize_external_env_path_value", side_effect=lambda _repo, _key, value: value,
            ):
                env = AUTOREVIEW.safe_engine_env(repo, engine="grok", extra=extra)
            for key in (
                "XAI_API_KEY", "GROK_DEPLOYMENT_KEY", "GROK_CONFIG", "GROK_CONFIG_PATH",
                "GROK_AUTH_PROVIDER_COMMAND", "AWS_CONFIG_FILE", "GOOGLE_APPLICATION_CREDENTIALS",
            ):
                self.assertNotIn(key, env)
            self.assertEqual(env["SSL_CERT_FILE"], "/tmp/cert")
            for key, value in AUTOREVIEW.GROK_RUNTIME_ENV.items():
                self.assertEqual(env[key], value)

    def test_grok_command_shape(self) -> None:
        args = argparse.Namespace(
            grok_bin="grok", model="grok-4.7", thinking="high",
            stream_engine_output=False, engine_timeout_seconds=None,
        )
        observed: dict[str, object] = {}

        def fake_stream(command: list[str], cwd: Path, **kwargs: object) -> subprocess.CompletedProcess[str]:
            observed["command"] = command
            observed["cwd"] = cwd
            observed["env"] = kwargs["env"]
            prompt_path = Path(command[command.index("--prompt-file") + 1])
            observed["prompt"] = prompt_path.read_text(encoding="utf-8")
            runtime_home = Path(str(kwargs["env"]["GROK_HOME"]))  # type: ignore[index]
            observed["home_entries"] = {path.name for path in runtime_home.iterdir()}
            stream = "\n".join(
                json.dumps(event)
                for event in (
                    {"type": "system", "subtype": "init", "tools": [], "skills": [], "mcp_servers": []},
                    {"type": "result", "subtype": "success", "is_error": False,
                     "result": "review-result",
                     "usage": {"server_tool_use": {"web_search_requests": 0}}},
                )
            ) + "\n"
            return subprocess.CompletedProcess(command, 0, stream, "")

        def fake_local_run(
            command: list[str], *_args: object, **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            if "--version" in command:
                return subprocess.CompletedProcess(command, 0, "grok 1.0.41 (test)", "")
            return subprocess.CompletedProcess(
                command, 0, json.dumps({"skills": [{"name": "bundled"}]}), "",
            )

        with tempfile.TemporaryDirectory(prefix="autoreview-grok-command-test.") as tmpdir:
            root = Path(tmpdir)
            repo = root / "repo"
            source = root / "source"
            repo.mkdir()
            source.mkdir()
            (source / "auth.json").write_bytes(b"fixture-auth")
            with mock.patch.dict(os.environ, {"GROK_HOME": str(source)}, clear=False), mock.patch.object(
                AUTOREVIEW, "ensure_grok_isolation_supported", return_value="/usr/bin/grok",
            ), mock.patch.object(
                AUTOREVIEW, "validate_grok_managed_layers", return_value=None,
            ), mock.patch.object(
                AUTOREVIEW, "run", side_effect=fake_local_run,
            ), mock.patch.object(
                AUTOREVIEW, "run_grok_with_guard", side_effect=fake_stream,
            ):
                self.assertEqual(AUTOREVIEW.run_grok(args, repo, "review prompt"), "review-result")

        command = observed["command"]
        assert isinstance(command, list)
        self.assertEqual(
            observed["prompt"],
            "review prompt\n\n"
            "You have no tools and this is your only turn: everything you need is in this prompt. "
            "Reply with the JSON object only: no preamble, no commentary, no code fence.",
        )
        self.assertNotIn("--tools", command)
        self.assertIn("--verbatim", command)
        self.assertIn("--no-subagents", command)
        self.assertIn("--disable-web-search", command)
        self.assertEqual(command[command.index("--max-turns") + 1], "1")
        self.assertEqual(command[command.index("--output-format") + 1], "streaming-messages-json")
        self.assertEqual(command[command.index("--reasoning-effort") + 1], "high")
        self.assertEqual(command[command.index("--model") + 1], "grok-4.7")
        self.assertEqual(command[command.index("--cwd") + 1], str(observed["cwd"]))
        self.assertNotEqual(Path(str(observed["cwd"])), repo)
        self.assertEqual(command[command.index("--disallowed-tools") + 1], ",".join(AUTOREVIEW.GROK_DENIED_TOOLS))
        self.assertEqual(observed["home_entries"], {"auth.json", "config.toml"})

    def test_grok_prelaunch_refuses_version_change(self) -> None:
        args = argparse.Namespace(
            grok_bin="grok", model="grok-4.7", thinking="high",
            stream_engine_output=False, engine_timeout_seconds=None,
        )
        observed_env: dict[str, str] = {}

        def fake_local_run(
            command: list[str], *_args: object, **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            if "--version" in command:
                observed_env.update(kwargs["env"])  # type: ignore[arg-type]
                runtime_home = Path(observed_env["GROK_HOME"])
                self.assertEqual(
                    {path.name for path in runtime_home.iterdir()},
                    {"auth.json", "config.toml"},
                )
                return subprocess.CompletedProcess(command, 0, "grok 1.0.42 (changed)", "")
            return subprocess.CompletedProcess(
                command, 0, json.dumps({"skills": [{"name": "bundled"}]}), "",
            )

        with tempfile.TemporaryDirectory(prefix="autoreview-grok-version-change-test.") as tmpdir:
            root = Path(tmpdir)
            repo = root / "repo"
            source = root / "source"
            repo.mkdir()
            source.mkdir()
            (source / "auth.json").write_bytes(b"fixture-auth")
            with mock.patch.dict(
                os.environ, {"GROK_HOME": str(source)}, clear=False,
            ), mock.patch.object(
                AUTOREVIEW, "ensure_grok_isolation_supported", return_value="/usr/bin/grok",
            ), mock.patch.object(
                AUTOREVIEW, "validate_grok_managed_layers", return_value=None,
            ), mock.patch.object(
                AUTOREVIEW, "run", side_effect=fake_local_run,
            ), mock.patch.object(
                AUTOREVIEW, "run_grok_with_guard",
            ) as guarded_run, self.assertRaisesRegex(
                SystemExit, "unverified Grok Build version 1.0.42",
            ):
                AUTOREVIEW.run_grok(args, repo, "review prompt")

        guarded_run.assert_not_called()
        self.assertNotEqual(Path(observed_env["GROK_HOME"]), source)
        self.assertNotEqual(Path(observed_env["HOME"]), source)
        self.assertEqual(observed_env["HOME"], observed_env["USERPROFILE"])

    def test_grok_runtime_guard(self) -> None:
        guarded = (
            [{"type": "system", "subtype": "init", "tools": ["read_file"], "skills": [], "mcp_servers": []}],
            [{"type": "system", "subtype": "init", "tools": [], "skills": ["bundled"], "mcp_servers": []}],
            [{"type": "system", "subtype": "init", "tools": [], "skills": [], "mcp_servers": ["foreign"]}],
            [{"type": "system", "subtype": "init", "tools": [], "skills": [], "mcp_servers": None}],
            [{"type": "system", "subtype": "init", "tools": [], "skills": []}],
            [
                {"type": "system", "subtype": "init", "tools": [], "skills": [], "mcp_servers": []},
                {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "read_file"}]}},
            ],
            [
                {"type": "system", "subtype": "init", "tools": [], "skills": [], "mcp_servers": []},
                {"type": "assistant", "message": {"content": [{"type": "server_tool_use", "name": "x_search"}]}},
            ],
            [
                {"type": "system", "subtype": "init", "tools": [], "skills": [], "mcp_servers": []},
                {"type": "user", "message": {"content": [{"type": "tool_result"}]}},
            ],
            [
                {"type": "system", "subtype": "init", "tools": [], "skills": [], "mcp_servers": []},
                {"type": "result", "subtype": "success", "is_error": False, "result": "x",
                 "usage": {"server_tool_use": {"web_search_requests": 1}}},
            ],
            [
                {"type": "system", "subtype": "init", "tools": [], "skills": [], "mcp_servers": []},
                {"type": "result", "subtype": "success", "is_error": False, "result": "x",
                 "usage": {"server_tool_use": {"web_search_requests": "0"}}},
            ],
            [
                {"type": "system", "subtype": "init", "tools": [], "skills": [], "mcp_servers": []},
                {"type": "result", "subtype": "error_max_turns", "is_error": True, "result": "",
                 "usage": {"server_tool_use": {"web_search_requests": 1}}},
            ],
        )
        for events in guarded:
            with self.subTest(event=events[-1].get("type")):
                with self.assertRaises(AUTOREVIEW.ReviewerUnavailable) as caught:
                    AUTOREVIEW.validate_grok_stream_result(
                        subprocess.CompletedProcess(
                            ["grok"], 17, "\n".join(json.dumps(event) for event in events) + "\n", "",
                        ),
                    )
                self.assertEqual(caught.exception.reason, "runtime_validation_failed")

        valid_init = "{'type':'system','subtype':'init','tools':[],'skills':[],'mcp_servers':[]}"
        violating_outputs = {
            "inventory": (
                "print(json.dumps({'type':'system','subtype':'init','tools':['read_file'],"
                "'skills':[],'mcp_servers':[]}), flush=True)"
            ),
            "garbage_before_init": "print('not-json', flush=True)",
            "missing_mcp_inventory": (
                "print(json.dumps({'type':'system','subtype':'init','tools':[],"
                "'skills':[]}), flush=True)"
            ),
            "second_init": (
                f"print(json.dumps({valid_init}), flush=True); "
                f"print(json.dumps({valid_init}), flush=True)"
            ),
        }
        with tempfile.TemporaryDirectory(prefix="autoreview-grok-guard-test.") as tmpdir:
            root = Path(tmpdir)
            for scenario, output_code in violating_outputs.items():
                fake = (
                    "import json, pathlib, sys, time; " + output_code + "; "
                    "time.sleep(5); pathlib.Path(sys.argv[1]).write_text('not killed')"
                )
                for stream_output in (False, True):
                    marker = root / f"completed-{scenario}-{stream_output}"
                    started = time.monotonic()
                    displayed = io.StringIO()
                    with self.subTest(
                        actual_process=scenario, stream_output=stream_output,
                    ), contextlib.redirect_stdout(displayed), self.assertRaises(
                        AUTOREVIEW.ReviewerUnavailable,
                    ) as caught:
                        AUTOREVIEW.run_grok_with_guard(
                            [sys.executable, "-c", fake, str(marker)], root, label="grok",
                            max_runtime_seconds=10, stream_output=stream_output, env=dict(os.environ),
                        )
                    self.assertEqual(caught.exception.reason, "runtime_validation_failed")
                    self.assertLess(time.monotonic() - started, 4)
                    self.assertFalse(marker.exists())
                    self.assertEqual(bool(displayed.getvalue()), stream_output)

            ignores_sigterm = (
                "import json, pathlib, signal, sys, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "print(json.dumps({'type':'system','subtype':'init','tools':['read_file'],"
                "'skills':[],'mcp_servers':[]}), flush=True); "
                "time.sleep(0.8); pathlib.Path(sys.argv[1]).write_text('deadline missed')"
            )
            marker = root / "deadline-missed"
            terminate_calls = 0
            real_terminate = AUTOREVIEW.terminate_process_group

            def staged_terminate(proc: subprocess.Popen[str]) -> None:
                nonlocal terminate_calls
                terminate_calls += 1
                if terminate_calls == 1:
                    proc.send_signal(signal.SIGTERM)
                else:
                    real_terminate(proc, grace_seconds=0.01)

            started = time.monotonic()
            with mock.patch.object(
                AUTOREVIEW, "terminate_process_group", side_effect=staged_terminate,
            ), self.assertRaises(AUTOREVIEW.ReviewerUnavailable) as caught:
                AUTOREVIEW.run_grok_with_guard(
                    [sys.executable, "-c", ignores_sigterm, str(marker)], root, label="grok",
                    max_runtime_seconds=0.05, stream_output=False, env=dict(os.environ),
                )
            self.assertEqual(caught.exception.reason, "runtime_validation_failed")
            self.assertGreaterEqual(terminate_calls, 2)
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertFalse(marker.exists())

    def test_grok_stream_output_redacts_credentials_before_display(self) -> None:
        bearer_fixture = "fixture-stream-bearer"
        access_fixture = "fixture-stream-access"
        prefixed_fixture = "fixture-stream-prefixed"
        events = (
            {"type": "system", "subtype": "init", "tools": [], "skills": [], "mcp_servers": []},
            {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "errors": [
                    f"Authorization: Bearer {bearer_fixture}",
                    f"access_token={access_fixture}",
                    f"xai_api_key={prefixed_fixture}",
                ],
            },
        )
        fake = (
            "import json; events = " + repr(events) + "; "
            "[print(json.dumps(event), flush=True) for event in events]"
        )
        displayed_stdout = io.StringIO()
        displayed_stderr = io.StringIO()
        with tempfile.TemporaryDirectory(prefix="autoreview-grok-stream-redaction-test.") as tmpdir:
            with contextlib.redirect_stdout(displayed_stdout), contextlib.redirect_stderr(
                displayed_stderr,
            ):
                result = AUTOREVIEW.run_grok_with_guard(
                    [sys.executable, "-c", fake],
                    Path(tmpdir),
                    label="grok",
                    max_runtime_seconds=5,
                    stream_output=True,
                    env=dict(os.environ),
                )
                with self.assertRaises(AUTOREVIEW.ReviewerUnavailable):
                    AUTOREVIEW.validate_grok_stream_result(result)

        displayed = displayed_stdout.getvalue() + displayed_stderr.getvalue()
        self.assertNotIn(bearer_fixture, displayed)
        self.assertNotIn(access_fixture, displayed)
        self.assertNotIn(prefixed_fixture, displayed)
        self.assertIn("[REDACTED]", displayed)

    def test_grok_engine_error_result_is_engine_failed(self) -> None:
        events = (
            {"type": "system", "subtype": "init", "tools": [], "skills": [], "mcp_servers": []},
            {
                "type": "result",
                "subtype": "error_max_turns",
                "is_error": True,
                "result": "",
                "errors": ["maximum turns reached", "access_token=synthetic-secret"],
            },
        )
        for returncode in (0, 9):
            result = subprocess.CompletedProcess(
                ["grok"], returncode,
                "\n".join(json.dumps(event) for event in events) + "\n",
                "access_token=stderr-secret",
            )

            with self.subTest(returncode=returncode), self.assertRaises(
                AUTOREVIEW.ReviewerUnavailable,
            ) as caught:
                AUTOREVIEW.validate_grok_stream_result(result)

            self.assertEqual(caught.exception.reason, "engine_failed")
            message = str(caught.exception)
            self.assertIn("error_max_turns", message)
            self.assertIn("maximum turns reached", message)
            self.assertNotIn("synthetic-secret", message)
            self.assertNotIn("stderr-secret", message)
            self.assertIn("[REDACTED]", message)

    def test_grok_engine_error_redacts_auth_schemes_and_prefixed_keys(self) -> None:
        secret_cases = (
            ("Authorization: Bearer fixture-authorization", "fixture-authorization"),
            (
                '{"authorization": "Bearer fixture-json-authorization"}',
                "fixture-json-authorization",
            ),
            ("token: Bearer fixture-token-bearer", "fixture-token-bearer"),
            ("bearer fixture-lowercase", "fixture-lowercase"),
            ("Bearer\tfixture-tab", "fixture-tab"),
            ("Authorization: Basic fixture-basic", "fixture-basic"),
            ("token: Token fixture-token-scheme", "fixture-token-scheme"),
            ("xai_api_key=fixture-xai-api-key", "fixture-xai-api-key"),
            ('{"x_api_key": "fixture-x-api-key"}', "fixture-x-api-key"),
            ("client_secret=fixture-client-secret", "fixture-client-secret"),
            ("user_password=fixture-user-password", "fixture-user-password"),
            ("oauth_refresh_token=fixture-oauth-refresh", "fixture-oauth-refresh"),
            ("__xai_api_key=fixture-leading-underscore", "fixture-leading-underscore"),
            ("--xai-api-key=fixture-leading-hyphen", "fixture-leading-hyphen"),
        )
        init = {"type": "system", "subtype": "init", "tools": [], "skills": [], "mcp_servers": []}
        for error_text, fixture in secret_cases:
            terminal = {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "errors": [error_text],
            }
            result = subprocess.CompletedProcess(
                ["grok"],
                0,
                "\n".join(json.dumps(event) for event in (init, terminal)) + "\n",
                "",
            )

            with self.subTest(error_text=error_text):
                with self.assertRaises(AUTOREVIEW.ReviewerUnavailable) as caught:
                    AUTOREVIEW.validate_grok_stream_result(result)

                self.assertEqual(caught.exception.reason, "engine_failed")
                message = str(caught.exception)
                self.assertNotIn(fixture, message)
                self.assertIn("[REDACTED]", message)

        for ordinary_text in (
            "turkey_keynote=public-value",
            "monkey_key_count=public-value",
            "client_secretary=public-value",
            "oauth_refresh_tokenizer=public-value",
        ):
            with self.subTest(ordinary_text=ordinary_text):
                self.assertEqual(AUTOREVIEW.redact_grok_error_text(ordinary_text), ordinary_text)

    def test_grok_non_string_nested_event_types_do_not_crash(self) -> None:
        init = {"type": "system", "subtype": "init", "tools": [], "skills": [], "mcp_servers": []}
        terminal = {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "review-result",
            "usage": {"server_tool_use": {"web_search_requests": 0}},
        }
        for malformed_type in ({"x": 1}, ["tool_use"]):
            assistant = {"type": "assistant", "content": [{"type": malformed_type}]}
            result = subprocess.CompletedProcess(
                ["grok"],
                0,
                "\n".join(json.dumps(event) for event in (init, assistant, terminal)) + "\n",
                "",
            )
            with self.subTest(malformed_type=malformed_type):
                self.assertEqual(
                    AUTOREVIEW.validate_grok_stream_result(result),
                    "review-result",
                )

    def test_grok_deeply_nested_events_fail_cleanly(self) -> None:
        init = {"type": "system", "subtype": "init", "tools": [], "skills": [], "mcp_servers": []}
        original_recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(original_recursion_limit, 5000))
        try:
            nested_assistant: object = {"type": "text", "text": "deep"}
            nested_error: object = "deep engine error"
            for _ in range(1000):
                nested_assistant = {"content": [nested_assistant]}
                nested_error = {"detail": nested_error}

            cases = (
                (
                    {"type": "assistant", "message": nested_assistant},
                    "runtime_validation_failed",
                ),
                (
                    {
                        "type": "result",
                        "subtype": "error_during_execution",
                        "is_error": True,
                        "errors": [nested_error],
                    },
                    "invalid_report",
                ),
            )
            for event, reason in cases:
                result = subprocess.CompletedProcess(
                    ["grok"], 0,
                    "\n".join(json.dumps(item) for item in (init, event)) + "\n",
                    "",
                )
                with self.subTest(reason=reason), self.assertRaises(
                    AUTOREVIEW.ReviewerUnavailable,
                ) as caught:
                    AUTOREVIEW.validate_grok_stream_result(result)
                self.assertEqual(caught.exception.reason, reason)
        finally:
            sys.setrecursionlimit(original_recursion_limit)

    def test_grok_inspect_failure_stops_preparation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autoreview-grok-inspect-test.") as tmpdir:
            root = Path(tmpdir)
            home = root / "GROK_HOME"
            home.mkdir()
            (home / "auth.json").write_bytes(b"fixture-auth")
            cases = (
                subprocess.CompletedProcess([], 1, "", "inspect failed"),
                subprocess.CompletedProcess([], 0, "not-json", ""),
                subprocess.CompletedProcess([], 0, "{}", ""),
            )
            for result in cases:
                with self.subTest(result=result), mock.patch.object(AUTOREVIEW, "run", return_value=result):
                    with self.assertRaises(SystemExit):
                        AUTOREVIEW.write_grok_review_files(
                            "/usr/bin/grok", root / "repo", root / "workspace", home, root / "home",
                        )

    def test_grok_inspect_probe_timeout_stops_preparation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autoreview-grok-inspect-timeout-test.") as tmpdir:
            root = Path(tmpdir)
            repo = root / "repo"
            workspace = root / "workspace"
            runtime_home = root / "GROK_HOME"
            runtime_user_home = root / "home"
            for directory in (repo, workspace, runtime_home):
                directory.mkdir()
            (runtime_home / "auth.json").write_bytes(b"fixture-auth")
            fake_grok = root / "grok"
            fake_grok.write_text(
                f"#!{sys.executable}\nimport time\ntime.sleep(0.5)\n",
                encoding="utf-8",
            )
            fake_grok.chmod(0o755)

            started = time.monotonic()
            with mock.patch.object(
                AUTOREVIEW, "GROK_PROBE_TIMEOUT_SECONDS", 0.05,
            ), self.assertRaisesRegex(
                SystemExit,
                r"Grok Build inspect probe timed out after 0.05s",
            ):
                AUTOREVIEW.write_grok_review_files(
                    str(fake_grok), repo, workspace, runtime_home, runtime_user_home,
                )
            self.assertLess(time.monotonic() - started, 0.4)

    def test_grok_collects_assistant_text_before_result_fallback(self) -> None:
        init = {"type": "system", "subtype": "init", "tools": [], "skills": [], "mcp_servers": []}
        report = {**FINAL_REPORT, "review_completion": "complete"}
        report_json = json.dumps(report)

        def validate(
            assistant_events: list[dict[str, object]],
            result_text: str,
        ) -> str:
            terminal = {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": result_text,
                "usage": {"server_tool_use": {"web_search_requests": 0}},
            }
            stream = "\n".join(
                json.dumps(event) for event in (init, *assistant_events, terminal)
            ) + "\n"
            return AUTOREVIEW.validate_grok_stream_result(
                subprocess.CompletedProcess(["grok"], 0, stream, ""),
            )

        two_blocks = [{
            "type": "assistant",
            "message": {"content": [
                {"type": "text", "text": "I will review the change.\n"},
                {"type": "thinking", "thinking": "private reasoning"},
                {"type": "text", "text": report_json},
            ]},
        }]
        self.assertEqual(validate(two_blocks, "short terminal text"), report_json)

        two_events = [
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Reviewing the supplied diff.\n"},
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": report_json},
            ]}},
        ]
        self.assertEqual(validate(two_events, "short terminal text"), report_json)

        longer_assistant = [{
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": report_json}]},
        }]
        self.assertEqual(validate(longer_assistant, "short"), report_json)
        self.assertEqual(validate([], report_json), report_json)

        preamble_only = [{
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "I am checking the supplied diff..."}]},
        }]
        self.assertEqual(validate(preamble_only, report_json), report_json)
        unstructured = validate(preamble_only, "No JSON report was produced.")
        with mock.patch.object(AUTOREVIEW, "run_engine", return_value=unstructured), self.assertRaises(
            AUTOREVIEW.ReviewerUnavailable,
        ) as caught:
            AUTOREVIEW.run_reviewer(
                argparse.Namespace(engine="grok", max_priority="P0"),
                Path.cwd(),
                "synthetic",
                set(),
                [],
            )
        self.assertEqual(caught.exception.reason, "invalid_report")

        thinking_only = [{
            "type": "assistant",
            "message": {"content": [{"type": "thinking", "thinking": "draft"}]},
        }]
        with self.assertRaises(AUTOREVIEW.ReviewerUnavailable) as caught:
            validate(thinking_only, "   ")
        self.assertEqual(caught.exception.reason, "invalid_report")

    def test_grok_extracts_report_object_from_surrounding_text(self) -> None:
        init = {"type": "system", "subtype": "init", "tools": [], "skills": [], "mcp_servers": []}
        report = {
            **FINAL_REPORT,
            "overall_explanation": 'brace text { nested } and escaped quote: " }',
            "review_completion": "complete",
        }
        report_json = json.dumps(report)
        earlier_report = {**report, "overall_explanation": "earlier report"}
        earlier_report_json = json.dumps(earlier_report)
        findings_example = json.dumps({"findings": []})
        cases = (
            ("Preamble before the report.\n" + report_json, report_json),
            ("Preamble about the `if (x) {` block.\n" + report_json, report_json),
            ("Preamble with an unmatched } brace.\n" + report_json, report_json),
            ('Preamble quotes "{not a JSON object}" in prose.\n' + report_json, report_json),
            (earlier_report_json + "\nBetween reports.\n" + report_json, report_json),
            (report_json + "\nFor example: " + findings_example, report_json),
            (findings_example, findings_example),
            (report_json + "\nPostscript after the report.", report_json),
            ("Preamble.\n```json\n" + report_json + "\n```", report_json),
            ("No JSON report was produced; {not-json} remains prose.",
             "No JSON report was produced; {not-json} remains prose."),
        )
        for assistant_text, expected in cases:
            assistant = {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": assistant_text}]},
            }
            terminal = {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "short terminal text",
                "usage": {"server_tool_use": {"web_search_requests": 0}},
            }
            stream = "\n".join(json.dumps(event) for event in (init, assistant, terminal)) + "\n"
            with self.subTest(assistant_text=assistant_text[:30]):
                self.assertEqual(
                    AUTOREVIEW.validate_grok_stream_result(
                        subprocess.CompletedProcess(["grok"], 0, stream, ""),
                    ),
                    expected,
                )

    def test_grok_invalid_output_is_unavailable_after_launch(self) -> None:
        init = {"type": "system", "subtype": "init", "tools": [], "skills": [], "mcp_servers": []}
        cases = (
            (0, [init], "invalid_report"),
            (0, [init, "not-json"], "invalid_report"),
            (0, [init, []], "invalid_report"),
            (0, [init, None], "invalid_report"),
            (0, [init, {"type": "result", "subtype": "success", "is_error": False, "result": 3,
                        "usage": {"server_tool_use": {"web_search_requests": 0}}}], "invalid_report"),
            (0, [init, {"type": "result", "subtype": "success", "is_error": False, "result": {},
                        "usage": {"server_tool_use": {"web_search_requests": 0}}}], "invalid_report"),
            (0, [init, {"type": "result", "subtype": "success", "is_error": False, "result": [],
                        "usage": {"server_tool_use": {"web_search_requests": 0}}}], "invalid_report"),
            (0, [init, {"type": "result", "subtype": "success", "is_error": False, "result": None,
                        "usage": {"server_tool_use": {"web_search_requests": 0}}}], "invalid_report"),
            (0, [init, {"type": "result", "subtype": "success", "is_error": False, "result": "  ",
                        "usage": {"server_tool_use": {"web_search_requests": 0}}}], "invalid_report"),
            (0, [init, {"type": "result", "subtype": "success", "is_error": False, "result": "ok",
                        "usage": {"server_tool_use": {"web_search_requests": 0}}},
                 {"type": "assistant", "message": "late"}], "invalid_report"),
            (9, [init], "engine_failed"),
        )
        for returncode, events, reason in cases:
            with self.subTest(returncode=returncode, events=events):
                result = subprocess.CompletedProcess(
                    ["grok"], returncode, "\n".join(json.dumps(event) for event in events) + "\n", "boom",
                )
                with self.assertRaises(AUTOREVIEW.ReviewerUnavailable) as caught:
                    AUTOREVIEW.validate_grok_stream_result(result)
                self.assertEqual(caught.exception.reason, reason)

    def test_grok_success_returns_result_text(self) -> None:
        usage_variants = (
            {},
            {"usage": {}},
            {"usage": {"server_tool_use": {}}},
            {"usage": {"server_tool_use": {"web_search_requests": 0}}},
        )
        for usage in usage_variants:
            stream = "\n".join(
                json.dumps(event)
                for event in (
                    {"type": "system", "subtype": "init", "tools": [], "skills": [], "mcp_servers": []},
                    {"type": "result", "subtype": "success", "is_error": False,
                     "result": "  final report  ", **usage},
                )
            ) + "\n"
            result = subprocess.CompletedProcess(["grok"], 0, stream, "")
            with self.subTest(usage=usage):
                self.assertEqual(AUTOREVIEW.validate_grok_stream_result(result), "  final report  ")

    def test_codex_config_status_exposes_keys_only(self) -> None:
        args = argparse.Namespace(codex_config=['model_verbosity="low"'])
        self.assertEqual(AUTOREVIEW.codex_config_keys(args), ["model_verbosity"])

    def test_codex_retries_luna_after_default_sol_access_failure(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            sys, "argv", ["autoreview", "--no-web-search"],
        ):
            args = AUTOREVIEW.reviewer_args(AUTOREVIEW.parse_args())[0]
        prompt = "complete retry pack: unicode \u03c0\r\n-deleted line\n unchanged context\n"
        with tempfile.TemporaryDirectory(prefix="autoreview-codex-fallback.") as tmpdir:
            events = []

            def fake_run(command, _cwd, **kwargs):
                self.assertEqual(kwargs["input_text"], prompt)
                model = command[command.index("--model") + 1]
                events.append(model)
                self.assertIn('model_reasoning_effort="high"', command)
                if model == "gpt-6-sol":
                    return subprocess.CompletedProcess(
                        command, 1, "",
                        "The model `gpt-6-sol` does not exist or you do not have access to it.",
                    )
                output_path = Path(command[command.index("--output-last-message") + 1])
                output_path.write_text(json.dumps({**FINAL_REPORT, "review_completion": "complete"}))
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.object(AUTOREVIEW, "resolve_command", return_value="/usr/bin/codex"), \
                    mock.patch.object(AUTOREVIEW, "ensure_codex_isolation_supported", return_value="/usr/bin/codex"), \
                    mock.patch.object(AUTOREVIEW, "codex_auth_config_flags", return_value=[]), \
                    mock.patch.object(AUTOREVIEW, "prepare_codex_runtime_auth", return_value=None), \
                    mock.patch.object(AUTOREVIEW, "find_command", side_effect=AssertionError("unexpected scanner lookup")), \
                    mock.patch.object(AUTOREVIEW, "run_with_heartbeat", side_effect=fake_run):
                result = AUTOREVIEW.run_reviewer(args, Path(tmpdir), prompt, set(), [])
                self.assertTrue(result.complete)
                self.assertEqual(result.report["findings"], [])
            self.assertEqual(events, ["gpt-6-sol", "gpt-6-luna"])

    def test_codex_runs_outside_repo_with_bundle_only_workspace(self) -> None:
        args = argparse.Namespace(
            codex_bin="codex",
            codex_config=None,
            codex_speed=None,
            fallback_model=None,
            model="gpt-5.6-sol",
            stream_engine_output=False,
            thinking="high",
            tools=True,
            web_search=False,
        )
        observed: dict[str, object] = {}

        def fake_run(
            command: list[str],
            cwd: Path,
            *_args: object,
            **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            observed["cwd"] = cwd
            observed["command"] = command
            observed["command_cwd"] = Path(command[command.index("-C") + 1])
            observed["workspace_entries"] = list(cwd.iterdir())
            observed["env"] = kwargs["env"]
            observed["schema"] = json.loads(Path(command[command.index("--output-schema") + 1]).read_text())
            output_path = Path(command[command.index("--output-last-message") + 1])
            output_path.write_text(json.dumps(FINAL_REPORT))
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory(prefix="autoreview-codex-workspace-test.") as tmpdir:
            repo = Path(tmpdir)
            (repo / ".env").write_text("ignored environment fixture\n")
            with mock.patch.dict(
                os.environ,
                {"CODEX_HOME": ""},
                clear=False,
            ), mock.patch.object(
                AUTOREVIEW,
                "resolve_command",
                return_value="/usr/bin/codex",
            ), mock.patch.object(
                AUTOREVIEW,
                "ensure_codex_isolation_supported",
                return_value="/usr/bin/codex",
            ), mock.patch.object(
                AUTOREVIEW,
                "codex_auth_config_flags",
                return_value=[],
            ), mock.patch.object(
                AUTOREVIEW,
                "prepare_codex_runtime_auth",
                return_value=None,
            ), mock.patch.object(
                AUTOREVIEW,
                "codex_source_home",
                return_value=None,
            ), mock.patch.object(
                AUTOREVIEW,
                "run_with_heartbeat",
                side_effect=fake_run,
            ):
                output = AUTOREVIEW.run_codex(args, repo, "review")

            self.assertEqual(json.loads(output), FINAL_REPORT)
            self.assertEqual(observed["schema"], AUTOREVIEW.PROVIDER_SCHEMA)
            observed_cwd = observed["cwd"]
            command_cwd = observed["command_cwd"]
            self.assertIsInstance(observed_cwd, Path)
            self.assertIsInstance(command_cwd, Path)
            assert isinstance(observed_cwd, Path)
            assert isinstance(command_cwd, Path)
            self.assertNotEqual(observed_cwd.resolve(), repo.resolve())
            self.assertEqual(observed_cwd, command_cwd)
            self.assertEqual(observed["workspace_entries"], [])
            env = observed["env"]
            self.assertIsInstance(env, dict)
            assert isinstance(env, dict)
            self.assertNotEqual(env["HOME"], os.environ.get("HOME"))
            self.assertEqual(env["USERPROFILE"], env["HOME"])
            self.assertNotEqual(env.get("CODEX_HOME"), str(repo.resolve()))
            self.assertEqual(Path(env["CODEX_HOME"]).name, "codex-home")
            self.assertNotEqual(env["CODEX_HOME"], str((Path.home() / ".codex").resolve()))
            self.assertIn("features.shell_snapshot=false", observed["command"])
            self.assertIn("features.hooks=false", observed["command"])
            self.assertIn("features.plugins=false", observed["command"])
            self.assertIn("skills.include_instructions=false", observed["command"])

    def test_codex_does_not_fallback_after_unrelated_failure(self) -> None:
        args = argparse.Namespace(
            codex_bin="codex",
            codex_config=None,
            codex_speed=None,
            fallback_model="gpt-6-luna",
            model="gpt-6-sol",
            stream_engine_output=False,
            thinking="high",
            tools=True,
            web_search=False,
        )
        models: list[str] = []

        def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            models.append(command[command.index("--model") + 1])
            return subprocess.CompletedProcess(command, 1, "", "network timeout")

        with tempfile.TemporaryDirectory(prefix="autoreview-codex-fallback.") as tmpdir, mock.patch.object(
            AUTOREVIEW,
            "resolve_command",
            return_value="/usr/bin/codex",
        ), mock.patch.object(
            AUTOREVIEW,
            "ensure_codex_isolation_supported",
            return_value="/usr/bin/codex",
        ), mock.patch.object(AUTOREVIEW, "codex_auth_config_flags", return_value=[]), mock.patch.object(
            AUTOREVIEW,
            "prepare_codex_runtime_auth",
            return_value=None,
        ), mock.patch.object(
            AUTOREVIEW,
            "run_with_heartbeat",
            side_effect=fake_run,
        ):
            with self.assertRaisesRegex(SystemExit, "network timeout"):
                AUTOREVIEW.run_codex(args, Path(tmpdir), "review")

        self.assertEqual(models, ["gpt-6-sol"])

    def test_codex_does_not_fallback_after_model_capacity_failure(self) -> None:
        args = argparse.Namespace(
            codex_bin="codex",
            codex_config=None,
            codex_speed=None,
            fallback_model="gpt-6-luna",
            model="gpt-6-sol",
            stream_engine_output=False,
            thinking="high",
            tools=True,
            web_search=False,
        )
        models: list[str] = []

        def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            models.append(command[command.index("--model") + 1])
            return subprocess.CompletedProcess(
                command,
                1,
                "",
                "model_not_available: gpt-6-sol is temporarily unavailable due to capacity",
            )

        with tempfile.TemporaryDirectory(prefix="autoreview-codex-fallback.") as tmpdir, mock.patch.object(
            AUTOREVIEW,
            "resolve_command",
            return_value="/usr/bin/codex",
        ), mock.patch.object(
            AUTOREVIEW,
            "ensure_codex_isolation_supported",
            return_value="/usr/bin/codex",
        ), mock.patch.object(AUTOREVIEW, "codex_auth_config_flags", return_value=[]), mock.patch.object(
            AUTOREVIEW,
            "prepare_codex_runtime_auth",
            return_value=None,
        ), mock.patch.object(
            AUTOREVIEW,
            "run_with_heartbeat",
            side_effect=fake_run,
        ):
            with self.assertRaisesRegex(SystemExit, "temporarily unavailable"):
                AUTOREVIEW.run_codex(args, Path(tmpdir), "review")

        self.assertEqual(models, ["gpt-6-sol"])

    def test_codex_access_fallback_ignores_structured_output_text(self) -> None:
        result = subprocess.CompletedProcess(
            ["codex"],
            1,
            '{"type":"agent_message","text":"gpt-5.6-sol does not exist or you do not have access"}',
            '{"type":"agent_message","message":"gpt-5.6-sol does not exist or you do not have access"}',
        )

        self.assertFalse(
            AUTOREVIEW.codex_model_access_failure(result, "gpt-5.6-sol")
        )

    def test_codex_access_fallback_accepts_terminal_error_event(self) -> None:
        result = subprocess.CompletedProcess(
            ["codex"],
            1,
            '{"type":"error","message":"gpt-5.6-sol does not exist or you do not have access"}',
            "",
        )

        self.assertTrue(
            AUTOREVIEW.codex_model_access_failure(result, "gpt-5.6-sol")
        )

    def test_codex_access_fallback_accepts_account_model_list_error(self) -> None:
        result = subprocess.CompletedProcess(
            ["codex"],
            1,
            "",
            (
                "The model gpt-5.6-sol does not appear in the list of models "
                "available to your account"
            ),
        )

        self.assertTrue(
            AUTOREVIEW.codex_model_access_failure(result, "gpt-5.6-sol")
        )

    def test_codex_access_fallback_ignores_plain_stdout(self) -> None:
        message = "gpt-5.6-sol does not exist or you do not have access"
        stdout_result = subprocess.CompletedProcess(["codex"], 1, message, "")
        stderr_result = subprocess.CompletedProcess(["codex"], 1, "", message)

        self.assertFalse(
            AUTOREVIEW.codex_model_access_failure(stdout_result, "gpt-5.6-sol")
        )
        self.assertTrue(
            AUTOREVIEW.codex_model_access_failure(stderr_result, "gpt-5.6-sol")
        )

    def test_extract_json_accepts_dict_result_payload(self) -> None:
        payload = {
            "type": "result",
            "subtype": "success",
            "result": FINAL_REPORT,
            "session_id": "session-id",
            "request_id": "request-id",
        }
        self.assertEqual(AUTOREVIEW.extract_json(json.dumps(payload)), FINAL_REPORT)

    def test_extract_json_rejects_result_string_with_preamble(self) -> None:
        payload = {
            "type": "result",
            "subtype": "success",
            "result": "Inspecting the diff first.\n" + json.dumps(FINAL_REPORT),
        }
        with self.assertRaisesRegex(SystemExit, "result was not structured JSON"):
            AUTOREVIEW.extract_json(json.dumps(payload))

if __name__ == "__main__":
    unittest.main()
