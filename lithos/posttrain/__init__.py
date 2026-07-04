"""Post-training: SFT, DPO, RLVR, and the shared verifier/sandbox (Phase 11-12)."""

from lithos.posttrain.decontam_gate import (
    PostTrainDecontaminator,
    messages_text,
    prefs_text,
)
from lithos.posttrain.sandbox import (
    ExecutionResult,
    octave_available,
    run_octave,
    run_python,
    run_tool,
)
from lithos.posttrain.taskbank import (
    Task,
    assert_disjoint,
    filter_by_level,
    load_tasks,
    split_by_year,
    task_from_record,
    verify,
    verify_batch,
)
from lithos.posttrain.verifier import (
    CheckResult,
    MathVerifier,
    check_code,
    check_numeric,
    check_symbolic,
    check_units,
    extract_final,
    extract_number,
    gen_arithmetic,
    heuristic_gaming_check,
    shaped_reward,
)

__all__ = [
    "CheckResult",
    "ExecutionResult",
    "MathVerifier",
    "PostTrainDecontaminator",
    "Task",
    "assert_disjoint",
    "check_code",
    "check_numeric",
    "check_symbolic",
    "check_units",
    "extract_final",
    "extract_number",
    "filter_by_level",
    "gen_arithmetic",
    "heuristic_gaming_check",
    "load_tasks",
    "messages_text",
    "octave_available",
    "prefs_text",
    "run_octave",
    "run_python",
    "run_tool",
    "shaped_reward",
    "split_by_year",
    "task_from_record",
    "verify",
    "verify_batch",
]
