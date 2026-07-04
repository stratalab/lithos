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
from lithos.posttrain.sft_corpus import (
    SFTCorpusBuildConfig,
    SFTShardWriter,
    SFTSourceSpec,
    build_sft_corpus,
    sft_corpus_manifest,
)
from lithos.posttrain.sft_dataset import (
    IGNORE_INDEX,
    PackedSFTDataset,
    SFTDataset,
    build_xy,
    load_sft_shard_specs,
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
from lithos.posttrain.tir_rollout import (
    RolloutResult,
    parse_tool_call,
    tir_rollout,
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
from lithos.posttrain.verifier_prefs import (
    build_verifier_prefs,
    make_pairs,
)

__all__ = [
    "IGNORE_INDEX",
    "CheckResult",
    "ExecutionResult",
    "MathVerifier",
    "PackedSFTDataset",
    "PostTrainDecontaminator",
    "RolloutResult",
    "SFTCorpusBuildConfig",
    "SFTDataset",
    "SFTShardWriter",
    "SFTSourceSpec",
    "Task",
    "assert_disjoint",
    "build_sft_corpus",
    "build_verifier_prefs",
    "build_xy",
    "check_code",
    "check_numeric",
    "check_symbolic",
    "check_units",
    "extract_final",
    "extract_number",
    "filter_by_level",
    "gen_arithmetic",
    "heuristic_gaming_check",
    "load_sft_shard_specs",
    "load_tasks",
    "make_pairs",
    "messages_text",
    "octave_available",
    "parse_tool_call",
    "prefs_text",
    "run_octave",
    "run_python",
    "run_tool",
    "sft_corpus_manifest",
    "shaped_reward",
    "split_by_year",
    "task_from_record",
    "tir_rollout",
    "verify",
    "verify_batch",
]
