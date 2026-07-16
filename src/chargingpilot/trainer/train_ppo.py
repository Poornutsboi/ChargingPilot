from __future__ import annotations

from chargingpilot.cli import (
    HierarchicalTrainingInputs,
    PPOTrainingResult,
    ValidationCheckpoint,
    build_hierarchical_inputs,
    build_hierarchical_trainer,
    main,
    parse_args,
    preflight_hierarchical_training,
    record_selected_checkpoint,
    require_split_access,
    run_hierarchical_training,
    run_post_selection_test,
    run_rule_policy_checks,
    run_training,
    select_validation_checkpoint,
    validate_hierarchical_checkpoint,
)


__all__ = [
    "PPOTrainingResult",
    "HierarchicalTrainingInputs",
    "ValidationCheckpoint",
    "build_hierarchical_inputs",
    "build_hierarchical_trainer",
    "main",
    "parse_args",
    "preflight_hierarchical_training",
    "record_selected_checkpoint",
    "require_split_access",
    "run_hierarchical_training",
    "run_post_selection_test",
    "run_rule_policy_checks",
    "run_training",
    "select_validation_checkpoint",
    "validate_hierarchical_checkpoint",
]


if __name__ == "__main__":
    main()
