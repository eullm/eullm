"""Tests for the distillation cost estimator."""

from eullm_forge.distill import estimate_distillation_cost


def test_estimate_14b_to_7b():
    cost = estimate_distillation_cost(
        teacher_params_b=14.0,
        student_params_b=7.0,
        num_tokens_b=50.0,
    )
    assert cost["gpu_hours"] > 0
    assert cost["num_gpus"] >= 1
    assert cost["wall_hours"] > 0
    assert cost["estimated_cost"] > 0


def test_estimate_70b_to_14b():
    cost = estimate_distillation_cost(
        teacher_params_b=70.0,
        student_params_b=14.0,
        num_tokens_b=50.0,
    )
    # 70B needs more GPUs than 14B
    assert cost["num_gpus"] >= 3


def test_estimate_custom_gpu_cost():
    cost_cheap = estimate_distillation_cost(14.0, 7.0, 50.0, gpu_cost_per_hour=1.0)
    cost_expensive = estimate_distillation_cost(14.0, 7.0, 50.0, gpu_cost_per_hour=5.0)
    assert cost_expensive["estimated_cost"] > cost_cheap["estimated_cost"]
