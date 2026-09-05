def test_threshold_policy_rotation_exports_are_public() -> None:
    import tiny_tensor_compiler as ttc

    assert ttc.NativeBundlePolicyRotationError.__name__ == "NativeBundlePolicyRotationError"
    assert ttc.ThresholdPolicyTransition.__name__ == "ThresholdPolicyTransition"
    assert ttc.ThresholdPolicyRotationStateStore.__name__ == "ThresholdPolicyRotationStateStore"
    assert callable(ttc.create_threshold_policy_transition)
    assert callable(ttc.verify_threshold_policy_transition)
