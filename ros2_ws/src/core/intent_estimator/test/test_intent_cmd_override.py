from intent_estimator.intent_cmd_override import TimedIntentXVelocity


def test_backward_intent_holds_negative_velocity_for_duration():
    state = TimedIntentXVelocity(hold_s=1.0, backward_x_vel=-0.5)

    state.on_intent(1, now_s=10.0)

    assert state.current_x_vel(10.2) == -0.5
    assert state.current_x_vel(11.1) == 0.0


def test_forward_intent_holds_positive_velocity_for_duration():
    state = TimedIntentXVelocity(hold_s=1.0, forward_x_vel=0.5)

    state.on_intent(2, now_s=3.0)

    assert state.current_x_vel(3.5) == 0.5
    assert state.current_x_vel(4.1) == 0.0


def test_idle_intent_clears_active_velocity():
    state = TimedIntentXVelocity(hold_s=1.0)

    state.on_intent(2, now_s=5.0)
    assert state.current_x_vel(5.1) == 0.5

    state.on_intent(0, now_s=5.2)

    assert state.current_x_vel(5.3) == 0.0
