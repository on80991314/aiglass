"""High-level FSM transition tests (no YOLO, no MediaPipe)."""

from state_machine.find_grab import FGState, FindGrabConfig, FindGrabFSM


def _silent_emit(_msg: str) -> None:
    pass


def _new_fsm() -> FindGrabFSM:
    return FindGrabFSM(FindGrabConfig(), emit=_silent_emit)


def test_initial_state():
    f = _new_fsm()
    assert f.state == FGState.WAITING_FOR_COMMAND


def test_set_target_starts_search():
    f = _new_fsm()
    f.set_target("杯子", "cup")
    assert f.state == FGState.SEARCHING_OBJECT
    assert f.target_zh == "杯子"
    assert f.target_en == "cup"


def test_target_switch_mid_search():
    """Setting a new target while already searching must switch, not be a no-op."""
    f = _new_fsm()
    f.set_target("杯子", "cup")
    f.set_target("筆電", "laptop")
    assert f.state == FGState.SEARCHING_OBJECT
    assert f.target_en == "laptop"


def test_cancel_resets():
    f = _new_fsm()
    f.set_target("杯子", "cup")
    f.cancel()
    assert f.state == FGState.WAITING_FOR_COMMAND
    assert f.target_en is None


def test_hotword_resets_anywhere():
    f = _new_fsm()
    f.set_target("杯子", "cup")
    f.hotword_reset()
    assert f.state == FGState.WAITING_FOR_COMMAND


def test_confirm_grab_yes_goes_to_success():
    f = _new_fsm()
    f.set_target("杯子", "cup")
    f._goto(FGState.CONFIRM_GRAB)
    f.confirm_grab(True)
    assert f.state == FGState.GRAB_SUCCESS


def test_confirm_grab_no_back_to_hand():
    f = _new_fsm()
    f.set_target("杯子", "cup")
    f._goto(FGState.CONFIRM_GRAB)
    f.confirm_grab(False)
    assert f.state == FGState.GUIDING_HAND


def test_subtitle_thread_safe_basics():
    f = _new_fsm()
    f.set_subtitle("hello", duration_s=0.5)
    assert f.get_subtitle() == "hello"
