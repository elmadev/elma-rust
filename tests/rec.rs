extern crate elma;

use elma::rec::*;
use elma::Position;
use std::env;
use std::fs;

const PATH_TEST_1: &str = "tests/assets/replays/test_1.rec";
const PATH_TEST_2: &str = "tests/assets/replays/test_2.rec";
const PATH_TEST_3: &str = "tests/assets/replays/test_3.rec";
const PATH_INVALID_EVENT: &str = "tests/assets/replays/invalid_event.rec";
const PATH_UNFINISHED: &str = "tests/assets/replays/unfinished.rec";
const PATH_EVENT_UNFINISHED: &str = "tests/assets/replays/event_unfinished.rec";
const PATH_MULTI_EVENT_UNFINISHED_1: &str = "tests/assets/replays/multi_event_unfinished.rec";
const PATH_MULTI_EVENT_UNFINISHED_2: &str = "tests/assets/replays/multi_event_unfinished_2.rec";

#[test]
// Probably redundant, but maybe some new fields are added in the future.
// Doesn't hurt or impact anything.
fn rec_default_values() {
    let frame = Frame::new();
    assert_eq!(
        frame,
        Frame {
            bike: Position::new(0_f32, 0_f32),
            left_wheel: Position::new(0, 0),
            right_wheel: Position::new(0, 0),
            head: Position::new(0, 0),
            rotation: 0,
            left_wheel_rotation: 0,
            right_wheel_rotation: 0,
            throttle_and_dir: 0,
            back_wheel_rot_speed: 0,
            collision_strength: 0,
        }
    );
    assert_eq!(frame.throttle(), false);
    assert_eq!(frame.direction(), Direction::Left);
    let event = Event::new();
    assert_eq!(
        event,
        Event {
            time: 0_f64,
            event_type: EventType::ObjectTouch(0),
        }
    );
    let replay = Replay::new();
    assert_eq!(
        replay,
        Replay {
            flag_tag: false,
            link: 0,
            level: String::new(),
            rides: vec![],
        }
    );
}

#[test]
fn load_invalid_replay_path() {
    assert_eq!(
        Replay::load("tests/assets/replays/missing.rec").unwrap_err(),
        elma::ElmaError::Io(std::io::ErrorKind::NotFound)
    );
}

#[test]
fn load_valid_replay_1() {
    let replay = Replay::load(PATH_TEST_1).unwrap();
    assert_eq!(replay.is_multi(), false);
    assert_eq!(replay.flag_tag, false);
    assert_eq!(replay.link, 2549082363);
    assert_eq!(replay.level, "tutor14.lev");

    // Some random frames.
    let frames = &replay.rides[0].frames;
    let events = &replay.rides[0].events;
    assert_eq!(frames.len(), 440);
    assert_eq!(
        frames[0],
        Frame {
            bike: Position::new(34.30250_f32, -1.1253119_f32),
            left_wheel: Position::new(-850, -524),
            right_wheel: Position::new(849, -524),
            head: Position::new(0, 439),
            rotation: 10000,
            left_wheel_rotation: 250,
            right_wheel_rotation: 0,
            throttle_and_dir: 205,
            back_wheel_rot_speed: 0,
            collision_strength: 0,
        }
    );
    assert_eq!(frames[0].throttle(), true);
    assert_eq!(frames[0].direction(), Direction::Left);
    assert_eq!(
        frames[100],
        Frame {
            bike: Position::new(27.142517089844_f32, -1.1152113676071_f32),
            left_wheel: Position::new(-903, -514),
            right_wheel: Position::new(586, -534),
            head: Position::new(74, 397),
            rotation: 9826,
            left_wheel_rotation: 248,
            right_wheel_rotation: 238,
            throttle_and_dir: 173,
            back_wheel_rot_speed: 114,
            collision_strength: 0,
        }
    );
    assert_eq!(frames[100].throttle(), true);
    assert_eq!(frames[100].direction(), Direction::Left);
    assert_eq!(
        frames[201],
        Frame {
            bike: Position::new(11.07129573822_f32, 2.8753623962402_f32),
            left_wheel: Position::new(-511, 917),
            right_wheel: Position::new(-692, -789),
            head: Position::new(471, 10),
            rotation: 7325,
            left_wheel_rotation: 25,
            right_wheel_rotation: 23,
            throttle_and_dir: 221,
            back_wheel_rot_speed: 234,
            collision_strength: 0,
        }
    );
    assert_eq!(frames[201].throttle(), true);
    assert_eq!(frames[201].direction(), Direction::Left);
    assert_eq!(
        frames[439],
        Frame {
            bike: Position::new(-34.779712677002_f32, 11.526465415955_f32),
            left_wheel: Position::new(-1050, -33),
            right_wheel: Position::new(286, -757),
            head: Position::new(226, 376),
            rotation: 9047,
            left_wheel_rotation: 73,
            right_wheel_rotation: 163,
            throttle_and_dir: 29,
            back_wheel_rot_speed: 136,
            collision_strength: 22,
        }
    );
    assert_eq!(frames[439].throttle(), true);
    assert_eq!(frames[439].direction(), Direction::Left);

    // Some random event.
    assert_eq!(events.len(), 24);
    assert_eq!(
        events[0],
        Event {
            time: 1.57728480001688_f64,
            event_type: EventType::VoltRight,
        }
    );
    assert_eq!(
        events[1],
        Event {
            time: 1.6974048000097273_f64,
            event_type: EventType::Ground(0.72119284),
        }
    );
    assert_eq!(
        events[11],
        Event {
            time: 3.9464880000114437_f64,
            event_type: EventType::VoltLeft,
        }
    );
    assert_eq!(
        events[23],
        Event {
            time: 6.398683200001716_f64,
            event_type: EventType::ObjectTouch(3),
        }
    );
}

#[test]
fn load_valid_multi_replay_1() {
    let replay = Replay::load(PATH_TEST_2).unwrap();
    assert_eq!(replay.is_multi(), true);
    assert_eq!(replay.flag_tag, false);
    assert_eq!(replay.link, 2549082363);
    assert_eq!(replay.level, "tutor14.lev");
    let frames = &replay.rides[0].frames;
    let frames_2 = &replay.rides[1].frames;
    let events = &replay.rides[0].events;
    let events_2 = &replay.rides[1].events;
    assert_eq!(frames.len(), 440);
    assert_eq!(
        frames[439],
        Frame {
            bike: Position::new(-34.779712677002_f32, 11.526465415955_f32),
            left_wheel: Position::new(-1050, -33),
            right_wheel: Position::new(286, -757),
            head: Position::new(226, 376),
            rotation: 9047,
            left_wheel_rotation: 73,
            right_wheel_rotation: 163,
            throttle_and_dir: 29,
            back_wheel_rot_speed: 136,
            collision_strength: 22,
        }
    );
    assert_eq!(frames[439].throttle(), true);
    assert_eq!(frames[439].direction(), Direction::Left);
    assert_eq!(events.len(), 24);
    assert_eq!(frames_2.len(), 441);
    assert_eq!(frames_2[100].bike.x, 27.138593673706_f32);
    assert_eq!(frames_2[0].bike.y, -1.1253118515015_f32);
    assert_eq!(events_2.len(), 23);
}

#[test]
fn load_valid_replay_1_and_save() {
    let mut replay = Replay::load(PATH_TEST_1).unwrap();
    let mut dir = env::temp_dir();
    dir.push("save_replay_1.rec");
    replay.save(&dir).unwrap();
    let replay_saved = Replay::load(&dir).unwrap();
    assert_eq!(replay, replay_saved);
}

#[test]
fn load_valid_replay_1_from_buffer() {
    let replay = Replay::load(PATH_TEST_1).unwrap();
    let buffer = fs::read(PATH_TEST_1).unwrap();
    assert_eq!(replay, Replay::from_bytes(&buffer).unwrap());
}

#[test]
fn check_save_load_same_replay_1() {
    let replay = Replay::load(PATH_TEST_1).unwrap();
    let buffer = fs::read(PATH_TEST_1).unwrap();
    assert_eq!(buffer, replay.to_bytes().unwrap());
}

#[test]
fn check_save_load_same_replay_2() {
    let replay = Replay::load(PATH_TEST_2).unwrap();
    let buffer = fs::read(PATH_TEST_2).unwrap();
    assert_eq!(buffer, replay.to_bytes().unwrap());
}

#[test]
fn check_save_load_same_replay_3() {
    let replay = Replay::load(PATH_TEST_3).unwrap();
    let buffer = fs::read(PATH_TEST_3).unwrap();
    assert_eq!(buffer, replay.to_bytes().unwrap());
}

#[test]
fn load_valid_replay_2_and_save() {
    let mut replay = Replay::load(PATH_TEST_3).unwrap();
    let mut dir = env::temp_dir();
    dir.push("save_replay_2.rec");
    replay.save(&dir).unwrap();
    let replay_saved = Replay::load(&dir).unwrap();
    assert_eq!(replay, replay_saved);
}

#[test]
fn load_valid_multi_replay_1_and_save() {
    let mut replay = Replay::load(PATH_TEST_2).unwrap();
    let mut dir = env::temp_dir();
    dir.push("save_multi_replay_2.rec");
    replay.save(&dir).unwrap();
    let replay_saved = Replay::load(&dir).unwrap();
    assert_eq!(replay, replay_saved);
}

#[test]
fn load_invalid_event_replay() {
    assert_eq!(
        Replay::load(PATH_INVALID_EVENT).unwrap_err(),
        elma::ElmaError::InvalidReplayFile
    );
}

#[test]
fn replay_get_time_ms_finished_single() {
    let replay = Replay::load(PATH_TEST_1).unwrap();
    let (time, finished) = replay.get_time_ms();
    assert_eq!(time, 14649);
    assert_eq!(finished, true);
}

#[test]
fn replay_get_time_ms_finished_multi() {
    let replay = Replay::load(PATH_TEST_2).unwrap();
    let (time, finished) = replay.get_time_ms();
    assert_eq!(time, 14671);
    assert_eq!(finished, true);
}

#[test]
fn replay_get_time_ms_unfinished_no_event() {
    let replay = Replay::load(PATH_UNFINISHED).unwrap();
    let (time, finished) = replay.get_time_ms();
    assert_eq!(time, 533);
    assert_eq!(finished, false);
}

#[test]
fn replay_get_time_ms_unfinished_event_single() {
    let replay = Replay::load(PATH_TEST_3).unwrap();
    let (time, finished) = replay.get_time_ms();
    assert_eq!(time, 4767);
    assert_eq!(finished, false);
}

#[test]
fn replay_get_time_ms_unfinished_event_multi() {
    let replay = Replay::load(PATH_MULTI_EVENT_UNFINISHED_1).unwrap();
    let (time, finished) = replay.get_time_ms();
    assert_eq!(time, 1600);
    assert_eq!(finished, false);
}

#[test]
fn replay_get_time_ms_unfinished_event_multi_2() {
    let replay = Replay::load(PATH_MULTI_EVENT_UNFINISHED_2).unwrap();
    let (time, finished) = replay.get_time_ms();
    assert_eq!(time, 3233);
    assert_eq!(finished, false);
}

#[test]
fn replay_get_time_ms_unfinished_event_single_2_frame_diff() {
    let replay = Replay::load(PATH_EVENT_UNFINISHED).unwrap();
    let (time, finished) = replay.get_time_ms();
    assert_eq!(time, 8567);
    assert_eq!(finished, false);
}

#[test]
fn replay_get_time_hs_finished_single() {
    let replay = Replay::load(PATH_TEST_1).unwrap();
    let (time, finished) = replay.get_time_hs();
    assert_eq!(time, 1464);
    assert_eq!(finished, true);
}

#[test]
fn replay_get_time_hs_finished_multi() {
    let replay = Replay::load(PATH_TEST_2).unwrap();
    let (time, finished) = replay.get_time_hs();
    assert_eq!(time, 1467);
    assert_eq!(finished, true);
}

#[test]
fn replay_get_time_hs_unfinished_no_event() {
    let replay = Replay::load(PATH_UNFINISHED).unwrap();
    let (time, finished) = replay.get_time_hs();
    assert_eq!(time, 53);
    assert_eq!(finished, false);
}

#[test]
fn replay_get_time_hs_unfinished_event_single() {
    let replay = Replay::load(PATH_TEST_3).unwrap();
    let (time, finished) = replay.get_time_hs();
    assert_eq!(time, 476);
    assert_eq!(finished, false);
}

#[test]
fn replay_get_time_hs_unfinished_event_multi() {
    let replay = Replay::load(PATH_MULTI_EVENT_UNFINISHED_1).unwrap();
    let (time, finished) = replay.get_time_hs();
    assert_eq!(time, 160);
    assert_eq!(finished, false);
}

#[test]
fn replay_get_time_hs_unfinished_event_multi_2() {
    let replay = Replay::load(PATH_MULTI_EVENT_UNFINISHED_2).unwrap();
    let (time, finished) = replay.get_time_hs();
    assert_eq!(time, 323);
    assert_eq!(finished, false);
}

#[test]
fn replay_get_time_hs_unfinished_event_single_2_frame_diff() {
    let replay = Replay::load(PATH_EVENT_UNFINISHED).unwrap();
    let (time, finished) = replay.get_time_hs();
    assert_eq!(time, 856);
    assert_eq!(finished, false);
}
