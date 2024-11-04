extern crate elma;
extern crate nom;

use elma::state::*;
use elma::{BestTimes, TimeEntry};
use std::env;
use std::fs;

#[test]
/// Load state.dat, save it and compare without changes.
fn load_parse_save_state() {
    let mut orig_state = State::load("tests/assets/state/state.dat").unwrap();
    orig_state.path = None; // remove path for easier equality check
    let mut dir = env::temp_dir();
    dir.push("saved.dat");
    let mut state = orig_state.clone();
    state.save(&dir).unwrap();

    let file_original = fs::read("tests/assets/state/state.dat").unwrap();
    let file_saved = fs::read(&dir).unwrap();

    let mut saved_state = State::load(&dir).unwrap();
    saved_state.path = None; // remove path for easier equality check

    let mut expected_times = BestTimes::new();
    expected_times
        .single
        .push(TimeEntry::new(("proman", "proman"), 1465));
    expected_times
        .single
        .push(TimeEntry::new(("proman", "proman"), 1487));
    expected_times
        .multi
        .push(TimeEntry::new(("proman", "proman"), 1492));
    expected_times
        .multi
        .push(TimeEntry::new(("proman", "proman"), 1494));
    assert_eq!(expected_times, orig_state.times[0]);
    assert_eq!(PlayMode::Single, orig_state.play_mode);
    assert_eq!(false, orig_state.sound_enabled);
    assert_eq!(
        SoundOptimization::Compatibility,
        orig_state.sound_optimization
    );
    assert_eq!(true, orig_state.animated_menus);
    assert_eq!(VideoDetail::High, orig_state.video_detail);
    assert_eq!(true, orig_state.animated_objects);
    assert_eq!(false, orig_state.swap_bikes);
    file_original
        .iter()
        .zip(&file_saved)
        .enumerate()
        .for_each(|(i, (o, s))| {
            if o != s {
                assert!(false, "State files differ at: {}", i)
            }
        });
    assert_eq!(&file_original, &file_saved);
    assert_eq!(&file_original.len(), &file_saved.len());
    assert_eq!(orig_state, saved_state);
}

#[test]
fn load_state_from_bytes() {
    let mut state = State::load("tests/assets/state/state.dat").unwrap();
    state.path = None; // remove path for easier equality check
    let buffer = fs::read("tests/assets/state/state.dat").unwrap();
    assert_eq!(state, State::from_bytes(&buffer).unwrap());
}

#[test]
fn default_state() {
    let mut state = State::load("tests/assets/state/state_default.dat").unwrap();
    state.path = None; // remove path for easier equality check
    assert_eq!(State::new(), state);
}

#[test]
fn state_5_skips_max_lev_tag() {
    let state = State::load("tests/assets/state/state_skipped_max_tag.dat").unwrap();
    assert_eq!(
        &state.players[0].skipped_internals[..8],
        &[false, false, true, true, true, true, true, false]
    );
    assert_eq!(state.players[0].last_internal, 8);
}
