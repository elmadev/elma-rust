use super::{BestTimes, ElmaError};
use crate::{
    constants::TOP10_SIZE,
    utils::{boolean, null_padded_string, string_null_pad, to_bool, write_top10},
    TimeEntry,
};
use byteorder::{WriteBytesExt, LE};
use nom::{
    bytes::complete::take,
    combinator::{map, verify},
    multi::count,
    number::complete::{le_i32, le_u32, le_u8},
    IResult,
};
use std::fs;
use std::path::PathBuf;
use std::str;

const PLAYER_STRUCT_SIZE: usize = 116;
const PLAYERENTRY_PADDING: usize = 38;
const NUM_INTERNALS: usize = 54;
const PLAYER_NAME_SIZE: usize = 15;
const PLAYERENTRY_NAME_SIZE: usize = 16;
const LEVEL_NAME_SIZE: usize = 20;
const NUM_PLAYERS: usize = 50;
const NUM_LEVELS: usize = 90;
const STATE_START: u32 = 200;
const STATE_END: u32 = 123_432_221;
const STATE_END_ALT: u32 = 123_432_112;
const TOP10_ENTRIES: usize = 10;

#[derive(Debug, Clone, PartialEq, Copy)]
/// Play mode.
pub enum PlayMode {
    /// Single player.
    Single = 1,
    /// Multi player.
    Multi = 0,
}

impl Default for PlayMode {
    fn default() -> Self {
        PlayMode::Single
    }
}

#[derive(Debug, Clone, PartialEq, Copy)]
/// Sound optimization.
pub enum SoundOptimization {
    /// Compatibility mode.
    Compatibility = 1,
    /// Best quality mode.
    BestQuality = 0,
}

impl Default for SoundOptimization {
    fn default() -> Self {
        SoundOptimization::BestQuality
    }
}

#[derive(Debug, Clone, PartialEq, Copy)]
/// Video detail.
pub enum VideoDetail {
    /// Low details.
    Low = 0,
    /// High details.
    High = 1,
}

impl Default for VideoDetail {
    fn default() -> Self {
        VideoDetail::High
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
/// Player entry in state.dat.
pub struct PlayerEntry {
    /// Player name.
    pub name: String,
    /// Skipped internals.
    pub skipped_internals: Vec<bool>,
    /// The index of last internal the player has reached so far.
    pub last_internal: i32,
    /// The last played (selected) internal.
    pub selected_internal: i32,
}

#[derive(Default, Debug, Clone, PartialEq)]
/// Key settings of a player.
pub struct PlayerKeys {
    /// Throttle key.
    pub throttle: u32,
    /// Brake key.
    pub brake: u32,
    /// Rotate right key.
    pub rotate_right: u32,
    /// Rotate left key.
    pub rotate_left: u32,
    /// Change direction key.
    pub change_direction: u32,
    /// Toggle navigator key.
    pub toggle_navigator: u32,
    /// Toggle timer key.
    pub toggle_timer: u32,
    /// Toggle show/hide key.
    pub toggle_show_hide: u32,
}

/// State.dat struct
#[derive(Default, Debug, Clone, PartialEq)]
pub struct State {
    /// Path to State file.
    pub path: Option<PathBuf>,
    /// State file version; the only supported value is 200.
    pub version: u32,
    /// Best times lists. state.dat has a fixed-size array of 90 of these.
    pub times: Vec<BestTimes>,
    /// List of players. state.dat has a fixed-size array of 50 of these.
    pub players: Vec<PlayerEntry>,
    /// Name of player A, maximum 14 characters.
    pub player_a_name: String,
    /// Name of player B, maximum 14 characters.
    pub player_b_name: String,
    /// Keys for player A.
    pub player_a_keys: PlayerKeys,
    /// Keys for player B.
    pub player_b_keys: PlayerKeys,
    /// Whether sound is enabled.
    pub sound_enabled: bool,
    /// Sound optimization.
    pub sound_optimization: SoundOptimization,
    /// Play mode.
    pub play_mode: PlayMode,
    /// Whether flag tag mode is enabled.
    pub flagtag: bool,
    /// Whether bikes are swapped.
    pub swap_bikes: bool,
    /// Video detail.
    pub video_detail: VideoDetail,
    /// Whether objects are animated.
    pub animated_objects: bool,
    /// Whether menus are animated.
    pub animated_menus: bool,
    /// Key for increasing screen size.
    pub inc_screen_size_key: u32,
    /// Key for decreasing screen size.
    pub dec_screen_size_key: u32,
    /// Key for taking a screenshot.
    pub screenshot_key: u32,
    /// Name of last edited level.
    pub last_edited_lev_name: String,
    /// Name of last played external level.
    pub last_played_external: String,
}

fn parse_playerkeys(input: &[u8]) -> IResult<&[u8], PlayerKeys> {
    let (input, throttle) = le_u32(input)?;
    let (input, brake) = le_u32(input)?;
    let (input, rotate_right) = le_u32(input)?;
    let (input, rotate_left) = le_u32(input)?;
    let (input, change_direction) = le_u32(input)?;
    let (input, toggle_navigator) = le_u32(input)?;
    let (input, toggle_timer) = le_u32(input)?;
    let (input, toggle_show_hide) = le_u32(input)?;
    Ok((
        input,
        PlayerKeys {
            throttle,
            brake,
            rotate_right,
            rotate_left,
            change_direction,
            toggle_navigator,
            toggle_timer,
            toggle_show_hide,
        },
    ))
}

fn parse_playerentry(input: &[u8]) -> IResult<&[u8], PlayerEntry> {
    let (input, name) = null_padded_string(input, PLAYERENTRY_NAME_SIZE)?;
    let (input, skipped_internals) =
        count(map(le_u8, |x| to_bool(i32::from(x))), NUM_INTERNALS)(input)?;
    let (input, _) = take(PLAYERENTRY_PADDING)(input)?;
    let (input, last_internal) = le_i32(input)?;
    let (input, selected_internal) = le_i32(input)?;

    Ok((
        input,
        PlayerEntry {
            name: name.to_string(),
            skipped_internals,
            last_internal,
            selected_internal,
        },
    ))
}

fn parse_player_names(input: &[u8], num: usize) -> IResult<&[u8], Vec<&str>> {
    let (input, names) = count(|i| null_padded_string(i, PLAYER_NAME_SIZE), num)(input)?;
    let (input, _) = count(
        |i| null_padded_string(i, PLAYER_NAME_SIZE),
        TOP10_ENTRIES - num,
    )(input)?;
    Ok((input, names))
}

fn parse_top10_part(input: &[u8]) -> IResult<&[u8], Vec<TimeEntry>> {
    let (input, num_times) = map(le_u32, |x| x as usize)(input)?;
    let (input, times) = count(map(le_i32, |x| x.into()), num_times)(input)?;
    let (input, _) = count(le_i32, TOP10_ENTRIES - num_times)(input)?;
    let (input, player_a_names) = parse_player_names(input, num_times)?;
    let (input, player_b_names) = parse_player_names(input, num_times)?;

    Ok((
        input,
        times
            .into_iter()
            .zip(player_a_names)
            .zip(player_b_names)
            .map(|((time, a), b)| TimeEntry {
                names: (a.to_string(), b.to_string()),
                time,
            })
            .collect(),
    ))
}

fn parse_top10(input: &[u8]) -> IResult<&[u8], BestTimes> {
    let (input, single) = parse_top10_part(input)?;
    let (input, multi) = parse_top10_part(input)?;

    Ok((input, BestTimes { single, multi }))
}

fn parse_state(input: &[u8]) -> IResult<&[u8], State> {
    let (input, version) = verify(le_u32, |x| *x == STATE_START)(input)?;
    let (input, times) = count(parse_top10, NUM_LEVELS)(input)?;
    let (input, mut players) = count(parse_playerentry, NUM_PLAYERS)(input)?;
    let (input, num_players) = le_u32(input)?;
    players.truncate(num_players as usize);
    let (input, player_a_name) = null_padded_string(input, PLAYER_NAME_SIZE)?;
    let (input, player_b_name) = null_padded_string(input, PLAYER_NAME_SIZE)?;
    let (input, sound_enabled) = boolean(input)?;
    let (input, sound_optimization) = map(le_i32, |i| {
        if i == 0 {
            SoundOptimization::BestQuality
        } else {
            SoundOptimization::Compatibility
        }
    })(input)?;
    let (input, play_mode) = map(le_i32, |i| {
        if i == 0 {
            PlayMode::Multi
        } else {
            PlayMode::Single
        }
    })(input)?;
    let (input, flagtag) = boolean(input)?;
    let (input, swap_bikes) = map(boolean, |x| !x)(input)?;
    let (input, video_detail) = map(le_i32, |i| {
        if i == 0 {
            VideoDetail::Low
        } else {
            VideoDetail::High
        }
    })(input)?;
    let (input, animated_objects) = boolean(input)?;
    let (input, animated_menus) = boolean(input)?;
    let (input, player_a_keys) = parse_playerkeys(input)?;
    let (input, player_b_keys) = parse_playerkeys(input)?;
    let (input, inc_screen_size_key) = le_u32(input)?;
    let (input, dec_screen_size_key) = le_u32(input)?;
    let (input, screenshot_key) = le_u32(input)?;
    let (input, last_edited_lev_name) = null_padded_string(input, LEVEL_NAME_SIZE)?;
    let (input, last_played_external) = null_padded_string(input, LEVEL_NAME_SIZE)?;
    let (input, _) = verify(le_u32, |x| *x == STATE_END || *x == STATE_END_ALT)(input)?;

    Ok((
        input,
        State {
            path: None,
            version,
            times,
            players,
            player_a_name: player_a_name.to_string(),
            player_b_name: player_b_name.to_string(),
            player_a_keys,
            player_b_keys,
            sound_enabled,
            sound_optimization,
            play_mode,
            flagtag,
            swap_bikes,
            video_detail,
            animated_objects,
            animated_menus,
            dec_screen_size_key,
            inc_screen_size_key,
            last_edited_lev_name: last_edited_lev_name.to_string(),
            last_played_external: last_played_external.to_string(),
            screenshot_key,
        },
    ))
}

impl State {
    /// Create a new state.dat.
    pub fn new() -> Self {
        State {
            path: None,
            version: 200,
            times: vec![
                BestTimes {
                    single: vec![],
                    multi: vec![],
                };
                90
            ],
            players: vec![],
            player_a_name: "".to_string(),
            player_b_name: "".to_string(),
            player_a_keys: PlayerKeys {
                throttle: 200,
                brake: 208,
                rotate_right: 205,
                rotate_left: 203,
                change_direction: 57,
                toggle_navigator: 47,
                toggle_timer: 20,
                toggle_show_hide: 2,
            },
            player_b_keys: PlayerKeys {
                throttle: 76,
                brake: 80,
                rotate_right: 81,
                rotate_left: 79,
                change_direction: 82,
                toggle_navigator: 48,
                toggle_timer: 21,
                toggle_show_hide: 3,
            },
            sound_enabled: true,
            sound_optimization: SoundOptimization::BestQuality,
            play_mode: PlayMode::Single,
            flagtag: false,
            swap_bikes: false,
            video_detail: VideoDetail::High,
            animated_objects: true,
            animated_menus: true,
            inc_screen_size_key: 13,
            dec_screen_size_key: 12,
            screenshot_key: 23,
            last_edited_lev_name: "".to_string(),
            last_played_external: "".to_string(),
        }
    }

    /// Load a state.dat file.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use elma::state::*;
    /// let state = State::load("state.dat").unwrap();
    /// ```
    pub fn load<P: Into<PathBuf>>(path: P) -> Result<Self, ElmaError> {
        let path = path.into();
        let buffer = fs::read(path.as_path())?;
        let mut state = State::parse(&buffer)?;
        state.path = Some(path);
        Ok(state)
    }

    /// Load a state.dat file from bytes.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use elma::state::*;
    /// let state = State::from_bytes(&[0,1,2]).unwrap();
    /// ```
    pub fn from_bytes<B: AsRef<[u8]>>(buffer: B) -> Result<Self, ElmaError> {
        State::parse(buffer.as_ref())
    }

    fn parse(buffer: &[u8]) -> Result<Self, ElmaError> {
        let mut v = buffer.to_vec();
        {
            let mut buf = &mut v[..];
            crypt_whole_state(&mut buf);
        }
        match parse_state(&v) {
            Ok((_, state)) => Ok(state),
            Err(_) => Err(ElmaError::InvalidStateFile),
        }
    }

    /// Returns state.dat as a stream of bytes.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use elma::state::*;
    /// let mut state = State::load("state.dat").unwrap();
    /// let buffer = state.to_bytes().unwrap();
    /// ```
    pub fn to_bytes(&self) -> Result<Vec<u8>, ElmaError> {
        let mut buffer = vec![];
        buffer.write_u32::<LE>(STATE_START)?;

        for level in &self.times {
            let top10_bytes = write_top10(&level)?;
            buffer.extend(top10_bytes);
        }
        buffer.extend(vec![0; (NUM_LEVELS - self.times.len()) * TOP10_SIZE]);

        for p in &self.players {
            buffer.extend(string_null_pad(&p.name, PLAYERENTRY_NAME_SIZE)?);
            p.skipped_internals
                .iter()
                .for_each(|&x| buffer.push(x as u8));
            buffer.extend(vec![0; NUM_INTERNALS - p.skipped_internals.len()]);
            buffer.extend_from_slice(&[0; PLAYERENTRY_PADDING]);
            buffer.write_i32::<LE>(p.last_internal)?;
            buffer.write_i32::<LE>(p.selected_internal)?;
        }
        buffer.extend(vec![
            0;
            (NUM_PLAYERS - self.players.len()) * PLAYER_STRUCT_SIZE
        ]);
        buffer.write_u32::<LE>(self.players.len() as u32)?;
        buffer.extend(string_null_pad(&self.player_a_name, PLAYER_NAME_SIZE)?);
        buffer.extend(string_null_pad(&self.player_b_name, PLAYER_NAME_SIZE)?);
        buffer.write_i32::<LE>(self.sound_enabled as i32)?;
        buffer.write_i32::<LE>(self.sound_optimization as i32)?;
        buffer.write_i32::<LE>(self.play_mode as i32)?;
        buffer.write_i32::<LE>(self.flagtag as i32)?;
        buffer.write_i32::<LE>(!self.swap_bikes as i32)?;
        buffer.write_i32::<LE>(self.video_detail as i32)?;
        buffer.write_i32::<LE>(self.animated_objects as i32)?;
        buffer.write_i32::<LE>(self.animated_menus as i32)?;
        for k in &[&self.player_a_keys, &self.player_b_keys] {
            buffer.write_u32::<LE>(k.throttle)?;
            buffer.write_u32::<LE>(k.brake)?;
            buffer.write_u32::<LE>(k.rotate_right)?;
            buffer.write_u32::<LE>(k.rotate_left)?;
            buffer.write_u32::<LE>(k.change_direction)?;
            buffer.write_u32::<LE>(k.toggle_navigator)?;
            buffer.write_u32::<LE>(k.toggle_timer)?;
            buffer.write_u32::<LE>(k.toggle_show_hide)?;
        }
        buffer.write_u32::<LE>(self.inc_screen_size_key)?;
        buffer.write_u32::<LE>(self.dec_screen_size_key)?;
        buffer.write_u32::<LE>(self.screenshot_key)?;
        buffer.extend(string_null_pad(
            &self.last_edited_lev_name,
            LEVEL_NAME_SIZE,
        )?);
        buffer.extend(string_null_pad(
            &self.last_played_external,
            LEVEL_NAME_SIZE,
        )?);
        buffer.write_u32::<LE>(STATE_END)?;
        crypt_whole_state(&mut buffer[..]);
        Ok(buffer)
    }

    /// Save state.dat
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use elma::state::*;
    /// let mut state = State::load("state.dat").unwrap();
    /// state.save("newstate.dat").unwrap();
    /// ```
    pub fn save<P: Into<PathBuf>>(&mut self, path: P) -> Result<(), ElmaError> {
        let buffer = self.to_bytes()?;
        let path = path.into();
        fs::write(path.as_path(), &buffer)?;
        self.path = Some(path);
        Ok(())
    }
}

fn crypt_whole_state(buf: &mut [u8]) {
    let state_pieces = [
        4,
        61920,
        5800,
        4,
        PLAYER_NAME_SIZE,
        PLAYER_NAME_SIZE,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        32,
        32,
        4,
        4,
        4,
        LEVEL_NAME_SIZE,
        LEVEL_NAME_SIZE,
    ];
    let mut curr = 0;
    for p in &state_pieces {
        crypt_state(&mut buf[curr..curr + p]);
        curr += p;
    }
}

fn crypt_state(buffer: &mut [u8]) {
    let mut ebp8: i16 = 0x17;
    let mut ebp10: i16 = 0x2636;

    for t in buffer.iter_mut() {
        *t ^= (ebp8 & 0xFF) as u8;
        ebp10 = ebp10.wrapping_add((ebp8.wrapping_rem(0xD3F)).wrapping_mul(0xD3F));
        ebp8 = ebp10.wrapping_mul(0x1F).wrapping_add(0xD3F);
    }
}
