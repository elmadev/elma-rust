use super::{BestTimes, ElmaError};
use byteorder::{WriteBytesExt, LE};
use constants::TOP10_SIZE;
use nom::{le_i32, le_u32, le_u8};
use shared::TimeEntry;
use std::fs;
use std::path::Path;
use std::str;
use utils::{string_null_pad, write_top10};

const PLAYER_STRUCT_SIZE: usize = 116;
const PLAYERENTRY_PADDING: usize = 38;
const NUM_INTERNALS: usize = 54;
const PLAYER_NAME_SIZE: usize = 15;
const PLAYERENTRY_NAME_SIZE: usize = 16;
const LEVEL_NAME_SIZE: usize = 20;
const NUM_PLAYERS: usize = 50;
const NUM_LEVELS: usize = 90;
const STATE_START: u32 = 200;
const STATE_END: u32 = 123432221;
const STATE_END_ALT: u32 = 123432112;
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

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(playerkeys<PlayerKeys>,
  do_parse!(
    throttle: le_u32 >>
    brake: le_u32 >>
    rotate_right: le_u32 >>
    rotate_left: le_u32 >>
    change_direction: le_u32 >>
    toggle_navigator: le_u32 >>
    toggle_timer: le_u32 >>
    toggle_show_hide: le_u32 >>
    (PlayerKeys {
      throttle,
      brake,
      rotate_right,
      rotate_left,
      change_direction,
      toggle_navigator,
      toggle_timer,
      toggle_show_hide,
    })
  )
);

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(playerentry<PlayerEntry>,
  do_parse!(
    name: apply!(null_padded_string, PLAYERENTRY_NAME_SIZE) >>
    cond_reduce!(!name.is_empty(), take!(0)) >>
    skipped_internals: many_m_n!(NUM_INTERNALS, NUM_INTERNALS, map!(le_u8, |x| to_bool(x as i32))) >>
    take!(PLAYERENTRY_PADDING) >>
    last_internal: le_i32 >>
    selected_internal: le_i32 >>
    (PlayerEntry {
      name: name.to_string(),
      skipped_internals,
      last_internal,
      selected_internal,
    })
  )
);

#[cfg_attr(rustfmt, rustfmt_skip)]
named_args!(null_padded_string(n: usize)<&str>,
  do_parse!(
    s: map_res!(take_while!(is_nonzero), str::from_utf8) >>
    cond_reduce!(n >= s.len(), take!(n - s.len())) >>
    (s)
  )
);

#[cfg_attr(rustfmt, rustfmt_skip)]
named_args!(playernames(num: usize)<Vec<&str>>,
  do_parse!(
    names: many_m_n!(num, num, apply!(null_padded_string, PLAYER_NAME_SIZE)) >>
    cond_reduce!(num <= TOP10_ENTRIES, count!(apply!(null_padded_string, PLAYER_NAME_SIZE), TOP10_ENTRIES - num)) >>
    (names)
  )
);

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(top10part<Vec<TimeEntry>>,
  do_parse!(
    num_times: map!(le_u32, |x| x as usize) >>
    times: many_m_n!(num_times, num_times, map!(le_i32, |x| x.into())) >>
    cond_reduce!(num_times <= TOP10_ENTRIES, count!(le_i32, TOP10_ENTRIES - num_times)) >>
    player_a_names: apply!(playernames, num_times) >>
    player_b_names: apply!(playernames, num_times) >>
    (izip!(times, player_a_names, player_b_names).map(|(time, a, b)| TimeEntry {names: (a.to_string(),b.to_string()), time}).collect())
  )
);

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(top10<BestTimes>,
  do_parse!(
    single: top10part >>
    multi: top10part >>
    (BestTimes {
      single,
      multi,
    })
  )
);

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(boolean<bool>,
  map!(le_i32, to_bool)
);

#[cfg_attr(rustfmt, rustfmt_skip)]
named!(parse_state<State>,
  do_parse!(
    version: verify!(le_u32, |x| x == STATE_START) >>
    times: many_m_n!(NUM_LEVELS, NUM_LEVELS, top10) >>
    // Using 0 here instead of 1 generates a warning. Workaround with map_opt + opt.
    players: map_opt!(opt!(many_m_n!(1, NUM_PLAYERS, playerentry)), |x: Option<_>| x.or_else(|| Some(vec![]))) >>
    cond_reduce!(players.len() <= NUM_PLAYERS, take!((NUM_PLAYERS - players.len()) * PLAYER_STRUCT_SIZE)) >>
    num_players: le_u32 >>
    cond_reduce!(players.len() == num_players as usize, take!(0)) >>
    player_a_name: apply!(null_padded_string, PLAYER_NAME_SIZE) >>
    player_b_name: apply!(null_padded_string, PLAYER_NAME_SIZE) >>
    sound_enabled: boolean >>
    sound_optimization: map!(le_i32, |i| if i == 0 {SoundOptimization::BestQuality} else {SoundOptimization::Compatibility}) >>
    play_mode: map!(le_i32, |i| if i == 0 {PlayMode::Multi} else {PlayMode::Single}) >>
    flagtag: boolean >>
    swap_bikes: map!(boolean, |x| !x) >>
    video_detail: map!(le_i32, |i| if i == 0 {VideoDetail::Low} else {VideoDetail::High}) >>
    animated_objects: boolean >>
    animated_menus: boolean >>
    player_a_keys: playerkeys >>
    player_b_keys: playerkeys >>
    inc_screen_size_key: le_u32 >>
    dec_screen_size_key: le_u32 >>
    screenshot_key: le_u32 >>
    last_edited_lev_name: apply!(null_padded_string, LEVEL_NAME_SIZE) >>
    last_played_external: apply!(null_padded_string, LEVEL_NAME_SIZE) >>
    verify!(le_u32, |x| x == STATE_END || x == STATE_END_ALT) >>
  (State {
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
  })
  )
);

fn to_bool(i: i32) -> bool {
    if i == 0 {
        false
    } else {
        true
    }
}

fn is_nonzero(u: u8) -> bool {
    to_bool(u as i32)
}

impl State {
    /// Create a new state.dat.
    pub fn new() -> Self {
        State {
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
    pub fn load<P: AsRef<Path>>(filename: P) -> Result<Self, ElmaError> {
        let buffer = fs::read(filename)?;
        State::parse(&buffer)
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

        for level in self.times.iter() {
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
        for k in [&self.player_a_keys, &self.player_b_keys].iter() {
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
    pub fn save<P: AsRef<Path>>(&mut self, filename: P) -> Result<(), ElmaError> {
        let buffer = self.to_bytes()?;
        fs::write(filename, &buffer)?;

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
    for p in state_pieces.iter() {
        crypt_state(&mut buf[curr..curr + p]);
        curr += p;
    }
}

fn crypt_state(buffer: &mut [u8]) {
    let mut ebp8: i16 = 0x17;
    let mut ebp10: i16 = 0x2636;

    for mut t in buffer.iter_mut() {
        *t ^= (ebp8 & 0xFF) as u8;
        ebp10 = ebp10.wrapping_add((ebp8.wrapping_rem(0xD3F)).wrapping_mul(0xD3F));
        ebp8 = ebp10.wrapping_mul(0x1F).wrapping_add(0xD3F);
    }
}

#[cfg(test)]
mod tests {
    use super::null_padded_string;
    use nom::simple_errors::Context::Code;
    use nom::Err::Error;
    use nom::Err::Incomplete;
    use nom::ErrorKind::CondReduce;
    use nom::Needed::Size;
    #[test]
    fn null_pad_string() {
        assert_eq!(
            null_padded_string(b"Elma\0\0\0\0\0\0", 10),
            Ok((&[][..], "Elma"))
        );
        assert_eq!(
            null_padded_string(b"Elma\0\0\0\0\0\0\0\0", 10),
            Ok((&[0, 0][..], "Elma"))
        );
        assert_eq!(
            null_padded_string(b"\0\0\0\0\0\0\0\0\0\0", 10),
            Ok((&[][..], ""))
        );
        assert_eq!(
            null_padded_string(b"Elma\0\0\0\0\0", 10),
            Err(Incomplete(Size(6)))
        );
        assert_eq!(
            null_padded_string(b"\0\0\0\0\0\0\0\0\0", 10),
            Err(Incomplete(Size(10)))
        );
        assert_eq!(
            null_padded_string(b"ElastoMani", 10),
            Err(Incomplete(Size(1)))
        );
        assert_eq!(
            null_padded_string(b"ElastoMania", 10),
            Err(Incomplete(Size(1)))
        );
        assert_eq!(
            null_padded_string(b"ElastoMania\0", 10),
            Err(Error(Code(&[0][..], CondReduce)))
        );
    }
}
