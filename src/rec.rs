use super::{utils::string_null_pad, ElmaError, Position};
use crate::utils::{boolean, null_padded_string};
use byteorder::{WriteBytesExt, LE};
use itertools::izip;
use nom::{
    bytes::complete::take,
    combinator::{map, map_opt, verify},
    multi::{count, many_m_n},
    number::complete::{le_f32, le_f64, le_i16, le_i32, le_u32, le_u8},
    IResult,
};
use std::fs;
use std::path::PathBuf;

// Magic arbitrary number to signify end of player data in a replay file.
const END_OF_PLAYER: i32 = 0x00_49_2F_75;
// Replay version number that all valid replays need to have.
const REPLAY_VERSION: u32 = 0x83;

/// Bike direction.
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum Direction {
    /// Right.
    Right,
    /// Left.
    Left,
}

impl Default for Direction {
    fn default() -> Self {
        Direction::Left
    }
}

/// One frame of replay.
#[derive(Debug, Default, PartialEq, Clone)]
pub struct Frame {
    /// Bike position.
    pub bike: Position<f32>,
    /// Left wheel position.
    pub left_wheel: Position<i16>,
    /// Right wheel position.
    pub right_wheel: Position<i16>,
    /// Head position.
    pub head: Position<i16>,
    /// Bike rotation. Range 0..10000.
    pub rotation: i16,
    /// Left wheel rotation. Range 0..255.
    pub left_wheel_rotation: u8,
    /// Right wheel rotation. Range 0..255.
    pub right_wheel_rotation: u8,
    /// State of throttle and direction.
    pub throttle_and_dir: u8,
    /// Rotation speed of back wheel.
    pub back_wheel_rot_speed: u8,
    /// Collision strength.
    pub collision_strength: u8,
}

impl Frame {
    /// Returns a new Frame struct with default values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use elma::rec::*;
    /// let frame = Frame::new();
    /// ```
    pub fn new() -> Self {
        Frame::default()
    }

    /// Returns whether throttle is on.
    pub fn throttle(&self) -> bool {
        self.throttle_and_dir & 1 != 0
    }

    /// Returns the current direction.
    pub fn direction(&self) -> Direction {
        if self.throttle_and_dir & (1 << 1) != 0 {
            Direction::Right
        } else {
            Direction::Left
        }
    }
}

#[derive(Debug, Default, PartialEq, Clone)]
/// Replay events.
pub struct Event {
    /// Time of event.
    pub time: f64,
    /// Event type.
    pub event_type: EventType,
}

#[derive(Debug, PartialEq, Clone)]
/// Type of event.
pub enum EventType {
    /// Object touch, with index of the object. The index corresponds to a sorted object array having the order: killers, apples, flowers, start.
    ObjectTouch(i16),
    /// Apple take. An apple take in replay always generates 2 events (an ObjectTouch and an AppleTake).
    Apple,
    /// Bike turn.
    Turn,
    /// Bike volt right.
    VoltRight,
    /// Bike volt left.
    VoltLeft,
    /// Ground touch. The float is in range [0, 0.99] and possibly denotes the strength of the touch.
    Ground(f32),
}

impl<'a> From<&'a EventType> for u8 {
    fn from(event_type: &'a EventType) -> Self {
        match *event_type {
            EventType::Apple => 4,
            EventType::Ground(_) => 1,
            EventType::ObjectTouch(_) => 0,
            EventType::Turn => 5,
            EventType::VoltLeft => 7,
            EventType::VoltRight => 6,
        }
    }
}

impl Default for EventType {
    fn default() -> EventType {
        EventType::ObjectTouch(0)
    }
}

impl Event {
    /// Returns a new Event struct with default values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use elma::rec::*;
    /// let event = Event::new();
    /// ```
    pub fn new() -> Self {
        Event::default()
    }
}

#[derive(Debug, PartialEq)]
pub(crate) struct ReplayHeader {
    pub multi: bool,
    pub flag_tag: bool,
    pub link: u32,
    pub level: String,
}

/// Player ride information (frames and events).
#[derive(Debug, PartialEq, Default, Clone)]
pub struct Ride {
    /// Player frames.
    pub frames: Vec<Frame>,
    /// Player events.
    pub events: Vec<Event>,
}

impl Ride {
    /// Creates an empty Ride.
    pub fn new() -> Self {
        Ride {
            frames: vec![],
            events: vec![],
        }
    }

    /// Gets the time based on frame count.
    pub fn get_frame_time(&self) -> f64 {
        self.frames.len() as f64 * 33.333
    }

    /// Gets the time based on last ObjectTouch event or 0 if the last event is not ObjectTouch.
    pub fn get_time(&self) -> f64 {
        let last_event = self.events.last();
        let time = match last_event {
            Some(e) => match e.event_type {
                EventType::ObjectTouch { .. } => e.time,
                _ => 0_f64,
            },
            None => 0_f64,
        };
        time * 2_289.377_289_38
    }
}

/// Replay struct
#[derive(Debug, PartialEq, Clone)]
pub struct Replay {
    /// Whether replay is flag-tag or not.
    pub flag_tag: bool,
    /// Random number to link with level file.
    pub link: u32,
    /// Full level filename.
    pub level: String,
    /// Rides of players.
    pub rides: Vec<Ride>,
}

impl Replay {
    /// Returns whether this is a multiplayer replay.
    pub fn is_multi(&self) -> bool {
        self.rides.len() > 1
    }
}

impl Default for Replay {
    fn default() -> Replay {
        Replay::new()
    }
}

fn parse_header_and_ride(input: &[u8]) -> IResult<&[u8], (ReplayHeader, Ride)> {
    let (input, frame_count) = map(le_i32, |x| x as usize)(input)?;
    let (input, _) = verify(le_u32, |x| *x == REPLAY_VERSION)(input)?;
    let (input, multi) = boolean(input)?;
    let (input, flag_tag) = boolean(input)?;
    let (input, link) = le_u32(input)?;
    let (input, level) = null_padded_string(input, 16)?;

    let (input, bodyx) = count(le_f32, frame_count)(input)?;
    let (input, bodyy) = count(le_f32, frame_count)(input)?;
    let (input, leftwheelx) = count(le_i16, frame_count)(input)?;
    let (input, leftwheely) = count(le_i16, frame_count)(input)?;
    let (input, rightwheelx) = count(le_i16, frame_count)(input)?;
    let (input, rightwheely) = count(le_i16, frame_count)(input)?;
    let (input, headx) = count(le_i16, frame_count)(input)?;
    let (input, heady) = count(le_i16, frame_count)(input)?;
    let (input, rotation) = count(le_i16, frame_count)(input)?;
    let (input, leftwheelrotation) = count(le_u8, frame_count)(input)?;
    let (input, rightwheelrotation) = count(le_u8, frame_count)(input)?;
    let (input, dir_and_throttle) = count(le_u8, frame_count)(input)?;
    let (input, back_wheel) = count(le_u8, frame_count)(input)?;
    let (input, collision_strength) = count(le_u8, frame_count)(input)?;

    let (input, num_events) = map(le_i32, |x| x as usize)(input)?;
    let (input, events) = count(parse_event, num_events)(input)?;

    let (input, _) = verify(le_i32, |x| *x == END_OF_PLAYER)(input)?;

    let header = ReplayHeader {
        multi,
        flag_tag,
        link,
        level: level.to_string(),
    };

    let frames = izip!(
        bodyx,
        bodyy,
        leftwheelx,
        leftwheely,
        rightwheelx,
        rightwheely,
        headx,
        heady,
        rotation,
        leftwheelrotation,
        rightwheelrotation,
        dir_and_throttle,
        back_wheel,
        collision_strength
    )
    .map(
        |(bx, by, lx, ly, rx, ry, hx, hy, r, lr, rr, dt, bw, cs)| Frame {
            bike: Position::new(bx, by),
            left_wheel: Position::new(lx, ly),
            right_wheel: Position::new(rx, ry),
            head: Position::new(hx, hy),
            rotation: r,
            left_wheel_rotation: lr,
            right_wheel_rotation: rr,
            throttle_and_dir: dt,
            back_wheel_rot_speed: bw,
            collision_strength: cs,
        },
    )
    .collect();

    let ride = Ride { frames, events };

    Ok((input, (header, ride)))
}

fn parse_event(input: &[u8]) -> IResult<&[u8], Event> {
    let (input, time) = le_f64(input)?;
    let (input, info) = le_i16(input)?;
    let (input, mut event_type) = map_opt(le_u8, |event_type| match event_type {
        0 => Some(EventType::ObjectTouch(info)),
        1 => Some(EventType::Ground(0.0)), // will be filled below
        4 => Some(EventType::Apple),
        5 => Some(EventType::Turn),
        6 => Some(EventType::VoltRight),
        7 => Some(EventType::VoltLeft),
        _ => None,
    })(input)?;
    let (input, _) = take(1u8)(input)?;
    let (input, info2) = le_f32(input)?;
    if let EventType::Ground(ref mut val) = event_type { *val = info2 }

    Ok((input, Event { time, event_type }))
}

fn parse_replay(input: &[u8]) -> IResult<&[u8], Replay> {
    let (input, players) = many_m_n(1, 2, parse_header_and_ride)(input)?;

    let replay = Replay {
        flag_tag: players[0].0.flag_tag,
        link: players[0].0.link,
        level: players[0].0.level.clone(),
        rides: players.into_iter().map(|(_, ride)| ride).collect(),
    };

    Ok((input, replay))
}

impl Replay {
    /// Return a new Replay struct.
    ///
    /// # Examples
    ///
    /// ```
    /// let rec = elma::rec::Replay::new();
    /// ```
    pub fn new() -> Self {
        Replay {
            flag_tag: false,
            link: 0,
            level: String::new(),
            rides: vec![],
        }
    }

    /// Loads a replay file and returns a Replay struct.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use elma::rec::*;
    /// let rec = Replay::load("tests/assets/replays/test_1.rec").unwrap();
    /// ```
    pub fn load<P: Into<PathBuf>>(path: P) -> Result<Self, ElmaError> {
        let path = path.into();
        let buffer = fs::read(path.as_path())?;
        let rec = Replay::parse_replay(&buffer)?;
        Ok(rec)
    }

    /// Load a replay from bytes.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use elma::rec::*;
    /// let rec = Replay::from_bytes(&[0,1,2]).unwrap();
    /// ```
    pub fn from_bytes<B: AsRef<[u8]>>(buffer: B) -> Result<Self, ElmaError> {
        Replay::parse_replay(buffer.as_ref())
    }

    /// Parses the raw binary data into Replay struct fields.
    fn parse_replay(buffer: &[u8]) -> Result<Self, ElmaError> {
        match parse_replay(buffer) {
            Ok((_, replay)) => Ok(replay),
            Err(_) => Err(ElmaError::InvalidReplayFile),
        }
    }

    /// Returns replay data as a buffer of bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ElmaError> {
        let mut bytes: Vec<u8> = vec![];
        for r in &self.rides {
            // Number of frames.
            bytes.write_i32::<LE>(r.frames.len() as i32)?;
            // Replay version.
            bytes.write_u32::<LE>(REPLAY_VERSION)?;
            // Multi-player replay or not.
            bytes.write_i32::<LE>(if self.is_multi() { 1_i32 } else { 0_i32 })?;
            // Flag-tag replay or not.
            bytes.write_i32::<LE>(if self.flag_tag { 1_i32 } else { 0_i32 })?;
            // Link.
            bytes.write_u32::<LE>(self.link)?;
            // Level name.
            bytes.extend_from_slice(&string_null_pad(&self.level, 12)?);
            // Garbage value.
            bytes.write_i32::<LE>(0x00_i32)?;

            // Frames and events.
            bytes.extend_from_slice(&write_frames(&r.frames)?);
            bytes.extend_from_slice(&write_events(&r.events)?);

            // End of player marker.
            bytes.write_i32::<LE>(END_OF_PLAYER)?;
        }
        Ok(bytes)
    }

    /// Save replay as a file.
    pub fn save<P: Into<PathBuf>>(&mut self, path: P) -> Result<(), ElmaError> {
        let path = path.into();
        fs::write(path.as_path(), &self.to_bytes()?)?;
        Ok(())
    }

    /// Get time of replay. Returns tuple with milliseconds and whether replay was finished,
    /// caveat being that there is no way to tell if a replay was finished or not just from the
    /// replay file with a 100% certainty. Merely provided for convenience.
    /// # Examples
    ///
    /// ```rust
    /// # use elma::rec::*;
    /// let replay = Replay::load("tests/assets/replays/test_1.rec").unwrap();
    /// let (time, finished) = replay.get_time_ms();
    /// assert_eq!(time, 14649);
    /// assert_eq!(finished, true);
    /// ```
    pub fn get_time_ms(&self) -> (usize, bool) {
        // First check if last event was a touch event in either event data.
        let times = self
            .rides
            .iter()
            .map(|r| (r.get_time(), r.get_frame_time()))
            .collect::<Vec<_>>();
        let (event_time_max, frame_time_max) =
            times.iter().fold((0_f64, 0_f64), |(acc_a, acc_b), (a, b)| {
                (a.max(acc_a), b.max(acc_b))
            });

        // If neither had a touch event, return approximate frame time.
        if event_time_max == 0. {
            return (frame_time_max.round() as usize, false);
        }

        // If event difference to frame time is >1 frames of time, probably not finished?
        if frame_time_max > (event_time_max + 33.333) {
            return (frame_time_max.round() as usize, false);
        }

        (event_time_max.round() as usize, true)
    }

    /// Get time of replay. Returns tuple with hundredths and whether replay was finished,
    /// caveat being that there is no way to tell if a replay was finished or not just from the
    /// replay file with a 100% certainty. Merely provided for convinience.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use elma::rec::*;
    /// let replay = Replay::load("tests/assets/replays/test_1.rec").unwrap();
    /// let (time, finished) = replay.get_time_hs();
    /// assert_eq!(time, 1464);
    /// assert_eq!(finished, true);
    /// ```
    pub fn get_time_hs(&self) -> (usize, bool) {
        let (time, finished) = self.get_time_ms();
        (time / 10, finished)
    }
}

/// Function for writing frame data.
fn write_frames(frame_data: &[Frame]) -> Result<Vec<u8>, ElmaError> {
    let mut bytes = vec![];

    let mut bike_x = vec![];
    let mut bike_y = vec![];
    let mut left_x = vec![];
    let mut left_y = vec![];
    let mut right_x = vec![];
    let mut right_y = vec![];
    let mut head_x = vec![];
    let mut head_y = vec![];
    let mut rotation = vec![];
    let mut left_rotation = vec![];
    let mut right_rotation = vec![];
    let mut data = vec![];
    let mut back_wheel = vec![];
    let mut collision = vec![];

    for frame in frame_data {
        bike_x.write_f32::<LE>(frame.bike.x)?;
        bike_y.write_f32::<LE>(frame.bike.y)?;

        left_x.write_i16::<LE>(frame.left_wheel.x)?;
        left_y.write_i16::<LE>(frame.left_wheel.y)?;

        right_x.write_i16::<LE>(frame.right_wheel.x)?;
        right_y.write_i16::<LE>(frame.right_wheel.y)?;

        head_x.write_i16::<LE>(frame.head.x)?;
        head_y.write_i16::<LE>(frame.head.y)?;

        rotation.write_i16::<LE>(frame.rotation)?;
        left_rotation.write_u8(frame.left_wheel_rotation)?;
        right_rotation.write_u8(frame.right_wheel_rotation)?;

        data.write_u8(frame.throttle_and_dir)?;

        back_wheel.write_u8(frame.back_wheel_rot_speed)?;
        collision.write_u8(frame.collision_strength)?;
    }

    bytes.extend_from_slice(&bike_x);
    bytes.extend_from_slice(&bike_y);
    bytes.extend_from_slice(&left_x);
    bytes.extend_from_slice(&left_y);
    bytes.extend_from_slice(&right_x);
    bytes.extend_from_slice(&right_y);
    bytes.extend_from_slice(&head_x);
    bytes.extend_from_slice(&head_y);
    bytes.extend_from_slice(&rotation);
    bytes.extend_from_slice(&left_rotation);
    bytes.extend_from_slice(&right_rotation);
    bytes.extend_from_slice(&data);
    bytes.extend_from_slice(&back_wheel);
    bytes.extend_from_slice(&collision);

    Ok(bytes)
}

/// Function for writing event data.
fn write_events(event_data: &[Event]) -> Result<Vec<u8>, ElmaError> {
    let mut bytes = vec![];

    // Number of events.
    bytes.write_i32::<LE>(event_data.len() as i32)?;

    for event in event_data {
        bytes.write_f64::<LE>(event.time)?;
        let default_info = -1;
        let default_info2 = 0.99;
        let event_type = (&event.event_type).into();
        match event.event_type {
            EventType::ObjectTouch(info) => {
                bytes.write_i16::<LE>(info)?;
                bytes.write_u8(event_type)?;
                bytes.write_u8(0)?;
                bytes.write_f32::<LE>(0.0)?; // always 0 for an ObjectTouch event
            }
            EventType::Ground(info2) => {
                bytes.write_i16::<LE>(default_info)?;
                bytes.write_u8(event_type)?;
                bytes.write_u8(0)?;
                bytes.write_f32::<LE>(info2)?;
            }
            _ => {
                bytes.write_i16::<LE>(default_info)?;
                bytes.write_u8(event_type)?;
                bytes.write_u8(0)?;
                bytes.write_f32::<LE>(default_info2)?;
            }
        }
    }

    Ok(bytes)
}
