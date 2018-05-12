use super::{ElmaError, Position, constants::EOR, utils::{string_null_pad, trim_string}};
use byteorder::{ReadBytesExt, WriteBytesExt, LE};
use rand::random;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Bike direction.
#[derive(Debug, Eq, PartialEq)]
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
#[derive(Debug, Default, PartialEq)]
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
    /// Throttle.
    pub throttle: bool,
    /// Direction.
    pub direction: Direction,
    /// Spring sound effect volume.
    pub volume: i16,
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
}

#[derive(Debug, Default, PartialEq)]
/// Replay events.
pub struct Event {
    /// Time of event.
    pub time: f64,
    /// Event type.
    pub event_type: EventType,
}

#[derive(Debug, PartialEq)]
/// Type of event.
pub enum EventType {
    /// Apple or flower touch, with index of event.
    Touch(i16),
    /// Bike turn.
    Turn,
    /// Bike volt right.
    VoltRight,
    /// Bike volt left.
    VoltLeft,
    /// Ground touch, for sound effects. Two types.
    // TODO: consider making two separate enums instead?
    Ground {
        /// If alternative is true, uses the second type.
        alternative: bool,
    },
}

impl Default for EventType {
    fn default() -> EventType {
        EventType::Touch(0)
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

/// Replay struct
#[derive(Debug, PartialEq)]
pub struct Replay {
    /// Whether replay is multi-player or not.
    pub multi: bool,
    /// Whether replay is flag-tag or not.
    pub flag_tag: bool,
    /// Random number to link with level file.
    pub link: u32,
    /// Full level filename.
    pub level: String,
    /// Player one frames.
    pub frames: Vec<Frame>,
    /// Player one events.
    pub events: Vec<Event>,
    /// Player two frames.
    pub frames_2: Vec<Frame>,
    /// Player two events.
    pub events_2: Vec<Event>,
}

impl Default for Replay {
    fn default() -> Replay {
        Replay::new()
    }
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
            multi: false,
            flag_tag: false,
            link: random::<u32>(),
            level: String::new(),
            frames: vec![],
            events: vec![],
            frames_2: vec![],
            events_2: vec![],
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
    pub fn load<P: AsRef<Path>>(filename: P) -> Result<Self, ElmaError> {
        let mut file = File::open(filename)?;
        let mut buffer = vec![];
        file.read_to_end(&mut buffer)?;
        Replay::parse_replay(&buffer)
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
    fn parse_replay(mut buffer: &[u8]) -> Result<Self, ElmaError> {
        let mut replay = Self::new();

        // Frame count.
        let frame_count = buffer.read_i32::<LE>()?;
        // Some unused value, always 0x83.
        let (_, mut remaining) = buffer.split_at(4);
        // Multi-player replay.
        replay.multi = remaining.read_i32::<LE>()? > 0;
        // Flag-tag replay.
        replay.flag_tag = remaining.read_i32::<LE>()? > 0;
        // Level link.
        replay.link = remaining.read_u32::<LE>()?;
        // Level file name, including extension.
        let (level, remaining) = remaining.split_at(12);
        replay.level = trim_string(level)?;
        // Unknown, unused.
        let (_, remaining) = remaining.split_at(4);
        // Frames.
        replay.frames = parse_frames(remaining, frame_count)?;
        let (_, mut remaining) = remaining.split_at(27 * frame_count as usize);
        // Events.
        let event_count = remaining.read_i32::<LE>()?;
        replay.events = parse_events(remaining, event_count)?;
        let (_, mut remaining) = remaining.split_at(16 * event_count as usize);
        // End of replay marker.
        let expected = remaining.read_i32::<LE>()?;
        if expected != EOR {
            return Err(ElmaError::EORMismatch);
        }

        // If multi-rec, parse frame and events, while skipping other fields?
        if replay.multi {
            // Frame count.
            let frame_count = remaining.read_i32::<LE>()?;
            // Skip other fields.
            let (_, remaining) = remaining.split_at(32);
            // Frames.
            replay.frames_2 = parse_frames(remaining, frame_count)?;
            let (_, mut remaining) = remaining.split_at(27 * frame_count as usize);
            // Events.
            let event_count = remaining.read_i32::<LE>()?;
            replay.events_2 = parse_events(remaining, event_count)?;
            let (_, mut remaining) = remaining.split_at(16 * event_count as usize);
            // End of replay marker.
            let expected = remaining.read_i32::<LE>()?;
            if expected != EOR {
                return Err(ElmaError::EORMismatch);
            }
        }
        Ok(replay)
    }

    fn write_rec(&self, multi: bool) -> Result<Vec<u8>, ElmaError> {
        let mut bytes: Vec<u8> = vec![];

        // Number of frames.
        if multi {
            bytes.write_i32::<LE>(self.frames_2.len() as i32)?;
        } else {
            bytes.write_i32::<LE>(self.frames.len() as i32)?;
        }
        // Garbage value.
        bytes.write_i32::<LE>(0x83_i32)?;
        // Multi-player replay or not.
        bytes.write_i32::<LE>(if self.multi { 1_i32 } else { 0_i32 })?;
        // Flag-tag replay or not.
        bytes.write_i32::<LE>(if self.flag_tag { 1_i32 } else { 0_i32 })?;
        // Link.
        bytes.write_u32::<LE>(self.link)?;
        // Level name.
        bytes.extend_from_slice(&string_null_pad(&self.level, 12)?);
        // Garbage value.
        bytes.write_i32::<LE>(0x00_i32)?;

        // Frames and events.
        if multi {
            bytes.extend_from_slice(&write_frames(&self.frames_2)?);
            bytes.extend_from_slice(&write_events(&self.events_2)?);
        } else {
            bytes.extend_from_slice(&write_frames(&self.frames)?);
            bytes.extend_from_slice(&write_events(&self.events)?);
        }

        // EOR marker.
        bytes.write_i32::<LE>(EOR)?;

        Ok(bytes)
    }

    /// Save replay as a file.
    pub fn save<P: AsRef<Path>>(&self, filename: P) -> Result<(), ElmaError> {
        let mut bytes = self.write_rec(false)?;
        if self.multi {
            bytes.extend_from_slice(&self.write_rec(true)?);
        }
        let mut file = File::create(filename)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    /// Get time of replay. Returns tuple with milliseconds and whether replay was finished,
    /// caveat being that there is no way to tell if a replay was finished or not just from the
    /// replay file with a 100% certainty. Merely provided for convinience.
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
        let last_event_1 = self.events.last();
        let last_event_2 = self.events_2.last();
        let time_1 = match last_event_1 {
            Some(last_event_1) => match last_event_1.event_type {
                EventType::Touch { .. } => last_event_1.time,
                _ => 0_f64,
            },
            None => 0_f64,
        };

        let time_2 = match last_event_2 {
            Some(last_event_2) => match last_event_2.event_type {
                EventType::Touch { .. } => last_event_2.time,
                _ => 0_f64,
            },
            None => 0_f64,
        };

        // Highest frame time.
        let frames_1_len = self.frames.len();
        let frames_2_len = self.frames_2.len();
        let frame_time_max = if frames_1_len > frames_2_len {
            frames_1_len
        } else {
            frames_2_len
        } as f64 * 33.333;

        // If neither had a touch event, return approximate frame time.
        if (time_1 == 0.) && (time_2 == 0.) {
            return (frame_time_max.round() as usize, false);
        }

        // Set to highest event time.
        let event_time_max = if time_1 > time_2 { time_1 } else { time_2 } * 2289.37728938;
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

/// Function for parsing frame data from either single-player or multi-player replays.
fn parse_frames(frame_data: &[u8], frame_count: i32) -> Result<Vec<Frame>, ElmaError> {
    let mut frames: Vec<Frame> = vec![];

    let (mut bike_x, remaining) = frame_data.split_at((frame_count * 4) as usize);
    let (mut bike_y, remaining) = remaining.split_at((frame_count * 4) as usize);
    let (mut left_x, remaining) = remaining.split_at((frame_count * 2) as usize);
    let (mut left_y, remaining) = remaining.split_at((frame_count * 2) as usize);
    let (mut right_x, remaining) = remaining.split_at((frame_count * 2) as usize);
    let (mut right_y, remaining) = remaining.split_at((frame_count * 2) as usize);
    let (mut head_x, remaining) = remaining.split_at((frame_count * 2) as usize);
    let (mut head_y, remaining) = remaining.split_at((frame_count * 2) as usize);
    let (mut rotation, remaining) = remaining.split_at((frame_count * 2) as usize);
    let (mut left_rotation, remaining) = remaining.split_at((frame_count) as usize);
    let (mut right_rotation, remaining) = remaining.split_at((frame_count) as usize);
    let (mut data, remaining) = remaining.split_at((frame_count) as usize);
    let (mut volume, _) = remaining.split_at((frame_count * 2) as usize);

    for _ in 0..frame_count {
        // Bike X and Y.
        let x = bike_x.read_f32::<LE>()?;
        let y = bike_y.read_f32::<LE>()?;
        let bike = Position { x: x, y: y };
        // Left wheel X and Y.
        let x = left_x.read_i16::<LE>()?;
        let y = left_y.read_i16::<LE>()?;
        let left_wheel = Position { x: x, y: y };
        // Right wheel X and Y.
        let x = right_x.read_i16::<LE>()?;
        let y = right_y.read_i16::<LE>()?;
        let right_wheel = Position { x: x, y: y };
        // Head X and Y.
        let x = head_x.read_i16::<LE>()?;
        let y = head_y.read_i16::<LE>()?;
        let head = Position { x: x, y: y };
        // Rotations.
        let rotation = rotation.read_i16::<LE>()?;
        let left_wheel_rotation = left_rotation.read_u8()?;
        let right_wheel_rotation = right_rotation.read_u8()?;
        // Throttle and turn right.
        let data = data.read_u8()?;
        let throttle = data & 1 != 0;
        let direction = if data & (1 << 1) != 0 {
            Direction::Right
        } else {
            Direction::Left
        };
        // Sound effect volume.
        let volume = volume.read_i16::<LE>()?;

        frames.push(Frame {
            bike,
            left_wheel,
            right_wheel,
            head,
            rotation,
            left_wheel_rotation,
            right_wheel_rotation,
            throttle,
            direction,
            volume,
        });
    }

    Ok(frames)
}

/// Function for parsing event data from either single-player or multi-player replays.
fn parse_events(mut event_data: &[u8], event_count: i32) -> Result<Vec<Event>, ElmaError> {
    let mut events: Vec<Event> = vec![];

    for _ in 0..event_count {
        // Event time
        let time = event_data.read_f64::<LE>()?;
        // Event details
        let info = event_data.read_i16::<LE>()?;
        let event = event_data.read_u8()?;
        // Unknown values
        let _ = event_data.read_u8()?;
        let _ = event_data.read_f32::<LE>()?;
        let event_type = match event {
            0 => EventType::Touch(info),
            1 => EventType::Ground { alternative: false },
            4 => EventType::Ground { alternative: true },
            5 => EventType::Turn,
            6 => EventType::VoltRight,
            7 => EventType::VoltLeft,
            _ => return Err(ElmaError::InvalidEvent(event)),
        };

        events.push(Event {
            time: time,
            event_type: event_type,
        });
    }

    Ok(events)
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
    let mut volume = vec![];

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

        let mut data_temp = random::<u8>() & 0xFC;
        if frame.throttle {
            data_temp |= 1;
        }
        if frame.direction == Direction::Right {
            data_temp |= 2;
        }
        data.write_u8(data_temp)?;

        volume.write_i16::<LE>(frame.volume)?;
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
    bytes.extend_from_slice(&volume);

    Ok(bytes)
}

/// Function for writing event data.
fn write_events(event_data: &[Event]) -> Result<Vec<u8>, ElmaError> {
    let mut bytes = vec![];

    // Number of events.
    bytes.write_i32::<LE>(event_data.len() as i32)?;

    for event in event_data {
        bytes.write_f64::<LE>(event.time)?;
        match event.event_type {
            EventType::Touch(info) => {
                bytes.write_u32::<LE>(info as u32)?;
                bytes.write_u32::<LE>(0 as u32)?;
            }
            EventType::Ground { alternative: false } => {
                bytes.write_u32::<LE>(131071 as u32)?;
                bytes.write_u32::<LE>(1050605825 as u32)?;
            }
            EventType::Ground { alternative: true } => {
                bytes.write_u32::<LE>(327679 as u32)?;
                bytes.write_u32::<LE>(1065185444 as u32)?;
            }
            EventType::Turn => {
                bytes.write_u32::<LE>(393215 as u32)?;
                bytes.write_u32::<LE>(1065185444 as u32)?;
            }
            EventType::VoltRight => {
                bytes.write_u32::<LE>(458751 as u32)?;
                bytes.write_u32::<LE>(1065185444 as u32)?;
            }
            EventType::VoltLeft => {
                bytes.write_u32::<LE>(524287 as u32)?;
                bytes.write_u32::<LE>(1065185444 as u32)?;
            }
        }
    }

    Ok(bytes)
}
