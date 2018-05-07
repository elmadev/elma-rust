use std::cmp::Ordering;
use std::fmt;
use std::i32;
use std::ops::{Add, Sub};

/// Best times struct.
#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub struct BestTimes {
    /// Single-player times.
    pub single: Vec<TimeEntry>,
    /// Multi-player times.
    pub multi: Vec<TimeEntry>,
}

impl BestTimes {
    /// Create a new best times struct.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Wrapper for time in hundredths.
/// Supports add/sub ops.
///
/// # Examples
///
/// ```rust
/// # use elma::Time;
/// let time_x = Time(100); // 1 second
/// let time_y = Time::from("00:00,01"); // 1 hundredth
///
/// assert_eq!(Time::from("0..,0:099"), time_x - time_y); // from string impl allows somewhat malformed input
/// assert_eq!("01:20,00", Time(8000).to_string()); // .to_string() pretty prints in 00:00,00 format
/// ```
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct Time(pub i32);

impl Time {
    /// Returns a tuple with `hours`, `mins`, `secs`, `hundredths`.
    pub fn to_parts(&self) -> (i32, i32, i32, i32) {
        let h = self.0 % 100;
        let s = (self.0 / 100) % 60;
        let m = (self.0 / (100 * 60)) % 60;
        let hr = self.0 / (100 * 60 * 60);

        (hr, m, s, h)
    }
}

impl From<i32> for Time {
    fn from(i: i32) -> Self {
        Time(i)
    }
}

impl<'a> From<&'a str> for Time {
    fn from(s: &'a str) -> Self {
        let parts: Vec<_> = s.split(|c: char| !c.is_numeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<i32>().unwrap())
            .inspect(|v| println!("{:?}", v))
            .collect();
        let mut time = 0;
        for (n, val) in parts.iter().rev().enumerate() {
            match n {
                n if n == 0 => time += val,
                n if n == 1 => time += val * 100,
                n if n == 2 => time += val * 6000,
                n if n == 3 => time += val * 360000,
                n if n == 4 => time += val * 8640000,
                _ => time = time.saturating_add(i32::MAX),
            }
        }
        if s.starts_with('-') {
            time *= -1
        }
        Time(time)
    }
}

impl Add for Time {
    type Output = Time;

    fn add(self, other: Self) -> Self {
        Time(self.0 + other.0)
    }
}

impl Sub for Time {
    type Output = Time;

    fn sub(self, other: Self) -> Self {
        Time(self.0 - other.0)
    }
}

impl From<Time> for i32 {
    fn from(t: Time) -> Self {
        t.0
    }
}

impl fmt::Display for Time {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let h = self.0 % 100;
        let s = (self.0 / 100) % 60;
        let m = (self.0 / (100 * 60)) % 60;
        let hr = self.0 / (100 * 60 * 60);

        write!(
            f,
            "{}{}{}{:02},{:02}",
            if self.0 < 0 { "-" } else { "" },
            if hr > 0 {
                format!("{:02}:", hr)
            } else {
                "".into()
            },
            if m > 0 {
                format!("{:02}:", m)
            } else {
                "".into()
            },
            s,
            h
        )
    }
}

/// Shared position struct used in both sub-modules.
///
/// # Examples
/// ```
/// let vertex = elma::Position { x: 23.1928_f64, y: -199.200019_f64 };
/// ```
#[derive(Debug, Default, PartialEq)]
pub struct Position<T> {
    /// X-position.
    pub x: T,
    /// Y-position.
    pub y: T,
}

/// Top10 list entry struct.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TimeEntry {
    /// Player names.
    pub names: (String, String),
    /// Time.
    pub time: Time,
}

impl Ord for TimeEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.time.cmp(&other.time)
    }
}

impl PartialOrd for TimeEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl TimeEntry {
    /// Creates a new TimeEntry.
    pub fn new<T: Into<Time>, S: Into<String>>(names: (S, S), time: T) -> Self {
        TimeEntry {
            names: (names.0.into(), names.1.into()),
            time: time.into(),
        }
    }
}