#![deny(missing_docs)]

//! Library for reading and writing Elasto Mania files.

use std::{io, string};

/// Various constant values used throughout the game and library.
pub mod constants;
/// Read and write Elasto Mania level files.
pub mod lev;
/// Read and write Elasto Mania LGR files.
pub mod lgr;
/// Read and write Elasto Mania replay files.
pub mod rec;
/// Read and write Elasto Mania state.dat files.
pub mod state;
/// Various utility functions.
pub mod utils;

mod shared;
use lgr::LGRError;
pub use shared::{BestTimes, Clip, Position, Time, TimeEntry, Version};

/// General errors.
#[derive(Debug, PartialEq)]
pub enum ElmaError {
    /// Across files are not supported.
    AcrossUnsupported,
    /// Invalid level file.
    InvalidLevelFile,
    /// Invalid level file name.
    InvalidLevelFilename,
    /// Invalid replay file.
    InvalidReplayFile,
    /// Invalid state file.
    InvalidStateFile,
    /// Invalid LGR file.
    InvalidLGRFile(LGRError),
    /// Invalid gravity value.
    InvalidGravity(i32),
    /// Invalid object value.
    InvalidObject(i32),
    /// Invalid clipping value.
    InvalidClipping(i32),
    /// End-of-data marker mismatch.
    EODMismatch,
    /// End-of-file marker mismatch.
    EOFMismatch,
    /// Invalid event value.
    InvalidEvent(u8),
    /// End-of-replay marker mismatch.
    EORMismatch,
    /// Invalid time format.
    InvalidTimeFormat,
    /// Too short padding.
    PaddingTooShort(isize),
    /// String contains non-ASCII characters.
    NonASCII,
    /// Input/output errors from std::io use.
    Io(std::io::ErrorKind),
    /// String errors from std::String.
    StringFromUtf8(usize),
}

impl From<io::Error> for ElmaError {
    fn from(err: io::Error) -> ElmaError {
        ElmaError::Io(err.kind())
    }
}

impl From<string::FromUtf8Error> for ElmaError {
    fn from(err: string::FromUtf8Error) -> ElmaError {
        ElmaError::StringFromUtf8(err.utf8_error().valid_up_to())
    }
}
