use byteorder::{ByteOrder, WriteBytesExt, LE};
use nom::bytes::complete::take_while_m_n;
use nom::character::complete::char;
use nom::combinator::map_res;
use nom::IResult;
use nom::{bytes::complete::take, combinator::map, number::complete::le_i32};
use std::str;

use super::{BestTimes, ElmaError, Time, TimeEntry};

/// Parse top10 lists and return a vector of `TimeEntry`s
pub fn parse_top10(top10: &[u8]) -> Result<Vec<TimeEntry>, ElmaError> {
    let mut list: Vec<TimeEntry> = vec![];
    let times = LE::read_i32(&top10[0..4]);
    for n in 0..times as usize {
        let time_offset = 4 + n * 4;
        let time_end = time_offset + 4;
        let name_1_offset = 44 + n * 15;
        let name_1_end = name_1_offset + 15;
        let name_2_offset = 194 + n * 15;
        let name_2_end = name_2_offset + 15;

        let name_1 = &top10[name_1_offset..name_1_end];
        let name_2 = &top10[name_2_offset..name_2_end];
        let time = &top10[time_offset..time_end];
        list.push(TimeEntry {
            time: Time(LE::read_i32(time)),
            names: (trim_string(name_1)?, trim_string(name_2)?),
        });
    }
    Ok(list)
}

/// Write `best times` data as bytes.
pub fn write_top10(best_times: &BestTimes) -> Result<Vec<u8>, ElmaError> {
    let mut top10_bytes: Vec<u8> = vec![];

    // Single-player times.
    let single_times = best_times.single.len();
    top10_bytes.write_i32::<LE>(if 10 < single_times { 10 } else { single_times } as i32)?;
    let mut times = [0_i32; 10];
    let mut names_1 = vec![];
    let mut names_2 = vec![];
    for (n, entry) in best_times.single.iter().enumerate() {
        if n < 10 {
            times[n] = entry.time.into();
            names_1.extend_from_slice(&string_null_pad(&entry.names.0, 15)?);
            names_2.extend_from_slice(&string_null_pad(&entry.names.1, 15)?);
        }
    }
    // Pad with null bytes if less than 10 entries.
    if single_times < 10 {
        for _ in 0..10 - single_times {
            names_1.extend_from_slice(&[0u8; 15]);
            names_2.extend_from_slice(&[0u8; 15]);
        }
    }

    for time in &times {
        top10_bytes.write_i32::<LE>(*time)?;
    }

    top10_bytes.extend_from_slice(&names_1);
    top10_bytes.extend_from_slice(&names_2);

    // Multi-player times.
    let multi_times = best_times.multi.len();
    top10_bytes.write_i32::<LE>(if 10 < multi_times { 10 } else { multi_times } as i32)?;
    let mut times = [0_i32; 10];
    let mut names_1 = vec![];
    let mut names_2 = vec![];
    for (n, entry) in best_times.multi.iter().enumerate() {
        if n < 10 {
            times[n] = entry.time.into();
            names_1.extend_from_slice(&string_null_pad(&entry.names.0, 15)?);
            names_2.extend_from_slice(&string_null_pad(&entry.names.1, 15)?);
        }
    }
    // Pad with null bytes if less than 10 entries.
    if multi_times < 10 {
        for _ in 0..10 - multi_times {
            names_1.extend_from_slice(&[0u8; 15]);
            names_2.extend_from_slice(&[0u8; 15]);
        }
    }

    for time in &times {
        top10_bytes.write_i32::<LE>(*time)?;
    }

    top10_bytes.extend_from_slice(&names_1);
    top10_bytes.extend_from_slice(&names_2);

    Ok(top10_bytes)
}

/// Trims trailing bytes after and including null byte.
///
/// # Examples
/// As all strings in Elma files are C-strings with padded null-bytes, you can use this function
/// to remove null-bytes and any potential garbage data follwing it and return a String.
///
/// ```
/// let cstring: [u8; 10] = [0x45, 0x6C, 0x6D, 0x61, 0x00, 0x00, 0x00, 0x7E, 0x7E, 0x7E];
/// let trimmed = elma::utils::trim_string(&cstring).unwrap();
/// assert_eq!(trimmed, "Elma");
/// ```
pub fn trim_string(data: &[u8]) -> Result<String, ElmaError> {
    let bytes: Vec<u8> = data.into_iter().take_while(|&&d| d != 0).cloned().collect();

    let trimmed = String::from_utf8(bytes)?;
    Ok(trimmed)
}

/// Pads a string with null bytes.
///
/// # Examples
/// When converting strings to bytes for use in an Elma file, you need to pad it to a certain
/// length depending on the field. This function creates a new zero-filled vector to `pad` size,
/// then fills in the string.
///
/// ```
/// let string = String::from("Elma");
/// let padded = elma::utils::string_null_pad(&string, 10).unwrap();
/// assert_eq!(&padded, &[0x45, 0x6C, 0x6D, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
/// ```
pub fn string_null_pad(name: &str, pad: usize) -> Result<Vec<u8>, ElmaError> {
    let name = name.as_bytes();

    // first check if string is ASCII
    if !name.is_ascii() {
        return Err(ElmaError::NonASCII);
    }
    // padding shorter than string
    if name.len() > pad {
        return Err(ElmaError::PaddingTooShort(
            (pad as isize - name.len() as isize) as isize,
        ));
    }

    let mut bytes = name.to_vec();
    bytes.resize(pad, 0);
    Ok(bytes)
}

pub(crate) fn null_padded_string(input: &[u8], n: usize) -> IResult<&[u8], &str> {
    let (input, s) = map_res(take_while_m_n(0, n - 1, |u| u != 0), str::from_utf8)(input)?;
    let (input, _) = char('\0')(input)?;
    let remaining_len = n - s.len() - 1;
    let (input, _) = take(remaining_len)(input)?;
    Ok((input, s))
}

pub(crate) fn boolean(input: &[u8]) -> IResult<&[u8], bool> {
    map(le_i32, to_bool)(input)
}

pub(crate) fn to_bool(i: i32) -> bool {
    i != 0
}

#[cfg(test)]
mod tests {
    use super::null_padded_string;
    use nom::error::{Error, ErrorKind};
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
            Err(nom::Err::Error(Error {
                input: [0, 0, 0, 0].as_slice(),
                code: ErrorKind::Eof
            }))
        );
        assert_eq!(
            null_padded_string(b"\0\0\0\0\0\0\0\0\0", 10),
            Err(nom::Err::Error(Error {
                input: [0, 0, 0, 0, 0, 0, 0, 0].as_slice(),
                code: ErrorKind::Eof
            }))
        );
        assert_eq!(
            null_padded_string(b"ElastoMani", 10),
            Err(nom::Err::Error(Error {
                input: [105].as_slice(),
                code: ErrorKind::Char
            }))
        );
        assert_eq!(
            null_padded_string(b"ElastoMania", 10),
            Err(nom::Err::Error(Error {
                input: [105, 97].as_slice(),
                code: ErrorKind::Char
            }))
        );
        assert_eq!(
            null_padded_string(b"ElastoMania\0", 10),
            Err(nom::Err::Error(Error {
                input: [105, 97, 0].as_slice(),
                code: ErrorKind::Char
            }))
        );
    }
}
