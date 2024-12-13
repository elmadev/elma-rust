use super::{
    constants::{OBJECT_RADIUS, PLAYER_TOP10_SIZE, TOP10_SIZE},
    utils::{parse_top10, string_null_pad, trim_string, write_top10},
    BestTimes, Clip, ElmaError, Position, Version,
};
use byteorder::{ReadBytesExt, WriteBytesExt, LE};
use rand::random;
use std::fs;
use std::path::PathBuf;

// Magic arbitrary number signifying end-of-data in level file.
const EOD: i32 = 0x00_67_10_3A;
// Magic arbitrary number signifying end-of-file in level file.
const EOF: i32 = 0x00_84_5D_52;

/// Topology related errors.
#[derive(Debug, PartialEq)]
pub enum TopologyError {
    /// Polygon has too few or too many vertices, with list of erroneous polygons' indexes.
    InvalidVertexCount(Vec<usize>),
    /// Too many objects, with number of excess object count.
    MaxObjects(usize),
    /// Too many pictures, with number of excess picture count.
    MaxPictures(usize),
    /// Too many polygons, with number of excess polygon count.
    MaxPolygons(usize),
    /// Too many players/starts, with number of excess player count.
    InvalidPlayerCount(usize),
    /// Missing exit/flower.
    MissingExit,
    /// Level is too wide, with excess width.
    TooWide(f64),
    /// Level is too high, with excess height.
    TooHigh(f64),
}

/// This trait specifies something having a rectangle bounding box.
pub trait BoundingBox {
    /// Bounding box of `&self`, going from top-left, top-right, bottom-left to bottom-right.
    fn bounding_box(&self) -> [Position<f64>; 4];
}

/// Top10 save option.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Top10Save {
    /// Yes. Will save best times into the file's top10 list.
    Yes,
    /// No. Will give an empty top10 best times list in level file.
    No,
}

/// Type of object.
#[derive(Debug, PartialEq, Clone)]
pub enum ObjectType {
    /// Apple.
    Apple {
        /// Gravity change.
        gravity: GravityDirection,
        /// Animation number.
        animation: i32,
    },
    /// Flower/exit.
    Exit,
    /// Killer.
    Killer,
    /// Player/start.
    Player,
}

impl Default for ObjectType {
    fn default() -> ObjectType {
        ObjectType::Apple {
            gravity: GravityDirection::default(),
            animation: 1,
        }
    }
}

/// Apple direction object.
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum GravityDirection {
    /// No gravity change.
    None,
    /// Gravity up.
    Up,
    /// Gravity down.
    Down,
    /// Gravity left.
    Left,
    /// Gravity right.
    Right,
}

impl Default for GravityDirection {
    fn default() -> GravityDirection {
        GravityDirection::None
    }
}

/// Object struct. Every level requires one `ObjectType::Player` Object and at least one `ObjectType::Exit` Object.
#[derive(Debug, Default, PartialEq, Clone)]
pub struct Object {
    /// Position. See `Position` struct.
    pub position: Position<f64>,
    /// Type of Object, see `ObjectType`.
    pub object_type: ObjectType,
}

impl Object {
    /// Create a new `Object`.
    pub fn new() -> Self {
        Object::default()
    }

    /// Returns whether this object is an Apple.
    pub fn is_apple(&self) -> bool {
        match self.object_type {
            ObjectType::Apple { .. } => true,
            _ => false,
        }
    }

    /// Returns whether this object is Player.
    pub fn is_player(&self) -> bool {
        match self.object_type {
            ObjectType::Player => true,
            _ => false,
        }
    }
}

/// Polygon struct.
#[derive(Debug, Default, PartialEq, Clone)]
pub struct Polygon {
    /// Grass polygon.
    pub grass: bool,
    /// Vector with all vertices, see `Position` struct.
    pub vertices: Vec<Position<f64>>,
}

impl BoundingBox for Polygon {
    fn bounding_box(&self) -> [Position<f64>; 4] {
        let mut max_x = 0_f64;
        let mut max_y = 0_f64;
        let mut min_x = 0_f64;
        let mut min_y = 0_f64;

        for vertex in &self.vertices {
            if vertex.x > max_x {
                max_x = vertex.x
            }
            if vertex.x < min_x {
                min_x = vertex.x
            }
            if vertex.y > max_y {
                max_y = vertex.y
            }
            if vertex.y < min_y {
                min_y = vertex.y
            }
        }

        [
            Position::new(min_x, max_y),
            Position::new(max_x, max_y),
            Position::new(min_x, min_y),
            Position::new(max_x, min_y),
        ]
    }
}

impl Polygon {
    /// Create a new empty polygon.
    pub fn new() -> Self {
        Polygon {
            grass: false,
            vertices: vec![],
        }
    }
}

/// Picture struct.
#[derive(Debug, Default, PartialEq, Clone)]
pub struct Picture {
    /// Picture name.
    pub name: String,
    /// Texture name.
    pub texture: String,
    /// Mask name.
    pub mask: String,
    /// Position. See `Position` struct.
    pub position: Position<f64>,
    /// Z-distance
    pub distance: i32,
    /// Clipping.
    pub clip: Clip,
}

impl Picture {
    /// Creates a new picture with default values.
    pub fn new() -> Self {
        Picture {
            name: "barrel".into(),
            distance: 600,
            ..Default::default()
        }
    }
}

/// Level struct that contains all level information.
#[derive(Debug, PartialEq, Clone)]
pub struct Level {
    /// Elma or Across level.
    pub version: Version,
    /// Random number that links level file to replay files.
    pub link: u32,
    /// Contains four integrity checks.
    pub integrity: [f64; 4],
    /// Level title.
    pub title: String,
    /// LGR file name.
    pub lgr: String,
    /// Ground texture name.
    pub ground: String,
    /// Sky texture name.
    pub sky: String,
    /// Vector with all polygons (See `Polygon`).
    pub polygons: Vec<Polygon>,
    /// Vector with all objects (See `Object`).
    pub objects: Vec<Object>,
    /// Vector with all pictures (See `Picture`).
    pub pictures: Vec<Picture>,
    /// Best times lists.
    pub best_times: BestTimes,
}

impl Default for Level {
    fn default() -> Level {
        Level::new()
    }
}

impl BoundingBox for Level {
    fn bounding_box(&self) -> [Position<f64>; 4] {
        let mut max_x = 0_f64;
        let mut max_y = 0_f64;
        let mut min_x = 0_f64;
        let mut min_y = 0_f64;

        for polygon in &self.polygons {
            let polygon_box = polygon.bounding_box();
            for vertex in &polygon_box {
                if vertex.x > max_x {
                    max_x = vertex.x
                }
                if vertex.x < min_x {
                    min_x = vertex.x
                }
                if vertex.y > max_y {
                    max_y = vertex.y
                }
                if vertex.y < min_y {
                    min_y = vertex.y
                }
            }
        }

        [
            Position::new(min_x, max_y),
            Position::new(max_x, max_y),
            Position::new(min_x, min_y),
            Position::new(max_x, min_y),
        ]
    }
}

impl Level {
    /// Returns a new `Level` struct.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use elma::lev::*;
    /// let level = Level::new();
    /// ```
    pub fn new() -> Self {
        Level {
            version: Version::Elma,
            link: random::<u32>(),
            integrity: [0f64; 4],
            title: "".into(),
            lgr: "default".into(),
            ground: "ground".into(),
            sky: "sky".into(),
            polygons: vec![Polygon {
                grass: false,
                vertices: vec![
                    Position::new(10., 0.),
                    Position::new(10., 7.),
                    Position::new(0., 7.),
                    Position::new(0., 0.),
                ],
            }],
            objects: vec![
                Object {
                    position: Position::new(2., 0. + OBJECT_RADIUS),
                    object_type: ObjectType::Player,
                },
                Object {
                    position: Position::new(8., 0. + OBJECT_RADIUS),
                    object_type: ObjectType::Exit,
                },
            ],
            pictures: vec![],
            best_times: BestTimes::default(),
        }
    }

    /// Loads a level file and returns a `Level` struct.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use elma::lev::*;
    /// let level = Level::load("tests/assets/levels/test_1.lev").unwrap();
    /// ```
    pub fn load<P: Into<PathBuf>>(path: P) -> Result<Self, ElmaError> {
        let path = path.into();
        let buffer = fs::read(path.as_path())?;
        let lev = Level::parse_level(&buffer)?;
        Ok(lev)
    }

    /// Load a level from bytes.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use elma::lev::*;
    /// let lev = Level::from_bytes(&[0,1,2]).unwrap();
    /// ```
    pub fn from_bytes<B: AsRef<[u8]>>(buffer: B) -> Result<Self, ElmaError> {
        Level::parse_level(buffer.as_ref())
    }

    /// Parses the raw binary data into `Level` struct fields.
    fn parse_level(buffer: &[u8]) -> Result<Self, ElmaError> {
        let mut level = Level::new();
        // Version.
        let (version, remaining) = buffer.split_at(5);
        level.version = match version {
            b"POT14" => Version::Elma,
            b"POT06" => return Err(ElmaError::AcrossUnsupported),
            _ => return Err(ElmaError::InvalidLevelFile),
        };

        // Link.
        let (_, mut remaining) = remaining.split_at(2); // Never used
        level.link = remaining.read_u32::<LE>()?;

        // Integrity checksums.
        for i in 0..4 {
            level.integrity[i] = remaining.read_f64::<LE>()?;
        }

        // Level name.
        let (name, remaining) = remaining.split_at(51);
        level.title = trim_string(name)?;
        // LGR name.
        let (lgr, remaining) = remaining.split_at(16);
        level.lgr = trim_string(lgr)?;
        // Ground texture name.
        let (ground, remaining) = remaining.split_at(10);
        level.ground = trim_string(ground)?;
        // Sky texture name.
        let (sky, mut remaining) = remaining.split_at(10);
        level.sky = trim_string(sky)?;

        // Polygons.
        let poly_count = (remaining.read_f64::<LE>()? - 0.464_364_3).round() as usize;
        let (polygons, read_bytes) = Level::parse_polygons(remaining, poly_count)?;
        level.polygons = polygons;
        let (_, mut remaining) = remaining.split_at(read_bytes);

        // Objects.
        let object_count = (remaining.read_f64::<LE>()? - 0.464_364_3).round() as usize;
        let (object_data, mut remaining) = remaining.split_at(object_count * 28);
        level.objects = Level::parse_objects(object_data, object_count)?;

        // Pictures.
        let picture_count = (remaining.read_f64::<LE>()? - 0.234_567_2).round() as usize;
        let (picture_data, mut remaining) = remaining.split_at(picture_count * 54);
        level.pictures = Level::parse_pictures(picture_data, picture_count)?;

        // EOD marker expected at this point.
        let expected = remaining.read_i32::<LE>()?;
        if expected != EOD {
            return Err(ElmaError::EODMismatch);
        }

        // First decrypt the top10 blocks.
        let (top10, mut remaining) = remaining.split_at(TOP10_SIZE);
        let decrypted_top10_data = crypt_top10(top10);

        // Single-player list.
        let single = &decrypted_top10_data[0..PLAYER_TOP10_SIZE];
        level.best_times.single = parse_top10(single)?;

        // Multi-player list.
        let multi = &decrypted_top10_data[PLAYER_TOP10_SIZE..TOP10_SIZE];
        level.best_times.multi = parse_top10(multi)?;

        // EOF marker expected at this point.
        let expected = remaining.read_i32::<LE>()?;
        if expected != EOF {
            return Err(ElmaError::EOFMismatch);
        }

        Ok(level)
    }

    fn parse_polygons(mut buffer: &[u8], n: usize) -> Result<(Vec<Polygon>, usize), ElmaError> {
        let mut polygons = vec![];
        let mut read_bytes = 0;
        for _ in 0..n {
            read_bytes += 8;
            let grass = buffer.read_i32::<LE>()? > 0;
            let vertex_count = buffer.read_i32::<LE>()?;
            let mut vertices: Vec<Position<f64>> = vec![];
            for _ in 0..vertex_count {
                read_bytes += 16;
                let x = buffer.read_f64::<LE>()?;
                let y = buffer.read_f64::<LE>()?;
                vertices.push(Position::new(x, -y));
            }
            polygons.push(Polygon { grass, vertices });
        }
        Ok((polygons, read_bytes))
    }

    fn parse_objects(mut buffer: &[u8], n: usize) -> Result<Vec<Object>, ElmaError> {
        let mut objects = vec![];
        for _ in 0..n {
            let x = buffer.read_f64::<LE>()?;
            let y = buffer.read_f64::<LE>()?;
            let position = Position::new(x, -y);
            let object_type = buffer.read_i32::<LE>()?;
            let gravity = buffer.read_i32::<LE>()?;
            let gravity = match gravity {
                0 => GravityDirection::None,
                1 => GravityDirection::Up,
                2 => GravityDirection::Down,
                3 => GravityDirection::Left,
                4 => GravityDirection::Right,
                other => return Err(ElmaError::InvalidGravity(other)),
            };
            let animation = buffer.read_i32::<LE>()? + 1;
            let object_type = match object_type {
                1 => ObjectType::Exit,
                2 => ObjectType::Apple { gravity, animation },
                3 => ObjectType::Killer,
                4 => ObjectType::Player,
                other => return Err(ElmaError::InvalidObject(other)),
            };

            objects.push(Object {
                position,
                object_type,
            });
        }
        Ok(objects)
    }

    fn parse_pictures(mut buffer: &[u8], n: usize) -> Result<Vec<Picture>, ElmaError> {
        let mut pictures = vec![];
        for _ in 0..n {
            let (name, temp_remaining) = buffer.split_at(10);
            let name = trim_string(name)?;
            let (texture, temp_remaining) = temp_remaining.split_at(10);
            let texture = trim_string(texture)?;
            let (mask, temp_remaining) = temp_remaining.split_at(10);
            let mask = trim_string(mask)?;
            buffer = temp_remaining;
            let x = buffer.read_f64::<LE>()?;
            let y = buffer.read_f64::<LE>()?;
            let distance = buffer.read_i32::<LE>()?;
            let clipping = buffer.read_i32::<LE>()?;
            let clip = match clipping {
                0 => Clip::Unclipped,
                1 => Clip::Ground,
                2 => Clip::Sky,
                other => return Err(ElmaError::InvalidClipping(other)),
            };

            pictures.push(Picture {
                name,
                texture,
                mask,
                position: Position::new(x, -y),
                distance,
                clip,
            });
        }
        Ok(pictures)
    }

    /// Converts all struct fields into raw binary form and returns the raw data.
    ///
    /// # Arguments
    ///
    /// * `top10` - Specifies whether to keep the top10 list (true), or write an empty list (false).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use elma::lev::*;
    /// let mut level = Level::new();
    /// let raw_bytes = level.to_bytes(Top10Save::No).unwrap();
    /// ```
    pub fn to_bytes(&self, top_10: Top10Save) -> Result<Vec<u8>, ElmaError> {
        let mut buffer = vec![];

        // Level version.
        match self.version {
            Version::Elma => buffer.extend_from_slice(&[80, 79, 84, 49, 52]),
            Version::Across => return Err(ElmaError::AcrossUnsupported),
        };

        // Lower short of link.
        buffer.write_i16::<LE>((self.link & 0xFFFF) as i16)?;
        // Link.
        buffer.write_u32::<LE>(self.link)?;
        // Integrity checksums.
        for sum in &self.calculate_integrity_sums(true) {
            buffer.write_f64::<LE>(*sum)?;
        }

        // Level name.
        buffer.extend_from_slice(&string_null_pad(&self.title, 51)?);
        // LGR name.
        buffer.extend_from_slice(&string_null_pad(&self.lgr, 16)?);
        // Ground name.
        buffer.extend_from_slice(&string_null_pad(&self.ground, 10)?);
        // Sky name.
        buffer.extend_from_slice(&string_null_pad(&self.sky, 10)?);

        // Polygons.
        buffer.extend_from_slice(&self.write_polygons()?);
        // Objects.
        buffer.extend_from_slice(&self.write_objects()?);
        // Pictures.
        buffer.extend_from_slice(&self.write_pictures()?);

        // EOD marker.
        buffer.write_i32::<LE>(EOD)?;

        // Top10 lists.
        match top_10 {
            Top10Save::Yes => {
                // Order lists first.
                let mut best_times = self.best_times.clone();
                best_times.single.sort();
                best_times.multi.sort();
                // Encrypt the data before writing.
                let top10_bytes = write_top10(&best_times)?;
                buffer.extend_from_slice(&crypt_top10(&top10_bytes));
            }
            Top10Save::No => buffer.extend(crypt_top10(&[0; TOP10_SIZE])),
        }

        // EOF marker.
        buffer.write_i32::<LE>(EOF)?;

        Ok(buffer)
    }

    fn write_polygons(&self) -> Result<Vec<u8>, ElmaError> {
        let mut buffer = vec![];
        // Number of polygons.
        buffer.write_f64::<LE>(self.polygons.len() as f64 + 0.464_364_3_f64)?;
        for poly in &self.polygons {
            // Grass poly.
            buffer.write_i32::<LE>(if poly.grass { 1 } else { 0 })?;
            // Number of vertices.
            buffer.write_i32::<LE>(poly.vertices.len() as i32)?;
            // Vertices.
            for vertex in &poly.vertices {
                buffer.write_f64::<LE>(vertex.x)?;
                buffer.write_f64::<LE>(-vertex.y)?;
            }
        }
        Ok(buffer)
    }

    fn write_objects(&self) -> Result<Vec<u8>, ElmaError> {
        let mut buffer = vec![];
        // Number of objects.
        buffer.write_f64::<LE>(self.objects.len() as f64 + 0.464_364_3_f64)?;
        for obj in &self.objects {
            // Position.
            buffer.write_f64::<LE>(obj.position.x)?;
            buffer.write_f64::<LE>(-obj.position.y)?;
            // Object type.
            buffer.write_i32::<LE>(match obj.object_type {
                ObjectType::Exit => 1,
                ObjectType::Apple { .. } => 2,
                ObjectType::Killer => 3,
                ObjectType::Player => 4,
            })?;
            // Apple gravity.
            buffer.write_i32::<LE>(match obj.object_type {
                ObjectType::Apple {
                    gravity: GravityDirection::Up,
                    ..
                } => 1,
                ObjectType::Apple {
                    gravity: GravityDirection::Down,
                    ..
                } => 2,
                ObjectType::Apple {
                    gravity: GravityDirection::Left,
                    ..
                } => 3,
                ObjectType::Apple {
                    gravity: GravityDirection::Right,
                    ..
                } => 4,
                _ => 0,
            })?;
            // Apple animation.
            buffer.write_i32::<LE>(match obj.object_type {
                ObjectType::Apple { animation: n, .. } => (n - 1) as i32,
                _ => 0,
            })?;
        }
        Ok(buffer)
    }

    fn write_pictures(&self) -> Result<Vec<u8>, ElmaError> {
        let mut buffer = vec![];
        // Number of pictures.
        buffer.write_f64::<LE>(self.pictures.len() as f64 + 0.234_567_2_f64)?;
        for pic in &self.pictures {
            // Picture name.
            buffer.extend_from_slice(&string_null_pad(&pic.name, 10)?);
            // Texture name.
            buffer.extend_from_slice(&string_null_pad(&pic.texture, 10)?);
            // Mask name.
            buffer.extend_from_slice(&string_null_pad(&pic.mask, 10)?);
            // Position.
            buffer.write_f64::<LE>(pic.position.x)?;
            buffer.write_f64::<LE>(-pic.position.y)?;
            // Z-distance.
            buffer.write_i32::<LE>(pic.distance)?;
            // Clipping.
            buffer.write_i32::<LE>(match pic.clip {
                Clip::Unclipped => 0,
                Clip::Ground => 1,
                Clip::Sky => 2,
            })?;
        }
        Ok(buffer)
    }

    /// Width of level based on left- and right-most vertices.
    pub fn width(&self) -> f64 {
        let level_box = &self.bounding_box();
        (level_box[0].x + level_box[1].x).abs()
    }

    /// Height of level based on top and bottom-most vertices.
    pub fn height(&self) -> f64 {
        let level_box = &self.bounding_box();
        (level_box[2].y + level_box[0].y).abs()
    }

    /// Check topology of level.
    pub fn check_topology(&self) -> Result<(), TopologyError> {
        self.check_objects()?;
        if self.width() > 188_f64 {
            return Err(TopologyError::TooWide(self.width() - 188_f64));
        }
        if self.height() > 188_f64 {
            return Err(TopologyError::TooHigh(self.height() - 188_f64));
        }
        self.check_vertex_count()?;
        // TODO: check line segment overlaps
        // TODO: check if head inside ground
        // TODO: check if apples fully inside ground
        Ok(())
    }

    /// Returns a vector with the indexes of polygons containing too few vertices.
    fn check_vertex_count(&self) -> Result<(), TopologyError> {
        let mut error_polygons = vec![];
        for (n, polygon) in self.polygons.iter().enumerate() {
            if polygon.vertices.len() < 3 {
                error_polygons.push(n);
            }
        }

        if !error_polygons.is_empty() {
            return Err(TopologyError::InvalidVertexCount(error_polygons));
        }

        Ok(())
    }

    fn check_objects(&self) -> Result<(), TopologyError> {
        if self.polygons.len() > 1000 {
            return Err(TopologyError::MaxPolygons(&self.polygons.len() - 1000));
        }

        if self.objects.len() > 252 {
            return Err(TopologyError::MaxObjects(&self.objects.len() - 252));
        }

        if self.pictures.len() > 5000 {
            return Err(TopologyError::MaxPictures(&self.pictures.len() - 5000));
        }

        let player_count = self.objects.iter().fold(0, |total, object| {
            if object.object_type == ObjectType::Player {
                total + 1
            } else {
                total
            }
        });
        if player_count != 1 {
            return Err(TopologyError::InvalidPlayerCount(player_count));
        }

        let exit_count = self.objects.iter().fold(0, |total, object| {
            if object.object_type == ObjectType::Exit {
                total + 1
            } else {
                total
            }
        });
        if exit_count < 1 {
            return Err(TopologyError::MissingExit);
        }

        Ok(())
    }

    /// Calculate integrity sums for level.
    fn calculate_integrity_sums(&self, valid_topology: bool) -> [f64; 4] {
        let mut pol_sum = 0_f64;
        let mut obj_sum = 0_f64;
        let mut pic_sum = 0_f64;

        for poly in &self.polygons {
            for vertex in &poly.vertices {
                pol_sum += vertex.x + vertex.y;
            }
        }

        for obj in &self.objects {
            let obj_type = match obj.object_type {
                ObjectType::Exit => 1,
                ObjectType::Apple { .. } => 2,
                ObjectType::Killer => 3,
                ObjectType::Player => 4,
            };
            obj_sum += obj.position.x + obj.position.y + f64::from(obj_type);
        }

        for pic in &self.pictures {
            pic_sum += pic.position.x + pic.position.y;
        }

        let sum = (pol_sum + obj_sum + pic_sum) * 3_247.764_325_643;
        [
            sum,
            f64::from(random::<u32>() % 5871) + 11877. - sum,
            if valid_topology {
                f64::from(random::<u32>() % 5871) + 11877. - sum
            } else {
                f64::from(random::<u32>() % 4982) + 20961. - sum
            },
            f64::from(random::<u32>() % 6102) + 12112. - sum,
        ]
    }

    /// Generate a random link number. When you save a level, it will keep the original link
    /// number unless explicitly changed manually or by running this function before saving.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use elma::lev::*;
    /// let mut level = Level::new();
    /// level.generate_link();
    /// level.save("newlink.lev", Top10Save::No).unwrap();
    /// ```
    pub fn generate_link(&mut self) {
        self.link = random::<u32>();
    }

    /// Saves level as a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save as.
    /// * `top10` - Specifies whether to keep the top10 list (true), or write an empty list (false).
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use elma::lev::*;
    /// let mut level = Level::new();
    /// level.save("newlevel.lev", Top10Save::No).unwrap();
    /// ```
    pub fn save<P: Into<PathBuf>>(&mut self, path: P, top10: Top10Save) -> Result<(), ElmaError> {
        let bytes = self.to_bytes(top10)?;
        let path = path.into();
        fs::write(&path.as_path(), &bytes)?;
        Ok(())
    }
}

/// Decrypt and encrypt top10 list data. Same algorithm for both.
pub fn crypt_top10(top10_data: &[u8]) -> Vec<u8> {
    let mut top10: Vec<u8> = Vec::with_capacity(TOP10_SIZE);
    top10.extend_from_slice(top10_data);

    // Some variable names to match the original c/asm code?
    let mut ebp8: i16 = 0x15;
    let mut ebp10: i16 = 0x2637;

    for t in top10.iter_mut().take(TOP10_SIZE) {
        *t ^= (ebp8 & 0xFF) as u8;
        ebp10 = ebp10.wrapping_add((ebp8.wrapping_rem(0xD3D)).wrapping_mul(0xD3D));
        ebp8 = ebp10.wrapping_mul(0x1F).wrapping_add(0xD3D);
    }

    top10
}
