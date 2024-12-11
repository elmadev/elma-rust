extern crate elma;
extern crate rand;

use elma::constants::TOP10_SIZE;
use elma::lev::*;
use elma::{BestTimes, Clip, ElmaError, Position, TimeEntry, Version};
use rand::random;
use std::env;
use std::fs;

#[test]
/// Generate random u8 data to simulate top10 lists, encrypting it and decrypting it,
/// and testing whether it returns the same unencrypted data.
fn decrypt_encrypt_top10() {
    let mut initial: Vec<u8> = vec![];
    for _ in 0..TOP10_SIZE {
        initial.push(random::<u8>());
    }
    let decrypted = crypt_top10(&initial);
    let encrypted = crypt_top10(&decrypted);
    assert_eq!(initial, encrypted);
}

#[test]
/// Generate a level with some arbitrary values and see if it saves.
fn construct_level_and_save() {
    let mut level = Level {
        version: Version::default(),
        link: random::<u32>(),
        integrity: [0f64; 4],
        title: String::new(),
        lgr: String::from("default"),
        ground: String::from("ground"),
        sky: String::from("sky"),
        polygons: vec![],
        objects: vec![],
        pictures: vec![],
        best_times: BestTimes::default(),
    };

    level.polygons.push(Polygon::new());
    level.objects.push(Object::new());
    level.objects.push(Object {
        position: Position::new(0_f64, 0_f64),
        object_type: ObjectType::Apple {
            gravity: GravityDirection::Down,
            animation: 1,
        },
    });
    level.pictures.push(Picture::new());
    level.pictures.push(Picture::new());
    level.pictures.push(Picture::new());
    level.pictures[1].clip = Clip::Unclipped;
    level.pictures[2].clip = Clip::Ground;
    level
        .best_times
        .single
        .push(TimeEntry::new(("", ""), 100000));
    level
        .best_times
        .multi
        .push(TimeEntry::new(("", ""), 100000));
    level.generate_link();
    let _ = level.to_bytes(Top10Save::No).unwrap();
    let mut dir = env::temp_dir();
    dir.push("constructed.lev");
    level.save(&dir, Top10Save::Yes).unwrap();
}

#[test]
/// Test top 10 saving when lists are longer than 10 entries, and whether they get sorted.
fn overflow_top10_and_sort() {
    let mut level = Level::new();
    // Create more than 10 entries unordered.
    let mut top10_single = vec![];
    top10_single.push(TimeEntry::new(("test_p1", "test_p2"), 2221));
    top10_single.push(TimeEntry::new(("test_p1", "test_p2"), 231));
    top10_single.push(TimeEntry::new(("test_p1", "test_p2"), 1221));
    top10_single.push(TimeEntry::new(("test_p1", "test_p2"), 10221));
    top10_single.push(TimeEntry::new(("test_p1", "test_p2"), 2321));
    top10_single.push(TimeEntry::new(("test_p1", "test_p2"), 22211));
    top10_single.push(TimeEntry::new(("test_p1", "test_p2"), 2201));
    top10_single.push(TimeEntry::new(("test_p1", "test_p2"), 5));
    top10_single.push(TimeEntry::new(("test_p1", "test_p2"), 5121));
    top10_single.push(TimeEntry::new(("test_p1", "test_p2"), 918));
    top10_single.push(TimeEntry::new(("test_p1", "test_p2"), 17));
    top10_single.push(TimeEntry::new(("test_p1", "test_p2"), 8172));

    // Make multi list shorter, but still unordered.
    let mut top10_multi = vec![];
    top10_multi.push(TimeEntry::new(("test_p1", "test_p2"), 2221));
    top10_multi.push(TimeEntry::new(("test_p1", "test_p2"), 231));
    top10_multi.push(TimeEntry::new(("test_p1", "test_p2"), 2321));
    top10_multi.push(TimeEntry::new(("test_p1", "test_p2"), 22211));
    top10_multi.push(TimeEntry::new(("test_p1", "test_p2"), 918));
    top10_multi.push(TimeEntry::new(("test_p1", "test_p2"), 17));
    top10_multi.push(TimeEntry::new(("test_p1", "test_p2"), 8172));

    level.best_times.single = top10_single;
    level.best_times.multi = top10_multi;
    // Save and then load it again to see whether it worked.
    let mut dir = env::temp_dir();
    dir.push("top10_overflow_and_sort.lev");
    level.save(&dir, Top10Save::Yes).unwrap();
    let level = Level::load(&dir).unwrap();
    // Check if we get the expected sorted times.
    let mut expected_single = vec![];
    expected_single.push(TimeEntry::new(("test_p1", "test_p2"), 5));
    expected_single.push(TimeEntry::new(("test_p1", "test_p2"), 17));
    expected_single.push(TimeEntry::new(("test_p1", "test_p2"), 231));
    expected_single.push(TimeEntry::new(("test_p1", "test_p2"), 918));
    expected_single.push(TimeEntry::new(("test_p1", "test_p2"), 1221));
    expected_single.push(TimeEntry::new(("test_p1", "test_p2"), 2201));
    expected_single.push(TimeEntry::new(("test_p1", "test_p2"), 2221));
    expected_single.push(TimeEntry::new(("test_p1", "test_p2"), 2321));
    expected_single.push(TimeEntry::new(("test_p1", "test_p2"), 5121));
    expected_single.push(TimeEntry::new(("test_p1", "test_p2"), 8172));

    let mut expected_multi = vec![];
    expected_multi.push(TimeEntry::new(("test_p1", "test_p2"), 17));
    expected_multi.push(TimeEntry::new(("test_p1", "test_p2"), 231));
    expected_multi.push(TimeEntry::new(("test_p1", "test_p2"), 918));
    expected_multi.push(TimeEntry::new(("test_p1", "test_p2"), 2221));
    expected_multi.push(TimeEntry::new(("test_p1", "test_p2"), 2321));
    expected_multi.push(TimeEntry::new(("test_p1", "test_p2"), 8172));
    expected_multi.push(TimeEntry::new(("test_p1", "test_p2"), 22211));

    assert_eq!(level.best_times.single, expected_single);
    assert_eq!(level.best_times.multi, expected_multi);
}

#[test]
fn load_invalid_level_path() {
    assert_eq!(
        Level::load("tests/assets/levels/missing.lev").unwrap_err(),
        ElmaError::Io(std::io::ErrorKind::NotFound)
    );
}

#[test]
/// This should error until Across support is added, if ever.
fn load_across_level_1() {
    assert_eq!(
        Level::load("tests/assets/levels/across.lev").unwrap_err(),
        ElmaError::AcrossUnsupported
    );
}

#[test]
/// Until Across is supported, should generate error when you try to save a Across level.
fn save_across_level_1() {
    let mut level = Level::new();
    level.version = Version::Across;
    let mut dir = env::temp_dir();
    dir.push("save_across_level_1.lev");
    assert_eq!(
        level.save(&dir, Top10Save::No).unwrap_err(),
        ElmaError::AcrossUnsupported
    );
}

#[test]
fn load_valid_level_1() {
    let level = Level::load("tests/assets/levels/test_1.lev").unwrap();
    assert_eq!(level.version, Version::Elma);
    assert_eq!(level.link, 1524269776);
    assert_eq!(
        level.integrity,
        [
            -1148375.210607791_f64,
            1164056.210607791_f64,
            1162467.210607791_f64,
            1162283.210607791_f64,
        ]
    );
    assert_eq!(level.title, "Rust test");
    assert_eq!(level.lgr, "default");
    assert_eq!(level.ground, "ground");
    assert_eq!(level.sky, "sky");

    // Polygon tests.
    assert_eq!(level.polygons.len(), 2);
    assert_eq!(
        level.polygons,
        vec![
            Polygon {
                grass: false,
                vertices: vec![
                    Position::new(-23.993693053024586_f64, 3.135779367971911_f64),
                    Position::new(-15.989070625361132_f64, 3.135779367971911_f64),
                    Position::new(-15.989070625361132_f64, -1.995755366905195_f64),
                    Position::new(-24_f64, -2_f64),
                ],
            },
            Polygon {
                grass: true,
                vertices: vec![
                    Position::new(-23.83645939819548_f64, -2.310222676563402_f64),
                    Position::new(-17.60428907951465_f64, -2.2816347393217473_f64),
                    Position::new(-17.53281923641051_f64, -1.8956975865594021_f64),
                    Position::new(-23.96510511578293_f64, -1.924285523801057_f64),
                ],
            },
        ]
    );

    // Object tests.
    assert_eq!(level.objects.len(), 8);
    assert_eq!(
        level.objects,
        vec![
            Object {
                position: Position::new(-23.221818747499896_f64, 1.3204453531268072_f64),
                object_type: ObjectType::Killer,
            },
            Object {
                position: Position::new(-20.37252715482359_f64, 0.3124543521844827_f64),
                object_type: ObjectType::Apple {
                    gravity: GravityDirection::None,
                    animation: 9,
                },
            },
            Object {
                position: Position::new(-20.3914786548306_f64, -0.5277288147929609_f64),
                object_type: ObjectType::Apple {
                    gravity: GravityDirection::Up,
                    animation: 1,
                },
            },
            Object {
                position: Position::new(-19.526026821177144_f64, -0.36348248139887396_f64),
                object_type: ObjectType::Apple {
                    gravity: GravityDirection::Right,
                    animation: 5,
                },
            },
            Object {
                position: Position::new(-21.269564821822065_f64, -0.38243398140588436_f64),
                object_type: ObjectType::Apple {
                    gravity: GravityDirection::Left,
                    animation: 1,
                },
            },
            Object {
                position: Position::new(-19.55761265452216_f64, 0.4387976855645497_f64),
                object_type: ObjectType::Apple {
                    gravity: GravityDirection::Up,
                    animation: 1,
                },
            },
            Object {
                position: Position::new(-20.075620321380434_f64, 1.2473950191969765_f64),
                object_type: ObjectType::Exit,
            },
            Object {
                position: Position::new(-22.94993115577695_f64, -1.5068896484884773_f64),
                object_type: ObjectType::Player,
            },
        ]
    );

    // Picture tests.
    assert_eq!(level.pictures.len(), 2);
    assert_eq!(
        level.pictures,
        vec![
            Picture {
                name: String::from("barrel"),
                texture: String::new(),
                mask: String::new(),
                position: Position::new(-19.37674118849727_f64, -0.895119783101471_f64),
                distance: 380,
                clip: Clip::Sky,
            },
            Picture {
                name: String::new(),
                texture: String::from("stone1"),
                mask: String::from("maskbig"),
                position: Position::new(-24.465394017511894_f64, 3.964829547979911_f64),
                distance: 750,
                clip: Clip::Sky,
            },
        ]
    );

    // Top10 tests.
    assert_eq!(level.best_times.single.len(), 10);
    assert_eq!(
        level.best_times.single[0],
        TimeEntry::new(("Rust", "Cargo"), 201)
    );
    assert_eq!(
        level.best_times.single[2],
        TimeEntry::new(("Cargo", "Rust"), 206)
    );
    assert_eq!(
        level.best_times.single[9],
        TimeEntry::new(("Rust", "Cargo"), 308)
    );
}

#[test]
fn load_valid_level_2() {
    let level = Level::load("tests/assets/levels/test_2.lev").unwrap();
    assert_eq!(level.version, Version::Elma);
    assert_eq!(level.link, 1505288190);
    assert_eq!(level.title, "");
    assert_eq!(level.ground, "brick");
    assert_eq!(level.sky, "ground");
    assert_eq!(level.polygons.len(), 5);
    assert_eq!(level.polygons[0].grass, false);
    assert_eq!(level.polygons[0].vertices.len(), 4);
    assert_eq!(level.polygons[0].vertices[0].x, 18.507991950076164);
    assert_eq!(level.polygons[0].vertices[1].y, -17.978810742022475);
    assert_eq!(level.objects.len(), 17);
    assert_eq!(level.pictures.len(), 3);
    assert_eq!(level.best_times.single.len(), 0);
}

#[test]
fn load_valid_level_2_from_bytes() {
    let level = Level::load("tests/assets/levels/test_2.lev").unwrap();
    let buffer = fs::read("tests/assets/levels/test_2.lev").unwrap();
    let buf_lev = Level::from_bytes(&buffer).unwrap();
    assert_eq!(level, buf_lev);
}

#[test]
fn load_valid_level_1_and_save_with_top10() {
    let mut level = Level::load("tests/assets/levels/test_1.lev").unwrap();
    let mut dir = env::temp_dir();
    dir.push("save_level_1_wtop10.lev");
    level.save(&dir, Top10Save::Yes).unwrap();
    let level_saved = Level::load(&dir).unwrap();
    assert_eq!(level.title, level_saved.title);
    assert_eq!(level.ground, level_saved.ground);
    assert_eq!(level.sky, level_saved.sky);
    assert_eq!(level.polygons, level_saved.polygons);
    assert_eq!(level.objects, level_saved.objects);
    assert_eq!(level.pictures, level_saved.pictures);
    assert_eq!(level.best_times.single, level_saved.best_times.single);
    assert_eq!(level.best_times.multi, level_saved.best_times.multi);
}

#[test]
fn load_valid_level_1_and_save_without_top10() {
    let mut level = Level::load("tests/assets/levels/test_1.lev").unwrap();
    let mut dir = env::temp_dir();
    dir.push("save_level_1_notop10.lev");
    level.save(&dir, Top10Save::No).unwrap();
    let level_saved = Level::load(&dir).unwrap();
    assert_eq!(level.title, level_saved.title);
    assert_eq!(level.ground, level_saved.ground);
    assert_eq!(level.sky, level_saved.sky);
    assert_eq!(level.polygons, level_saved.polygons);
    assert_eq!(level.objects, level_saved.objects);
    assert_eq!(level.pictures, level_saved.pictures);
    assert!(level.best_times.single != level_saved.best_times.single);
    assert!(level.best_times.multi != level_saved.best_times.multi);
    assert_eq!(level_saved.best_times.single.len(), 0);
    assert_eq!(level_saved.best_times.multi.len(), 0);
}

#[test]
fn load_invalid_level_1() {
    assert_eq!(
        Level::load("tests/assets/levels/invalid_1.lev").unwrap_err(),
        ElmaError::InvalidLevelFile
    );
}

#[test]
fn load_invalid_gravity_level_1() {
    assert_eq!(
        Level::load("tests/assets/levels/invalid_grav.lev").unwrap_err(),
        ElmaError::InvalidGravity(6)
    );
}

#[test]
fn load_invalid_object_level_1() {
    assert_eq!(
        Level::load("tests/assets/levels/invalid_obj.lev").unwrap_err(),
        ElmaError::InvalidObject(6)
    );
}

#[test]
fn load_invalid_clip_level_1() {
    assert_eq!(
        Level::load("tests/assets/levels/invalid_clip.lev").unwrap_err(),
        ElmaError::InvalidClipping(3)
    );
}

#[test]
fn is_apple() {
    let mut lev = Level::new();
    lev.objects.push(Object::new());
    assert_eq!(true, lev.objects[0].is_player());
    assert_eq!(
        false,
        lev.objects[1].is_apple() || lev.objects[1].is_player()
    );
    assert_eq!(true, lev.objects[2].is_apple());
}
