extern crate elma;
#[cfg(test)]
mod tests {
    use elma::{ lev, rec };

    #[test]
    fn test_lev_default_values_1 () {
        let level = lev::Level::new();
        assert_eq!(level.link, 0);
    }

    #[test]
    #[should_panic]
    fn test_lev_default_values_2 () {
        let level = lev::Level::new();
        assert_eq!(level.link, 1);
    }

    #[test]
    fn test_lev_load_level_1 () {
        let level = lev::Level::load_level("tests/test.lev");
        assert_eq!(level.version, "Elma".to_string());
    }

    #[test]
    #[should_panic]
    fn test_lev_load_level_2 () {
        let level = lev::Level::load_level("tests/test.lev");
        assert_eq!(level.version, "Across".to_string());
    }

    #[test]
    fn test_rec_default_values_1 () {
        let rec = rec::Rec::new();
        assert_eq!(true, true);
    }
}
