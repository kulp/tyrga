use std::error::Error;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() -> Result<(), Box<Error>> {
    let test_dir = "test";
    let out_dir = std::env::var("OUT_DIR").unwrap();

    for entry in fs::read_dir(Path::new(test_dir))? {
        let path = entry?.path();
        let name = &path.to_str().unwrap();
        if let Some(e) = path.extension() {
            if e == "java" {
                Command::new("javac")
                    .args(&[ "-d", &out_dir ])
                    .arg("-g:none")
                    .arg(name)
                    .status()?;
            }
        }
    }

    Ok(())
}
