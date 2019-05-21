use std::error::Error;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() -> Result<(), Box<Error>> {
    let test_dir = "test";
    let out_dir = std::env::var("OUT_DIR")?;
    let out_path = Path::new(&out_dir);

    for entry in fs::read_dir(Path::new(test_dir))? {
        let path = entry?.path();
        let name = &path.to_str().ok_or("path is not a UTF-8 string")?;
        if let Some(e) = path.extension() {
            if e == "java" {
                let stem = &path.file_stem().ok_or("filename is empty")?;
                let dest = out_path.join(stem).with_extension("class");
                let source_meta = &path.metadata()?;
                // run javac only if files need to be updated
                if !dest.exists() || source_meta.modified()? > dest.metadata()?.modified()? {
                    Command::new("javac")
                        .args(&[ "-d", &out_dir ])
                        .arg("-g:none")
                        .arg(name)
                        .status()?;
                }
            }
        }
    }

    Ok(())
}
