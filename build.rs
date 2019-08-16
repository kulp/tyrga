use std::error::Error;
use std::fs;
use std::path::Path;
use std::process::Command;

use walkdir::WalkDir;

fn main() -> Result<(), Box<dyn Error>> {
    let test_dir = Path::new("../test");
    let out_dir = std::env::var("OUT_DIR")?;
    let out_path = Path::new(&out_dir);

    println!("cargo:rerun-if-changed={}", test_dir.display());
    for entry in WalkDir::new(test_dir) {
        let entry = entry?;
        let path = entry.path();
        let name = path.to_str().ok_or("path is not a UTF-8 string")?;
        let stem = path.file_stem().ok_or("filename is empty")?;
        let source_meta = path.metadata()?;

        // Watch for new contents of found directories
        if source_meta.is_dir() {
            println!("cargo:rerun-if-changed={}", path.display());
        }

        match path.extension() {
            Some(e) if e == "java" => {
                let dest = out_path.join(stem).with_extension("class");
                let tas_in = path.with_extension("tas");
                println!("cargo:rerun-if-changed={}", tas_in.display());
                println!("cargo:rerun-if-changed={}", path.display());

                // run javac only if files need to be updated
                if !dest.exists()
                    || source_meta.modified()? > dest.metadata()?.modified()?
                    || tas_in.metadata()?.modified()? > dest.metadata()?.modified()?
                {
                    let output = Command::new("javac")
                        .args(&[ "-d", &out_dir ])
                        .arg("-g:none")
                        .arg("-verbose")
                        .arg(name)
                        .output()?;

                    // javac -verbose writes out a line like this, which we parse to determine the
                    // actual output location for the generated class:
                    //     [wrote RegularFileObject[xyz/interesting/module/name/Packaged.class]]
                    // This lets us copy the .tas file to the right location alongside the .class.
                    let all = std::str::from_utf8(&output.stderr)?;
                    let lines = all.lines();
                    let wrote = lines
                        .filter(|x| x.starts_with(&"[wrote"))
                        .next()
                        .expect("unexpected output from javac");
                    // Different versions of javac use different syntaxes for demarcating paths
                    let first = wrote.rfind(|c| c == '[' || c == ' ').expect("could not find `[`");
                    let last  = wrote.rfind(']').expect("could not find `]`");
                    let out = Path::new(&wrote[first+1..last-1]);
                    let tas = out.with_extension("tas");

                    fs::copy(tas_in, tas)?;
                }
            },
            Some(e) if e == "tas" => {}, // handled in the .class case
            _ => eprintln!("Skipping unrecognized file {}", path.display()),
        }
    }

    Ok(())
}
