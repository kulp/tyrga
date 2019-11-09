#![deny(clippy::needless_borrow)]

use std::fs::File;
use std::path::Path;

type TerminatingResult = std::result::Result<(), Box<dyn std::error::Error>>;

fn translate_file(input_filename : &Path, output_filename : &Path) -> TerminatingResult {
    let stem = Path::new(input_filename).with_extension("");
    let stem = stem.to_str().ok_or("expected Unicode filename")?;
    let class = classfile_parser::parse_class(stem)?;

    let mut outfile = File::create(output_filename)?;
    tyrga::translate_class(class, &mut outfile)?;

    Ok(())
}

#[test]
fn test_translate_file() -> TerminatingResult {
    let is_dir_or_class = |e : &walkdir::DirEntry| {
        e.metadata().map(|e| e.is_dir()).unwrap_or(false) ||
            e.file_name().to_str().map(|s| s.ends_with(".class")).unwrap_or(false)
    };
    for from in walkdir::WalkDir::new(env!("OUT_DIR")).into_iter().filter_entry(is_dir_or_class) {
        let from = from?;
        let from = from.path();
        if ! from.metadata()?.is_dir() {
            use std::io::Read;

            let to   = from.with_extension("tas-test");
            let gold = from.with_extension("tas");
            translate_file(from, &to)?;

            let mut translated = Vec::new();
            File::open(to)?.read_to_end(&mut translated)?;
            let mut expected = Vec::new();
            File::open(gold)?.read_to_end(&mut expected)?;

            assert_eq!(
                &std::str::from_utf8(&translated),
                &std::str::from_utf8(&expected)
            );
        }
    }

    Ok(())
}

fn main() -> TerminatingResult {
    use clap::*;

    let m =
        app_from_crate!()
            .subcommand(
                SubCommand::with_name("translate")
                    .about("Translates JVM .class files into tenyr .tas assembly files")
                    .arg(Arg::with_name("output")
                            .short("o")
                            .long("output")
                            .help("Specifies the output file name")
                            .takes_value(true)
                        )
                    .arg(Arg::with_name("classes")
                            .help("Names .class files as input")
                            .multiple(true)
                            .required(true)
                        )
                )
            .subcommand(
                SubCommand::with_name("mangle")
                    .about("Mangles strings of bytes into valid tenyr symbols")
                    .arg(Arg::with_name("strings")
                            .help("Provides string inputs for mangling")
                            .multiple(true)
                            .required(true)
                        )
                )
            .subcommand(
                SubCommand::with_name("demangle")
                    .about("Decodes mangled tenyr symbols into strings")
                    .arg(Arg::with_name("strings")
                            .help("Provides string inputs for demangling")
                            .multiple(true)
                            .required(true)
                        )
                )
            .get_matches();

    if let Some(m) = m.subcommand_matches("translate") {
        let ins = m.values_of("classes").ok_or("expected at least one input file")?;
        let user_out = m.value_of("output");
        if user_out.is_some() && ins.len() > 1 {
            return Err("The `-o,--output` flag is incompatible with multiple input files".into());
        }
        for file in ins {
            let file = Path::new(&file);
            let stem;
            let out = if let Some(f) = user_out {
                Path::new(f)
            } else {
                stem = file.with_extension("tas");
                Path::new(stem.file_name().ok_or("expected path to have a filename")?)
            };
            println!("Creating {} from {} ...", out.display(), file.display());
            translate_file(file, out)?;
        }
    } else if let Some(m) = m.subcommand_matches("mangle") {
        for string in m.values_of("strings").ok_or("expected at least one string to mangle")? {
            println!("{}", tyrga::mangling::mangle(string.bytes()));
        }
    } else if let Some(m) = m.subcommand_matches("demangle") {
        for string in m.values_of("strings").ok_or("expected at least one string to mangle")? {
            let de = tyrga::mangling::demangle(string)?;
            let st = String::from_utf8(de)?;
            println!("{}", st);
        }
    }

    Ok(())
}
