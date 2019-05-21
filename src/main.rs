fn main() -> std::result::Result<(), Box<std::error::Error>> {
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;

    use clap::*;

    let m =
        app_from_crate!()
            .subcommand(
                SubCommand::with_name("translate")
                    .about("Translates JVM .class files into tenyr .tas assembly files")
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
        for file in m.values_of("classes").ok_or("expected at least one input file")? {
            let stem = Path::new(&file).with_extension("");
            let out = stem.with_extension("tas");
            let out = out.file_name().ok_or("failed to format name for output file")?;
            let stem = stem.to_str().ok_or("expected Unicode filename")?;
            let class = classfile_parser::parse_class(&stem)?;

            println!("Creating {} from {} ...", out.to_str().ok_or("expected Unicode filename")?, file);
            let mut file = File::create(out)?;
            for method in &class.methods {
                let mm = tyrga::translate_method(&class, method)?;
                writeln!(file, "{}", mm)?;
            }
        }
    } else if let Some(m) = m.subcommand_matches("mangle") {
        for string in m.values_of("strings").ok_or("expected at least one string to mangle")? {
            println!("{}", tyrga::mangling::mangle(string.bytes())?);
        }
    } else if let Some(m) = m.subcommand_matches("demangle") {
        for string in m.values_of("strings").ok_or("expected at least one string to mangle")? {
            let de = tyrga::mangling::demangle(&string)?;
            let st = String::from_utf8(de)?;
            println!("{}", st);
        }
    }

    Ok(())
}

