use std::process::Command;

fn main() {
    Command::new("make")
        .args(&["-C", "test"])
        .status()
        .expect("failed to build test artifacts");
}
