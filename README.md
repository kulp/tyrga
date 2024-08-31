# tyrga README

[![Rust CI](https://github.com/kulp/tyrga/actions/workflows/rust.yml/badge.svg)](https://github.com/kulp/tyrga/actions/workflows/rust.yml)

tyrga is a project to translate [Java Virtual Machine] bytecode into [tenyr assembly], in order to provide high-level language support on the [tenyr] CPU.

Like tenyr, tyrga is a personal project meant for didactic use. It should not be used to do real work, but it aspires to be fun and educational.

[Java Virtual Machine]: https://en.wikipedia.org/wiki/Java_virtual_machine
[tenyr assembly]: https://github.com/kulp/tenyr/wiki/Assembly-language
[tenyr]: https://github.com/kulp/tenyr

## Status

The capability envelope is defined by the [test suite], which demonstrates both Java-language capabilities (like [static blocks] or [switches]) and algorithms (like [GCD] or a [sieve of Eratosthenes]).

Many capabilities of the JVM remain unimplemented, including basic things like memory allocation (although [references to allocation][allocation] can be translated).

[test suite]: https://github.com/kulp/tyrga/tree/develop/test
[static blocks]: https://github.com/kulp/tyrga/blob/develop/test/Static.java
[switches]: https://github.com/kulp/tyrga/blob/develop/test/Switch.java
[GCD]: https://github.com/kulp/tyrga/blob/develop/test/GCD.java
[sieve of Eratosthenes]: https://github.com/kulp/tyrga/blob/develop/test/Sieve.java
[allocation]: https://github.com/kulp/tyrga/blob/develop/test/Allocate.java

## Basic Usage

The `tyrga-cli` binary drives the `tyrga-lib` library. The main entry point is `tyrga-cli translate`, which takes Java 11-compatible `.class` files on input and produces `.tas` output files.

    javac --release 11 -g:none test/Sieve.java
    tyrga-cli translate --output test/Sieve.tas test/Sieve.class

Compare [`Sieve.java`](https://github.com/kulp/tyrga/blob/develop/test/Sieve.java) and [`Sieve.tas`](https://github.com/kulp/tyrga/blob/develop/test/Sieve.tas).
