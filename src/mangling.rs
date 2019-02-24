extern crate rand;
extern crate regex;

#[cfg(test)]
use rand::distributions::{Alphanumeric, Distribution, Normal};
#[cfg(test)]
use rand::{thread_rng, Rng};
use regex::Regex;

#[cfg(test)]
const MANGLE_LIST : &[(&str, &str)] = &[
    ( "()V"                        , "_02_28291V"                                            ),
    ( "(II)I"                      , "_01_282II01_291I"                                      ),
    ( "<init>"                     , "_01_3c4init01_3e"                                      ),
    ( "<init>:()V"                 , "_01_3c4init04_3e3a28291V"                              ),
    ( "Code"                       , "_4Code"                                                ),
    ( "GCD"                        , "_3GCD"                                                 ),
    ( "StackMapTable"              , "_13StackMapTable"                                      ),
    ( "gcd"                        , "_3gcd"                                                 ),
    ( "java/lang/Object"           , "_4java01_2f4lang01_2f6Object"                          ),
    ( "java/lang/Object.<init>:()V", "_4java01_2f4lang01_2f6Object02_2e3c4init04_3e3a28291V" ),
];

#[test]
fn test_mangle() {
    for (unmangled, mangled) in MANGLE_LIST {
        assert_eq!(&mangle(unmangled.as_ref()), mangled);
    }
}

pub fn mangle(name : &[u8]) -> String {
    let mut offset = 0;
    let mut out = String::with_capacity(2 * name.len()); // heuristic
    let re_token    = Regex::new(r"^(?i)[a-z_]\w*").unwrap();
    let re_nontoken = Regex::new(r"^[0-9]*\W*").unwrap();

    out.push('_');

    let begin_ok = |x| char::from(x).is_alphabetic() || char::from(x) == '_';
    let within_ok = |x| begin_ok(x) || char::from(x).is_numeric();

    while offset < name.len() {
        let mut len = 0;
        if begin_ok(name[offset + len]) {
            while offset + len < name.len() && within_ok(name[offset + len]) {
                len += 1;
            }
            match &String::from_utf8(name[offset..offset + len].to_vec()) {
                Ok(s) => out.push_str(&format!("{}{}", len, s)),
                Err(s) => panic!("TODO"),
            }
            offset += len;
        }
        if let Some(m) = re_nontoken.find(&String::from_utf8_lossy(&name[offset..])) {
            if m.as_str().len() > 0 {
                let s = m.as_str();
                let len = s.len();
                offset += len;
                out.push_str(&format!("0{}_{}", len, hexify(&s)));
            } else if offset < name.len() {
                panic!("Unable to progress");
            }
        }
    }

    out.shrink_to_fit();
    return out;
}

#[test]
fn test_demangle() {
    for (unmangled, mangled) in MANGLE_LIST {
        let got : Vec<u8> = demangle(mangled);
        let want : Vec<u8> = unmangled.to_owned().to_string().into();
        assert_eq!(want, got);
    }
}

pub fn demangle(name : &str) -> Vec<u8> { // TODO Option<Vec<u8>>
    let mut offset = 0;
    let mut out = Vec::with_capacity(name.len());
    let num = Regex::new(r"^\d+").unwrap();

    if &name[0..1] != "_" {
        panic!("Bad identifier (expected `_`)");
    } else {
        offset += 1;
    }

    let mut is_hex = false;
    while offset < name.len() {
        if &name[offset..offset+1] == "0" {
            offset += 1;
            is_hex = true;
        }
        let m = num.find(&name[offset..])
                   .expect("Bad identifier (expected number)");
        let len = usize::from_str_radix(m.as_str(), 10)
            .expect("Hex parse failure");
        offset += m.as_str().len();
        if is_hex {
            if &name[offset..offset+1] != "_" {
                panic!("Bad identifier (expected `_`)");
            }
            offset += 1;
            let nybbles = 2 * len;
            out.append(&mut dehexify(&name[offset..offset+nybbles]));
            offset += nybbles;
        } else {
            out.append(&mut Vec::from(&name[offset..offset+len]));
            offset += len;
        }
        is_hex = false;
    }

    out.shrink_to_fit();
    return out;
}

#[test]
fn test_round_trip() {
    let mut rng = thread_rng();
    let norm = Normal::new(20.0, 5.0);
    for i in 1..10 {
        let len = norm.sample(&mut rng) as usize;
        let rs : String = rng.sample_iter(&Alphanumeric).take(len).collect();
        let rst : Vec<u8> = rs.to_string().into();

        assert_eq!(&rst, &demangle(&mangle(&rs.as_ref())));
    }
}

fn hexify(bytes : &[u8]) -> String {
    let mut out = String::new();

    for &b in bytes {
        out.push_str(&format!("{:02x}", &b));
    }

    return out;
}

fn dehexify(s : &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() / 2);

    for i in 0..(s.len() / 2) {
        let hex = u8::from_str_radix(&s[i*2..i*2+2], 16)
                    .expect("Hex parse failure");
        out.push(hex);
    }

    return out;
}

