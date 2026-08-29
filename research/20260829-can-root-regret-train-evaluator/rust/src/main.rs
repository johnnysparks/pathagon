use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use pathagon_root_regret_pilot::{collect_roots, emit_targets};

fn main() {
    let args = parse_args();
    let command = args.get("command").map(String::as_str).unwrap_or("collect");
    let result = match command {
        "collect" => {
            let roots = collect_roots(
                &required_path(&args, "games-dir"),
                &required_path(&args, "seeded"),
                &required_path(&args, "human"),
                number(&args, "canonical-limit", 24_usize),
                number(&args, "seeded-limit", 6_usize),
            )
            .unwrap_or_else(|error| fail(&error));
            let output = required_path(&args, "output");
            let mut text = String::new();
            for root in &roots {
                text.push_str(&serde_json::to_string(root).expect("root serializes"));
                text.push('\n');
            }
            fs::write(&output, text)
                .unwrap_or_else(|error| fail(&format!("write roots {}: {error}", output.display())));
            println!("{{\"schemaVersion\":1,\"roots\":{}}}", roots.len());
            Ok(())
        }
        "label" => {
            let count = emit_targets(
                &required_path(&args, "roots"),
                &required_path(&args, "output"),
                number(&args, "teacher-depth", 5_u8),
                number(&args, "teacher-nodes", 2_000_u64),
                number(&args, "teacher-beam", 16_usize),
            )
            .unwrap_or_else(|error| fail(&error));
            println!("{{\"schemaVersion\":1,\"targets\":{count}}}");
            Ok(())
        }
        _ => Err(format!("unknown command {command}; use collect or label")),
    };
    if let Err(error) = result {
        fail(&error);
    }
}

fn parse_args() -> HashMap<String, String> {
    let values: Vec<String> = env::args().skip(1).collect();
    let mut args = HashMap::new();
    let mut index = 0;
    while index < values.len() {
        let value = &values[index];
        if let Some(option) = value.strip_prefix("--") {
            if let Some((key, inline)) = option.split_once('=') {
                args.insert(key.to_owned(), inline.to_owned());
            } else if values.get(index + 1).is_some_and(|next| !next.starts_with("--")) {
                args.insert(option.to_owned(), values[index + 1].clone());
                index += 1;
            } else {
                args.insert(option.to_owned(), "true".to_owned());
            }
        }
        index += 1;
    }
    args
}

fn required_path(args: &HashMap<String, String>, key: &str) -> PathBuf {
    args.get(key)
        .map(PathBuf::from)
        .unwrap_or_else(|| fail(&format!("missing --{key}")))
}

fn number<T: std::str::FromStr>(args: &HashMap<String, String>, key: &str, fallback: T) -> T {
    args.get(key).map_or(fallback, |value| {
        value
            .parse()
            .unwrap_or_else(|_| fail(&format!("invalid --{key}: {value}")))
    })
}

fn fail(message: &str) -> ! {
    eprintln!("pathagon-root-regret-pilot: {message}");
    std::process::exit(2);
}
