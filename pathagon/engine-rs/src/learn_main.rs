use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use pathagon_engine::learned::LearnedBook;

fn main() {
    let args = parse_args();
    let games = PathBuf::from(
        args.get("games")
            .map(String::as_str)
            .unwrap_or("work/rust-v1/current/games.tsv"),
    );
    let output = PathBuf::from(
        args.get("out")
            .map(String::as_str)
            .unwrap_or("work/rust-v1/learned-current"),
    );
    let agent = args
        .get("agent")
        .map(String::as_str)
        .unwrap_or("rust-learned-tabular-v0.1.0");
    let book = LearnedBook::from_games_file(&games)
        .unwrap_or_else(|error| fail(&format!("cannot read game corpus: {error}")));
    fs::create_dir_all(&output)
        .unwrap_or_else(|error| fail(&format!("cannot create output directory: {error}")));
    book.write(&output.join("learned.tsv"))
        .unwrap_or_else(|error| fail(&format!("cannot write learned book: {error}")));
    let manifest = format!(
        "{{\"schemaVersion\":1,\"agent\":\"{}\",\"games\":{},\"moves\":{},\"positions\":{}}}\n",
        agent,
        book.games(),
        book.moves(),
        book.len(),
    );
    fs::write(output.join("manifest.json"), manifest)
        .unwrap_or_else(|error| fail(&format!("cannot write learned manifest: {error}")));
    println!(
        "{{\"schemaVersion\":1,\"agent\":\"{}\",\"games\":{},\"moves\":{},\"positions\":{},\"path\":\"{}\"}}",
        agent,
        book.games(),
        book.moves(),
        book.len(),
        output.display(),
    );
}

fn parse_args() -> HashMap<String, String> {
    let mut parsed = HashMap::new();
    let values: Vec<String> = env::args().skip(1).collect();
    let mut index = 0;
    while index < values.len() {
        if let Some(option) = values[index].strip_prefix("--") {
            if let Some((key, value)) = option.split_once('=') {
                parsed.insert(key.to_owned(), value.to_owned());
            } else if values
                .get(index + 1)
                .is_some_and(|next| !next.starts_with("--"))
            {
                parsed.insert(option.to_owned(), values[index + 1].clone());
                index += 1;
            }
        }
        index += 1;
    }
    parsed
}

fn fail(message: &str) -> ! {
    eprintln!("pathagon-learn: {message}");
    std::process::exit(2)
}
