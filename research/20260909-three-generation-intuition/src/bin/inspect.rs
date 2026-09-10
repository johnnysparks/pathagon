//! Human-readable snapshots from the authoritative native replay transitions.
use pathagon_engine::{winning_path, Action, BoardConfig, GameState, Player};
use pathagon_engine::contract::{ContractAction, ReplayRecord};
use std::{env, fs};
fn board(s: GameState) {
    println!("ply {} turn {:?} reserves {:?} winner {:?}",s.ply,s.turn,s.reserve,s.winner);
    for y in 0..s.config.board_size {
        for x in 0..s.config.board_size {
            print!("{}",match s.board_at(y*s.config.board_size+x) {Some(Player::Light)=>'L',Some(Player::Dark)=>'D',None=>'.'});
        }
        println!();
    }
    if let Some(w)=s.winner {println!("winning path {:?}",winning_path(s,w));}
}
fn main() -> Result<(),Box<dyn std::error::Error>> {
    let args:Vec<_>=env::args().collect();
    if args.len()!=3 {return Err("usage: inspect ARENA ZERO_BASED_GAME_INDEX".into());}
    let index:usize=args[2].parse()?;
    let data=fs::read_to_string(&args[1])?;
    let r=ReplayRecord::from_json(data.lines().nth(index).ok_or("missing game index")?)?;
    let mut s=match &r.initial_position {Some(p)=>GameState::from_position(p)?,None=>GameState::with_config(BoardConfig::from_contract(&r.config)?)};
    println!("index {index} seed {} agents {:?} reason {} winner {:?}",r.seed,r.agents,r.reason,r.winner);
    let mut first_relocation=true;
    for (i,m) in r.moves.iter().enumerate() {
        let relocation=matches!(m.action,ContractAction::Relocate{..});
        let show=i+6>=r.moves.len() || (first_relocation && relocation);
        if first_relocation && relocation {println!("first relocation, before move:");board(s);first_relocation=false;}
        let a=match m.action {ContractAction::Place{to}=>Action::Place{to},ContractAction::Relocate{from,to}=>Action::Relocate{from,to}};
        if show {println!("move {} {:?} {} captures {:?}",m.ply,m.player,a,m.captured);}
        s=s.apply(a)?.state;
        if show {board(s);}
    }
    Ok(())
}
