use std::{env,fs,io::{BufRead,BufReader,Write},collections::HashMap};
use pathagon_engine::{Action,GameState,BoardConfig};
use pathagon_engine::contract::{ReplayRecord,ContractAction};
use pathagon_engine::corpus::{encode_state,encode_action};
use pathagon_engine::search::{SearchConfig,EvaluationWeights};
use pathagon_engine::transition_policy::{TransitionPolicyModel,action_features};
use serde_json::json;
fn main(){
 let a:Vec<String>=env::args().skip(1).collect();
 let args:HashMap<_,_>=a.chunks(2).map(|x|(x[0].as_str(),x[1].as_str())).collect();
 let model=TransitionPolicyModel::from_path(std::path::Path::new(args["--model"])).unwrap();
 let mut out=fs::File::create(args["--output"]).unwrap();
 let depth:u8=args.get("--depth").unwrap_or(&"6").parse().unwrap();
 let nodes:u64=args.get("--nodes").unwrap_or(&"512000").parse().unwrap();
 let config=SearchConfig{depth,max_nodes:nodes,beam_width:256,weights:EvaluationWeights{path:241,material:112,capture:887,structure:40,threat:154,edge:74},tactical_proof_horizon:None};
 let mut count=0;
 for (game_index,line) in BufReader::new(fs::File::open(args["--input"]).unwrap()).lines().enumerate(){
  let record=ReplayRecord::from_json(&line.unwrap()).unwrap();
  let mut state=GameState::with_config(BoardConfig::new(record.config.board_size,record.config.reserve_per_player).unwrap().with_max_plies(record.config.max_plies).unwrap());
  let mut roots=Vec::new();
  for m in &record.moves {
   let action=match m.action{ContractAction::Place{to}=>Action::Place{to},ContractAction::Relocate{from,to}=>Action::Relocate{from,to}};
   assert!(state.legal_actions().contains(&action));
   assert_eq!(state.turn.as_str(),match m.player{pathagon_engine::contract::ContractPlayer::Light=>"light",_=>"dark"});
   if state.ply>=4 {
    let ranked=model.ranked_actions(state,config.weights);
    if ranked.len()>1 {
     let gap=(ranked[0].score-ranked[1].score).abs();
     roots.push((state,action,gap,m.completed_depth));
    }
   }
   let next=state.apply_legal(action);
   let captured:Vec<u8>=(0..49).filter(|i|next.captured & (1u64<<i)!=0).collect();
   assert_eq!(captured,m.captured);
   state=next.state;
  }
  assert_eq!(state.winner.map(|p|p.as_str()),record.winner.map(|p|match p{pathagon_engine::contract::ContractPlayer::Light=>"light",_=>"dark"}));
  // Four evenly spaced roots plus four uncertain roots, at most eight per game.
  let mut selected=Vec::new();
  for i in 0..4 {if !roots.is_empty(){selected.push(i*(roots.len()-1)/3);}}
  let mut uncertain:Vec<usize>=(0..roots.len()).collect();
  uncertain.sort_by(|&i,&j|roots[i].2.total_cmp(&roots[j].2));
  for i in uncertain {if selected.len()>=8{break;} if !selected.contains(&i){selected.push(i);}}
  selected.sort_unstable();selected.dedup();
  for i in selected {
   let (s,played,gap,source_depth)=roots[i];
   let ranked=model.ranked_actions(s,config.weights);
   let result=model.search(s,config,None);
   let teacher=result.action.unwrap();
   let features:Vec<_>=ranked.iter().map(|r|action_features(s,r.action,r.safe,false).to_vec()).collect();
   let target=ranked.iter().position(|r|r.action==teacher).unwrap();
   writeln!(out,"{}",json!({"id":format!("{}-{}-{}",record.seed,game_index,s.ply),"seed":record.seed,"gameIndex":game_index,"state":encode_state(s),"ply":s.ply,"turn":s.turn.as_str(),"features":features,"actions":ranked.iter().map(|r|encode_action(r.action)).collect::<Vec<_>>(),"target":target,"played":encode_action(played),"teacherAction":encode_action(teacher),"changedAction":teacher!=played,"scoreGap":gap,"sourceCompletedDepth":source_depth,"completedDepth":result.completed_depth,"nodes":result.nodes,"exhausted":result.exhausted,"teacherDepth":depth,"teacherNodeBudget":nodes,"reason":record.reason})).unwrap();
   count+=1;
  }
  out.flush().unwrap();eprintln!("labeled game {}: {} roots total",game_index,count);
 }
}
