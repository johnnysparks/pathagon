//! Read-only cross-runtime parity probe; gameplay features and inference stay native.
use std::{env,fs::File,io::{BufRead,BufReader}};
use pathagon_engine::{corpus::{decode_state,decode_action},transition_policy::{TransitionPolicyModel,action_features}};
use serde_json::{Value,json};
fn main(){
 let a:Vec<String>=env::args().collect();
 let model=TransitionPolicyModel::from_path(std::path::Path::new(&a[1])).unwrap();
 for line in BufReader::new(File::open(&a[2]).unwrap()).lines(){
  let r:Value=serde_json::from_str(&line.unwrap()).unwrap();
  let mut state=decode_state(r["state"].as_str().unwrap()).unwrap();
  state.config=state.config.with_max_plies(196).unwrap();
  let scores:Vec<f32>=r["actions"].as_array().unwrap().iter().enumerate().map(|(i,a)|{
   let action=decode_action(a.as_str().unwrap()).unwrap();
   assert!(state.legal_actions().contains(&action));
   let f=action_features(state,action,true,false);
   for (j,value) in f.iter().enumerate(){assert!((*value-r["features"][i][j].as_f64().unwrap() as f32).abs()<1e-6);}
   model.score(state,action,true)
  }).collect();
  println!("{}",json!({"id":r["id"],"scores":scores}));
 }
}
