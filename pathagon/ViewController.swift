//
//  ViewController.swift
//  pathagon
//
//  Created by John Sparks on 1/20/18.
//  Copyright © 2018 solocast. All rights reserved.
//

import UIKit

enum Player: Int {
    case light = 0
    case dark = 1
    
    func other() -> Player {
        return self == .light ? .dark : .light
    }
}

enum Move {
    case place
    case pickup
}

struct GameState {
    
    static let handSize: Int = 14
    static let boardSize: Int = 7
    
    let turn: Player
    let peices: [Peice]
    let removed: [Peice]
    
    init(turn: Player = .light, peices: [Peice] = [], removed: [Peice] = []) {
        self.turn = turn
        self.peices = peices
        self.removed = removed
    }
    
    func handleActionAt(position: Position) -> GameState {
        let peice = Peice(pos: position, player: turn)
        if isValidPlacement(peice: peice) {
            return place(peice: peice)
        } else if isValidPickup(peice: peice) {
            return pickup(peice: peice)
        }
        return self
    }
    
    func siblings(peice: Peice, forPlayer player: Player) -> [Peice] {
        return peices.filter {
            // Above
            (peice.x == $0.x && peice.y == $0.y - 1 && peice.player == player) ||
            // Below
            (peice.x == $0.x && peice.y == $0.y + 1 && peice.player == player) ||
            // Left
            (peice.x == $0.x - 1 && peice.y == $0.y && peice.player == player) ||
            // Right
            (peice.x == $0.x + 1 && peice.y == $0.y && peice.player == player)
        }
    }
    
    func peices(for player: Player) -> [Peice] {
        return peices.filter { $0.player == player }
    }
    
    func peiceAt(x: Int, y: Int) -> Peice? {
        return peiceAt(Position(x: x, y: y))
    }
    
    func peiceAt(_ pos: Position) -> Peice? {
        return peices.filter { $0.x == pos.x && $0.y == pos.y }.first
    }
    
    func handCount(for player: Player) -> Int {
        return GameState.handSize - peices(for: player).count
    }
    
    func nextTurn() -> Player {
        return turn == .dark ? .light : .dark
    }
    
    func isValidPlacement(peice: Peice) -> Bool {
        // is on board
        if peice.x >= GameState.boardSize
            || peice.x < 0
            || peice.y >= GameState.boardSize
            || peice.y < 0 {
            return false
        }
        // if was the last move
        if removed.contains(peice) {
            return false
        }
        // if a peice is already there
        if peiceAt(peice.pos) != nil {
            return false
        }
        
        // You need a peice in hand
        if handCount(for: peice.player) < 1 {
            return false
        }
        
        // Otherwise we're good!
        return true
    }
    
    func capturesForPlace(peice: Peice) -> [Peice] {
        var captures: [Peice] = []
        // above
        if let capturer = peiceAt(peice.pos.above().above()),
            let victim = peiceAt(peice.pos.above()),
            capturer.player == peice.player,
            victim.player == peice.player.other() {
            captures.append(victim)
        }
        // left
        if let capturer = peiceAt(peice.pos.left().left()),
            let victim = peiceAt(peice.pos.left()),
            capturer.player == peice.player,
            victim.player == peice.player.other() {
            captures.append(victim)
        }
        // below
        if let capturer = peiceAt(peice.pos.below().below()),
            let victim = peiceAt(peice.pos.below()),
            capturer.player == peice.player,
            victim.player == peice.player.other() {
            captures.append(victim)
        }
        // right
        if let capturer = peiceAt(peice.pos.right().right()),
            let victim = peiceAt(peice.pos.right()),
            capturer.player == peice.player,
            victim.player == peice.player.other() {
            captures.append(victim)
        }
        return captures
    }
    
    func place(peice: Peice) -> GameState {
        if isValidPlacement(peice: peice) {
            let captures = capturesForPlace(peice: peice)
            var nextPeices = Array(Set(peices).subtracting(Set(captures)))
            nextPeices.append(peice)
            return GameState(turn: nextTurn(), peices: nextPeices, removed: captures)
        }
        return self
    }
    
    func isValidPickup(peice: Peice) -> Bool {
        // it's got to be your peice
        if peice.player != turn {
            return false
        }
        // it's got to exist
        if !peices.contains(peice) {
            return false
        }
        
        // it can't have just been removed
        if removed.contains(peice) {
            return false
        }
        
        // You can't have peices in hand
        if handCount(for: turn) > 0 {
            return false
        }

        // Otherwise we're good!
        return true
    }
    
    func pickup(peice: Peice) -> GameState {
        if isValidPickup(peice: peice) {
            var nextPeices = peices
            nextPeices.remove(at: peices.index(of: peice)!)
            return GameState(turn: turn, peices: nextPeices, removed: [peice])
        }
        return self
    }
    
    func pathExists(for player: Player) -> Bool {
        
        let heads = Set(peices(for: player).filter({ (player == .dark ? $0.x : $0.y) == 0 }))
        let tails = Set(peices(for: player).filter({ (player == .dark ? $0.x : $0.y) == GameState.boardSize - 1 }))
        
        // If there are no begining or end nodes, then there is no path
        if heads.count == 0 || tails.count == 0 {
            return false
        }
        
        var visited = Set<Peice>()
        var toVisit = heads
        
        while !toVisit.isEmpty {
            let peice = toVisit.popFirst()!
            visited.insert(peice)
            
            if tails.contains(peice) {
                return true
            }
            
            let siblingsToVisit = Set(siblings(peice: peice, forPlayer: player)).subtracting(visited)
            
            for sibling in siblingsToVisit {
                toVisit.insert(sibling)
            }
        }
        
        return false
    }
}

struct Peice: Equatable, Hashable {
    var hashValue: Int {
        return pos.hashValue + (1000 * player.rawValue)
    }
    
    var y: Int {
        return pos.y
    }
    
    var x: Int {
        return pos.x
    }

    let pos: Position
    let player: Player
    
    init(pos: Position, player: Player) {
        self.pos = pos
        self.player = player
    }
    
    init(x: Int, y: Int, player: Player) {
        self.init(pos: Position(x: x, y: y), player: player)
    }
}

func ==(lhs: Peice, rhs: Peice) -> Bool {
    return lhs.pos == rhs.pos && lhs.player == rhs.player
}

struct Position: Equatable, Hashable {
    let x: Int
    let y: Int
    
    var hashValue: Int {
        return x + (100 * y)
    }
    
    func above() -> Position {
        return Position(x: x, y: y - 1)
    }
    
    func below() -> Position {
        return Position(x: x, y: y + 1)
    }
    
    func left() -> Position {
        return Position(x: x - 1, y: y)
    }
    
    func right() -> Position {
        return Position(x: x + 1, y: y)
    }
}

func ==(lhs: Position, rhs: Position) -> Bool {
    return lhs.x == rhs.x && lhs.y == rhs.y
}





class Game {
    
    var currentState = GameState()
    
    
    
}


class GameStateView: UIView {
    
    var state: GameState = GameState() {
        didSet {
            collectionView.reloadData()
        }
    }
    
    let flow = UICollectionViewFlowLayout()
    let collectionView = UICollectionView(frame: CGRect.zero, collectionViewLayout: UICollectionViewFlowLayout())
    
    override init(frame: CGRect) {
        super.init(frame: frame)
     
        flow.minimumInteritemSpacing = 2
        flow.minimumLineSpacing = 2
        flow.itemSize = CGSize(width: 40, height: 40)
        flow.sectionInset = UIEdgeInsets(top: 2, left: 0, bottom: 2, right: 0)
        collectionView.collectionViewLayout = flow
        collectionView.delegate = self
        collectionView.dataSource = self
        
        collectionView.register(UICollectionViewCell.self, forCellWithReuseIdentifier: "cell")
        
        addSubview(collectionView)
    }
    required init?(coder aDecoder: NSCoder) { fatalError("init(coder:) has not been implemented") }
    
    override func layoutSubviews() {
        super.layoutSubviews()
        collectionView.frame = bounds.insetBy(dx: 2, dy: 2)
    }
}

extension GameStateView: UICollectionViewDelegate {
    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        state = state.handleActionAt(position: Position(x: indexPath.item, y: indexPath.section))
    }
}

extension GameStateView: UICollectionViewDataSource {
    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        return GameState.boardSize
    }
    
    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "cell", for: indexPath)
        
        if let peice = state.peiceAt(x: indexPath.item, y: indexPath.section) {
            cell.backgroundColor = peice.player == .dark ? UIColor.darkGray : UIColor.lightGray
        } else {
            cell.backgroundColor = .white
        }
        return cell
    }
    
    func numberOfSections(in collectionView: UICollectionView) -> Int {
        return GameState.boardSize
    }

}


class ViewController: UIViewController {

    let gameView = GameStateView()
    
    var timer: Timer?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        view.backgroundColor = UIColor.brown
        view.addSubview(gameView)
        gameView.state = GameState()
//        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true, block: { _ in
//            // none
//        })
    }
    
    override func viewDidLayoutSubviews() {
        gameView.frame = CGRect(x: 0, y: 20, width: view.bounds.width, height: view.bounds.width)
    }
}

