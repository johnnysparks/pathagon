//
//  Primatives.swift
//  pathagon
//
//  Created by John Sparks on 1/20/18.
//  Copyright © 2018 solocast. All rights reserved.
//

import Foundation

// MARK - Player
enum PlayerColor: Int {
    case light = 0
    case dark = 1
    
    func other() -> PlayerColor {
        return self == .light ? .dark : .light
    }
}

// MARK - MoveType
enum MoveType {
    case pickup
    case place
}

// MARK - Move
struct Move {
    let type: MoveType
    let peice: Peice
}

// MARK - Peice
struct Peice {
    let pos: Position
    let player: PlayerColor
}

extension Peice {
    var y: Int { return pos.y }
    var x: Int { return pos.x }
    
    init(x: Int, y: Int, player: PlayerColor) {
        self.init(pos: Position(x: x, y: y), player: player)
    }
}

extension Peice: Hashable {
    var hashValue: Int { return pos.hashValue + (1000 * player.rawValue) }
}

extension Peice: Equatable { }
func ==(lhs: Peice, rhs: Peice) -> Bool {
    return lhs.pos == rhs.pos && lhs.player == rhs.player
}

// MARK - Position
struct Position {
    let x: Int
    let y: Int

    func above() -> Position { return Position(x: x, y: y - 1) }
    func below() -> Position { return Position(x: x, y: y + 1) }
    func left() -> Position  { return Position(x: x - 1, y: y) }
    func right() -> Position { return Position(x: x + 1, y: y) }
}

extension Position {
    var hashValue: Int { return x + (100 * y) }
}

extension Position: Equatable { }
func ==(lhs: Position, rhs: Position) -> Bool {
    return lhs.x == rhs.x && lhs.y == rhs.y
}
