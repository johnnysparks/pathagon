//
//  Protocols.swift
//  pathagon
//
//  Created by John Sparks on 1/20/18.
//  Copyright © 2018 solocast. All rights reserved.
//

import Foundation

protocol BoardProtocol {
    
    // Configuration
    func boardSize() -> Int
    func handSize() -> Int
    
    // Turn State
    var currentTurn: PlayerColor { get }
    func turn() -> PlayerColor
    func moveType() -> MoveType

    func peiceAt(_ position: Position) -> Peice?
    func isRemoved(_ peice: Peice) -> Bool
    func handCount(for player: PlayerColor) -> Int
    func peices(for: PlayerColor) -> [Peice]
    
    // Winning
    func winExists() -> PlayerColor?
    func pathExists(for: PlayerColor) -> Bool
    
    // Next State
    func nextTurn() -> PlayerColor
    func boardAfter(move: Move) -> BoardProtocol?
    func pickup(peice: Peice) -> BoardProtocol
    func place(peice: Peice) -> BoardProtocol
    func validNextBoards() -> [BoardProtocol]
    
    // Validation
    func isOnBoard(position: Position) -> Bool
    func isValid(move: Move) -> Bool
    func isValidPickup(peice: Peice) -> Bool
    func isValidPlacement(peice: Peice) -> Bool
}

extension BoardProtocol {
    func boardSize() -> Int {
        return 7
    }
    
    func handSize() -> Int {
        return 14
    }
    
    func turn() -> PlayerColor {
        return currentTurn
    }
    
    func nextTurn() -> PlayerColor {
        return currentTurn.other()
    }
    
    func moveType() -> MoveType {
        return handCount(for: turn()) > 0 ? .place : .pickup
    }
    
    func winExists() -> PlayerColor? {
        if pathExists(for: .light) {
            return .light
        }
        if pathExists(for: .dark) {
            return .dark
        }
        return nil
    }
    
    func boardAfter(move: Move) -> BoardProtocol? {
        if isValid(move: move) {
            switch move.type {
            case .pickup:
                return pickup(peice: move.peice)
            case .place:
                return place(peice: move.peice)
            }
        }
        return nil
    }
    
    func isValid(move: Move) -> Bool {
        switch move.type {
        case .pickup:
            return isValidPickup(peice: move.peice)
        case .place:
            return isValidPlacement(peice: move.peice)
        }
    }
    
    func isOnBoard(position: Position) -> Bool {
        return position.x < boardSize()
            && position.x >= 0
            && position.y < boardSize()
            && position.y >= 0
    }
    
    func isValidPlacement(peice: Peice) -> Bool {
        
        // is current turn
        if peice.player != turn() {
            return false
        }
        
        // is on board
        if !isOnBoard(position: peice.pos) {
            return false
        }
        
        // if was the last move
        if isRemoved(peice) {
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
    
    func isValidPickup(peice: Peice) -> Bool {
        // it's got to be your peice
        if peice.player != turn() {
            return false
        }
        
        // Got to be on board
        if !isOnBoard(position: peice.pos) {
            return false
        }
        
        // it's got to exist
        if peiceAt(peice.pos) != peice {
            return false
        }
        
        // it can't have just been removed
        if isRemoved(peice) {
            return false
        }
        
        // You can't have peices in hand
        if handCount(for: turn()) > 0 {
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
    
    func validNextBoards() -> [BoardProtocol] {
        var boards: [BoardProtocol] = []
        for x in 0..<boardSize() {
            for y in 0..<boardSize() {
                let move = Move(type: moveType(), peice: Peice(x: x, y: y, player: turn()))
                if let board = boardAfter(move: move) {
                    switch moveType() {
                    case .place:
                        boards.append(board)
                    case .pickup:
                        boards += board.validNextBoards()
                    }
                }
            }
        }
        return boards
    }
    
    func pathExists(for player: PlayerColor) -> Bool {
        
        let heads = Set(peices(for: player).filter({ (player == .dark ? $0.x : $0.y) == 0 }))
        let tails = Set(peices(for: player).filter({ (player == .dark ? $0.x : $0.y) == boardSize() - 1 }))
        
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
    
    func siblings(peice: Peice, forPlayer player: PlayerColor) -> [Peice] {
        return peices(for: player).filter {
            // Above
            (peice.x == $0.x && peice.y == $0.y - 1) ||
            // Below
            (peice.x == $0.x && peice.y == $0.y + 1) ||
            // Left
            (peice.x == $0.x - 1 && peice.y == $0.y) ||
            // Right
            (peice.x == $0.x + 1 && peice.y == $0.y)
        }
    }
}

