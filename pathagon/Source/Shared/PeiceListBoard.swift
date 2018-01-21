//
//  PeiceListBoard.swift
//  pathagon
//
//  Created by John Sparks on 1/20/18.
//  Copyright © 2018 solocast. All rights reserved.
//

import Foundation


struct PeiceIntBoard {
    let currentTurn: PlayerColor
    let dark: UInt64
    let light: UInt64
    let darkRemoved: UInt64
    let lightRemoved: UInt64
    
    init(currentTurn: PlayerColor = .light,
         dark: UInt64 = 0,
         light: UInt64 = 0,
         darkRemoved: UInt64 = 0,
         lightRemoved: UInt64 = 0) {
        self.currentTurn = currentTurn
        self.dark = dark
        self.light = light
        self.darkRemoved = darkRemoved
        self.lightRemoved = lightRemoved
    }
}

extension PeiceIntBoard: BoardProtocol {
    
    func toInt(_ position: Position) -> UInt64 {
        return 1 << (UInt64(boardSize() * position.y) + UInt64(position.x))
    }
    
    func hitTest(int: UInt64, position: Position) -> Bool {
        let loc = toInt(position)
        return int & loc == loc
    }
    
    func add(int: UInt64, position: Position) -> UInt64 {
        return int + toInt(position)
    }
    
    func add(int: UInt64, positions: [Position]) -> UInt64 {
        var next = int
        positions.forEach { next = add(int: next, position: $0) }
        return next
    }
    
    func remove(int: UInt64, position: Position) -> UInt64 {
        return int - toInt(position)
    }
    
    func remove(int: UInt64, positions: [Position]) -> UInt64 {
        var next = int
        positions.forEach { next = remove(int: next, position: $0) }
        return next
    }
    
    func peiceAt(_ position: Position) -> Peice? {
        
        if !isOnBoard(position: position) {
            return nil
        }
        
        if hitTest(int: light, position: position) {
            return Peice(pos: position, player: .light)
        }
        
        if hitTest(int: dark, position: position) {
            return Peice(pos: position, player: .dark)
        }
        
        return nil
    }
    
    func isRemoved(_ peice: Peice) -> Bool {
        
        switch peice.player {
        case .light:
            return hitTest(int: lightRemoved, position: peice.pos)
        case .dark:
            return hitTest(int: darkRemoved, position: peice.pos)
        }
    }
    
    func handCount(for player: PlayerColor) -> Int {
        switch player {
        case .light:
            return handSize() - light.nonzeroBitCount
        case .dark:
            return handSize() - dark.nonzeroBitCount
        }
    }
    
    func peices(for player: PlayerColor) -> [Peice] {
        var peices: [Peice] = []
        for x in 0..<boardSize() {
            for y in 0..<boardSize() {
                if let peice = peiceAt(Position(x: x, y: y)), peice.player == player {
                    peices.append(peice)
                }
            }
        }
        return peices
    }
    
    func pickup(peice: Peice) -> BoardProtocol {
        switch peice.player {
        case .light:
            return PeiceIntBoard(currentTurn: turn(),
                                 dark: dark,
                                 light: remove(int: light, position: peice.pos),
                                 darkRemoved: 0,
                                 lightRemoved: toInt(peice.pos))
        case .dark:
            return PeiceIntBoard(currentTurn: turn(),
                                 dark: remove(int: dark, position: peice.pos),
                                 light: light,
                                 darkRemoved: toInt(peice.pos),
                                 lightRemoved: 0)
        }
    }
    
    func place(peice: Peice) -> BoardProtocol {
        let capturePos = capturesForPlace(peice: peice).map { $0.pos }
        
        switch peice.player {
        case .light:
            return PeiceIntBoard(currentTurn: nextTurn(),
                                 dark: remove(int: dark, positions: capturePos),
                                 light: add(int: light, position: peice.pos),
                                 darkRemoved: add(int: 0, positions: capturePos),
                                 lightRemoved: 0)
        case .dark:
            return PeiceIntBoard(currentTurn: nextTurn(),
                                 dark: add(int: dark, position: peice.pos),
                                 light: remove(int: light, positions: capturePos),
                                 darkRemoved: 0,
                                 lightRemoved: add(int: 0, positions: capturePos))
        }
    }
}


struct PeiceListBoard {
    let currentTurn: PlayerColor
    let peices: [Peice]
    let removed: [Peice]
    
    init(currentTurn: PlayerColor = .light, peices: [Peice] = [], removed: [Peice] = []) {
        self.currentTurn = currentTurn
        self.peices = peices
        self.removed = removed
    }
}

extension PeiceListBoard: Equatable { }
func ==(lhs: PeiceListBoard, rhs: PeiceListBoard) -> Bool {
    return lhs.currentTurn == rhs.currentTurn
        && Set(lhs.peices).elementsEqual(Set(rhs.peices))
        && Set(lhs.removed).elementsEqual(Set(rhs.removed))
}

extension PeiceListBoard: BoardProtocol {
    
    func peiceAt(_ position: Position) -> Peice? {
        return peices.filter { $0.x == position.x && $0.y == position.y }.first
    }
    
    func isRemoved(_ peice: Peice) -> Bool {
        return removed.contains(peice)
    }
    
    func handCount(for player: PlayerColor) -> Int {
        return handSize() - peices(for: player).count
    }
    
    func peices(for player: PlayerColor) -> [Peice] {
        return peices.filter { $0.player == player }
    }
    
    func place(peice: Peice) -> BoardProtocol {
        if isValidPlacement(peice: peice) {
            let captures = capturesForPlace(peice: peice)
            var nextPeices = Array(Set(peices).subtracting(Set(captures)))
            nextPeices.append(peice)
            return PeiceListBoard(currentTurn: nextTurn(), peices: nextPeices, removed: captures)
        }
        return self
    }
    
    func pickup(peice: Peice) -> BoardProtocol {
        if isValidPickup(peice: peice) {
            var nextPeices = peices
            nextPeices.remove(at: peices.index(of: peice)!)
            return PeiceListBoard(currentTurn: turn(), peices: nextPeices, removed: [peice])
        }
        return self
    }
}
