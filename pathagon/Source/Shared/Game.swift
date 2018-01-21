//
//  Game.swift
//  pathagon
//
//  Created by John Sparks on 1/20/18.
//  Copyright © 2018 solocast. All rights reserved.
//

import Foundation

class AIPlayer {
    static func makeMove(board: BoardProtocol) -> BoardProtocol {
        let boards = board.validNextBoards()
        let rand = Double(arc4random()) / Double(UInt32.max)
        let idx = Int(rand * Double(boards.count))
        return boards[idx]
    }
}

struct GameStat {
    
    private static let formatter = NumberFormatter()
    
    let turns: UInt
    let seconds: TimeInterval
    
    init(turns: UInt = 0, seconds: TimeInterval = 0) {
        self.turns = turns
        self.seconds = seconds
    }
    
    var frameRateMessage: String {
        let frames = (Double(turns) / seconds)
        GameStat.formatter.numberStyle = .decimal
        GameStat.formatter.maximumFractionDigits = 2
        let text = GameStat.formatter.string(from: NSNumber(value: frames))!
        return "\(text) TPS"
    }
}


class Game {
    
    var stat: GameStat = GameStat()
    var ended: Bool = false
    var board: BoardProtocol {
        didSet {
            onUpdate?(stat, board)

            if let player = board.winExists() {
                ended = true
                onWin?(player)
            }
        }
    }
    
    var onWin: ((PlayerColor) -> ())?
    var onUpdate: ((GameStat, BoardProtocol) -> ())?
    
    func autoplay() {
        
        guard !ended else { return }
        
        DispatchQueue.global(qos: .userInitiated).async {
            
            var timer = Stopwatch()
            timer.start()
            let aiNextBoard = AIPlayer.makeMove(board: self.board)
            timer.stop()
            self.stat = GameStat(turns: self.stat.turns + 1, seconds: self.stat.seconds + timer.seconds)
            
            DispatchQueue.main.async {
                self.board = aiNextBoard
                self.autoplay()
            }
        }
    }
    
    init(board: BoardProtocol) {
        self.board = board
    }
}
