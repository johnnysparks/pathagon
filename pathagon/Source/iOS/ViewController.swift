//
//  ViewController.swift
//  pathagon
//
//  Created by John Sparks on 1/20/18.
//  Copyright © 2018 solocast. All rights reserved.
//

import UIKit


class ViewController: UIViewController {

    let label = UILabel()
    let gameView = GameView()
//    var game = Game(board: PeiceListBoard())
    var game = Game(board: PeiceIntBoard())
    
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = UIColor.brown
        
        label.textColor = UIColor.white
        label.textAlignment = .center
        label.font = UIFont.boldSystemFont(ofSize: 18)
        
        [ gameView, label ].forEach { view.addSubview($0) }
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)

        game.onWin = { player in
            let alert = UIAlertController(title: "Game Over", message: "Player \(player == .light ? "Light" : "Dark") wins!", preferredStyle: .alert)
            self.present(alert, animated: true) {
                DispatchQueue.main.asyncAfter(deadline: .now() + 1, execute: {
                    
                    alert.dismiss(animated: true) {
                        self.game.ended = false
//                        self.game.board = PeiceListBoard()
                        self.game.board = PeiceIntBoard()
                        self.game.autoplay()
                    }
                })
            }
            
        }
        
        game.onUpdate = { stat, board in
            self.gameView.board = board
            self.label.text = stat.frameRateMessage
        }
        
        gameView.board = game.board
        gameView.onSelect = { position in
            
            self.game.autoplay()
            return
//

            let move = Move(type: self.game.board.moveType(), peice: Peice(pos: position, player: self.game.board.turn()))
            
            if let userNextBoard = self.game.board.boardAfter(move: move) {
                
                self.game.board = userNextBoard
                DispatchQueue.global(qos: .userInitiated).async {
                    
                    let aiNextBoard = AIPlayer.makeMove(board: userNextBoard)
                    
                    DispatchQueue.main.async {
                        self.game.board = aiNextBoard
                    }
                }
            }
        }
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        gameView.frame = CGRect(x: 0, y: 20, width: view.bounds.width, height: view.bounds.width)
        label.frame = CGRect(x: 0, y: view.bounds.width + 100, width: view.bounds.width, height: 100)
    }
}

// MARK
class GameView: UIView {
    
    public var onSelect:((Position) -> ())?
    public var board: BoardProtocol? {
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

extension GameView: UICollectionViewDelegate {
    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        let position = Position(x: indexPath.item, y: indexPath.section)
        onSelect?(position)
    }
}

extension GameView: UICollectionViewDataSource {
    
    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        return board?.boardSize() ?? 0
    }
    
    func numberOfSections(in collectionView: UICollectionView) -> Int {
        return board?.boardSize() ?? 0
    }
    
    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "cell", for: indexPath)
        
        if let peice = board?.peiceAt(Position(x: indexPath.item, y: indexPath.section)) {
            cell.backgroundColor = peice.player == .dark ? UIColor.darkGray : UIColor.lightGray
        } else {
            cell.backgroundColor = .white
        }
        return cell
    }
}
