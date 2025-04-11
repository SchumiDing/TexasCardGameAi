from neural import *
from model import play

class modelPlay(play):
    def print(self):
        print(f"pot: {self.pot}")
        print(f"Round: {self.round}")
        suitList = ["spade", "heart", "diamond", "club"]
        print("Table Card: ", end="")
        for card in self.tableCard:
            if card == None:
                print("None", end=" ")
                continue
            c = self.dealCard(card)
            print(f"{suitList[card.suit-1]} {c}", end=" ")
        decisionL = ["fold", "check", "call"]
        print("\nPlayer: ")
        for player in self.player:
            if player.handCard[0] == None:
                c1 = self.dealCard(player.foldCard[0])
                c2 = self.dealCard(player.foldCard[1])
                print(f"Player {player.playerNum}:\nfold")
                print(f"{suitList[player.foldCard[0].suit-1]} {c1}, {suitList[player.foldCard[1].suit-1]} {c2}")
                print(f"-"*20)
                continue
            print(f"Player {player.playerNum}:\n {decisionL[player.action[0]]}")
            c1 = self.dealCard(player.handCard[0])
            c2 = self.dealCard(player.handCard[1])
            print(f"{suitList[player.handCard[0].suit-1]} {c1}, {suitList[player.handCard[1].suit-1]} {c2}")
            print(f"-"*20)
        return
    def oneGame(self):
        # one game of poker
        # print("=="*10)
        self.createcardList()
        self.shuffle_card()
        self.tableCard = []
        self.init_table()
        for player in self.player:
            player.vpip = 0
            player.pfr = 0
            player.threeBet = 0
            player.folds = 0
            player.current_bet = 0
            player.handCard = [None, None]
            player.action = [None, None, None]
            player.init_network()
            player.history = []

        self.pot = 20
        self.player[1].money -= 15
        self.player[-1].money -= 5
        self.round = 0
        self.prebet = 5
        self.start()
        while True:
            for player in self.player:
                player.action = [None, None, None]
            ifallFold = self.oneRound()

            self.print()
            if ifallFold:
                break
            if self.round == 4:
                break
            

if __name__ == "__main__":
    play = modelPlay("cpu")
    num = 6
    version = "0.0"
    for i in range(num):
        play.add_player("cpu")
        play.player[-1].qnn = torch.load(f"v{version}/model_{play.player[-1].playerNum}.pth")
        # play.player[-1].qnn.load_state_dict(torch.load(f"v{version}/model_{play.player[-1].playerNum}.pth"))
    play.oneGame()