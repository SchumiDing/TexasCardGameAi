from neural import *
from model import play

class train(play):
    def trainOneRound(self):
        # one round of poker
        
        self.round += 1
        if self.round == 2:
            # deal the flop
            self.tableCard[0] = self.gain_card()
            self.tableCard[1] = self.gain_card()
            self.tableCard[2] = self.gain_card()
        elif self.round == 3:
            # deal the turn
            self.cutCard()
            self.tableCard[3] = self.gain_card()    
        elif self.round == 4:
            # deal the river
            self.cutCard()
            self.tableCard[4] = self.gain_card()
        self.trainPlayerMakeDecision()
        ifallFold = True
        for player in self.player:
            if player.action[0] != 0:
                ifallFold = False
        
        return ifallFold
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
            ifallFold = self.trainOneRound()
            if ifallFold:
                break
            
            # self.print()
            # a = 0
            if self.round == 4:
                break
        self.calculate_reward()
        return
    def playGame(self, num):
        for i in range(num):
            # print("Game: ", i)
            premoney = []
            for player in self.player:
                player.money = 100
                premoney.append(player.money)
            # random.shuffle(self.player)
            firstp = self.player[0]
            for k in range(len(self.player)-1):
                self.player[k] = self.player[k+1]
            self.player[-1] = firstp
            self.oneGame()
            for k, player in enumerate(self.player):
                objv = torch.tensor(player.money - premoney[k]).float().to(self.device)
                # preParam = []
                # for a in player.qnn.parameters():
                #     preParam.append(a.clone())
                player.qnn.optimizer.zero_grad()
                
                loss = torch.tensor(0).float().to(self.device)
                for j in range(len(player.history)):
                    reward = self.monteCaloReward(objv, j+1 , player.history[j][1][0].item(), len(player.history))
                    loss += player.qnn.criteration(player.history[j][1][0].float(), reward)
                    if player.history[j][0][0] == 2:
                        reward = self.monteCaloReward(objv, j+1 , player.history[j][1][1].item(), len(player.history))
                        loss += player.qnn.criteration(player.history[j][1][1].float(), reward)
                    
                loss.backward()
                # for param in player.qnn.parameters():
                #     print(param.grad)
                player.qnn.optimizer.step()
                player.qnn.scheduler.step()
                # newParam = []
                # for a in player.qnn.parameters():
                #     newParam.append(a.clone())
                # x = preParam[0] == newParam[0]
                player.history = []
            if i % 102 == 0 or i==1:
                print(f"Game: {i}")
                maxReward = torch.load(f"players/maxmodel.pth")[1]
                for k,p in enumerate(self.player):
                    torch.save(p.qnn.state_dict(), f"players/model_{p.playerNum}.pth")
                    if p.money - premoney[k] > maxReward:
                        maxReward = p.money - premoney[k]
                        torch.save((p.qnn.state_dict(),maxReward), f"players/maxmodel.pth")
                    print(f"Player {p}: {p.money}")
                print(f"loss: {loss.item()}")
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
            

if __name__ == '__main__':
    p = train("cpu")
    for i in range(6):
        p.add_player("cpu")
        # p.player[-1].qnn.load_state_dict(torch.load(f"players/maxmodel.pth")[0])
        # p.player[-1].qnn.load_state_dict(torch.load(f"players/model_{p.player[-1].playerNum}.pth"))
        # torch.save(p.player[-1].qnn.state_dict(), f"players/model_{p.player[-1].playerNum}_v0.0.pth")
        # torch.save(p.player[-1].qnn, f"v0.0/model_{p.player[-1].playerNum}.pth")
    p.playGame(100000000)