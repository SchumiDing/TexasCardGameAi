from neural import *
class play:
    def __init__(self, device='cpu'):
        self.player = []
        self.card = []
        self.tableCard = []
        self.pot = 0
        self.current_bet = 0
        self.round = 0
        self.playerNum = 0
        self.history = []
        
        self.action = None
        self.state = None
        self.reward = None
        self.done = None
        self.history = []
        
        self.device = device
        
        self.betList = [0.3, 0.5, 0.66, 0.9, 1.1]
        
    def createcardList(self):
        # 52 cardss
        card_list = []
        for i in range(1, 14):
            for j in range(1, 5):
                card_list.append(Card(i, j))
        # shuffle the card
        random.shuffle(card_list)
        self.card = card_list
        return
    
    def shuffle_card(self):
        # shuffle the card
        random.shuffle(self.card)
        return
    
    def add_player(self, device):
        player = TexasPocker(device=device)
        player.init_network()
        self.player.append(player)
        player.playerNum = len(self.player)
        player.money = 100
    
    def start(self):
        # deal the card
        for i in range(2):
            for player in self.player:
                player.handCard[i] = self.card.pop()
        return
    
    def create_input(self, handCard, tableCard, player):
        players = []
        for p in self.player:
            if p != player:
                players.append(p)
        card = torch.zeros(1, 119+100).to(self.device)
        for i in range(5):
            if tableCard[i] == None:
                continue
            if tableCard[i].value != 0:
                card[0][i*17 + tableCard[i].value] = 1
                card[0][i*17 + 13+ tableCard[i].suit] = 1
        for i in range(2):
            if handCard[i].value != 0:
                card[0][i*17 + handCard[i].value] = 1
                card[0][i*17 + 13+ handCard[i].suit] = 1
        for i in range(5):
            if p.action[0] != None:
                card[0][i*20 + p.action[0]] = 1
                if p.action[0] == 2:
                    card[0][i*20 + 3 + p.action[1]] = 1
                if p.action[0] == 0:
                    card[0][i*20 + 8 + p.action[2]] = 1
        player_state = torch.zeros(1, 15).to(self.device)
        for i,p in enumerate(players):
            player_state[0][i*3] = p.vpip/p.folds if p.folds != 0 else 0
            player_state[0][i*3 + 1] = p.pfr/p.folds if p.folds != 0 else 0
            player_state[0][i*3 + 2] = p.threeBet/p.folds if p.folds != 0 else 0
        return player_state, card
    
    def gain_card(self):
        return self.card.pop()
    
    def cutCard(self):
        self.card.pop()
        return
    
    def playerMakeDecision(self):
        # player make decision
        for player in self.player:
            if player.handCard[0] == None:
                continue
            sta, card = self.create_input(player.handCard, self.tableCard, player)
            action, memory = player.qnn.choose_action(sta, card)
            player.action = action
            player.history.append((action, memory))
            if action[0] == 0:
                # fold
                player.foldCard = [player.handCard[0], player.handCard[1]]
    
                # print(f"fold : {player.handCard[0].value} {player.handCard[0].suit}, {player.handCard[1].value} {player.handCard[1].suit}")
                player.handCard = [None, None]
            elif action[0] == 1:
                # check d
                pass
            elif action[0] == 2:
                # call
                bet = self.betList[action[1]]*self.pot
                player.current_bet = bet
                self.pot += bet
                player.money -= bet
                player.bet += bet
                player.vpip += 1
                # player.folds += 1
    
        return
    
    def trainPlayerMakeDecision(self):
        # player make decision
        for player in self.player:
            if player.handCard[0] == None:
                continue
            sta, card = self.create_input(player.handCard, self.tableCard, player)
            action, memory = player.qnn.train_choose_action(sta, card)
            player.action = action
            player.history.append((action, memory))
            if action[0] == 0:
                # fold
                player.foldCard = [player.handCard[0], player.handCard[1]]
    
                # print(f"fold : {player.handCard[0].value} {player.handCard[0].suit}, {player.handCard[1].value} {player.handCard[1].suit}")
                player.handCard = [None, None]
            elif action[0] == 1:
                # check d
                bet = self.prebet
                player.current_bet = bet
                self.pot += bet
                player.money -= bet
                player.bet += bet
                player.vpip += 1
                
                pass
            elif action[0] == 2:
                # call
                bet = self.betList[action[1]]*self.pot
                player.current_bet = bet
                self.pot += bet
                player.money -= bet
                player.bet += bet
                player.vpip += 1
                self.prebet = bet
                # player.folds += 1
    
        return
    
    def oneRound(self):
        # one round of poker
        self.playerMakeDecision()
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
        ifallFold = True
        for player in self.player:
            if player.action[0] != 0:
                ifallFold = False
        
        return ifallFold
    
    def trainOneRound(self):
        # one round of poker
        self.trainPlayerMakeDecision()
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
        ifallFold = True
        for player in self.player:
            if player.action[0] != 0:
                ifallFold = False
        
        return ifallFold
    
    def compareHandCard(self, handCard1, handCard2, tableCard):
        """
        比较两手牌的大小，返回 True 如果 handCard1 比 handCard2 大，否则返回 False。
        """
        def evaluate_hand(hand):
            """
            计算一手牌的评分，返回一个元组 (hand_rank, high_cards)，
            hand_rank 越大表示牌型越强，high_cards 用于比较同牌型的大小。
            """
            values = sorted([card.value for card in hand], reverse=True)
            suits = [card.suit for card in hand]

            value_counts = {v: values.count(v) for v in set(values)}
            suit_counts = {s: suits.count(s) for s in set(suits)}

            is_flush = max(suit_counts.values()) >= 5

            unique_values = sorted(set(values), reverse=True)
            is_straight = False
            for i in range(len(unique_values) - 4):
                if unique_values[i] - unique_values[i + 4] == 4:
                    is_straight = True
                    break

            if set([14, 2, 3, 4, 5]).issubset(set(values)):
                is_straight = True

            # 判断牌型
            if is_flush and is_straight:
                return (8, values)  # 同花顺
            if 4 in value_counts.values():
                return (7, values)  # 四条
            if 3 in value_counts.values() and 2 in value_counts.values():
                return (6, values)  # 葫芦
            if is_flush:
                return (5, values)  # 同花
            if is_straight:
                return (4, values)  # 顺子
            if 3 in value_counts.values():
                return (3, values)  # 三条
            if list(value_counts.values()).count(2) == 2:
                return (2, values)  # 两对
            if 2 in value_counts.values():
                return (1, values)  # 一对
            return (0, values)  # 高牌

        # 计算两手牌的评分
        score1 = evaluate_hand(handCard1 + tableCard)
        score2 = evaluate_hand(handCard2 + tableCard)

        # 比较评分
        return score1 > score2


    
    def calculate_reward(self):
        # calculate the reward
        bighandCard = None
        bigplayer = None
        for player in self.player[1:]:
            if player.handCard[0] == None:
                continue
            handCard = player.handCard
            if bighandCard == None:
                bighandCard = handCard
                bigplayer = player
            elif self.compareHandCard(bighandCard, handCard, self.tableCard):
                bighandCard = handCard
                bigplayer = player
        if bigplayer != None:
            # not all player fold
            bigplayer.money += self.pot
        self.pot = 0
    
        return

    def init_table(self):
        for i in range(5):
            self.tableCard.append(None)
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
            # ifallFold = self.trainOneRound()
            ifallFold = self.oneRound()
            if ifallFold:
                break
            
            self.print()
            a = 0
            if self.round == 4:
                break
        self.calculate_reward()
        return
    
    def monteCaloReward(self, objv, timeStep, modelOut, lenHis):
        # monte carlo reward
        # timeStep: 0-3
        # objv: 0-1
        t = lenHis
        reward = 0
        objv = torch.tensor(objv).float().to(self.device)
        alpha = (timeStep)/t
        reward =  alpha * modelOut + (1-alpha) * objv
        return reward
    
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
    def dealCard(self, card):
        
        c = 0
        if card.value == 13:
            c = 'A'
        else:
            c = card.value+1
        return c
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
    p = play("cpu")
    for i in range(6):
        p.add_player("cpu")
        # p.player[-1].qnn.load_state_dict(torch.load(f"players/maxmodel.pth")[0])
        p.player[-1].qnn.load_state_dict(torch.load(f"players/model_{p.player[-1].playerNum}.pth"))
        # torch.save(p.player[-1].qnn.state_dict(), f"players/model_{p.player[-1].playerNum}_v0.0.pth")
        # torch.save(p.player[-1].qnn, f"v0.0/model_{p.player[-1].playerNum}.pth")
    p.playGame(100000000)