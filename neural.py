import torch
from numpy import random
import numpy as np

class qnn(torch.nn.Module):
    
    def __init__(self, input_dim=18, betinput=16, output_dim=3, bet_dim=5, card_input=119,otherDecision=100, card_sol = 3, device='cpu'):
        '''
        input_dim: [1otherstastic(3*1),1otherstastic(3*1),1otherstastic(3*1),1otherstastic(3*1), 1otherstastic(3*1), card_sol]
        card_sol: w_i * 3, 1 <= w_i <= 13, if no card 0
        output_dim: [fold, check, call]
        bet_dim: 30%, 50%, 66%, 90%, 110% of the bet
        card_input : 5*(13+4) + 2*(13+4)
        otherDecision : (3+13+4)*5
        '''
        super(qnn, self).__init__()
        self.device = device
        self.card_decision = torch.nn.Sequential(
                torch.nn.Linear(card_input+otherDecision, card_input//2),
                torch.nn.ReLU(),
                torch.nn.Linear(card_input//2, card_input//4),
                torch.nn.ReLU(),
                torch.nn.Linear(card_input//4, card_input//8),
                torch.nn.ReLU(),
                torch.nn.Linear(card_input//8, card_sol),
        )
        # self.card_threw = torch.nn.Sequential(
        #         torch.nn.Linear(input_dim, input_dim//2),
        #         torch.nn.ReLU(),
        #         torch.nn.Linear(input_dim//2, input_dim//4),
        #         torch.nn.ReLU(),
        #         torch.nn.Linear(input_dim//4, 2),
        # )

        self.fc = torch.nn.Sequential(
                torch.nn.Linear(input_dim-output_dim+bet_dim, (input_dim + output_dim) // 2),
                torch.nn.ReLU(),
                torch.nn.Linear((input_dim + output_dim) // 2, (input_dim + output_dim) // 4),
                torch.nn.ReLU(),
                torch.nn.Linear((input_dim + output_dim) // 4, bet_dim),
        )
        
        self.fc1 = torch.nn.Sequential(
                torch.nn.Linear(input_dim+output_dim, (betinput + bet_dim) // 2),
                torch.nn.ReLU(),
                torch.nn.Linear((betinput + bet_dim) // 2, (betinput + bet_dim) // 4),
                torch.nn.ReLU(),
                torch.nn.Linear((betinput + bet_dim) // 4, output_dim),
        )
        
        self.fcC = torch.nn.Sequential(
                torch.nn.Linear(card_input+otherDecision, (card_input+otherDecision)//2),
                torch.nn.ReLU(),
                torch.nn.Linear((card_input+otherDecision)//2, card_input//2),
                torch.nn.ReLU(),
                torch.nn.Linear(card_input//2, bet_dim),
        )
        self.input_dim = input_dim
        self.output_dim = output_dim
    
        #lstm modules
        self.hidden_size = input_dim
        self.cell_size = input_dim
        self.hiddenState = [torch.zeros(1, self.hidden_size)]
        self.cellState = [torch.zeros(1, self.cell_size)]
        self.input_layer = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size+output_dim, self.input_dim*2),
                torch.nn.ReLU(),
                torch.nn.Linear(input_dim*2, self.hidden_size),
                torch.nn.Sigmoid()
        )
        self.forget_layer = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size+output_dim, self.input_dim*2),
                torch.nn.ReLU(),
                torch.nn.Linear(input_dim*2, self.hidden_size),
        )
        self.forget1 = torch.nn.Sigmoid()
        self.forget2 = torch.nn.Tanh()
        self.output_layer = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size+output_dim, self.input_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(input_dim, self.hidden_size),
                torch.nn.Sigmoid()
        )
        
        self.criteration = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
    
    def LSTMst(self, sta, card):
        # x: [batch_size, input_dim]
        # hiddenState: [batch_size, hidden_size]
        # cellState: [batch_size, cell_size]
        cD = self.card_decision(card)


        # x = torch.cat((sta, cD, self.hiddenState[-1]), dim=1)
        x = torch.cat((self.hiddenState[-1], cD), dim=1)
        
        input_gate = self.input_layer(x)
        forget_gate = self.forget_layer(x)
        output_gate = self.output_layer(x)

        self.cellState.append(self.cellState[-1].clone().detach().requires_grad_(True).to(self.device))
        self.hiddenState.append(self.hiddenState[-1].clone().detach().requires_grad_(True).to(self.device))
        
        # update cell state
        self.cellState[-1] = self.forget1(forget_gate) * self.cellState[-2] + self.forget2(input_gate) * torch.tanh(self.cellState[-2])
        self.hiddenState[-1] = output_gate * torch.tanh(self.cellState[-1])
        
        
        out1 = self.fc1(torch.cat((self.hiddenState[-1][0],cD[0])))
        # out2 = self.card_threw(torch.cat((sta, out1), dim=1))
        return out1, None
    

                
        

    def forward(self, x, card):
        # x: [batch_size, input_dim]
        x1, x3 = self.LSTMst(x, card)
        cB = self.fcC(card)
        inp = torch.cat((x, cB), dim=1).view(-1, 20).to(self.device)
        x2 = self.fc(inp)
        return x1, x2, x3
    
    def choose_action(self, x, card):
        # x: [batch_size, input_dim]
        # 0: fold, 1: check, 2: call
        
        x1, x2, x3 = self.forward(x, card)
        action = [0, 0, 0]
        memory = [0, 0, 0]
        action[0] = torch.argmax(x1)
        memory[0] = x1[action[0]]
        if action[0] == 2:
            action[1] = torch.argmax(x2, dim=1)
            memory[1] = x2[0][action[1]]
        # if action[0] == 0:
        #     action[2] = torch.argmax(x3, dim=1)
        #     memory[2] = x3[0][action[2]]

        return action, memory
    
    def train_choose_action(self, x, card):
        # x: [batch_size, input_dim]
        # 0: fold, 1: check, 2: call
        
        x1, x2, x3 = self.forward(x, card)
        action = [0, 0, 0]
        memory = [0, 0, 0]
        action[0] = np.random.choice([0, 1, 2])
        memory[0] = x1[action[0]]
        if action[0] == 2:
            action[1] = np.random.choice([0, 1, 2, 3, 4])
            memory[1] = x2[0][action[1]]
        # if action[0] == 0:
        #     action[2] = torch.argmax(x3, dim=1)
        #     memory[2] = x3[0][action[2]]

        return action, memory
    


class Card:
    def __init__(self, value, suit):
        # value: 0-13, 0: no card
        # suit: 0-4, 0: no card
        # 1: spade, 2: heart, 3: diamond, 4: club
        self.value = value
        self.suit = suit
        
    
    

class TexasPocker:
    def __init__(self, device='cpu'):
        self.qnn = qnn(device=device).to(device)
        self.qnn
        self.action = None
        self.state = None
        self.reward = None
        self.money = 0
        self.bet = 0
        self.handCard = (None, None)
        self.vpip = 0
        self.pfr = 0
        self.threeBet = 0
        self.folds = 0
    
    def init_network(self):
        self.qnn.hiddenState = [torch.zeros(1, self.qnn.hidden_size).to(self.qnn.device)]
        self.qnn.cellState = [torch.zeros(1, self.qnn.cell_size).to(self.qnn.device)]