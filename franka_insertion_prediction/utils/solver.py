import torch
import torch.nn as nn
import os
from torch import optim
import numpy as np
import torch_optimizer as torch_optim
from sklearn.metrics import confusion_matrix, accuracy_score

class Solver(object):
    def __init__(self, args, model, train_loader, test_loader):
        self.args = args
        self.train_loader, self.test_loader = train_loader, test_loader
        self.model = model#Transformer(args)#.cuda()
        self.mse = nn.MSELoss()

        self.model_save_path=os.path.join(self.args['model_path'], 'Transformer.pt')
        self.finalmodel_save_path=os.path.join(self.args['model_path'], 'Transformer_final.pt')

        print('--------Network--------')
        print(self.model)

        if args['load_model']:
            print("Using pretrained model")
            self.load_model()
            #self.model.load_state_dict(torch.load(os.path.join(self.args.model_path, 'Transformer.pt')))

    def test_dataset(self, db='test'):
        self.model.eval()

        actual = []
        pred = []
        mse_loss = 0

        if db.lower() == 'train':
            loader = self.train_loader
        elif db.lower() == 'test':
            loader = self.test_loader

        for (inputs, labels) in loader:
            #inputs = inputs.cuda()

            with torch.no_grad():
                preds = self.model(inputs)
            
            mse_loss += self.mse(preds, labels)
            print((preds - labels) / np.array([1000, 1000, 10]))

        return mse_loss

    def test(self):
        train_acc = self.test_dataset('train')
        print("Tr Acc: %.2f" % (train_acc))

        test_acc = self.test_dataset('test')
        print("Te Acc: %.2f" % (test_acc))
    
        return train_acc, test_acc

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_save_path))
        #model.eval()
        return

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        return

    def train(self):
        total_iters = 0
        best_acc = 0
        iter_per_epoch = len(self.train_loader)
        test_epoch = max(self.args['epochs'] // 10, 1)

        #optimizer = optim.Adam(self.model.parameters(), self.args.lr, weight_decay=1e-5)
        #cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs)

        optimizer = torch_optim.Yogi(self.model.parameters(),
                               self.args['lr'], weight_decay=1e-5)
        cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args['epochs'])

        
        
        for epoch in range(self.args['epochs']):

            self.model.train()

            for i, (inputs, labels) in enumerate(self.train_loader):
                total_iters += 1

                #inputs, labels = inputs.cuda(), labels.cuda()

                preds = self.model(inputs)
                clf_loss = self.mse(preds, labels)

                optimizer.zero_grad()
                clf_loss.backward()
                optimizer.step()

                if epoch % 100 == 0: #i == (iter_per_epoch - 1)
                    print('Ep: %d/%d, it: %d/%d, total_iters: %d, err: %.4f'
                          % (epoch + 1, self.args['epochs'], i + 1, iter_per_epoch, total_iters, clf_loss))

            if (epoch + 1) % 100 == 0:
                mse= self.test_dataset('test')
                print("Test acc: %0.2f" % (mse))

                if mse > best_acc:
                    best_acc = mse
                    self.save_model(self.model_save_path)

            cos_decay.step()
            self.save_model(self.finalmodel_save_path)

        
class SequenceSolver(object):
    def __init__(self, args, model, train_loader, test_loader):
        self.args = args
        self.train_loader, self.test_loader = train_loader, test_loader
        self.model = model#Transformer(args)#.cuda()
        self.mse = nn.MSELoss()

        self.model_save_path=os.path.join(self.args['model_path'], 'Transformer.pt')
        self.finalmodel_save_path=os.path.join(self.args['model_path'], 'Transformer_final.pt')

        print('--------Network--------')
        print(self.model)

        if args['load_model']:
            print("Using pretrained model")
            self.load_model()
            #self.model.load_state_dict(torch.load(os.path.join(self.args.model_path, 'Transformer.pt')))

    def test_dataset(self, db='test'):
        self.model.eval()

        actual = []
        pred = []
        mse_loss = 0

        if db.lower() == 'train':
            loader = self.train_loader
        elif db.lower() == 'test':
            loader = self.test_loader

        for (inputs, force, labels) in loader:
            #inputs = inputs.cuda()

            with torch.no_grad():
                preds = self.model(inputs, force)
            
            mse_loss += self.mse(preds, labels)
            print((preds - labels) / np.array([1000, 1000, 1]))

        return mse_loss

    def test(self):
        train_acc = self.test_dataset('train')
        print("Tr Acc: %.2f" % (train_acc))

        test_acc = self.test_dataset('test')
        print("Te Acc: %.2f" % (test_acc))
    
        return train_acc, test_acc

    def load_model(self):
        self.model.load_state_dict(torch.load(self.finalmodel_save_path))
        #model.eval()
        return

    def train(self):
        total_iters = 0
        best_acc = 0
        iter_per_epoch = len(self.train_loader)
        test_epoch = max(self.args['epochs'] // 10, 1)

        #optimizer = optim.Adam(self.model.parameters(), self.args.lr, weight_decay=1e-5)
        #cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs)

        optimizer = torch_optim.Yogi(self.model.parameters(),
                               self.args['lr'], weight_decay=1e-5)
        cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args['epochs'])

        
        
        for epoch in range(self.args['epochs']):

            self.model.train()

            for i, (inputs, force, labels) in enumerate(self.train_loader):
                total_iters += 1

                #inputs, labels = inputs.cuda(), labels.cuda()

                preds = self.model(inputs, force)
                clf_loss = self.mse(preds, labels)

                optimizer.zero_grad()
                clf_loss.backward()
                optimizer.step()

                if epoch % 100 == 0: #i == (iter_per_epoch - 1)
                    print('Ep: %d/%d, it: %d/%d, total_iters: %d, err: %.4f'
                          % (epoch + 1, self.args['epochs'], i + 1, iter_per_epoch, total_iters, clf_loss))

            if (epoch + 1) % 100 == 0:
                mse= self.test_dataset('test')
                print("Test acc: %0.2f" % (mse))

                if mse < best_acc:
                    best_acc = mse
                    torch.save(self.model.state_dict(), self.model_save_path)

            cos_decay.step()
            torch.save(self.model.state_dict(),  self.finalmodel_save_path)


    