import numpy as np
import time
   
class n_gram_modal():
    def __init__(self, n_gram: int):
        """n_gram >= 2"""
        if not n_gram >=2:
            raise Exception('n_gram must be >= 2')
        self.names = open('name_input.txt', 'r').read().splitlines()
        self.n_gram = n_gram
        self.chars = (list(set(''.join(self.names))))

        self.chars.extend(('.'))
        self.chars.sort()
        self.stoi = {s:i for i,s in enumerate(self.chars)}
        self.itos = {i:s for s,i in self.stoi.items()}

        self.W = np.random.randn(27, 27)
    
    def data_constructor(self):
        init = [0] * (self.n_gram - 1)
        xs = []
        ys = []
        for word in self.names:
            new_name = ['.'] + list(word) + ['.']
            for chs1, chs2 in zip(new_name, new_name[1:]):
                ix = self.stoi[chs1]
                iy = self.stoi[chs2]

                xs.append(init[1:] + [ix])
                ys.append(iy)

                init = xs[-1]
                if iy == 0:
                    init = [0] * self.n_gram
        return xs, ys

    def one_hot(self, idx: list, len_encode: int = 27):
        output = []
        if type(idx[0]) == int:
            for i in idx:
                a = np.zeros(len_encode)
                a[i] = 1
                output.append(a)
        else:
            for i in idx:
                a = np.zeros(len_encode)
                for x in i:
                    a[x] = 1
                output.append(a)
        return output

    def mini_batches(self, train_data, batch_size: int = 10):
        X = train_data[0]
        Y = train_data[1]
        pos = np.random.randint(len(X))
        end = pos + batch_size
        Xs = X[pos: end]
        Ys = Y[pos: end]
        return Xs,Ys


    def train(self):
        xs, ys = self.data_constructor()
        xenc = np.array(self.one_hot(xs, 27))
        prev_loss = 0
        train_data = [xenc, ys]
        X = xenc
        Y = ys
        
        start = time.perf_counter()
        for i in range(100):
            #X,Y = mini_batches(train_data, 100)
            #forward pass
            logits = X @ self.W 
            prob = np.exp(logits) / np.exp(logits).sum(axis = 1, keepdims = True)
            loss = np.mean(-np.log(prob[np.arange(len(X)), Y]))

            #backward pass
            dprob = np.zeros(prob.shape)
            dprob[np.arange(len(X)), Y] = -1/(len(prob)*prob[np.arange(len(X)), Y])  

            dlogits = np.zeros(dprob.shape)
            dlogits[np.arange(len(X))] = -prob[np.arange(len(X)), Y].reshape(-1, 1) * prob[np.arange(len(X))]
            dlogits[np.arange(len(X)), Y] = prob[np.arange(len(X)), Y]*(1-prob[np.arange(len(X)), Y])
            dlogits[np.arange(len(X))] *= dprob[np.arange(len(X)), Y].reshape(-1, 1)

            dw = X.transpose() @ dlogits

            #UPDATE
            self.W += 30 * -dw

            #STATUS    
            if i % 10 == 0:
                print(f'loss= {loss:4f}  /  iteration: {i}')
                
            if abs(loss - prev_loss) < 0.0001:
                print(f'loss threshold reached !  /  iteration: {i}')
                break
            prev_loss = loss
        
        print(f'Run time: {time.perf_counter() - start} s')

    def sample(self):
        for i in range(10):

          output = []
          ix = [0,0]
          while True:
            xenc = np.array(self.one_hot([ix], len_encode=27))
            logits = xenc @ self.W
            prb = np.exp(logits) / np.exp(logits).sum()

            out = np.argmax(np.random.multinomial(1, np.ravel(prb)))

            if type(ix) == int:
                output.append(self.itos[ix])

            if type(ix) == list:
                ix = ix[1:] + [out]
                output.append(self.itos[ix[-1]])

            if out == 0:
                break
          print(''.join(output))
          
NN = n_gram_modal(2)
NN.train()

NN.sample()