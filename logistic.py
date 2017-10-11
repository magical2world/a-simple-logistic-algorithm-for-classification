import numpy as np
import matplotlib.pyplot as plt

class logistic():
    def __init__(self,x,y,learn_rate=0.1):
        if x.shape[0]!=y.shape[0]:
            raise ValueError('The inputs should have same size of first dimensionality')
        self.x=np.matrix(x)#data
        self.y=y#label
        self.learn_rate=learn_rate
        shape = self.x.shape

        self.w = np.matrix(np.random.rand(shape[-1], 1))
        self.b = np.matrix(np.zeros((1,1)))

    def fit(self,train_step=1000):
        for i in range(train_step):
            y_ = 1 / (1 + np.exp(-(self.x * self.w + self.b)))
            d_w=np.array(self.x)*np.array(y_)*np.array((y_-1))*np.array((y_-self.y))#calculated partial derivative
            d_w=np.mean(d_w,0)
            d_w=np.reshape(d_w,[2,1])
            self.w = self.w +self.learn_rate*d_w#use gd to update w
            d_b=np.array(y_)*np.array((y_-1))*np.array((y_-self.y))#calculated partial derivative
            d_b=np.mean(d_b,0)
            d_b=np.reshape(d_b,[1,1])
            self.b=self.b+self.learn_rate*d_b#use gd to update b


        return np.array(self.w),np.array(self.b)

def main():
    x=np.array([[1,2],[2,3],[2,1],[4,3],[1,1.5],[2,1.5],[3,4],[3,1.5],[3,2],[0.5,1],[1,6]])
    y=np.array([[0],[0],[1],[1],[0],[1],[0],[1],[1],[0],[0]])
    clf=logistic(x,y)
    '''
    plot the line of classification model
    '''
    w,b=clf.fit()
    w1=-w[0]/w[1]
    w2=-b/w[1]
    x1=np.arange(7)
    y1=w1*x1+w2
    plt.plot(x1,y1[0])
    '''
    show the data
    '''
    for i,in_put in enumerate(x):
        if y[i]==0:
            plt.scatter(in_put[0],in_put[1],c='red')
        else:
            plt.scatter(in_put[0],in_put[1])
    plt.show()

if __name__=='__main__':
    main()



