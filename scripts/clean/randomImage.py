# %%
import random
import cupy as cp
#import numpy as cp
from matplotlib import pyplot as plt
from clean.paramsEllipses import ParamsEllipses
from clean.listEllipses import ListEllipses

import time

# %%
class RandomImage:
    def __init__(self,size_figure,index_random,device):
        self.size_figure = size_figure 
        self.device = self.init_device(device)
        self.index_random = cp.asnumpy(index_random).item(0)        
        
        
        params= ParamsEllipses(size_figure = size_figure , device = device)        
        listEllipses = ListEllipses(params = params,index_random =  index_random, device = device)
        
        self.recursion = 0
        self.percentage_info = listEllipses.params.percentage_info
        self.image,self.mask = self.random_figure(listEllipses)


 

    def init_device(self,device):
        cp.cuda.Device(device).use()
        return device 
            
    
    def normalize(self,figure):
        figure[figure[0][0] == figure] =0
        mask = figure == 0
        reverse = cp.logical_not(cp.asarray(mask))
        if (len(figure[reverse]) > 0):
            figure[reverse] -= cp.min(figure[reverse]) 
            if (cp.max(figure) == 0):
                figure = -1*figure
            figure = figure/cp.max(figure)
        return  figure
    
    def random_operation(self,figure_a,figure_b):
        random.seed(self.index_random)
        self.index_random = self.index_random+1
        operators = ['-','+','*']
        operator = random.choice(operators)
        if (operator == '+'):
              figure_a = figure_b + figure_a
        elif (operator == '*'):
              figure_a = figure_b * figure_a
        elif (operator == '-'):
              figure_a = figure_b - figure_a
        figure_a = self.normalize(figure_a)
        return figure_a

    def get_percentage_info(self,figure):
        n = len(figure)
        c = cp.sum(figure>0) 
        percentage=(c)/(n*n)
        return percentage
    
    def isNull(self,figure):
        if (self.get_percentage_info(figure) < self.percentage_info):
            return True
        return False
    
    def get_mask(self,image):  
        minimum = image[0][0]
        mask = image == minimum
        return mask
    
    def random_figure(self,list_figures):
        size_list_figure = list_figures.len_list()
        random.seed(self.index_random)
        random_index = random.randrange(0,size_list_figure-1,1);
        final_figure = list_figures.data[random_index].data    
        for figure in list_figures.data:
                final_figure = self.random_operation(final_figure,figure.data)
        final_figure_copy  = cp.copy(final_figure)
        final_figure[final_figure_copy ==0]=0       
        if (self.isNull(final_figure) == False):
            mask = self.get_mask(final_figure)
            final_figure = self.normalize(final_figure)
            return final_figure, mask
        else:
            self.recursion = self.recursion+1
            self.index_random  = self.index_random + 1
            return self.random_figure(list_figures)
        
    def get(self):
        return self.image
    
    def view(self):
        plt.imshow(cp.asnumpy(self.image))
    def viewMask(self):
        plt.imshow(cp.asnumpy(self.mask))
    def viewMaskReverse(self):
        plt.imshow(cp.asnumpy(1-self.mask))

           

# %%
#from listEllipses import ListEllipses
#from paramsEllipses import ParamsEllipses
#params= ParamsEllipses(100)
#listEllipses = ListEllipses(params,10)
#randomImage = RandomImage(listEllipses,99)
#randomImage.view()

# %%


# %%
