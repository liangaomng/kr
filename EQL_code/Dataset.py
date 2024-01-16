from torch.utils.data import Dataset, DataLoader
import torch
import scipy.io as io
from abc import abstractmethod
class My_Dataset():
    
    def __init__(self,train_mat_file):
        self.train_mat_name = train_mat_file
    @abstractmethod
    def read_data(self,mat_file)->dict:
        pass
    
class Physical_Dataset(My_Dataset):
    
    def __init__(self,train_mat_file):
        super().__init__(train_mat_file)
    @abstractmethod
    def read_data(self,mat_file)->dict:
        data=io.loadmat(mat_file)
        return data
    
class Adv_Dataset(Physical_Dataset):
    def __init__(self,**kwargs):
        super().__init__(kwargs["train_mat_file"])  
        
        self.train_dict=None
        self.input=None
        self.train_auxiliary_dict=None
        
        self.__train_mat_file=kwargs["train_mat_file"]
        self.train_dict,self.train_auxiliary_dict= self.__Read4tensor_data(self.__train_mat_file)
       
    def __Read4tensor_data(self,mat_file)->dict:
        
        #reshape 操作
        self.data=super().read_data(mat_file) #父类的方法
        

        #
        self.x_dim = torch.from_numpy(self.data['x']).shape[1]#256
        self.t_dim = torch.from_numpy(self.data['t']).shape[1]#512
        
        x=torch.from_numpy(self.data['x']).squeeze(0)
        t=torch.from_numpy(self.data['t']).squeeze(0)
        f_x=torch.from_numpy((self.data['f_x']).squeeze(0)) #[1,256]->[256]
        f_prime_x=torch.from_numpy((self.data['f_prime_x']).squeeze(0)) #[1,256]->[256]

        # 创建网格
        grid_x, grid_t = torch.meshgrid(x, t, indexing='xy')
        
        grid_f_x=f_x.repeat(self.t_dim,1)#[512,256]
        grid_f_prime_x=f_prime_x.repeat(self.t_dim,1)#[512,256]



        tensor_data={
            'f_x':grid_f_x.reshape(self.x_dim*self.t_dim,1).float(),
            'coord':grid_x.reshape(self.x_dim*self.t_dim,1).float()#[512,256]
        }
        
        auxiliary_data={
            't':grid_t.reshape(self.x_dim*self.t_dim).float(),#[512,256]
            'u':torch.from_numpy(self.data['u']).float().reshape(self.x_dim*self.t_dim,1),
            'u_x':torch.from_numpy(self.data['u_x']).float().reshape(self.x_dim*self.t_dim,1),
            'u_xx':torch.from_numpy(self.data['u_xx']).float().\
                  reshape(self.x_dim*self.t_dim,1),
            'u_t':torch.from_numpy(self.data['u_t']).float().reshape(self.x_dim*self.t_dim,1),
            'f_prime_x':grid_f_prime_x.reshape(self.x_dim*self.t_dim,1).float(),
        }
        return tensor_data,auxiliary_data
    
       
    def __len__(self):

       # 这个任务是输入是x 输出是f(x),loss 里面用pde
        return self.x_dim*self.t_dim#256*512

    def __getitem__(self, idx):
        x = self.train_dict["coord"][idx]
        label=self.train_dict["f_x"][idx]
        #auxil for pde
        auxil_u=self.train_auxiliary_dict["u"][idx]
        auxil_u_x=self.train_auxiliary_dict["u_x"][idx]
        auxil_u_xx=self.train_auxiliary_dict["u_xx"][idx]
        auxil_u_t=self.train_auxiliary_dict["u_t"][idx]
        auxil_f_prime=self.train_auxiliary_dict["f_prime_x"][idx]
        auxil_t=self.train_auxiliary_dict["t"][idx]
        return x,label,[auxil_u,auxil_u_x,auxil_u_xx,auxil_u_t,auxil_f_prime,auxil_t]
     


import matplotlib.pyplot as plt

if __name__ =="__main__":
   
   adv_dataset = Adv_Dataset(train_mat_file='Dataset/adv_diffu_grid_values_train.mat')

   train_loader = DataLoader(adv_dataset, batch_size=256, shuffle=True)  

   
         
       

    
    





   