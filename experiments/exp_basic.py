import os
import torch
from model import Transformer, Informer, Reformer, Flowformer, Flashformer, \
    iTransformer, iInformer, iReformer, iFlowformer, iFlashformer, S_Mamba, \
    Flashformer_M, Flowformer_M, Autoformer, Autoformer_M, Transformer_M, \
    Informer_M, Reformer_M, Aamba, Famba, Famba_2, Famba_3, Famba_4, Famba_5, Famba_6, Famba_7, \
    Damba, dropout_Damba, DFamba


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'iTransformer': iTransformer,
            'iInformer': iInformer,
            'iReformer': iReformer,
            'iFlowformer': iFlowformer,
            'iFlashformer': iFlashformer,

            'Transformer': Transformer,
            'Transformer_M': Transformer_M,

            'Informer': Informer,
            'Informer_M': Informer_M,

            'Reformer': Reformer,
            'Reformer_M': Reformer_M,

            'Flowformer': Flowformer,
            'Flashformer_M': Flashformer_M,

            'Flashformer': Flashformer,
            'Flowformer_M': Flowformer_M,

            'Autoformer': Autoformer,
            'Autoformer_M': Autoformer_M,

            'S_Mamba': S_Mamba,
            'Aamba': Aamba,
            'Famba': Famba,
            'Famba_2': Famba_2,
            'Famba_3': Famba_3,
            'Famba_4': Famba_4, # B,N,E*D 去掉了B,N,(E+1)*D
            'Famba_5': Famba_5, # 去掉了全局视野for patching and N
            'Famba_6': Famba_6,  # 证明了其实分解的作用最大，现在尝试分解加全局放进mamba
            'Famba_7': Famba_7,  # 跟Famba5对比，没有考虑Z最后直接用out作为结果
            'dropout_Damba':dropout_Damba,
            #在patch里面把全局变量放到dmodle形成两倍大小
            'Damba': Damba,  # 在patch里面把全局变量放到dmodle形成两倍大小
            'DFamba': DFamba  # 利用mask的思想把E固定为4，然后分类，然后把低概率的值mask成0
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
