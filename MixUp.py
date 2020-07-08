from fastai import *
from fastai.text import *

class MixUp(LearnerCallback):
    "Callback that creates the mixed-up input and target."
    def __init__(self, learn:Learner, alpha:float=0.4, stack_x:bool=False, stack_y:bool=True):
        super().__init__(learn)
        self.alpha,self.stack_x,self.stack_y = alpha,stack_x,stack_y
    
    def on_train_begin(self, **kwargs):
        if self.stack_y: self.learn.loss_func = MixUpLoss(self.learn.loss_func)
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):

        "Applies mixup to `last_input` and `last_target` if `train`."
        if not train: return
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
        lambd = last_input.float().new(lambd)
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        x1, y1 = last_input[shuffle], last_target[shuffle]
        if self.stack_x:
            new_input = [[last_input, last_input[shuffle], lambd]]
        else: 
            out_shape = [lambd.size(0)] + [1 for _ in range(len(x1.shape) - 1)]
            new_input = (last_input * lambd.view(out_shape) + x1 * (1-lambd).view(out_shape))
        if self.stack_y:
            new_target = torch.cat([last_target[:,None].float(), y1[:,None].float(), lambd[:,None].float()], 1)
        else:
            if len(last_target.shape) == 2:
                lambd = lambd.unsqueeze(1).float()
            new_target = last_target.float() * lambd + y1.float() * (1-lambd)
        return {'last_input': new_input, 'last_target': new_target}  
    
    def on_train_end(self, **kwargs):
        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()
            
class MixUpLoss(Module):
    "Adapt the loss function `crit` to go with mixup."
    
    def __init__(self, crit, reduction='mean'):
        super().__init__()
        if hasattr(crit, 'reduction'): 
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, 'reduction', 'none')
        else: 
            self.crit = partial(crit, reduction='none')
            self.old_crit = crit
        self.reduction = reduction
        
    def forward(self, output, target):
        if len(target.size()) == 2:
            loss1, loss2 = self.crit(output,target[:,0].long()), self.crit(output,target[:,1].long())
            d = loss1 * target[:,2] + loss2 * (1-target[:,2])
        else:  d = self.crit(output, target)
        if self.reduction == 'mean':    return d.mean()
        elif self.reduction == 'sum':   return d.sum()
        return d
    
    def get_old(self):
        if hasattr(self, 'old_crit'):  return self.old_crit
        elif hasattr(self, 'old_red'): 
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit
			
def mixup(x, x1, lambd):
    x = lambd[:,None,None] * x + (1 - lambd[:,None,None]) * x1
    return x   
	
def AWDforward(self, input:Tensor, from_embeddings:bool=False)->Tuple[List[Tensor],List[Tensor]]:
    use_mixup = False
    if from_embeddings: bs,sl,es = input.size()
    elif isinstance(input, list):
        use_mixup = True
        bs,sl = input[0].size()
    else: bs,sl = input.size()
    if bs!=self.bs:
        self.bs=bs
        self.reset()
    if use_mixup:    
        input,in1,lambd = input
        input = self.encoder_dp(input)
        in1 = self.encoder_dp(in1)
        raw_output = mixup(input,in1,lambd)
    else:
        raw_output = self.input_dp(input if from_embeddings else self.encoder_dp(input))
    new_hidden,raw_outputs,outputs = [],[],[]
    for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
        raw_output, new_h = rnn(raw_output, self.hidden[l])
        new_hidden.append(new_h)
        raw_outputs.append(raw_output)
        if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
        outputs.append(raw_output)
    self.hidden = to_detach(new_hidden, cpu=False)
    return raw_outputs, outputs



def MBEforward(self, input:LongTensor,from_embeddings:bool=False)->Tuple[List[Tensor],List[Tensor],Tensor]:
    use_mixup = False
    if isinstance(input, list): 
        bs,sl = input[0].size() 
        use_mixup = True
    else: bs,sl = input.size()
    self.reset()
    raw_outputs,outputs,masks = [],[],[]
    for i in range(0, sl, self.bptt):
        if use_mixup:
            r, o = self.module([input[0][:,i: min(i+self.bptt, sl)],input[1][:,i: min(i+self.bptt, sl)],input[2]])
        else:
            r, o = self.module(input[:,i: min(i+self.bptt, sl)])
        if i>(sl-self.max_len):
            if use_mixup:
                m1 = input[0][:,i: min(i+self.bptt, sl)] == self.pad_idx
                m2 = input[1][:,i: min(i+self.bptt, sl)] == self.pad_idx
                masks.append(m1 & m2)
            else:
                masks.append(input[:,i: min(i+self.bptt, sl)] == self.pad_idx)
            raw_outputs.append(r)
            outputs.append(o)
    return self.concat(raw_outputs),self.concat(outputs),torch.cat(masks,dim=1)
