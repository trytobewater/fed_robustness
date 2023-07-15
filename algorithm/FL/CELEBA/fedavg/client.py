import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch.nn as nn
import utils
import numpy as np
import copy
import torch
from models.model_utils import cifar10_loaders


from models.model_utils import cifar10_somebatch_loaders



class Client:

    def __init__(self, client_id, train_data, test_data, model=None, optimizer=None, model_type='int'):
        self.model = model
        self.id = client_id
        self.train_data = train_data
        self.test_data = test_data
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.model_type = model_type

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.test_data is None:
            return 0
        
        return len(self.test_data['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        
        return len(self.train_data['y'])

    
    
    
    
    
    
    
    
    
    
    
    
    
    

    def train(self, epoch, cids, num_epochs=1, batch_size=5, model_type='int'):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
            update_size: number of bytes in update
        """
        
        
        
        
        
        

        self.model.train()

        from meters import AverageMeter, accuracy
        losses = AverageMeter()
        top1 = AverageMeter()
        

        
        

        
        

        

        data_loader = cifar10_loaders('train', self.train_data)
        
        

        

        init_para = copy.deepcopy(self.model.state_dict())

        for num_epoch in range(num_epochs):
            for i, (inputs, target) in enumerate(data_loader):
                
                
                
                inputs = inputs.to('cuda:0')
                target = target.to('cuda:0')

                
                output = self.model(inputs)
                self.criterion.to('cuda:0')
                if model_type == 'int':
                    
                    output, output_exp = output
                    output = output.float()
                    loss = self.criterion(output*(2**output_exp.float()), target)
                else:
                    output_exp = 0
                    loss = self.criterion(output, target)

                
                losses.update(float(loss), inputs.size(0))
                
                
                
                
                


                if model_type == 'int':
                    self.model.backward(target)
                else:
                    self.optimizer.update(epoch, epoch * len(data_loader) + i)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        

        
        update = self.model.state_dict()

        
        

        
        

        
        
            

        
        return losses.avg, update
        


    def test(self, batch_size=5, set_to_use='test'):
        self.model.eval()
        

        from meters import AverageMeter, accuracy
        losses = AverageMeter()
        top1 = AverageMeter()

        import copy
        para = copy.deepcopy(self.model.state_dict())
        
        

        if set_to_use == 'train':
            data_loader = cifar10_loaders('train', self.train_data)
            
        elif set_to_use == 'test':
            data_loader = cifar10_loaders('test', self.test_data)
            

        for i, (inputs, target) in enumerate(data_loader):
            
            
            inputs = inputs.to('cuda:0')
            target = target.to('cuda:0')

            
            
            output = self.model(inputs)

            if self.model_type == 'int':
                
                output, output_exp = output
                output = output.float()
                loss = self.criterion(output*(2**output_exp.float()), target)
            else:
                output_exp = 0
                loss = self.criterion(output, target)

            
            losses.update(float(loss), inputs.size(0))
            
            prec1, _ = accuracy(output.detach(), target, topk=(1, 2))
            top1.update(float(prec1), inputs.size(0))
        
        return top1


    def forward(self, data, model, criterion, epoch, training, optimizer=None):
        if training:
            model.train()
        else:
            model.eval()

        from meters import AverageMeter, accuracy

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        

        

        for i, (inputs, target) in enumerate(data):
            
            
            inputs = inputs.to('cuda:0')
            target = target.to('cuda:0')

            
            output = model(inputs) 
            if model_type == 'int':
                
                output, output_exp = output
                output = output.float()
                loss = criterion(output*(2**output_exp.float()), target)
            else:
                output_exp = 0
                loss = criterion(output, target)

            
            losses.update(float(loss), inputs.size(0))
            prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
            top1.update(float(prec1), inputs.size(0))
            top5.update(float(prec5), inputs.size(0))

            if training:
                if model_type == 'int':
                    model.backward(target)

                elif model_type == 'hybrid':
                    
                    optimizer.update(epoch, epoch * len(data_loader) + i)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    model.backward()
                else:
                    optimizer.update(epoch, epoch * len(data_loader) + i)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_interval == 0 and training:
                logging.info('{model_type} [{0}][{1}/{2}] '
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Data {data_time.val:.2f} '
                            'loss {loss.val:.3f} ({loss.avg:.3f}) '
                            'e {output_exp:d} '
                            '@1 {top1.val:.3f} ({top1.avg:.3f}) '
                            '@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                epoch, i, len(data_loader),
                                model_type=model_type,
                                batch_time=batch_time,
                                data_time=data_time,
                                loss=losses,
                                output_exp=output_exp,
                                top1=top1, top5=top5))

                if args.grad_hist:
                    if args.model_type == 'int':
                        for idx, l in enumerate(model.forward_layers):
                            if hasattr(l,'weight'):
                                grad = l.grad_int32acc
                                writer.add_histogram('Grad/'+l.__class__.__name__ +'_'+str(idx), grad, epoch*total_steps+i)

                    elif args.model_type == 'float':
                        for idx, l in enumerate(model.layers):
                            if hasattr(l,'weight'):
                                writer.add_histogram('Grad/'+l.__class__.__name__ +'_'+str(idx), l.weight.grad, epoch*total_steps+i)
                        for idx, l in enumerate(model.classifier):
                            if hasattr(l,'weight'):
                                writer.add_histogram('Grad/'+l.__class__.__name__ +'_'+str(idx), l.weight.grad, epoch*total_steps+i)

        return losses.avg, top1.avg, top5.avg
