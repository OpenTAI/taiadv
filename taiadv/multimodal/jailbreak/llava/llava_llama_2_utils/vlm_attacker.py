import torch
from tqdm import tqdm
import random
from llava_llama_2_utils import prompt_wrapper, generator
from torchvision.utils import save_image


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
import seaborn as sns
from minimal_gcg.opt_utils import token_gradients, sample_control
from minimal_gcg.opt_utils import load_model_and_tokenizer, get_filtered_cands
import copy
import gc


def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images


class Attacker:

    def __init__(self, args, model, tokenizer, queries,targets, device='cuda:0', is_rtp=False, image_processor=None):

        self.args = args
        self.model = model
        self.tokenizer= tokenizer
        self.device = device
        self.is_rtp = is_rtp
        self.queries=queries
        self.targets = targets
        self.num_targets = len(targets)

        self.loss_buffer = []

        # freeze and set to eval model:
        self.model.eval()
        self.model.requires_grad_(False)

        self.image_processor = image_processor


    def attack_unconstrained(self, img, batch_size = 8, adv_string_init ='! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !',num_iter=2000, alpha=1/255,ratio=10):
        print('>>> batch_size:', batch_size)

        my_generator = generator.Generator(model=self.model, tokenizer=self.tokenizer)


        adv_noise = denormalize(img).clone().to(self.device)
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()
        
        adv_suffix = adv_string_init

        for t in tqdm(range(num_iter + 1)):


            batch_queries_targets = random.sample(list(zip(self.queries,self.targets)), batch_size)
            separator = ' | '
            batch_queries=[t[0]+separator for t in batch_queries_targets]#list of str
            
            batch_queries_adv=[t[0]+separator+adv_suffix for t in batch_queries_targets]
            
            batch_targets = [t[1] for t in batch_queries_targets]
            

            text_prompt_adv=[prompt_wrapper.prepare_text_prompt(q) for q in batch_queries_adv]

            prompt_adv = prompt_wrapper.Prompt( self.model, self.tokenizer, text_prompts=text_prompt_adv, device=self.device )


            x_adv = normalize(adv_noise)


            target_loss,inputs_embeds = self.attack_loss(prompt_adv, x_adv, batch_targets)   
            target_loss.backward()
            

            if t%ratio==0:
                with torch.no_grad():
                    embedding_layer = self.model.get_input_embeddings()
                    adv_suffix_ids=torch.tensor(self.tokenizer(adv_suffix, add_special_tokens=False).input_ids).cuda()
                    adv_embeds = embedding_layer(adv_suffix_ids)

                def find_pattern_start(sample, pattern):
                    seq_len = sample.shape[0]
                    pattern_len = pattern.shape[0]
                    min_dist = float('inf')
                    start_idx = 0
                

                    for i in range(seq_len - pattern_len + 1):
                        segment = sample[i:i + pattern_len, :]
                        dist = torch.norm(segment - pattern, dim=1).sum().item()
                        if dist < min_dist:
                            min_dist = dist
                            start_idx = i
                    return start_idx                 
                

                adv_tokens_grad = torch.zeros(batch_size, adv_embeds.shape[0], adv_embeds.shape[1]).cuda()
    
                
                for i in range(batch_size):
                    start_idx = find_pattern_start(inputs_embeds[i], adv_embeds)

                    adv_tokens_grad[i] = inputs_embeds.grad[i, start_idx:start_idx + adv_embeds.shape[0], :] 
                     
                adv_tokens_grad = adv_tokens_grad.mean(dim=0, keepdim=True) 

                text_grad=token_gradients(self.model, adv_suffix, self.tokenizer,adv_tokens_grad)#有问题么

                with torch.no_grad():
                    new_adv_suffix_toks=sample_control(torch.tensor(self.tokenizer(adv_suffix, add_special_tokens=False).input_ids),text_grad,
                                                batch_size=250, topk=256)
                    candidates = get_filtered_cands(self.tokenizer,
                                                        new_adv_suffix_toks,
                                                        filter_cand=True,
                                                        curr_control=adv_suffix)
                    curr_best_loss = 9999
                    curr_best_trigger = None
    
                    for cand in candidates:
                     
                        cand_text_prompts = [(batch_queries[i] + cand) for i in
                                        range(batch_size)]
                        cand_prepared_prompts=[prompt_wrapper.prepare_text_prompt(q) for q in cand_text_prompts]
                        cand_prompt = prompt_wrapper.Prompt(self.model, self.tokenizer, text_prompts=cand_prepared_prompts, device=self.device)

                        next_target_loss,_ = self.attack_loss(cand_prompt, x_adv, batch_targets)
                        curr_loss = next_target_loss
                        if curr_loss < curr_best_loss:
                            curr_best_loss = curr_loss
                            curr_best_trigger = cand
                        

                # Update overall best if the best current candidate is better
                print("curr_best_loss",curr_best_loss)
                print("target_loss",target_loss)
                if curr_best_loss < target_loss:
    
                    adv_suffix = curr_best_trigger
                    
                del embedding_layer,inputs_embeds,adv_tokens_grad,text_grad, new_adv_suffix_toks ; gc.collect()

            
            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(0, 1)
            adv_noise.grad.zero_()
            self.model.zero_grad()

            self.loss_buffer.append(target_loss.item())

            print("target_loss: %f" % (
                target_loss.item())
                  )


            if t % 100 == 0:
                print('######### Output - Iter = %d ##########' % t)

                
                adv_img_prompt = denormalize(x_adv).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))
                print("adv_suffix:",adv_suffix)

                with open("adv_suffix.txt", "a") as file:

                    file.write(adv_suffix + "\n")
                    
        return adv_img_prompt


    def plot_loss(self):

        sns.set_theme()
        num_iters = len(self.loss_buffer)

        x_ticks = list(range(0, num_iters))

        # Plot and label the training and validation loss values
        plt.plot(x_ticks, self.loss_buffer, label='Target Loss')

        # Add in a title and axes labels
        plt.title('Loss Plot')
        plt.xlabel('Iters')
        plt.ylabel('Loss')

        # Display the plot
        plt.legend(loc='best')
        plt.savefig('%s/loss_curve.png' % (self.args.save_dir))
        plt.clf()

        torch.save(self.loss_buffer, '%s/loss' % (self.args.save_dir))

    def attack_loss(self, prompts, images, targets):

        context_length = prompts.context_length
        context_input_ids = prompts.input_ids
        batch_size = len(targets)

        if len(context_input_ids) == 1:
            context_length = context_length * batch_size
            context_input_ids = context_input_ids * batch_size

        images = images.repeat(batch_size, 1, 1, 1)

        assert len(context_input_ids) == len(targets), f"Unmathced batch size of prompts and targets {len(context_input_ids)} != {len(targets)}"


        to_regress_tokens = [ torch.as_tensor([item[1:]]).cuda() for item in self.tokenizer(targets).input_ids] # get rid of the default <bos> in targets tokenization.


        seq_tokens_length = []
        labels = []
        input_ids = []

        for i, item in enumerate(to_regress_tokens):

            L = item.shape[1] + context_length[i]
            seq_tokens_length.append(L)

            context_mask = torch.full([1, context_length[i]], -100,
                                      dtype=to_regress_tokens[0].dtype,
                                      device=to_regress_tokens[0].device)
            labels.append( torch.cat( [context_mask, item], dim=1 ) )
            input_ids.append( torch.cat( [context_input_ids[i], item], dim=1 ) )

        # padding token
        pad = torch.full([1, 1], 0,
                         dtype=to_regress_tokens[0].dtype,
                         device=to_regress_tokens[0].device).cuda() # it does not matter ... Anyway will be masked out from attention...


        max_length = max(seq_tokens_length)
        attention_mask = []

        for i in range(batch_size):

            # padding to align the length
            num_to_pad = max_length - seq_tokens_length[i]

            padding_mask = (
                torch.full([1, num_to_pad], -100,
                       dtype=torch.long,
                       device=self.device)
            )
            labels[i] = torch.cat( [labels[i], padding_mask], dim=1 )

            input_ids[i] = torch.cat( [input_ids[i],
                                       pad.repeat(1, num_to_pad)], dim=1 )
            attention_mask.append( torch.LongTensor( [ [1]* (seq_tokens_length[i]) + [0]*num_to_pad ] ) )

        labels = torch.cat( labels, dim=0 ).cuda()
        input_ids = torch.cat( input_ids, dim=0 ).cuda()
        attention_mask = torch.cat(attention_mask, dim=0).cuda()

        outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                labels=labels,
                images=images.half(),
            )
        #loss = outputs.loss
        loss,inputs_embeds = outputs[0].loss,outputs[1]

        return loss,inputs_embeds