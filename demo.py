import uuid
import time
import torch
import numpy as np
import os
from collections import OrderedDict
import pandas as pd
import pickle
from config import CONFIG, PARAMS
from vae import VAE1, VAE2, VAE3
import helper
from loss import calculate_contrastive_loss, calculate_reconstruction_loss, calculate_SNL_loss


# Setting up working environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#These two options shoul be seed to ensure reproducible (If you are using cudnn backend)
#https://pytorch.org/docs/stable/notes/randomness.html
#We used 35888 as the seed when we conducted experiments
np.random.seed(35813)
torch.manual_seed(35813)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
MODEL_WEIGHT_BACKUP_PATH = "./output"
TEMP_FOLDER = "./temp"
work_path = MODEL_WEIGHT_BACKUP_PATH + "/" + CONFIG["model_name"]


if not os.path.exists(MODEL_WEIGHT_BACKUP_PATH):
    os.makedirs(MODEL_WEIGHT_BACKUP_PATH)
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)
if not os.path.exists(work_path):
    os.makedirs(work_path)
with open(work_path + "model_params.txt", 'w') as f:
    print(PARAMS, file=f)
with open(work_path + "model_config.txt", 'w') as f:
    print(CONFIG, file=f)

latent_dim = CONFIG['latent_dim']



class Client:
    def __init__(self, model, optimizer, params, train_data):
        self.params = params
        self.model = model
        self.optimizer = optimizer
        self.train_data = train_data

    def update_weights(self, data, freezed, ablated):
        data = data.to(device)
        template, recon = self.model(data)


        # Calculate SNL loss. i= batch_size, n=sampled_size
        sampled_embeddings = []
        for i in range(template.shape[0]):
            sampled_image_ids = helper.random_sampling(self.train_data, self.params['sample size'])

            sampled_image = torch.stack([self.train_data[i].to(device) for i in sampled_image_ids], dim=3).permute(3, 0, 1, 2)
            sampled_embeddings.append(self.model.enc(sampled_image))  # n*latent_dim
        sampled_embeddings = torch.stack(sampled_embeddings, dim=0)  # i*n*latent_dim

        expanded_template = template.expand(self.params['sample size'], template.shape[0], latent_dim)
        expanded_template = expanded_template.permute(1,0,2)  # i*n*latent_dim

        SNL_loss = calculate_SNL_loss(sampled_embeddings, expanded_template)

        # Calculate contrastive loss
        contrastive_loss = calculate_contrastive_loss(sampled_embeddings, expanded_template, True, margin=self.params['margin'] )

        # Calculate reconstruction loss
        reconstruction_loss = calculate_reconstruction_loss(data, recon)

        # Sum up three losses
        if ablated:
            loss = reconstruction_loss + self.params['lambda1']*SNL_loss 
        else:
            loss = reconstruction_loss + self.params['lambda1']*SNL_loss + self.params['lambda2']*contrastive_loss

        SNL_loss = SNL_loss.detach().cpu().clone().tolist()
        contrastive_loss = contrastive_loss.detach().cpu().clone().tolist()
        reconstruction_loss = reconstruction_loss.detach().cpu().clone().tolist()

        #backprop
        if freezed:
            pass
        else:
            self.optimizer.zero_grad()
            losses = torch.sum(loss)
            losses.backward()
            self.optimizer.step()
        
        return reconstruction_loss, SNL_loss, contrastive_loss
    
    def get_weights(self):
        return self.model.state_dict()
    
    def set_weights(self, state_dict):
        # current_state_dict = self.model.enc.state_dict()
        # batchnorm_keys = helper.batchnorm_keys(self.model.enc)

        updated_state_dict = {}
        for key,_ in state_dict.items():
            if key in self.model.state_dict().keys():
                updated_state_dict.update({key:_})
            else:
                pass
        
        # for key in batchnorm_keys:
        #     updated_state_dict.update({key: current_state_dict[key]})

        self.model.load_state_dict(updated_state_dict)
    



        
class Server:
    def __init__(self, clients):
        self.clients = clients

    def broadcast_weights(self):
        avg_state_dict = self.average_weights()
        for client in self.clients:
            client.set_weights(avg_state_dict)

    def average_weights(self):
        """
        Average the weights of the models from all clients for the common layers
        """
        # Get the state_dict from each client's model
        state_dicts = [client.model.state_dict() for client in self.clients]
        

        # Get keys from all state dict
        keys = set()
        for client in state_dicts:
            keys.update(set(client.keys()))
        
        # # Remove batchnorm keys
        # for client in self.clients:
        #     batchnorm_keys = helper.batchnorm_keys(client.model.enc)
        #     keys = keys - batchnorm_keys

        # Initialize a new state_dict to store the average weights
        average_state_dict = OrderedDict()

        # For each key (layer) in the state_dict
        for key in keys:
            # Get the list of state_dicts for this layer from clients that have this layer
            current_layer_state_dicts = [client[key].float() for client in state_dicts if key in client]

            # If no clients have this layer, skip it
            if not current_layer_state_dicts:
                continue

            # Average the weights from all clients for this layer
            average_state_dict[key] = torch.stack(current_layer_state_dicts).mean(dim=0)

        return average_state_dict
        


def train(model_id, thin_train, thin_test, thic_train, thic_test, raw_train, raw_test, swel_train, swel_test, frac_train, frac_test, is_federated, ablated):

    if is_federated:
        print("FEDERATION ")
        if ablated:
            print("Ablated")
            save_path = work_path + "/" + "ablated_fed"
        else:
            print("constrastive centeredness loss")
            save_path = work_path + "/" + "contras_fed"
    else:
        print("NO FEDERATION ")
        if ablated:
            print("Ablated")
            save_path = work_path + "/" + "ablated_nofed"
        else:
            print("constrastive centeredness loss")
            save_path = work_path + "/" + "contras_nofed"

    
    # cbt_path1 = save_path + "/" + "client1_cbts" + "/"
    # cbt_path2 = save_path + "/" + "client2_cbts" + "/"
    # cbt_path3 = save_path + "/" + "client3_cbts" + "/"
    # cbt_path4 = save_path + "/" + "client4_cbts" + "/"
    image_template_path = save_path + "/" + "image_templates" + "/"
    embedding_template_path = save_path + "/" + "embedding_templates" + "/"
    loss_path = save_path + "/" + "loss" +"/"
    eval_path = save_path + "/" + "eval" +"/"
    model_path = save_path + "/" + "model" +"/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(image_template_path):
        os.makedirs(image_template_path)
    if not os.path.exists(embedding_template_path):
        os.makedirs(embedding_template_path)
    # if not os.path.exists(cbt_path2):
    #     os.makedirs(cbt_path2)
    # if not os.path.exists(cbt_path3):
    #     os.makedirs(cbt_path3)
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    total_epochs = CONFIG['fed_rounds']*CONFIG['epochs_per_round']

    # create trainset and testset loader
    thin_train_loader = helper.get_loader(thin_train, CONFIG['batch_size'])
    thin_test_loader = helper.get_loader(thin_test, CONFIG['batch_size'])

    thic_train_loader = helper.get_loader(thic_train, CONFIG['batch_size'])
    thic_test_loader = helper.get_loader(thic_test, CONFIG['batch_size'])

    raw_train_loader = helper.get_loader(raw_train, CONFIG['batch_size'])
    raw_test_loader = helper.get_loader(raw_test, CONFIG['batch_size'])

    swel_train_loader = helper.get_loader(swel_train, CONFIG['batch_size'])
    swel_test_loader = helper.get_loader(swel_test, CONFIG['batch_size'])

    frac_train_loader = helper.get_loader(frac_train, CONFIG['batch_size'])
    frac_test_loader = helper.get_loader(frac_test, CONFIG['batch_size'])

    

    # create vae model
    thin_model = VAE1(CONFIG['latent_dim']).to(device)
    thic_model = VAE1(CONFIG['latent_dim']).to(device)
    raw_model = VAE2(CONFIG['latent_dim']).to(device)
    swel_model = VAE3(CONFIG['latent_dim']).to(device)
    frac_model = VAE3(CONFIG['latent_dim']).to(device)

    # initialise optimizer
    thin_optimizer = torch.optim.AdamW(thin_model.parameters(), lr= PARAMS['thin']["learning rate"], weight_decay= 0.001)
    thic_optimizer = torch.optim.AdamW(thic_model.parameters(), lr= PARAMS['thic']["learning rate"], weight_decay= 0.001)
    raw_optimizer = torch.optim.AdamW(raw_model.parameters(), lr= PARAMS['raw']["learning rate"], weight_decay= 0.001)
    swel_optimizer = torch.optim.AdamW(swel_model.parameters(), lr= PARAMS['swel']["learning rate"], weight_decay= 0.001)
    frac_optimizer = torch.optim.AdamW(frac_model.parameters(), lr= PARAMS['frac']["learning rate"], weight_decay= 0.001)

    # initilise client
    client1 = Client(thin_model, thin_optimizer, PARAMS['thin'], thin_train)
    client2 = Client(thic_model, thic_optimizer, PARAMS['thic'], thic_train)
    client3 = Client(raw_model, raw_optimizer, PARAMS['raw'], raw_train)
    client4 = Client(swel_model, swel_optimizer, PARAMS['swel'], swel_train)
    client5 = Client(frac_model, frac_optimizer, PARAMS['frac'], frac_train)

    # initialise server
    server = Server([client1, client2, client3, client4, client5])


    test_errors1 = []
    test_errors2 = []
    test_errors3 = []
    test_errors4 = []
    test_errors5 = []

    freezed1 = False
    freezed2 = False
    freezed3 = False
    freezed4 = False
    freezed5 = False

    epoch_log = []
    EPOCH_log = []

    # client1 
    recon_loss_log1 = []
    SNL_loss_log1 = []
    contra_loss_log1 = []

    recon_eval_log1 = []
    local_center_eval_log1 = []

    # client2 
    recon_loss_log2 = []
    SNL_loss_log2 = []
    contra_loss_log2 = []

    recon_eval_log2 = []
    local_center_eval_log2 = []

    # client3
    recon_loss_log3 = []
    SNL_loss_log3 = []
    contra_loss_log3 = []

    recon_eval_log3 = []
    local_center_eval_log3 = []

    # client4
    recon_loss_log4 = []
    SNL_loss_log4 = []
    contra_loss_log4 = []

    recon_eval_log4 = []
    local_center_eval_log4 = []

    # client5 
    recon_loss_log5 = []
    SNL_loss_log5 = []
    contra_loss_log5 = []

    recon_eval_log5 = []
    local_center_eval_log5 = []

    # global center
    global_center_eval_log = []

    tick = time.time()
    for epoch in range(total_epochs):
        epoch_log.append(epoch)
        client1.model.train()
        client2.model.train()
        client3.model.train()
        client4.model.train()
        client5.model.train()

        recon_losses1 = []
        SNL_losses1 = []
        contra_losses1 = []

        recon_losses2 = []
        SNL_losses2 = []
        contra_losses2 = []

        recon_losses3 = []
        SNL_losses3 = []
        contra_losses3 = []

        recon_losses4 = []
        SNL_losses4 = []
        contra_losses4 = []

        recon_losses5 = []
        SNL_losses5 = []
        contra_losses5 = []

        for batch_idx, (data1, data2, data3, data4, data5) in enumerate(zip(thin_train_loader, thic_train_loader, raw_train_loader, swel_train_loader, frac_train_loader)):
            reconstruction_loss1, SNL_loss1, contrastive_loss1 = client1.update_weights(data1, freezed1, ablated)
            torch.cuda.empty_cache()
            reconstruction_loss2, SNL_loss2, contrastive_loss2 = client2.update_weights(data2, freezed2, ablated)
            torch.cuda.empty_cache()
            reconstruction_loss3, SNL_loss3, contrastive_loss3 = client3.update_weights(data3, freezed3, ablated)
            torch.cuda.empty_cache()
            reconstruction_loss4, SNL_loss4, contrastive_loss4 = client4.update_weights(data4, freezed4, ablated)
            torch.cuda.empty_cache()
            reconstruction_loss5, SNL_loss5, contrastive_loss5 = client5.update_weights(data5, freezed5, ablated)
            torch.cuda.empty_cache()

            recon_losses1.append(reconstruction_loss1)
            recon_losses2.append(reconstruction_loss2)
            recon_losses3.append(reconstruction_loss3)
            recon_losses4.append(reconstruction_loss4)
            recon_losses5.append(reconstruction_loss5)
            
            # recon_losses1.extend(reconstruction_loss1)
            # recon_losses2.extend(reconstruction_loss2)
            # recon_losses3.extend(reconstruction_loss3)
            # recon_losses4.extend(reconstruction_loss4)
            # recon_losses5.extend(reconstruction_loss5)

            # SNL_losses1.extend(SNL_loss1)
            # SNL_losses2.extend(SNL_loss2)
            # SNL_losses3.extend(SNL_loss3)
            # SNL_losses4.extend(SNL_loss4)
            # SNL_losses5.extend(SNL_loss5)

            SNL_losses1.append(SNL_loss1)
            SNL_losses2.append(SNL_loss2)
            SNL_losses3.append(SNL_loss3)
            SNL_losses4.append(SNL_loss4)
            SNL_losses5.append(SNL_loss5)

            contra_losses1.extend(contrastive_loss1)
            contra_losses2.extend(contrastive_loss2)
            contra_losses3.extend(contrastive_loss3)
            contra_losses4.extend(contrastive_loss4)
            contra_losses5.extend(contrastive_loss5)
        
        # track client loss
        recon_loss_log1.append(sum(recon_losses1)/len(recon_losses1))
        recon_loss_log2.append(sum(recon_losses2)/len(recon_losses2))
        recon_loss_log3.append(sum(recon_losses3)/len(recon_losses3))
        recon_loss_log4.append(sum(recon_losses4)/len(recon_losses4))
        recon_loss_log5.append(sum(recon_losses5)/len(recon_losses5))

        SNL_loss_log1.append(sum(SNL_losses1)/len(SNL_losses1))
        SNL_loss_log2.append(sum(SNL_losses2)/len(SNL_losses2))
        SNL_loss_log3.append(sum(SNL_losses3)/len(SNL_losses3))
        SNL_loss_log4.append(sum(SNL_losses4)/len(SNL_losses4))
        SNL_loss_log5.append(sum(SNL_losses5)/len(SNL_losses5))

        contra_loss_log1.append(sum(contra_losses1)/len(contra_losses1))
        contra_loss_log2.append(sum(contra_losses2)/len(contra_losses2))
        contra_loss_log3.append(sum(contra_losses3)/len(contra_losses3))
        contra_loss_log4.append(sum(contra_losses4)/len(contra_losses4))
        contra_loss_log5.append(sum(contra_losses5)/len(contra_losses5))

        if is_federated:
            if epoch % CONFIG['epochs_per_round'] == 0:
                server.broadcast_weights()

        

        # if epoch % 5 == 0:
        #     for batch_idx, (data1, data2, data3, data4, data5) in enumerate(zip(thin_test_loader, thic_test_loader, raw_test_loader, swel_test_loader, frac_test_loader)):
        #         if batch_idx == 0:
        #             data1 = data1.to(device)
        #             data2 = data2.to(device)
        #             data3 = data3.to(device)
        #             data4 = data4.to(device)
        #             data5 = data5.to(device)
        #             z1, recon1 = client1.model(data1)
        #             z2, recon2 = client2.model(data2)
        #             z3, recon3 = client3.model(data3)
        #             z4, recon4 = client4.model(data4)
        #             z5, recon5 = client5.model(data5)

        #             from morphomnist.util import plot_grid, plot_digit
        #             import matplotlib.pyplot as plt
        #             plot_digit(recon1[0].detach().cpu().numpy())
        #             plt.show()
        #             plot_digit(recon2[0].detach().cpu().numpy())
        #             plt.show()
        #             plot_digit(recon3[0].detach().cpu().numpy())
        #             plt.show()
        #             plot_digit(recon4[0].detach().cpu().numpy())
        #             plt.show()
        #             plot_digit(recon5[0].detach().cpu().numpy())
        #             plt.show()

        # Evaluation
        if epoch % 1 == 0: 
            EPOCH_log.append(epoch)

            template1, _ = helper.generate_template_median(client1.model, thin_train_loader)
            template2, _ = helper.generate_template_median(client2.model, thic_train_loader)
            template3, _ = helper.generate_template_median(client3.model, raw_train_loader)
            template4, _ = helper.generate_template_median(client4.model, swel_train_loader)
            template5, _ = helper.generate_template_median(client5.model, frac_train_loader)

            # eval reconstruction
            recon_eval1 = helper.evaluate_reconstruction(client1.model, thin_test_loader)
            recon_eval2 = helper.evaluate_reconstruction(client2.model, thic_test_loader)
            recon_eval3 = helper.evaluate_reconstruction(client3.model, raw_test_loader)
            recon_eval4 = helper.evaluate_reconstruction(client4.model, swel_test_loader)
            recon_eval5 = helper.evaluate_reconstruction(client5.model, frac_test_loader)

            recon_eval_log1.append(recon_eval1)
            recon_eval_log2.append(recon_eval2)
            recon_eval_log3.append(recon_eval3)
            recon_eval_log4.append(recon_eval4)
            recon_eval_log5.append(recon_eval5)

            # eval local_centeredness
            local_center_eval1 = helper.evaluate_local_centeredness(client1.model, thin_test_loader, template1)
            local_center_eval2 = helper.evaluate_local_centeredness(client2.model, thic_test_loader, template2)
            local_center_eval3 = helper.evaluate_local_centeredness(client3.model, raw_test_loader, template3)
            local_center_eval4 = helper.evaluate_local_centeredness(client4.model, swel_test_loader, template4)
            local_center_eval5 = helper.evaluate_local_centeredness(client5.model, frac_test_loader, template5)

            local_center_eval_log1.append(local_center_eval1)
            local_center_eval_log2.append(local_center_eval2)
            local_center_eval_log3.append(local_center_eval3)
            local_center_eval_log4.append(local_center_eval4)
            local_center_eval_log5.append(local_center_eval5)

            # evaluate global centeredness
            client_templates = [template1, template2, template3, template4, template5]
            global_center_eval = helper.evaluate_global_centeredness(client_templates)

            global_center_eval_log.append(global_center_eval)

            tock = time.time()
            time_elapsed = tock - tick
            tick = tock

            test_errors1.append(float(local_center_eval1))
            test_errors2.append(float(local_center_eval2))
            test_errors3.append(float(local_center_eval3))
            test_errors4.append(float(local_center_eval4))
            test_errors5.append(float(local_center_eval5))

            print("Epoch: {}  | Local Centeredness: {:.3f} ; {:.3f} ; {:.3f} ; {:.3f} ; {:.3f} | Reconstruction: {:.3f} ; {:.3f} ; {:.3f} ; {:.3f} ; {:.3f} | Global Centeredness: {:.3f} | Time Elapsed: {:.2f}  |".format(
                epoch, local_center_eval1, local_center_eval2, local_center_eval3, local_center_eval4, local_center_eval5, recon_eval1, recon_eval2, recon_eval3, recon_eval4, recon_eval5, global_center_eval, time_elapsed))

            #  #Freeze client control
            # if len(test_errors1) > 6:
            #     torch.save(client1.model.state_dict(), TEMP_FOLDER + "/weight_"+ model_id + "_" + "client1" + "_" + str(local_center_eval1)[:5]  + ".model")
            #     last_6 = test_errors1[-6:]
            #     if CONFIG['freeze'] and not freezed1:
            #         if(all(last_6[i] < last_6[i + 1] for i in range(5))):
            #             print(f"Freeze model1")
            #             freezed1 = True
                        
            
            # if len(test_errors2) > 6:
            #     torch.save(client2.model.state_dict(), TEMP_FOLDER + "/weight_"+ model_id + "_" + "client2" + "_" + str(local_center_eval2)[:5]  + ".model")
            #     last_6 = test_errors2[-6:]
            #     if CONFIG['freeze'] and not freezed2:
            #         if(all(last_6[i] < last_6[i + 1] for i in range(5))):
            #             print(f"Freeze model2")
            #             freezed2 = True

            # if len(test_errors3) > 6:
            #     torch.save(client3.model.state_dict(), TEMP_FOLDER + "/weight_"+ model_id + "_" + "client3" + "_" + str(local_center_eval3)[:5]  + ".model")
            #     last_6 = test_errors3[-6:]
            #     if CONFIG['freeze'] and not freezed3:
            #         if(all(last_6[i] < last_6[i + 1] for i in range(5))):
            #             print(f"Freeze model3")
            #             freezed3 = True
            
            # if len(test_errors4) > 6:
            #     torch.save(client4.model.state_dict(), TEMP_FOLDER + "/weight_"+ model_id + "_" + "client4" + "_" + str(local_center_eval4)[:5]  + ".model")
            #     last_6 = test_errors4[-6:]
            #     if CONFIG['freeze'] and not freezed4:
            #         if(all(last_6[i] < last_6[i + 1] for i in range(5))):
            #             print(f"Freeze model3")
            #             freezed4 = True
            
            # if len(test_errors5) > 6:
            #     torch.save(client5.model.state_dict(), TEMP_FOLDER + "/weight_"+ model_id + "_" + "client5" + "_" + str(local_center_eval5)[:5]  + ".model")
            #     last_6 = test_errors5[-6:]
            #     if CONFIG['freeze'] and not freezed5:
            #         if(all(last_6[i] < last_6[i + 1] for i in range(5))):
            #             print(f"Freeze model3")
            #             freezed5 = True

    # save client training loss
    client1_loss = {'epoch': epoch_log,
                    'SNL': SNL_loss_log1,
                    'contrastive': contra_loss_log1,
                    'reconstruction': recon_loss_log1,
                    }
    
    client2_loss = {'epoch': epoch_log,
                    'SNL': SNL_loss_log2,
                    'contrastive': contra_loss_log2,
                    'reconstruction': recon_loss_log2,
                    }
    
    client3_loss = {'epoch': epoch_log,
                    'SNL': SNL_loss_log3,
                    'contrastive': contra_loss_log3,
                    'reconstruction': recon_loss_log3,
                    }
    
    client4_loss = {'epoch': epoch_log,
                    'SNL': SNL_loss_log4,
                    'contrastive': contra_loss_log4,
                    'reconstruction': recon_loss_log4,
                    }
    
    client5_loss = {'epoch': epoch_log,
                    'SNL': SNL_loss_log5,
                    'contrastive': contra_loss_log5,
                    'reconstruction': recon_loss_log5,
                    }
    
    client_loss = {'client1': client1_loss,
                   'client2': client2_loss,
                   'client3': client3_loss,
                   'client4': client4_loss,
                   'client5': client5_loss,
                   }
    
    with open(f'{loss_path}/client_loss_log.pkl', 'wb') as file:
                pickle.dump(client_loss, file)
    
    # save client test eval
    client1_eval = {'epoch': EPOCH_log,
                    'local_centeredness': local_center_eval_log1,
                    'reconstruction': recon_eval_log1}
    
    client2_eval = {'epoch': EPOCH_log,
                    'local_centeredness': local_center_eval_log2,
                    'reconstruction': recon_eval_log2}
    
    client3_eval = {'epoch': EPOCH_log,
                    'local_centeredness': local_center_eval_log3,
                    'reconstruction': recon_eval_log3}
    
    client4_eval = {'epoch': EPOCH_log,
                    'local_centeredness': local_center_eval_log4,
                    'reconstruction': recon_eval_log4}
    
    client5_eval = {'epoch': EPOCH_log,
                    'local_centeredness': local_center_eval_log5,
                    'reconstruction': recon_eval_log5}
    
    client_eval = {'client1': client1_eval,
                   'client2': client2_eval,
                   'client3': client3_eval,
                   'client4': client4_eval,
                   'client5': client5_eval
                   }

    with open(f'{eval_path}/client_eval_log.pkl', 'wb') as file:
                pickle.dump(client_eval, file)

    # save server test eval
    server_eval = {'epoch': EPOCH_log,
                   'global_centeredness': global_center_eval_log}
    with open(f'{eval_path}/server_eval_log.pkl', 'wb') as file:
                pickle.dump(server_eval, file)

    #Restore best model so far
    try:
        restore1 = "./temp/weight_" + model_id + "_" +  "client1" + "_" + str(min(test_errors1))[:5] + ".model"
        client1.model.load_state_dict(torch.load(restore1))
    except:
        pass

    try:
        restore2 = "./temp/weight_" + model_id + "_" + "client2" + "_" + str(min(test_errors2))[:5] + ".model"
        client2.model.load_state_dict(torch.load(restore2))
    except:
        pass
    
    try:
        restore3 = "./temp/weight_" + model_id + "_" + "client3" + "_" + str(min(test_errors3))[:5] + ".model"
        client3.model.load_state_dict(torch.load(restore3))
    except:
        pass

    try:
        restore4 = "./temp/weight_" + model_id + "_" + "client4" + "_" + str(min(test_errors4))[:5] + ".model"
        client4.model.load_state_dict(torch.load(restore4))
    except:
        pass

    try:
        restore5 = "./temp/weight_" + model_id + "_" + "client5" + "_" + str(min(test_errors5))[:5] + ".model"
        client5.model.load_state_dict(torch.load(restore5))
    except:
        pass

    torch.save(client1.model.state_dict(), model_path + "client1" + ".model")
    torch.save(client2.model.state_dict(), model_path + "client2" + ".model")
    torch.save(client3.model.state_dict(), model_path + "client3" + ".model")
    torch.save(client4.model.state_dict(), model_path + "client4" + ".model")
    torch.save(client5.model.state_dict(), model_path + "client5"  + ".model")


    # Generate and save refined CBT
    template1, _ = helper.generate_template_median(client1.model, thin_train_loader) # (1, latent_dim)
    template2, _ = helper.generate_template_median(client2.model, thic_train_loader)
    template3, _ = helper.generate_template_median(client3.model, raw_train_loader)
    template4, _ = helper.generate_template_median(client4.model, swel_train_loader)
    template5, _ = helper.generate_template_median(client5.model, frac_train_loader)

    np.save(embedding_template_path +  "client1_embedding_template", template1.detach().cpu().numpy())
    np.save(embedding_template_path +  "client2_embedding_template", template2.detach().cpu().numpy())
    np.save(embedding_template_path +  "client3_embedding_template", template3.detach().cpu().numpy())
    np.save(embedding_template_path +  "client4_embedding_template", template4.detach().cpu().numpy())
    np.save(embedding_template_path +  "client5_embedding_template", template5.detach().cpu().numpy())

    image_template1 = helper.generate_image_template(client1.model, template1)
    image_template2 = helper.generate_image_template(client2.model, template2)
    image_template3 = helper.generate_image_template(client3.model, template3)
    image_template4 = helper.generate_image_template(client4.model, template4)
    image_template5 = helper.generate_image_template(client5.model, template5)

    np.save(image_template_path +  "client1_image_template", image_template1)
    np.save(image_template_path +  "client2_image_template", image_template2)
    np.save(image_template_path +  "client3_image_template", image_template3)
    np.save(image_template_path +  "client4_image_template", image_template4)
    np.save(image_template_path +  "client5_image_template", image_template5)

    recon_eval1 = helper.evaluate_reconstruction(client1.model, thin_test_loader)
    recon_eval2 = helper.evaluate_reconstruction(client2.model, thic_test_loader)
    recon_eval3 = helper.evaluate_reconstruction(client3.model, raw_test_loader)
    recon_eval4 = helper.evaluate_reconstruction(client4.model, swel_test_loader)
    recon_eval5 = helper.evaluate_reconstruction(client5.model, frac_test_loader)

    local_center_eval1 = helper.evaluate_local_centeredness(client1.model, thin_test_loader, template1)
    local_center_eval2 = helper.evaluate_local_centeredness(client2.model, thic_test_loader, template2)
    local_center_eval3 = helper.evaluate_local_centeredness(client3.model, raw_test_loader, template3)
    local_center_eval4 = helper.evaluate_local_centeredness(client4.model, swel_test_loader, template4)
    local_center_eval5 = helper.evaluate_local_centeredness(client5.model, frac_test_loader, template5)

    client_templates = [template1, template2, template3, template4, template5]
    global_center_eval = helper.evaluate_global_centeredness(client_templates)

    test_embeddings1 = helper.generate_embedding(client1.model, thin_test_loader)
    test_embeddings2 = helper.generate_embedding(client2.model, thic_test_loader)
    test_embeddings3 = helper.generate_embedding(client3.model, raw_test_loader)
    test_embeddings4 = helper.generate_embedding(client4.model, swel_test_loader)
    test_embeddings5 = helper.generate_embedding(client5.model, frac_test_loader)

    test_embeddings = [test_embeddings1, test_embeddings2, test_embeddings3, test_embeddings4, test_embeddings5]
    recon_eval = [recon_eval1, recon_eval2, recon_eval3, recon_eval4, recon_eval5]
    local_center_eval = [local_center_eval1, local_center_eval2, local_center_eval3, local_center_eval4, local_center_eval5]

    print("Final result| Local Centeredness: {:.3f} ; {:.3f} ; {:.3f} ; {:.3f} ; {:.3f} | Reconstruction: {:.3f} ; {:.3f} ; {:.3f} ; {:.3f} ; {:.3f} | Global Centeredness: {:.3f} | Time Elapsed: {:.2f}  |".format(
                local_center_eval1, local_center_eval2, local_center_eval3, local_center_eval4, local_center_eval5, recon_eval1, recon_eval2, recon_eval3, recon_eval4, recon_eval5, global_center_eval, time_elapsed))

    #Clean interim model weights
    helper.clear_dir(TEMP_FOLDER)

    return  client_templates, test_embeddings, recon_eval, local_center_eval, global_center_eval




def demo(thin_train, thin_test, thic_train, thic_test, raw_train, raw_test, swel_train, swel_test, frac_train, frac_test):
    model_id = str(uuid.uuid4())



    templates_ablated_nofed, test_embeddings_ablated_nofed, recon_eval_ablated_nofed, local_center_eval_ablated_nofed, global_center_eval_ablated_nofed = train(model_id, 
            thin_train, thin_test, thic_train, thic_test, raw_train, raw_test, swel_train, swel_test, frac_train, frac_test, False, True)
    helper.plot_TSNE(test_embeddings_ablated_nofed, templates_ablated_nofed, 40, 'ablated_nofed', work_path)
    # helper.plot_PCA(test_embeddings_ablated_nofed, templates_ablated_nofed,'ablated_nofed', work_path)
    torch.cuda.empty_cache() 


    templates_ablated_fed, test_embeddings_ablated_fed, recon_eval_ablated_fed, local_center_eval_ablated_fed, global_center_eval_ablated_fed = train(model_id, 
            thin_train, thin_test, thic_train, thic_test, raw_train, raw_test, swel_train, swel_test, frac_train, frac_test, True, True)
    helper.plot_TSNE(test_embeddings_ablated_fed, templates_ablated_fed, 20, 'ablated_fed', work_path)
    torch.cuda.empty_cache() 

   

    recon_results = pd.DataFrame({'ablated nofed': recon_eval_ablated_nofed,
                                  'ablated fed': recon_eval_ablated_fed,},
                                  index = ['client1', 'client2', 'client3', 'client4', 'client5'])
    
    local_center_results = pd.DataFrame({'ablated nofed': local_center_eval_ablated_nofed,
                                  'ablated fed': local_center_eval_ablated_fed,},
                                  index = ['client1', 'client2', 'client3', 'client4', 'client5'])
    global_center_results = pd.DataFrame({'ablated nofed': [global_center_eval_ablated_nofed],
                                  'ablated fed': [global_center_eval_ablated_fed],},
                                  index = ['server'])

    return recon_results, local_center_results, global_center_results






