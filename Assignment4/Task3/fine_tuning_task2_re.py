'''
@Author: Abdelaziz Neamatallah
@Date: 17.12.25
@Description: The main goal of this task is to implement a GAN-based anomaly detection system for ICS network traffic

Phase 0: Design Decisions:
    - I will use PyTorch
    - Input shape convention is: (batch_size, channels, height, width)
        - channels = 1 for bytes vector
        - height = # of packets -> m
        - width = # of bytes per packet -> n
        - batch_size = number of samples in a batch (how many samples we feed to the model at once). 
    - Packet extraction rule:
        - First n bytes
    - Noise distribution:
        - Standard normal distribution (mean=0, std=1) (Gaussian)
'''
from fine_tuning_sheet4_task2_utility_tools_re import *
# Main function
def task2_sheet4_main():
    # Main logic of the task
    n, mode = phase1_read_arguments() 
    p_vals = [5,10,15]
    #! finetuning parameters. 
    
    m = [10,20,30,40,50]  # number of packets per sample
    epochs=[5,10,15,20,25]
    batch_size=[32,64,128]
    lr_D=[1e-4, 1e-5, 5e-5]
    lr_G=[3e-4, 3e-4,5e-4]
    D_threshold = [0.1,0.3,0.5]
    G_threshold = [0.1,0.3,0.5]
    K = [128,256,512]
    D_LOSS_TOO_LOW = [0.1, 0.05, 0.2]   # D dominating
    D_LOSS_TOO_HIGH = [0.7, 0.8, 0.9] # D too weak
    G_UPDATES = [1,2,3]
    configurations = [
        {'m': m_val, 'epochs': epoch_val, 'batch_size': batch_val, 'lr_D': lr_D_val, 'lr_G': lr_G_val, 'D_threshold': D_threshold_val, 'G_threshold': G_threshold_val, 'K': K_val, 'D_LOSS_TOO_LOW': D_LOSS_TOO_LOW_val, 'D_LOSS_TOO_HIGH': D_LOSS_TOO_HIGH_val, 'G_UPDATES': G_UPDATES_val} for m_val in m for epoch_val in epochs for batch_val in batch_size for lr_D_val in lr_D for lr_G_val in lr_G for D_threshold_val in D_threshold for G_threshold_val in G_threshold for K_val in K for D_LOSS_TOO_LOW_val in D_LOSS_TOO_LOW for D_LOSS_TOO_HIGH_val in D_LOSS_TOO_HIGH for G_UPDATES_val in G_UPDATES
    ] # generating all combinations
    print (configurations[:2])
    print(len(configurations)) # 5 * 5 * 3 * 3 * 3 * 3 * 3 = 6075 combinations
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu") # because GPU 0 is occupied
    print(f'[*] Using device: {device}')
    
    # define best parameters
    best_config = {}
    best_f1 = 5
    best_percision = 5
    best_recall = 5
    highest_f1 = 0
    highest_percision = 0
    highest_recall = 0
    # #! Todo: phase 7: perfomring hyperparameter tuning using validation set to get best results on test set.
    for p in p_vals: # for each value of p, perform the fine tuning
        for config in configurations:
            print('[*] Phase 7: Hyperparameter tuning iteration started...')
            print(f'[*] Validation configuration: {config}')
            m = config['m']
            print('[*] Phase 1: Starting data prepration...')
            training_data, validation_data, testing_data, _, validation_labels, test_labels = phase1_getting_data(n, m,p)
            print('[*] Phase 1: Prepration Done Successfully. \n --------------------------------------- \n ')
            # running phase 5 sanity check for GAN Training loop
            print('[*] Phase 5: Starting GAN Training loop...')
            D = Discriminator(input_shape=(1, m, n))
            G = Generator(m=m, n=n)
            print('before training:')
            print(training_data.min(), training_data.max())
            # normalize training data to [0, 1]
            training_data = (training_data - training_data.min()) / (training_data.max() - training_data.min())
            print('after normalization:')
            print(training_data.min(), training_data.max())

            # extracting hyperparameters from config
            epoch = config['epochs']
            d_lr = config['lr_D']
            g_lr = config['lr_G']
            batch = config['batch_size']
            D_LOSS_TOO_LOW_val = config['D_LOSS_TOO_LOW']
            D_LOSS_TOO_HIGH_val = config['D_LOSS_TOO_HIGH']
            G_UPDATES_val = config['G_UPDATES']
            d_threshold = config['D_threshold']
            g_threshold = config['G_threshold']
            k = config['K']
        
            print(f'[*] Current configuration: m={m}, epochs={epoch}, batch_size={batch}, lr_D={d_lr}, lr_G={g_lr}, D_LOSS_TOO_LOW={D_LOSS_TOO_LOW_val}, D_LOSS_TOO_HIGH={D_LOSS_TOO_HIGH_val}, G_UPDATES={G_UPDATES_val}')
            print('[*] Phase 5: GAN Training loop sanity check passed. \n --------------------------------------- \n ')
            # train the model on the current configurations        
            D,G = train_gan_on_training_data(
                training_data=training_data,
                D=D,
                G=G,
                m=m,
                n=n,
                p=p,
                epochs=epoch,
                batch_size=batch,
                lr_D=d_lr,
                lr_G=g_lr,
                D_LOSS_TOO_LOW=D_LOSS_TOO_LOW_val,
                D_LOSS_TOO_HIGH=D_LOSS_TOO_HIGH_val,
                G_UPDATES=G_UPDATES_val,
                device=device,
                # save_dir="models2"
            )
            # excuting the inference phase always 
            print('[*] Phase 6: Trained models found. Starting inference mode...')

            
            # D, G = load_trained_models(
            #         m=m,
            #         n=n,
            #         epochs=epoch,
            #         batch_size=batch,
            #         lr_D=d_lr,
            #         lr_G=g_lr,
            #         device=device,
            #         D_LOSS_TOO_LOW=D_LOSS_TOO_LOW_val,
            #         D_LOSS_TOO_HIGH=D_LOSS_TOO_HIGH_val,
            #         G_UPDATES=G_UPDATES_val,
            #         # model_dir="models"
            #     )
            # normalizing validation and testing data
            
            validation_data = (validation_data - validation_data.min()) / (validation_data.max() - validation_data.min())
            testing_data = (testing_data - testing_data.min()) / (testing_data.max() - testing_data.min())
            if mode == 'D':
                print("[*] Mode: INFERENCE (Discriminator-based)")
                percision, recall, f1 = phase6_discriminator_mode(
                    D = D,
                    p= p,
                    data = validation_data,
                    labels = validation_labels,
                    device = device,
                    d_lr=d_lr,
                    g_lr=g_lr,
                    epochs=epoch,
                    batch_size=batch,
                    validation=True,
                    threshold=d_threshold,
                    m=m
                )
                
                if f1 > highest_f1:
                    highest_f1 = f1
                if percision > highest_percision:
                    highest_percision = percision
                if recall > highest_recall:
                    highest_recall = recall


                if f1 > best_f1:
                    best_f1 = f1
                    best_percision = percision
                    best_recall = recall
                    best_config = config
                
            elif mode == 'G':
                print("[*] Mode: INFERENCE (Generator-based)")
                percision, recall, f1 = phase6_generator_mode(
                    D=D,
                    G=G,
                    p=p,
                    data=validation_data,
                    labels=validation_labels,
                    device=device,
                    m=m,
                    n=n,
                    K=k,
                    batch_size=batch,
                    epochs=epoch,
                    d_lr=d_lr,
                    g_lr=g_lr,
                    threshold=g_threshold,
                    validation=True,
                )
                if f1 > highest_f1:
                    highest_f1 = f1
                if percision > highest_percision:
                    highest_percision = percision
                if recall > highest_recall:
                    highest_recall = recall

                if f1 > best_f1:
                    best_f1 = f1
                    best_percision = percision
                    best_recall = recall
                    best_config = config

            else:
                print("[!] Invalid mode selected. Please choose 'D' for Discriminator-based inference or 'G' for Generator-based inference.")
        
        print(f'best results from validation: Precision: {best_percision}, Recall: {best_recall}, F1-score: {best_f1}')
        print(f' highest results from validation: Precision: {highest_percision}, Recall: {highest_recall}, F1-score: {highest_f1}')

        print(f'best configuration: {best_config}')

        # use best configuration to evaluate on test set
        print(f'[*] Evaluating best configuration on test set...')
        m = best_config['m']
        d_lr = best_config['lr_D']
        g_lr = best_config['lr_G']
        epoch = best_config['epochs']
        batch = best_config['batch_size']
        k = best_config['K']
        d_threshold = best_config['D_threshold']
        g_threshold = best_config['G_threshold']
        # load best trained models
        D, G = load_trained_models(
                m=m,
                n=n,
                device=device,
                p=p,
                epochs=epoch,
                batch_size=batch,
                lr_D=d_lr,
                lr_G=g_lr,
                D_LOSS_TOO_LOW=best_config['D_LOSS_TOO_LOW'],
                D_LOSS_TOO_HIGH=best_config['D_LOSS_TOO_HIGH'],
                G_UPDATES=best_config['G_UPDATES'],
                # model_dir="models"
            )
        testing_data = (testing_data - testing_data.min()) / (testing_data.max() - testing_data.min())
        if mode == 'D':
            print("[*] Mode: INFERENCE (Discriminator-based) on test set")
            percision, recall, f1 = phase6_discriminator_mode(
                D=D,
                p= p,   
                data=testing_data,
                labels=test_labels,
                device=device, # because GPU 0 is occupied 
                d_lr=d_lr,
                g_lr=g_lr,
                epochs=epoch,
                batch_size=batch,
                validation=False,
                threshold=d_threshold,
                m=m

            )
            print(f'[*] Test set results - Precision: {percision}, Recall: {recall}, F1-score: {f1}')
            
        elif mode == 'G':
            print("[*] Mode: INFERENCE (Generator-based) on test set")
            percision, recall, f1 = phase6_generator_mode(
                D=D,
                G=G,
                p=p,
                data=testing_data,
                labels=test_labels,
                device=device, # because GPU 0 is occupied
                m=m,
                n=n,
                K=k,
                batch_size=batch,
                epochs=epoch,
                d_lr=d_lr,
                g_lr=g_lr,
                threshold=g_threshold,
                validation=False,
            )
            print(f'[*] Test set results - Precision: {percision}, Recall: {recall}, F1-score: {f1}')
        else:
            print("[!] Invalid mode selected. Please choose 'D' for Discriminator-based inference or 'G' for Generator-based inference.")



if __name__ == "__main__":
    task2_sheet4_main()
