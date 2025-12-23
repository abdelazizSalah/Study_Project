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
from sheet4_task2_utility_tools import *
# Main function
def task2_sheet4_main():
    # Main logic of the task
    n, mode = phase1_read_arguments() 
    #! finetuning parameters. 
      
    m = [10,20]  # number of packets per sample
    epochs=[5,10,20]
    batch_size=[32,64,128]
    lr_D=[1e-4, 1e-5]
    lr_G=[3e-4, 3e-4,]
    D_threshold = [0.1,0.3]
    G_threshold = [0.9,0.7]
    K = [128,256]
    D_LOSS_TOO_LOW = [0.1, 0.05]   # D dominating
    D_LOSS_TOO_HIGH = [0.8, 0.9] # D too weak
    G_UPDATES = [2,3]


    # m = [10,20,30,40,50]  # number of packets per sample
    # epochs=[5,10,15,20,25]
    # batch_size=[32,64,128]
    # lr_D=[1e-4, 1e-5, 5e-5]
    # lr_G=[3e-4, 3e-4,5e-4]
    # D_threshold = [0.1,0.3,0.5]
    # G_threshold = [0.1,0.3,0.5]
    # K = [128,256,512]
    # D_LOSS_TOO_LOW = [0.1, 0.05, 0.2]    # D dominating
    # D_LOSS_TOO_HIGH = [0.7, 0.8, 0.9]   # D too weak
    # G_UPDATES = [1,2,3]

    # use single value only for faster testing
    # m = [10]  # number of packets per sample
    # epochs=[5]
    # batch_size=[32]
    # lr_D=[1e-4]
    # lr_G=[3e-4]
    # D_threshold = [0.1]
    # G_threshold = [0.1]
    # K = [128]
    # D_LOSS_TOO_LOW = [0.1]     # D dominating
    # D_LOSS_TOO_HIGH = [0.7]   # D too weak
    # G_UPDATES = [1]
    configurations = [
        {'m': m_val,
          'epochs': epoch_val,
            'batch_size': batch_val,
              'lr_D': lr_D_val,
                'lr_G': lr_G_val,
                  'D_threshold': D_threshold_val,
                    'G_threshold': G_threshold_val,
                      'K': K_val,
                        'D_LOSS_TOO_LOW': d_loss_val_low,
                          'D_LOSS_TOO_HIGH': d_loss_val_high,
                            'G_UPDATES': g_updates_val}
                              for m_val in m
                                for epoch_val in epochs
                                  for batch_val in batch_size
                                    for lr_D_val in lr_D 
                                     for lr_G_val in lr_G 
                                      for D_threshold_val in D_threshold
                                       for G_threshold_val in G_threshold 
                                        for K_val in K 
                                         for d_loss_val_low in D_LOSS_TOO_LOW 
                                          for d_loss_val_high in D_LOSS_TOO_HIGH 
                                           for g_updates_val in G_UPDATES
                              ] # generating all combinations
    print (configurations[:2])
    print(len(configurations)) # 5 * 5 * 3 * 3 * 3 * 3 * 3 = 6075 combinations
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # because GPU 0 is occupied
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
    for config in configurations:
        print('[*] Phase 7: Hyperparameter tuning iteration started...')
        print(f'[*] Validation configuration: {config}')
        m = config['m']
        print('[*] Phase 1: Starting data prepration...')
        training_data, validation_data, testing_data, _, validation_labels, test_labels = phase1_getting_data(n, m)
        print('[*] Phase 1: Prepration Done Successfully. \n --------------------------------------- \n ')


        ## check if ./models/discriminator.pth and ./models/generator.pth exist 

        ## Training_Models_Exist = os.path.exists(f'./models2/discriminator.pth') and os.path.exists(f'./models2/generator.pth')
        ## if not Training_Models_Exist:
        ##     # Running sanity checks for Discriminator implementation
        ##     # tring better training for GAN
        ##     phase2_sanity_checks(n , m=m)

        ##     # Running sanity checks for Generator implementation
        ##     phase3_sanity_checks(n , m=m)

        ##     # running sanity check for custom loss function
        ##     phase4_custom_generator_loss()

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
        d_loss_low = config['D_LOSS_TOO_LOW']
        d_loss_high = config['D_LOSS_TOO_HIGH']
        g_updates = config['G_UPDATES']
        d_threshold = config['D_threshold']
        g_threshold = config['G_threshold']
        k = config['K']
       
        '''
        'models/discriminator_m_10_d_lr0.0001_epochs_5_bs32_dl_thresh_low_0.1_dl_thresh_high_0.8_gupdates_1.pth' 
          'models/generator_m_10_g_lr0.0003_epochs_5_bs32_dl_thresh_low_0.1_dl_thresh_high_0.8_gupdates_1.pth'
        
        '''
        print(f'[*] Current configuration: m={m}, epochs={epoch}, batch_size={batch}, lr_D={d_lr}, lr_G={g_lr}, D_LOSS_TOO_LOW={d_loss_low}, D_LOSS_TOO_HIGH={d_loss_high}, G_UPDATES={g_updates}, D_threshold={d_threshold}, G_threshold={g_threshold}, K={k}')
        print('[*] Phase 5: GAN Training loop sanity check passed. \n --------------------------------------- \n ')
        # train the model on the current configurations     
        d_trained_model_path = f'./models/discriminator_m_{m}_d_lr{d_lr}_epochs_{epoch}_bs{batch}_dl_thresh_low_{d_loss_low}_dl_thresh_high_{d_loss_high}_gupdates_{g_updates}.pth'
        g_trained_model_path = f'./models/generator_m_{m}_g_lr{g_lr}_epochs_{epoch}_bs{batch}_dl_thresh_low_{d_loss_low}_dl_thresh_high_{d_loss_high}_gupdates_{g_updates}.pth'
        Training_Models_Exist = os.path.exists(d_trained_model_path) and os.path.exists(g_trained_model_path)
        if not Training_Models_Exist:
            print('[*] Phase 6: No trained models found. Starting training phase...')
             # training the GAN on the training data
            D,G = train_gan_on_training_data(
                training_data=training_data,
                D=D,
                G=G,
                m=m,
                n=n,
                device=device,
                epochs=epoch,
                batch_size=batch,
                lr_D=d_lr,
                lr_G=g_lr,
                D_LOSS_TOO_LOW=d_loss_low,
                D_LOSS_TOO_HIGH=d_loss_high,
                G_UPDATES=g_updates,
                # save_dir="models2"
            )
        else:
            # excuting the inference phase always 
            print('[*] Phase 6: Trained models found. Starting inference mode...')

            
            D, G = load_trained_models(
                    m=m,
                    n=n,
                    epochs=epoch,
                    batch_size=batch,
                    lr_D=d_lr,
                    lr_G=g_lr,
                    device=device,
                    D_LOSS_TOO_LOW=d_loss_low,
                    D_LOSS_TOO_HIGH=d_loss_high,
                    G_UPDATES=g_updates,
                    # model_dir="models"
                )
        # normalizing validation and testing data
        
        validation_data = (validation_data - validation_data.min()) / (validation_data.max() - validation_data.min())
        testing_data = (testing_data - testing_data.min()) / (testing_data.max() - testing_data.min())
        if mode == 'D':
            print("[*] Mode: INFERENCE (Discriminator-based)")
            percision, recall, f1 = phase6_discriminator_mode(
                D=D,
                batch_size=batch,
                data=validation_data,
                labels=validation_labels,
                device=device,
                d_lr=d_lr,
                g_lr=g_lr,
                epochs=epoch,
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
                data=validation_data,
                labels=validation_labels,
                device=device,
                batch_size=batch,
                d_lr=d_lr,
                g_lr=g_lr,
                epochs=epoch,
                m=m,
                n=n,
                K=k,
                validation=True,
                threshold=g_threshold,
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
            batch_size=batch,
            data=testing_data,
            labels=test_labels,
            device=device, # because GPU 0 is occupied 
            d_lr=d_lr,
            g_lr=g_lr,
            epochs=epoch,
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
            data=testing_data,
            labels=test_labels,
            device=device, # because GPU 0 is occupied
            batch_size=batch,
            d_lr=d_lr,
            g_lr=g_lr,
            epochs=epoch,
            m=m,
            n=n,
            K=k,
            threshold=g_threshold,
            validation=False,
        )
        print(f'[*] Test set results - Precision: {percision}, Recall: {recall}, F1-score: {f1}')
    else:
        print("[!] Invalid mode selected. Please choose 'D' for Discriminator-based inference or 'G' for Generator-based inference.")



if __name__ == "__main__":
    task2_sheet4_main()
