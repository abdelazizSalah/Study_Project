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
    m = 10  # number of packets per sample
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # because GPU 0 is occupied
    print(f'[*] Using device: {device}')
    
    
    print('[*] Phase 1: Starting data prepration...')
    training_data, validation_data, testing_data, training_labels, validation_labels, test_labels = phase1_getting_data(n, m)
    print('[*] Phase 1: Prepration Done Successfully. \n --------------------------------------- \n ')


    # check if ./models/discriminator.pth and ./models/generator.pth exist 
    Training_Models_Exist = os.path.exists('./models/discriminator.pth') and os.path.exists('./models/generator.pth')
    if Training_Models_Exist:
        print('[*] Phase 6: Trained models found. Starting inference mode...')
        D, G = load_trained_models(
                m=m,
                n=n,
                device=device,
                model_dir="models"
            )
        if mode == 'D':
            print("[*] Mode: INFERENCE (Discriminator-based)")
            phase6_discriminator_mode(
                D,
                validation_data,
                validation_labels,
                testing_data,
                test_labels,
                device
            )
        elif mode == 'G':
            print("[*] Mode: INFERENCE (Generator-based)")
            phase6_generator_mode(
                D,
                G,
                validation_data,
                validation_labels,
                testing_data,
                test_labels,
                m,
                n,
                device
            )
        else:
            print("[!] Invalid mode selected. Please choose 'D' for Discriminator-based inference or 'G' for Generator-based inference.")
        return 
    else: 
        # # Running sanity checks for Discriminator implementation
        # phase2_sanity_checks(n , m=m)

        # # Running sanity checks for Generator implementation
        # phase3_sanity_checks(n , m=m)

        # # running sanity check for custom loss function
        # phase4_custom_generator_loss()

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
        
        train_gan_on_training_data(
            training_data=training_data,
            D=D,
            G=G,
            m=m,
            n=n,
            epochs=30,
            batch_size=64,
            device=device
        )



if __name__ == "__main__":
    task2_sheet4_main()
