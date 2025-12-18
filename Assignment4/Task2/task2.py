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
    n, mode = phase1_read_arguments() # I think I should read n only when we should perform training, but for inference mode, I should let him select from the two other modes. 

    # Data preparation
    # check if ./models/discriminator.pth and ./models/generator.pth exist 
    Training_Models_Exist = os.path.exists('./models/discriminator.pth') and os.path.exists('./models/generator.pth')
    if Training_Models_Exist:
        return # TODO: implement inference mode here phase 6 and 7
    else: 
        print('[*] Phase 1: Starting data prepration...')
        # Load and preprocess data
        normal_data, attack_data = phase1_data_prepration(n)
        print(f"[*] Total normal samples: {len(normal_data)}")
        print(f"[*] Total attack samples: {len(attack_data)}")

        # Convert data to tensors
        normalData = prepare_tensors(normal_data, n, m=10)  # assuming m=10 packets per sample
        attackData = prepare_tensors(attack_data, n, m=10)  # assuming m=10 packets per sample
        training_data, validation_data, testing_data = phase1_dataset_splitting(
            # converting normal data to tensor
            normal_data = normalData,

            # converting attack data to tensor
            attack_data = attackData
        )

        print(f"[*] Training samples: {len(training_data)}")
        print(f"[*] Validation samples: {len(validation_data)}")
        print(f"[*] Testing samples: {len(testing_data)}")
        print(training_data.shape)
        print(validation_data.shape)
        print(testing_data.shape)

        print('[*] Phase 1: Prepration Done Successfully. \n --------------------------------------- \n ')

        # Running sanity checks for Discriminator implementation
        phase2_sanity_checks(n , m=10)

        # Running sanity checks for Generator implementation
        phase3_sanity_checks(n , m=10)

        # running sanity check for custom loss function
        phase4_custom_generator_loss()

        # running phase 5 sanity check for GAN Training loop
        print('[*] Phase 5: Starting GAN Training loop...')
        m = 10  # number of packets per sample
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        D = Discriminator(input_shape=(1, m, n))
        G = Generator(m=m, n=n)

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
