
# import pytorch libraries
TESTING = False
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import sys
import os
from discriminator_wrapper import Discriminator
from generator import Generator
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

# Phase 1: Data Prepration


def phase1_getting_data(n,m,p):
    '''
        This function is responsible for checking if the data and labels exist or not, and if not, to generate them
        Input:
            - n: number of bytes per packet
            - m: number of packets per sample
        Output:
            - training_data: tensor of shape (num_training_samples, 1, m, n)
            - validation_data: tensor of shape (num_validation_samples, 1, m, n)
            - testing_data: tensor of shape (num_testing_samples, 1, m, n)
            - training_labels: numpy array of shape (num_training_samples,)
            - validation_labels: numpy array of shape (num_validation_samples,)
            - test_labels: numpy array of shape (num_testing_samples,)
    '''


    # check if data and labels already exist.
    Data_Files_Exist = os.path.exists(f'./final_data/training_data_n_{n}_m_{m}_p_{p}.npy') and os.path.exists(f'./final_data/validation_data_n_{n}_m_{m}_p_{p}.npy') and os.path.exists(f'./final_data/testing_data_n_{n}_m_{m}_p_{p}.npy') and os.path.exists(f'./final_data/training_labels_n_{n}_m_{m}_p_{p}.npy') and os.path.exists(f'./final_data/validation_labels_n_{n}_m_{m}_p_{p}.npy') and os.path.exists(f'./final_data/testing_labels_n_{n}_m_{m}_p_{p}.npy')
    if Data_Files_Exist:
        # load them
        training_data, validation_data, testing_data, training_labels, validation_labels, test_labels = load_phase1_saved_data(p,n,m)
        # print unique labels for training, validation, and testing
        print(f"[*] Training labels unique values: {np.unique(training_labels)}")
        print(f"[*] Validation labels unique values: {np.unique(validation_labels)}")
        print(f"[*] Testing labels unique values: {np.unique(test_labels)}")
        
        print('[*] Phase 1: Data files found. Loaded successfully.')

    else: 
        # Load and preprocess data
        normal_data, attack_data = phase1_data_prepration(n,p)
        print(f"[*] Total normal samples: {len(normal_data)}")
        print(f"[*] Total attack samples: {len(attack_data)}")


        # Convert data to tensors
        normalData = prepare_tensors(normal_data, n, m=m)  
        attackData = prepare_tensors(attack_data, n, m=m)  

        print('[*] Data converted to tensors successfully.\n --------------------------------------- \n splitting data into training, validation, testing sets...')
        training_data, validation_data, testing_data, training_labels, validation_labels, test_labels = phase1_dataset_splitting(
            # converting normal data to tensor
            normal_data = normalData,

            # converting attack data to tensor
            attack_data = attackData, 
        )
        # Save training, validation, testing data into .npy
        # make data folders if not exist
        if not os.path.exists('./final_data'):
            os.makedirs('./final_data')
        np.save(f'./final_data/training_data_n_{n}_m_{m}_p_{p}.npy', training_data.numpy())
        np.save(f'./final_data/validation_data_n_{n}_m_{m}_p_{p}.npy', validation_data.numpy())
        np.save(f'./final_data/testing_data_n_{n}_m_{m}_p_{p}.npy', testing_data.numpy())
        print('Saved training, validation, testing data into .npy files.')

        # Save training, validation, testing labels into .npy
        np.save(f'./final_data/training_labels_n_{n}_m_{m}_p_{p}.npy', training_labels)
        np.save(f'./final_data/validation_labels_n_{n}_m_{m}_p_{p}.npy', validation_labels)
        np.save(f'./final_data/testing_labels_n_{n}_m_{m}_p_{p}.npy', test_labels)
        print('Saved training, validation, testing labels into .npy files.')



        # print unique labels for training, validation, and testing
        print(f"[*] Training labels unique values: {np.unique(training_labels)}")
        print(f"[*] Validation labels unique values: {np.unique(validation_labels)}")
        print(f"[*] Testing labels unique values: {np.unique(test_labels)}")
    return training_data, validation_data, testing_data, training_labels, validation_labels, test_labels

def load_phase1_saved_data(p,n,m):
    # load training, validation, testing data from .npy
    training_data = torch.from_numpy(np.load(f'./final_data/training_data_n_{n}_m_{m}_p_{p}.npy'))
    validation_data = torch.from_numpy(np.load(f'./final_data/validation_data_n_{n}_m_{m}_p_{p}.npy'))
    testing_data = torch.from_numpy(np.load(f'./final_data/testing_data_n_{n}_m_{m}_p_{p}.npy'))
    # load training, validation, testing labels from .npy
    training_labels = np.load(f'./final_data/training_labels_n_{n}_m_{m}_p_{p}.npy', allow_pickle=True)
    validation_labels = np.load(f'./final_data/validation_labels_n_{n}_m_{m}_p_{p}.npy', allow_pickle=True)
    test_labels = np.load(f'./final_data/testing_labels_n_{n}_m_{m}_p_{p}.npy', allow_pickle=True)
    return training_data, validation_data, testing_data, training_labels, validation_labels, test_labels


def phase1_read_arguments():
    '''
        This function reads the arguments from command line
        - --n: number of bytes per packet
        - --mode: anomaly detection mode (options: 'D' or 'G')
    '''
    parser = argparse.ArgumentParser(description='GAN-based Anomaly Detection for ICS Network Traffic')
    parser.add_argument('--n', type=int, required=True, help='Number of bytes per packet')
    parser.add_argument('--mode', type=str, choices=['D', 'G'], required=True, help='Anomaly detection mode (D for Discriminator, G for Generator)')
    args = parser.parse_args()
    return args.n, args.mode


def phase1_dataset_splitting(normal_data, attack_data):
    """
    Splits data into train / validation / test with proper shuffling.

    Training:
        - 70% normal only
    Validation:
        - 15% normal + 85% attack
    Test:
        - 15% normal + 15% attack
    """
    # converting labels into tensors for compatability
    normalLabels = torch.zeros(len(normal_data), dtype=torch.long)
    attackLabels = torch.ones(len(attack_data), dtype=torch.long)


    # --------------------------------------------------
    # Shuffle NORMAL data
    # --------------------------------------------------
    normal_perm = torch.randperm(len(normal_data))
    normal_data = normal_data[normal_perm]
    normalLabels = normalLabels[normal_perm]

    # --------------------------------------------------
    # Shuffle ATTACK data
    # --------------------------------------------------
    attack_perm = torch.randperm(len(attack_data))
    attack_data = attack_data[attack_perm]
    attackLabels = attackLabels[attack_perm]

    # --------------------------------------------------
    # NORMAL splits
    # --------------------------------------------------
    num_normal = len(normal_data)
    train_size = int(0.7 * num_normal)
    val_size = int(0.15 * num_normal)

    train_data = normal_data[:train_size]
    trainingLabels = normalLabels[:train_size]

    val_normal_data = normal_data[train_size:train_size + val_size]
    val_normal_labels = normalLabels[train_size:train_size + val_size]

    test_normal_data = normal_data[train_size + val_size:]
    test_normal_labels = normalLabels[train_size + val_size:]

    # --------------------------------------------------
    # ATTACK splits
    # --------------------------------------------------
    num_attack = len(attack_data)
    val_attack_size = int(0.85 * num_attack)
    test_attack_size = int(0.15 * num_attack)

    val_attack_data = attack_data[:val_attack_size]
    val_attack_labels = attackLabels[:val_attack_size]

    test_attack_data = attack_data[val_attack_size:val_attack_size + test_attack_size]
    test_attack_labels = attackLabels[val_attack_size:val_attack_size + test_attack_size]

    # --------------------------------------------------
    # Combine validation and test
    # --------------------------------------------------
    val_data = torch.cat((val_normal_data, val_attack_data), dim=0)
    validationLabels = torch.cat((val_normal_labels, val_attack_labels), dim=0)

    test_data = torch.cat((test_normal_data, test_attack_data), dim=0)
    testLabels = torch.cat((test_normal_labels, test_attack_labels), dim=0)

    return (
        train_data,
        val_data,
        test_data,
        trainingLabels,
        validationLabels,
        testLabels,
    )



def phase1_data_prepration(n,p):
    '''
    This function prepares the data for training and evaluation.
        Packets preprocessing:
            We should have a function that load the dataset in form of raw packets, and return it in shape of (m, n)
            And it should truncate/pad each packet to M bytes
            It is already implemented before, but we will just need to adapt it here.    
    '''
    def pad_or_truncate_packet(packet, n):
        if isinstance(packet, np.ndarray):
            packet = packet.tobytes()

        if len(packet) > n:
            return packet[:n]
        elif len(packet) < n:
            return packet + bytes(n - len(packet))
        else:
            return packet


    def load_and_label_data(n, p):
        normal_pcap_path = f"./final_data/final_processed_packets_{p}_dedup_normal_data.npy"
        attacked_pcap_path  = f"./final_data/final_processed_packets_{p}_dedup_attack_data.npy"
        data_normal_arrays = np.load(normal_pcap_path, allow_pickle=True)
        data_attacked = np.load(attacked_pcap_path, allow_pickle=True)

        labeled_data_normal =[ ]
        labeled_data_attack =[ ]
        for array in data_normal_arrays:
            labeled_data_normal.append( (pad_or_truncate_packet(array, n), 'normal') ) 

        
        for array in data_attacked:
            labeled_data_attack.append( (pad_or_truncate_packet(array,n), 'attack') )
        return labeled_data_normal, labeled_data_attack

    return load_and_label_data(n,p)

def prepare_tensors(data, n, m):
    '''
        This function converts the data to tensors.
        Input:
            - data: list of tuples (packet, label)
            - n: number of bytes per packet
            - m: number of packets per sample
        Output:
            - data_tensor: tensor of shape (num_samples, 1, m, n)
    '''
    print(f'[*] Converting data to tensors...\n data shape is : {len(data)}')
    print(type(data))
    print(f'shape of first packet: {len(data[0][0])} , label: {data[0][1]}')
    num_samples = len(data) // m
    data_tensor = torch.zeros((num_samples, 1, m, n), dtype=torch.float32)
    for i in range(num_samples):
        for j in range(m):
            packet, _ = data[i * m + j]
            data_tensor[i, 0, j, :] = torch.tensor(
                np.frombuffer(packet, dtype=np.uint8),
                dtype=torch.float32
            ) / 255.0
    return data_tensor


# ---------------- Phase 2: Discriminator implementation ---------------- #
def phase2_sanity_checks(n , m): 
    '''
        Logic: 
            - Instantiate the Discriminator model
            - Create a dummy batch with the same shape as real data
            - Run a forward pass through the Discriminator
            - Extract features using the feature extractor
            - verify shapes
            - verify output ranges
    '''
    print("[*] Phase 2: Discriminator sanity checks")

    # 1. Instantiate discriminator
    input_shape = (1, m, n)   # (channels, height, width)
    D = Discriminator(input_shape=input_shape)

    # 2. Create dummy input batch
    batch_size = 8
    x_dummy = torch.randn(batch_size, 1, m, n)

    # 3. Forward pass
    with torch.no_grad():
        output = D(x_dummy)

    # 4. Feature extraction
    with torch.no_grad():
        features = D.extract_features(x_dummy)

    # 5. Sanity checks
    print("[*] Discriminator output shape:", output.shape)
    print("[*] Feature tensor shape:", features.shape)
    print("[*] Output range:",
        output.min().item(),
        output.max().item())

    # 6. Assertions (fail fast if something is wrong)
    assert output.shape == (batch_size, 1), \
        "Discriminator output shape is wrong"

    assert features.shape[0] == batch_size, \
        "Feature batch dimension mismatch"

    assert 0.0 <= output.min() and output.max() < 1.0, \
        "Discriminator output not in [0,1]"

    print("[✓] Phase 2 sanity checks passed successfully")
    print("-----------------------------------------------\n")




# ---------------- Phase 3: Generator implementation ---------------- #
def phase3_sanity_checks(n, m):
    """
    Logic:
        - Instantiate the Generator model
        - Create a dummy noise batch
        - Run a forward pass through the Generator
        - Verify shapes
        - Verify output ranges
    """
    print("[*] Phase 3: Generator sanity checks")

    # 1. Instantiate generator
    G = Generator(m=m, n=n)

    # 2. Create dummy noise batch
    batch_size = 8
    noise_dim = m * n
    z_dummy = torch.randn(batch_size, noise_dim)

    # 3. Forward pass
    with torch.no_grad():
        generated_data = G(z_dummy)

    # 4. Sanity checks
    print("[*] Generated data shape:", generated_data.shape)
    print("[*] Generated data range:",
          generated_data.min().item(),
          generated_data.max().item())

    # 5. Assertions (fail fast)
    assert generated_data.shape == (batch_size, 1, m, n), \
        "Generator output shape is incorrect"

    assert 0.0 <= generated_data.min() and generated_data.max() <= 1.0, \
        "Generator output is not in [0,1]"

    print("[✓] Phase 3 sanity checks passed successfully")
    print("-----------------------------------------------\n")



# ---------------- Phase 4: Custom Generator Loss ---------------- #
def phase4_custom_generator_loss(x_real, x_fake, D):
    """
    Phase 4: Custom Generator Loss sanity check
    Logic:
        - Extract discriminator features for real and fake samples
        - Compute feature-matching loss using mean feature vectors
        - Do NOT use discriminator output
    """
    # print("[*] Phase 4: Custom Generator Loss sanity checks")

    # 1. Create dummy batches
    # batch_size = 8
    # m, n = 10, 50
    # x_real = torch.randn(batch_size, 1, m, n)
    # x_fake = torch.randn(batch_size, 1, m, n)

    # 2. Instantiate discriminator
    # D = Discriminator(input_shape=(1, m, n))

    # 3. Freeze discriminator
    # for p in D.parameters():
    #     p.requires_grad = False

    # 4. Feature matching loss
    def feature_matching_loss(D, x_real, x_fake):
        '''
        Input:
            D: Discriminator model
            x_real: real data batch
            x_fake: fake data batch
        Output:
            loss: feature matching loss value
        Logic:
            - Extract features from real and fake samples
            - Compute mean feature vectors
            - Compute L2 loss between mean feature vectors   
        '''
        f_real = D.extract_features(x_real)
        f_fake = D.extract_features(x_fake)

        mean_real = f_real.mean(dim=0)
        mean_fake = f_fake.mean(dim=0)

        loss = torch.mean((mean_real - mean_fake) ** 2) / mean_real.numel()  # L2 -> higher punishment
        return loss

    loss = feature_matching_loss(D, x_real, x_fake)

    # 5. Sanity checks
    # print("[*] Feature matching loss value:", loss.item())

    assert isinstance(loss.item(), float), "Loss is not scalar"
    assert loss.item() >= 0.0, "Loss must be non-negative"

    # print("[✓] Phase 4 sanity checks passed successfully")
    # print("-----------------------------------------------\n")
    return loss

# ---------------- Phase 5: GAN Training Loop ---------------- #
def discriminator_accuracy(D, x_real, G, m, n):
    batch_size = x_real.size(0)
    z = torch.randn(batch_size, m * n, device=x_real.device)

    with torch.no_grad():
        pred_real = D(x_real) > 0.5
        pred_fake = D(G(z)) < 0.5

    acc = (pred_real.sum() + pred_fake.sum()) / (2 * batch_size)
    return acc.item()


def save_models(D, G, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)

    torch.save(D.state_dict(), os.path.join(save_dir, "discriminator.pth"))
    torch.save(G.state_dict(), os.path.join(save_dir, "generator.pth"))

    print(f"[✓] Models saved to '{save_dir}/'")

def phase5_sanity_checks(m,n,batch_size=8, lr=1e-4, epochs=1):
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Instantiate models
    # --------------------------------------------------
    D = Discriminator(input_shape=(1, m, n)).to(device)
    G = Generator(m=m, n=n).to(device)

    # --------------------------------------------------
    # Optimizers
    # --------------------------------------------------
    optimizer_D = optim.Adam(D.parameters(), lr=1e-4)
    optimizer_G = optim.Adam(G.parameters(), lr=1e-4)

    # --------------------------------------------------
    # Dummy real batch (simulates DataLoader output)
    # --------------------------------------------------
    x_real = torch.rand(batch_size, 1, m, n).to(device)

    # --------------------------------------------------
    # Discriminator step
    # --------------------------------------------------
    def train_discriminator_step(D, G, x_real, optimizer_D, m, n):
        D.train()
        G.eval()

        batch_size = x_real.size(0)
        optimizer_D.zero_grad()

        # Real samples
        y_real = torch.ones(batch_size, 1, device=x_real.device)
        pred_real = D(x_real)
        loss_real = F.binary_cross_entropy(pred_real, y_real)

        # Fake samples
        z = torch.randn(batch_size, m * n, device=x_real.device)
        with torch.no_grad():
            x_fake = G(z)

        y_fake = torch.zeros(batch_size, 1, device=x_real.device)
        pred_fake = D(x_fake)
        loss_fake = F.binary_cross_entropy(pred_fake, y_fake)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        return loss_D.item()

    # --------------------------------------------------
    # Generator step (feature matching)
    # --------------------------------------------------
    def train_generator_step(D, G, x_real, optimizer_G, m, n):
        G.train()
        D.eval()

        # Freeze D
        for p in D.parameters():
            p.requires_grad = False

        optimizer_G.zero_grad()

        batch_size = x_real.size(0)
        z = torch.randn(batch_size, m * n, device=x_real.device)
        x_fake = G(z)

        f_real = D.extract_features(x_real)
        f_fake = D.extract_features(x_fake)

        mean_real = f_real.mean(dim=0)
        mean_fake = f_fake.mean(dim=0)
        loss_G = torch.mean((mean_real - mean_fake) ** 2)

        loss_G.backward()
        optimizer_G.step()

        # Unfreeze D
        for p in D.parameters():
            p.requires_grad = True

        return loss_G.item()

    # --------------------------------------------------
    # RUN CHECK 1: one-batch dry run
    # --------------------------------------------------
    print("\n[✓] Running Phase 5 – One-batch dry run\n")

    loss_D = train_discriminator_step(D, G, x_real, optimizer_D, m, n)
    loss_G = train_generator_step(D, G, x_real, optimizer_G, m, n)

    print(f"D loss: {loss_D}")
    print(f"G feature loss: {loss_G}")

    assert torch.isfinite(torch.tensor(loss_D)), "D loss is not finite"
    assert torch.isfinite(torch.tensor(loss_G)), "G loss is not finite"

    # save trained models 
    save_models(D, G)

    print("\n[✓] Phase 5 CHECK 1 passed successfully\n")

# -----------------------------
# Hyperparameters for control
# -----------------------------


def train_gan_on_training_data(
    training_data,
    D,
    G,
    m,
    n,
    p,
    epochs=10,
    batch_size=64,
    lr_D=1e-4,
    lr_G=3e-4,
    D_LOSS_TOO_LOW = 0.1,     # D dominating
    D_LOSS_TOO_HIGH = 0.7,   # D too weak
    G_UPDATES = 2,         # train G two times when allowed
    device="cuda",
    save_dir="models"
):
    # -----------------------------
    # DataLoader
    # -----------------------------
    train_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # -----------------------------
    # Optimizers
    # -----------------------------
    optimizer_D = optim.Adam(D.parameters(), lr=lr_D)
    optimizer_G = optim.Adam(G.parameters(), lr=lr_G)

    D.to(device)
    G.to(device)

    print("\n[*] Starting GAN training")
    print(f"Epochs: {epochs} | Batch size: {batch_size}")
    print("--------------------------------------------------")

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(epochs):
        D.train()
        G.train()

        epoch_loss_D = 0.0
        epoch_loss_G = 0.0

        print(f"\n[*] Epoch {epoch+1}/{epochs}")

        progress_bar = tqdm(train_loader, leave=False)

        for x_real in progress_bar:
            x_real = x_real.to(device)
            bs = x_real.size(0)

            # =========================
            # 1. Discriminator forward
            # =========================
            z = torch.randn(bs, m * n, device=device)
            with torch.no_grad():
                x_fake = G(z)

            y_real = torch.ones(bs, 1, device=device)
            y_fake = torch.zeros(bs, 1, device=device)

            pred_real = D(x_real)
            pred_fake = D(x_fake)

            loss_real = F.binary_cross_entropy_with_logits(pred_real, y_real)
            loss_fake = F.binary_cross_entropy_with_logits(pred_fake, y_fake)
            loss_D = loss_real + loss_fake

            # -------------------------
            # Decide who trains
            # -------------------------
            update_D = True
            update_G = True

            if loss_D.item() < D_LOSS_TOO_LOW:
                # D too strong → freeze D
                update_D = False
                update_G = True
            elif loss_D.item() > D_LOSS_TOO_HIGH:
                # D too weak → freeze G
                update_D = True
                update_G = False

            # =========================
            # 2. Update Discriminator
            # =========================
            if update_D:
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

            # =========================
            # 3. Update Generator
            # =========================
            if update_G:
                for p in D.parameters():
                    p.requires_grad = False

                for _ in range(G_UPDATES):
                    optimizer_G.zero_grad()

                    z = torch.randn(bs, m * n, device=device)
                    x_fake = G(z)

                    # Feature matching loss
                    f_real = D.extract_features(x_real)
                    f_fake = D.extract_features(x_fake)

                    mean_real = f_real.mean(dim=0)
                    mean_fake = f_fake.mean(dim=0)

                    loss_G = torch.mean((mean_real - mean_fake) ** 2)

                    loss_G.backward()
                    optimizer_G.step()

                for p in D.parameters():
                    p.requires_grad = True
            else:
                loss_G = torch.tensor(0.0)

            # =========================
            # Logging
            # =========================
            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()

            progress_bar.set_postfix({
                "D_loss": f"{loss_D.item():.4f}",
                "G_loss": f"{loss_G.item():.2e}",
                "upd_D": update_D,
                "upd_G": update_G
            })

        # -----------------------------
        # Epoch summary
        # -----------------------------
        num_batches = len(train_loader)
        print(
            f"[Epoch {epoch+1} DONE] "
            f"Avg D loss: {epoch_loss_D/num_batches:.4f} | "
            f"Avg G loss: {epoch_loss_G/num_batches:.2e}"
        )

    # -----------------------------
    # Save models
    # -----------------------------
    '''
    epochs=10,
    batch_size=64,
    lr_D=1e-4,
    lr_G=3e-4,
    D_LOSS_TOO_LOW = 0.1,     # D dominating
    D_LOSS_TOO_HIGH = 0.7,   # D too weak
    G_UPDATES = 2,         # train G two times when allowed
    
    '''
    os.makedirs(save_dir, exist_ok=True)
    d_file_name = f"p_{p}_m_{m}_discriminator_d_lr{lr_D}_epochs_{epochs}_bs{batch_size}_dl_thresh_low_{D_LOSS_TOO_LOW}_dl_thresh_high_{D_LOSS_TOO_HIGH}_gupdates_{G_UPDATES}.pth"
    g_file_name = f"p_{p}_m_{m}_generator_g_lr{lr_G}_epochs_{epochs}_bs{batch_size}_dl_thresh_low_{D_LOSS_TOO_LOW}_dl_thresh_high_{D_LOSS_TOO_HIGH}_gupdates_{G_UPDATES}.pth"
    torch.save(D.state_dict(), os.path.join(save_dir, d_file_name))
    torch.save(G.state_dict(), os.path.join(save_dir, g_file_name))

    print("\n[✓] Training finished successfully")
    print(f"[✓] Models saved to '{save_dir}/{d_file_name}' and '{save_dir}/{g_file_name}'")
    return D,G



# ---------------- Phase 6: Anomaly Detection modes ---------------- #
'''
Phase 6: Anomaly Detection Modes
    Mode 1 (Discriminator-based):
        For each test sample:
            Compute D(X)
            If D(X) < threshold -> Anomalous
    Mode 2 (Generator-based):
        For each test sample:
            Find closest G(z) to X
            Compute reconstruction error or feature mismatch
            If error > threshold -> Anomalous
'''

# ==================================================
# 1. LOAD TRAINED MODELS
# ==================================================
def load_trained_models(
    m,
    n,
    device,
    p,
    epochs=10,
    batch_size=64,
    lr_D=1e-4,
    lr_G=3e-4,
    D_LOSS_TOO_LOW = 0.1,     # D dominating
    D_LOSS_TOO_HIGH = 0.7,   # D too weak
    G_UPDATES = 2,         # train G two times when allowed
    model_dir="models"
):
    D = Discriminator(input_shape=(1, m, n)).to(device)
    G = Generator(m=m, n=n).to(device)
    d_file_name = f"p_{p}_m_{m}_discriminator_d_lr{lr_D}_epochs_{epochs}_bs{batch_size}_dl_thresh_low_{D_LOSS_TOO_LOW}_dl_thresh_high_{D_LOSS_TOO_HIGH}_gupdates_{G_UPDATES}.pth"
    g_file_name = f"p_{p}_m_{m}_generator_g_lr{lr_G}_epochs_{epochs}_bs{batch_size}_dl_thresh_low_{D_LOSS_TOO_LOW}_dl_thresh_high_{D_LOSS_TOO_HIGH}_gupdates_{G_UPDATES}.pth"
    D.load_state_dict(
        torch.load(os.path.join(model_dir, d_file_name), map_location=device)
    )
    G.load_state_dict(
        torch.load(os.path.join(model_dir, g_file_name), map_location=device)
    )

    # prepare them for the evaluation mode. 
    D.eval()
    G.eval()

    print("[✓] Trained Discriminator & Generator loaded successfully")
    return D, G


# ==================================================
# 2. MODE 1 — DISCRIMINATOR-BASED SCORES
# ==================================================
def discriminator_scores(D, data, device):
    scores = []

    with torch.no_grad():
        for x in data:
            x = x.unsqueeze(0).to(device)  # (1,1,m,n)
            score = D(x)
            scores.append(score.item())

    return np.array(scores)


# ==================================================
# 3. MODE 2 — GENERATOR FEATURE-MATCHING SCORES
# ==================================================
def generator_feature_scores(D, G, data, m, n, device):
    scores = []

    with torch.no_grad():
        for x in data:
            x = x.unsqueeze(0).to(device)

            z = torch.randn(1, m * n, device=device)
            x_fake = G(z)

            f_real = D.extract_features(x)
            f_fake = D.extract_features(x_fake)

            score = torch.mean((f_real - f_fake) ** 2).item()
            scores.append(score)

    return np.array(scores)


# ==================================================
# 4. THRESHOLD COMPUTATION
# ==================================================
def compute_threshold(scores, percentile):
    threshold = np.percentile(scores, percentile)
    print(f"[✓] Threshold ({percentile}th percentile): {threshold:.6f}")
    return threshold


# ==================================================
# 5. EVALUATION
# ==================================================
def evaluate(scores, threshold, true_labels, anomaly_if_lower):
    """
    anomaly_if_lower = True  -> anomaly if score < threshold  (Discriminator)
    anomaly_if_lower = False -> anomaly if score > threshold  (Generator)
    """
    if anomaly_if_lower:
        preds = (scores < threshold).astype(int)
    else:
        preds = (scores > threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, preds, average="binary"
    )

    return precision, recall, f1


# ==================================================
# 6. PHASE 6 PIPELINE ONLY
# We do not use labels for training, we use them for vaidation and testing only. 
# ==================================================
def filter_normal_samples(data, labels):
    # print unique values of labels
    print(f"Unique labels: {np.unique(labels)}")
    return data[labels == 0]


def phase6_discriminator_mode(
    D,
    p,
    data,
    labels,
    device,
    d_lr,
    g_lr,
    epochs,
    batch_size,
    validation:bool = False,
    threshold=0.5,
    m=10,
):
    print("\n[*] Phase 6 - Mode 1: Discriminator-based")
    # ---- test evaluation ----
    print('Testing on test data...')
    scores = []
    with torch.no_grad():
        for x in data:
            x = x.unsqueeze(0).to(device)
            logit = D(x)
            prob = torch.sigmoid(logit) # to get output in range [0,1]
            scores.append(prob.item())    
    scores = np.array(scores)

    preds = (scores < threshold).astype(int)  # 1 = anomaly
    y_true = labels

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary"
    )

    print(f"[D-mode] Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    # write results in metrics_D.txt
    final_label = 'final_testing' if not validation else 'validation'
    file_name = f"p_{p}_metrics_D_{final_label}_epochs_{epochs}_d_lr_{d_lr}_g_lr_{g_lr}_batch_size_{batch_size}_m_{m}.txt"
    with open(file_name, "w") as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
    return precision, recall, f1
    

def phase6_generator_mode(
    D,
    G,
    p,  
    data,
    labels,
    device,
    m,
    n,
    K,
    batch_size,
    epochs,
    d_lr,
    g_lr,
    threshold,
    validation:bool = False,  
):
    print("\n[*] Phase 6 - Mode 2: Generator-based (Feature Matching)")
    
    with torch.no_grad():
        z = torch.randn(K, m * n, device=device)
        x_fake = G(z)
        f_fake_mean = D.extract_features(x_fake).mean(dim=0)
        f_fake_mean = F.normalize(f_fake_mean, dim=0)

    scores = []
    with torch.no_grad():
        for x in data:
            x = x.unsqueeze(0).to(device)
            f_real = D.extract_features(x)
            f_real = F.normalize(f_real, dim=1)
            score = torch.mean((f_real - f_fake_mean) ** 2) / f_real.size(1)
            scores.append(score.item())

    scores = np.array(scores)

    preds = (scores > threshold).astype(int)  # 1 = anomaly
    y_true = labels

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )

    print(f"[G-mode] Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    # write results in metrics_G.txt
    final_label = 'final_testing' if not validation else 'validation'
    file_name = f"p_{p}_metrics_G_{final_label}_epochs_{epochs}_d_lr_{d_lr}_g_lr_{g_lr}_batch_size_{batch_size}_m_{m}.txt"
    with open(file_name, "w") as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
    return precision, recall, f1
    
