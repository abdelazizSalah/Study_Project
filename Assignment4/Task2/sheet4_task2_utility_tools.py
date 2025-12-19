
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

# Phase 1: Data Prepration
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
    '''
    Input:
        - normal_data: tensor of normal data samples
        - attack_data: tensor of attack data samples
    Output:
        - train_data: tensor of training data samples
        - val_data: tensor of validation data samples
        - test_data: tensor of test data samples
    Logic:
        This function splits the dataset into training, validation, and test sets.
        - Training set: 70% of normal.
        - Validation set: 15%  of normal and 85% of attack.
        - Test set: 15% of normal and 15% of attack.        
    '''
    # 70% of normal data for training
    num_normal = len(normal_data)
    train_size = int(0.7 * num_normal)
    train_data = normal_data[:train_size]

    # 15% of normal data for validation
    val_size = int(0.15 * num_normal)
    val_normal_data = normal_data[train_size:train_size + val_size]
    val_attack_size = int(0.85 * len(attack_data))
    val_attack_data = attack_data[:val_attack_size]
    val_data = torch.cat((val_normal_data, val_attack_data), dim=0)
    # 15% of normal data for testing
    test_normal_data = normal_data[train_size + val_size:]
    test_attack_size = int(0.15 * len(attack_data))
    test_attack_data = attack_data[val_attack_size:val_attack_size + test_attack_size]
    test_data = torch.cat((test_normal_data, test_attack_data), dim=0)
    return train_data, val_data, test_data
    


def phase1_data_prepration(n):
    '''
    This function prepares the data for training and evaluation.
        Packets preprocessing:
            We should have a function that load the dataset in form of raw packets, and return it in shape of (m, n)
            And it should truncate/pad each packet to M bytes
            It is already implemented before, but we will just need to adapt it here.    
    '''
    def load_all_modules():
        # print('loading all necessary modules')
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        sheet1_codes_path = os.path.abspath(os.path.join(curr_dir, '..', '..','Assignment1','Abdelaziz_Codes' ,'Sheet1_codes'))
        sys.path.append(sheet1_codes_path)
        
    
    def pad_or_truncate_packet(packet, n):
        '''
            This function pads or truncates a packet to have exactly n bytes.
            Input:
                - packet: byte array of the packet
            Output:
                - processed_packet: byte array of the packet with exactly n bytes
        '''
        if len(packet) > n:
            return packet[:n]
        elif len(packet) < n:
            return packet + bytes(n - len(packet))
        else:
            return packet


    def load_and_label_data(n):
        normal_pcap_path = "../../DataSets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set"
        attacked_pcap_path  = "../../DataSets/2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks"
        load_all_modules()
        from utilities import generate_bytes_array_from_packet_list

        # check if .npy files already exist, read it if yes, else generate it from pcap
        print(f"[*] Loading normal packets...")
        if os.path.exists("all_packets_control.npy"):
            data_normal_arrays = np.load("all_packets_control.npy", allow_pickle=True)
        else:
            data_normal_arrays = generate_bytes_array_from_packet_list(normal_pcap_path, pad = False, label = 'control')
    

        print(f"[*] Loading attacked packets...")
        data_attacked = []
        if os.path.exists("all_packets_attack.npy"):
            data_attacked = np.load("all_packets_attack.npy", allow_pickle=True)
        else:
            data_attacked = generate_bytes_array_from_packet_list(attacked_pcap_path, pad = False, label = 'attack')
        
        labeled_data_normal =[ ]
        labeled_data_attack =[ ]
        for array in data_normal_arrays:
            for pkt in array:
                labeled_data_normal.append( (pad_or_truncate_packet(pkt, n), 'normal') ) 

        if TESTING:
            for array in data_attacked[-2]:# this logic should be the same as normal.
                # for pkt in array:
                labeled_data_attack.append( (pad_or_truncate_packet(pkt, n), 'attack') )
        else:
            for array in data_attacked:
                for pkt in array:
                    labeled_data_attack.append( (pkt, 'attack') )
        return labeled_data_normal, labeled_data_attack

    load_all_modules()
    return load_and_label_data(n)

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
    num_samples = len(data) // m
    data_tensor = torch.zeros((num_samples, 1, m, n), dtype=torch.float32)
    for i in range(num_samples):
        for j in range(m):
            packet, label = data[i * m + j]
            data_tensor[i, 0, j, :] = torch.tensor(np.frombuffer(packet, dtype=np.uint8), dtype=torch.float32)
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
def phase4_custom_generator_loss():
    """
    Phase 4: Custom Generator Loss sanity check
    Logic:
        - Extract discriminator features for real and fake samples
        - Compute feature-matching loss using mean feature vectors
        - Do NOT use discriminator output
    """
    print("[*] Phase 4: Custom Generator Loss sanity checks")

    # 1. Create dummy batches
    batch_size = 8
    m, n = 10, 50
    x_real = torch.randn(batch_size, 1, m, n)
    x_fake = torch.randn(batch_size, 1, m, n)

    # 2. Instantiate discriminator
    D = Discriminator(input_shape=(1, m, n))

    # 3. Freeze discriminator
    for p in D.parameters():
        p.requires_grad = False

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

        loss = torch.mean((mean_real - mean_fake) ** 2)  # L2 -> higher punishment
        return loss

    loss = feature_matching_loss(D, x_real, x_fake)

    # 5. Sanity checks
    print("[*] Feature matching loss value:", loss.item())

    assert isinstance(loss.item(), float), "Loss is not scalar"
    assert loss.item() >= 0.0, "Loss must be non-negative"

    print("[✓] Phase 4 sanity checks passed successfully")
    print("-----------------------------------------------\n")

# ---------------- Phase 5: GAN Training Loop ---------------- #
def train_discriminator_step(D, G, x_real, optimizer_D, m, n):
    D.train()
    G.eval()  # generator is NOT updated here

    batch_size = x_real.size(0)
    device = x_real.device

    optimizer_D.zero_grad()

    # --- Real samples ---
    y_real = torch.ones(batch_size, 1, device=device)
    pred_real = D(x_real)
    loss_real = F.binary_cross_entropy(pred_real, y_real)

    # --- Fake samples ---
    z = torch.randn(batch_size, m * n, device=device)
    with torch.no_grad():  # do NOT update G here
        x_fake = G(z)

    y_fake = torch.zeros(batch_size, 1, device=device)
    pred_fake = D(x_fake)
    loss_fake = F.binary_cross_entropy(pred_fake, y_fake)

    # --- Total loss ---
    loss_D = loss_real + loss_fake
    loss_D.backward()
    optimizer_D.step()

    return loss_D.item()

def train_generator_step(D, G, x_real, optimizer_G, m, n):
    G.train()
    D.eval()  # discriminator is frozen here

    # Freeze discriminator parameters
    for p in D.parameters():
        p.requires_grad = False

    optimizer_G.zero_grad()

    batch_size = x_real.size(0)
    device = x_real.device

    # Generate fake samples
    z = torch.randn(batch_size, m * n, device=device)
    x_fake = G(z)

    # Extract features
    f_real = D.extract_features(x_real)
    f_fake = D.extract_features(x_fake)

    # Feature-matching loss (L2)
    mean_real = f_real.mean(dim=0)
    mean_fake = f_fake.mean(dim=0)
    loss_G = torch.mean((mean_real - mean_fake) ** 2)

    loss_G.backward()
    optimizer_G.step()

    # Unfreeze discriminator
    for p in D.parameters():
        p.requires_grad = True

    return loss_G.item()

def train_gan(
    D,
    G,
    train_loader,
    epochs,
    optimizer_D,
    optimizer_G,
    m,
    n,
    device
):
    D.to(device)
    G.to(device)

    for epoch in range(epochs):
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0

        for x_real in train_loader:
            x_real = x_real.to(device)

            # --- Discriminator step ---
            loss_D = train_discriminator_step(
                D, G, x_real, optimizer_D, m, n
            )

            # --- Generator step ---
            loss_G = train_generator_step(
                D, G, x_real, optimizer_G, m, n
            )

            epoch_loss_D += loss_D
            epoch_loss_G += loss_G

        # --- Logging ---
        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"D loss: {epoch_loss_D:.4f} | "
            f"G feature loss: {epoch_loss_G:.6f}"
        )

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



from tqdm import tqdm


def train_gan_on_training_data(
    training_data,
    D,
    G,
    m,
    n,
    epochs=30,
    batch_size=64,
    lr_D=1e-4,
    lr_G=1e-4,
    device="cpu",
    save_dir="models"
):
    """
    Phase 5: Full GAN training using ONLY training_data (normal samples)
    With detailed verbosity (epoch + batch progress).

    Inputs:
        - training_data: Tensor (N, 1, m, n)
        - D: Discriminator (fresh instance)
        - G: Generator (fresh instance)
    """

    # --------------------------------------------------
    # DataLoader
    # --------------------------------------------------
    train_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    num_batches = len(train_loader)

    # --------------------------------------------------
    # Optimizers
    # --------------------------------------------------
    optimizer_D = optim.Adam(D.parameters(), lr=lr_D)
    optimizer_G = optim.Adam(G.parameters(), lr=lr_G)

    D.to(device)
    G.to(device)

    print("\n[*] Starting GAN training")
    print(f"    Epochs: {epochs}")
    print(f"    Batch size: {batch_size}")
    print(f"    Training samples: {len(training_data)}")
    print(f"    Number of batches per epoch: {num_batches}")
    print("--------------------------------------------------")

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(epochs):
        D.train()
        G.train()

        epoch_loss_D = 0.0
        epoch_loss_G = 0.0

        print(f"\n[*] Epoch {epoch + 1}/{epochs}")

        progress_bar = tqdm(
            enumerate(train_loader),
            total=num_batches,
            desc=f"Epoch {epoch + 1}",
            leave=False
        )

        for batch_idx, x_real in progress_bar:
            x_real = x_real.to(device)
            curr_batch_size = x_real.size(0)

            # ============================
            # 1. Discriminator step
            # ============================
            optimizer_D.zero_grad()

            # Real samples
            y_real = torch.ones(curr_batch_size, 1, device=device)
            pred_real = D(x_real)
            loss_real = F.binary_cross_entropy(pred_real, y_real)

            # Fake samples
            z = torch.randn(curr_batch_size, m * n, device=device)
            with torch.no_grad():
                x_fake = G(z)

            y_fake = torch.zeros(curr_batch_size, 1, device=device)
            pred_fake = D(x_fake)
            loss_fake = F.binary_cross_entropy(pred_fake, y_fake)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            # ============================
            # 2. Generator step (Feature Matching)
            # ============================
            for p in D.parameters():
                p.requires_grad = False

            optimizer_G.zero_grad()

            z = torch.randn(curr_batch_size, m * n, device=device)
            x_fake = G(z)

            f_real = D.extract_features(x_real)
            f_fake = D.extract_features(x_fake)

            mean_real = f_real.mean(dim=0)
            mean_fake = f_fake.mean(dim=0)

            loss_G = torch.mean((mean_real - mean_fake) ** 2)
            loss_G.backward()
            optimizer_G.step()

            for p in D.parameters():
                p.requires_grad = True

            # ============================
            # Logging
            # ============================
            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()

            if batch_idx % 50 == 0:
                progress_bar.set_postfix({
                    "D_loss": f"{loss_D.item():.4f}",
                    "G_loss": f"{loss_G.item():.2e}"
                })

        # --------------------------------------------------
        # Epoch summary
        # --------------------------------------------------
        avg_loss_D = epoch_loss_D / num_batches
        avg_loss_G = epoch_loss_G / num_batches

        print(
            f"[Epoch {epoch + 1} DONE] "
            f"Avg D loss: {avg_loss_D:.4f} | "
            f"Avg G feature loss: {avg_loss_G:.2e}"
        )

    # --------------------------------------------------
    # Save trained models
    # --------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)

    torch.save(D.state_dict(), os.path.join(save_dir, "discriminator.pth"))
    torch.save(G.state_dict(), os.path.join(save_dir, "generator.pth"))

    print("\n[✓] Training finished successfully")
    print(f"[✓] Models saved to '{save_dir}/'")
