from sklearn.model_selection import train_test_split
import numpy as np

#use this to test models in task 2b, because task 2a is .... *****
def create_balanced_train_test_indices(labels, control_label="CONTROL", test_size=0.20, random_state=42):


    # 1. Indizes nach Klasse trennen
    all_indices = np.arange(len(labels))
    control_indices = all_indices[labels == control_label]
    attack_indices = all_indices[labels != control_label]

    print(f"Total Control Samples: {len(control_indices)}")
    print(f"Total Attack Samples: {len(attack_indices)}")

    # 2. Aufteilung der CONTROL-Indizes (80% Train, 20% Test)
    train_control_indices, test_control_indices = train_test_split(
        control_indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # 3. Aufteilung der ATTACK-Indizes (80% Train, 20% Test)
    # Wichtig: Wir teilen die Attacken jetzt auch auf!
    train_attack_indices, test_attack_indices = train_test_split(
        attack_indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # 4. Finales TRAININGS-Set (Control + Attack)
    train_indices = np.concatenate((train_control_indices, train_attack_indices))

    # Optional: Mischen der finalen Trainings-Indizes
    np.random.seed(random_state + 1)  # Verwende einen anderen Seed, um Reproduzierbarkeit zu wahren
    np.random.shuffle(train_indices)

    # 5. Finales TEST-Set (Control + Attack)
    test_indices = np.concatenate((test_control_indices, test_attack_indices))

    # Optional: Mischen der finalen Test-Indizes
    np.random.seed(random_state + 2)
    np.random.shuffle(test_indices)

    # 6. Kontrolle der Größen
    print("\n--- Finaler Split ---")
    print(
        f"TRAIN: Control {len(train_control_indices)} + Attack {len(train_attack_indices)} = {len(train_indices)} Samples")
    print(
        f"TEST: Control {len(test_control_indices)} + Attack {len(test_attack_indices)} = {len(test_indices)} Samples")

    return train_indices, test_indices


def create_balanced_subsampled_indices(labels, control_label="CONTROL", test_size=0.20, random_state=42):
    """
    Erstellt Indizes für eine normale SVM, indem die Control-Klasse per Undersampling
    reduziert wird, um eine 50:50 Balance zwischen Control und Attack zu erreichen.

    Args:
        labels (np.ndarray): Array der Labels (Text oder numerisch).
        control_label (str): Der Wert der Control-Klasse.
        test_size (float): Der Anteil der Daten, der ins Testset geht (1/5 = 0.20).
        random_state (int): Seed für die Reproduzierbarkeit.

    Returns:
        tuple: (train_indices, test_indices)
    """

    # 1. Indizes nach Klasse trennen
    all_indices = np.arange(len(labels))
    control_indices = all_indices[labels == control_label]
    attack_indices = all_indices[labels != control_label]

    print(f"Ursprüngliche Balance: Control={len(control_indices)}, Attack={len(attack_indices)}")

    # 2. Undersampling der Control-Indizes
    # Ziel: Die Control-Indizes auf die Größe der Attack-Indizes reduzieren.
    N_attack = len(attack_indices)

    # Sicherstellen, dass das Mischen reproduzierbar ist
    np.random.seed(random_state)

    # Control-Indizes mischen und nur N_attack auswählen
    np.random.shuffle(control_indices)

    # Den Teil der Control-Indizes auswählen, der dem Undersampling entspricht
    subsampled_control_indices = control_indices[:N_attack]

    # 3. Kontroll-Ausgabe nach Undersampling
    N_total_subsampled = len(subsampled_control_indices) + N_attack
    print(f"Nach Undersampling: {len(subsampled_control_indices)} Control und {N_attack} Attacken.")

    # 4. Gesamter unterabgetasteter Index-Pool
    balanced_indices = np.concatenate((subsampled_control_indices, attack_indices))

    # 5. Mischen des gesamten unterabgetasteten Pools VOR dem Split
    np.random.shuffle(balanced_indices)

    # 6. Aufteilung des balancierten Pools in Train und Test (4/5 zu 1/5)
    train_indices, test_indices = train_test_split(
        balanced_indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True  # Wichtig: Der Pool ist bereits gemischt, wird aber hier noch einmal gemischt
    )

    # 7. Kontrolle der finalen Balance (sollte sehr nah an 50:50 sein)

    # Die Balance der finalen Sets manuell prüfen:
    train_controls = np.sum(labels[train_indices] == control_label)
    train_attacks = len(train_indices) - train_controls

    test_controls = np.sum(labels[test_indices] == control_label)
    test_attacks = len(test_indices) - test_controls

    print("\n--- Finaler Balancierter Split (Sollte 50:50 sein) ---")
    print(f"TRAIN: Control {train_controls} + Attack {train_attacks} = {len(train_indices)} Samples")
    print(f"TEST: Control {test_controls} + Attack {test_attacks} = {len(test_indices)} Samples")

    return train_indices, test_indices

