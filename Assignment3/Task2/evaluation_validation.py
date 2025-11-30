import numpy as np


""""


RAW: (3657746 total)

CONTROL: 870682
ChangeLowerThreshold: 264
ChangeUpperThreshold: 152
ChangeUpperThreshold_Flooding: 79180
ConveyorBeltGateChangeDirection: 32
ConveyorBeltGateChangeDirection_Flooding: 92760
ConveyorBeltOff: 28
ConveyorBeltOff_Flooding: 236290
ConveyorBeltOn: 72
ConveyorBeltOn_Flooding: 495248
ConveyorBeltReset: 58
EmergencyStop: 8
GlobalReset: 16
ReactorOff: 16
ReactorOff_Flooding: 244581
ReactorOn: 114
ReactorOn_Flooding: 335590
WaterTankOff: 16
WaterTankOff_Flooding: 356464
WaterTankOnAuto: 40
WaterTankOnAuto_Flooding: 872374
WaterTankOnManu: 32
WaterTankOnManu_Flooding: 73729
"""


"""RE: (3185836 total)

CONTROL: 476344
ChangeLowerThreshold: 152
ChangeUpperThreshold: 88
ChangeUpperThreshold_Flooding: 76980
ConveyorBeltGateChangeDirection: 32
ConveyorBeltGateChangeDirection_Flooding: 90248
ConveyorBeltOff: 28
ConveyorBeltOff_Flooding: 229596
ConveyorBeltOn: 72
ConveyorBeltOn_Flooding: 481672
ConveyorBeltReset: 52
EmergencyStop: 8
GlobalReset: 16
ReactorOff: 16
ReactorOff_Flooding: 236896
ReactorOn: 112
ReactorOn_Flooding: 326404
WaterTankOff: 16
WaterTankOff_Flooding: 346668
WaterTankOnAuto: 40
WaterTankOnAuto_Flooding: 848676
WaterTankOnManu: 32
WaterTankOnManu_Flooding: 71688

Process finished with exit code 0
"""


def print_size_of_different_attack_types(labels):

    unique, counts = np.unique(labels, return_counts=True)

    for u, c in zip(unique, counts):
        print(f"{u}: {c}")

