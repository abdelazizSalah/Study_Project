# known_ports.py
KNOWN_PORTS = {
    102:    "s7comm",
    502:    "modbus",
    20000:  "dnp3",
    44818:  "ethernetip",
    34964:  "profinet",
    123: "NTP",
    137: "NBNS",
    138: "NBDS",
    546: "DHCPv6 Client",
    547: "DHCPv6 Server",
    5353: "mDNS",
    5355: "llmnr"

}


ATTACK_LABELS = [
    "ConveyorBeltOff",  #1
    "ConveyorBeltOn",   #2
    "ConveyorBeltGateChangeDirection", #3
    "ConveyorBeltReset",
    "WaterTankOff",
    "WaterTankOnAuto",
    "WaterTankOnManu",
    "ReactorOff",
    "ReactorOn",
    "ChangeUpperThreshold",
    "ChangeLowerThreshold",
    "GlobalReset",
    "EmergencyStop",

    # with flooding
    "WaterTankOnManu_Flooding",
    "ConveyorBeltGateChangeDirection_Flooding",
    "ReactorOn_Flooding",
    "WaterTankOnAuto_Flooding",
    "ConveyorBeltOff_Flooding",
    "ReactorOff_Flooding",
    "WaterTankOff_Flooding",
    "ConveyorBeltOn_Flooding",
    "ChangeUpperThreshold_Flooding",
]

ALL_POSSIBLE_LABELS = ["CONTROL"] + ATTACK_LABELS


""""
Distribution among attack type from evaluation:

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