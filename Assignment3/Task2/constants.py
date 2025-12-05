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
