class PhysicalProperty:
    def __init__(self, prop: str):
        self.prop = prop
        self.name = self.get_name()
        self.metric = self.get_metric()
        self.text_metric = self.get_text_metric()

    def get_name(self):
        if self.prop == "h":
            return "Enthalpy"
        elif self.prop == "g":
            return "Gibbs Free Energy"
        elif self.prop == "h298":
            return "Standard Enthalpy of Formation"
        elif self.prop == "s":
            return "Entropy"
        elif self.prop == "cp":
            return "Heat Capacity"
        elif self.prop == "bp":
            return "Boiling Point"
        elif self.prop == "tc":
            return "Critical Temperature"
        elif self.prop == "pc":
            return "Critical Pressure"
        elif self.prop == "mu":
            return "Dynamic Viscosity"
        elif self.prop == "af":
            return "Acentric Factor"
        elif self.prop == "nu":
            return "Kinematic Viscosity"
        elif self.prop == "st":
            return "Surface Tension"
        elif self.prop == "ea":
            return "Activation Energy"
        else:
            return self.prop

    def get_metric(self):
        if self.prop == "h":
            return "kJ mol$^{-1}$"
        elif self.prop == "g":
            return "kJ mol$^{-1}$"
        elif self.prop == "h298":
            return "kJ mol$^{-1}$"
        elif self.prop == "s":
            return "J mol$^{-1}$ K$^{-1}$"
        elif self.prop == "cp":
            return "J mol$^{-1}$ K$^{-1}$"
        elif self.prop == "bp":
            return "K"
        elif self.prop == "tc":
            return "K"
        elif self.prop == "pc":
            return "bar"
        elif self.prop == "mu":
            return "Pa.s"
        elif self.prop == "af":
            return "-"
        elif self.prop == "nu":
            return "m$^{2}$s^{-1}"
        elif self.prop == "st":
            return "N m^{-1}"
        elif self.prop == "ea":
            return "kJ mol$^{-1}$"
        else:
            return "-"

    def get_text_metric(self):
        if self.prop == "h":
            return "kJ/mol"
        elif self.prop == "g":
            return "kJ/mol"
        elif self.prop == "h298":
            return "kJ/mol"
        elif self.prop == "s":
            return "J/mol.K"
        elif self.prop == "cp":
            return "J/mol.K"
        elif self.prop == "bp":
            return "K"
        elif self.prop == "tc":
            return "K"
        elif self.prop == "pc":
            return "bar"
        elif self.prop == "mu":
            return "Pa.s"
        elif self.prop == "af":
            return "-"
        elif self.prop == "nu":
            return "m2/s"
        elif self.prop == "st":
            return "N/m"
        elif self.prop == "ea":
            return "kJ/mol"
        else:
            return "-"
